# Developed by Kenza Amara and Gyri Reiersen

import os
import PIL
import exifread
import lxml.etree as etree
import numpy as np
import pandas as pd
import slidingwindow
import skimage.color
import skimage.io
import cv2

from PIL import Image
PIL.Image.MAX_IMAGE_PIXELS = None


# Read .kml files for Orthomosaics RGB

def read_kml(dir, file):
    """
    Extract the position of the image in file
    Returns:
        list of the name of the file and the coordinates bounds
    """
    kml_file = os.path.join(dir, file)
    x = etree.parse(kml_file)
    name = file.replace('.kml', '')
    for el in x.iter(tag="{*}south"):
        lat_min = float(el.text)
    for el in x.iter(tag="{*}north"):
        lat_max = float(el.text)
    for el in x.iter(tag="{*}west"):
        lon_min = float(el.text)
    for el in x.iter(tag="{*}east"):
        lon_max = float(el.text)

    return ([name, lat_min, lat_max, lon_min, lon_max])


def read_orthomosaics(dir):
    data = []
    for file in os.listdir(dir):
        if file.endswith('.kml'):
            data.append(read_kml(dir, file))
    return (pd.DataFrame(data=data, columns=['name', 'lat_min', 'lat_max', 'lon_min', 'lon_max']))


# Split Orthomosaics in 4000x4000 tiles

def image_name_from_path(image_path):
    """Convert path to image name for use in indexing."""
    image_name = os.path.basename(image_path)
    image_name = os.path.splitext(image_name)[0]

    return image_name


def compute_windows(numpy_image, patch_size, patch_overlap):
    """Create a sliding window object from a raster tile.

    Args:
        numpy_image (numpy array): Raster object as numpy array to cut into crops

    Returns:
        windows (list): a sliding windows object
    """

    if patch_overlap > 1:
        raise ValueError("Patch overlap {} must be between 0 - 1".format(patch_overlap))

    # Generate overlapping sliding windows
    windows = slidingwindow.generate(numpy_image,
                                     slidingwindow.DimOrder.HeightWidthChannel,
                                     patch_size, patch_overlap)

    return (windows)


def save_crop(base_dir, image_name, index, tile_position, crop):
    """Save window crop as image file to be read by PIL.

    Filename should match the image_name + window index
    """
    # create dir if needed
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    im = Image.fromarray(crop)
    image_basename = os.path.splitext(image_name)[0]
    x0, y0, x1, y1 = tile_position
    filename = "{}/{}_{}_{}_{}_{}_{}.png".format(base_dir, image_basename, index, x0, y0, x1, y1)
    im.save(filename)

    return filename


def save_crop_annotations(base_dir, image_name, index, crop):
    """Save window crop as image file to be read by PIL.

    Filename should match the image_name + window index
    """
    # create dir if needed
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    im = Image.fromarray(crop)
    image_basename = os.path.splitext(image_name)[0]
    filename = "{}/{}_{}.png".format(base_dir, image_basename, index)
    im.save(filename)

    return filename


def select_annotations(annotations, windows, index, allow_empty=False):
    """Select annotations that overlap with selected image crop.

    Args:
        image_name (str): Name of the image in the annotations file to lookup.
        annotations_file: path to annotations file in
            the format -> image_path, xmin, ymin, xmax, ymax, label
        windows: A sliding window object (see compute_windows)
        index: The index in the windows object to use a crop bounds
        allow_empty (bool): If True, allow window crops
            that have no annotations to be included

    Returns:
        selected_annotations: a pandas dataframe of annotations
    """

    # Window coordinates - with respect to tile
    window_xmin, window_ymin, w, h = windows[index].getRect()
    window_xmax = window_xmin + w
    window_ymax = window_ymin + h

    # buffer coordinates a bit to grab boxes that might start just against
    # the image edge. Don't allow boxes that start and end after the offset
    offset = 40
    selected_annotations = annotations[(annotations.xmin > (window_xmin - offset)) &
                                       (annotations.xmin < (window_xmax)) &
                                       (annotations.xmax >
                                        (window_xmin)) & (annotations.ymin >
                                                          (window_ymin - offset)) &
                                       (annotations.xmax <
                                        (window_xmax + offset)) & (annotations.ymin <
                                                                   (window_ymax)) &
                                       (annotations.ymax >
                                        (window_ymin)) & (annotations.ymax <
                                                          (window_ymax + offset))].copy()

    # change the image name
    image_name = os.path.splitext("{}".format(annotations.image_path.unique()[0]))[0]
    image_basename = os.path.splitext(image_name)[0]
    selected_annotations.image_path = "{}_{}.png".format(image_basename, index)

    # If no matching annotations, return a line with the image name, but no records
    if selected_annotations.empty:
        if allow_empty:
            selected_annotations = pd.DataFrame(
                ["{}_{}.png".format(image_basename, index)], columns=["image_path"])
            selected_annotations["xmin"] = ""
            selected_annotations["ymin"] = ""
            selected_annotations["xmax"] = ""
            selected_annotations["ymax"] = ""
            selected_annotations["label"] = ""
        else:
            return None
    else:
        # update coordinates with respect to origin
        selected_annotations.xmax = (selected_annotations.xmin - window_xmin) + (
                selected_annotations.xmax - selected_annotations.xmin)
        selected_annotations.xmin = (selected_annotations.xmin - window_xmin)
        selected_annotations.ymax = (selected_annotations.ymin - window_ymin) + (
                selected_annotations.ymax - selected_annotations.ymin)
        selected_annotations.ymin = (selected_annotations.ymin - window_ymin)

        # cut off any annotations over the border.
        selected_annotations.loc[selected_annotations.xmin < 0, "xmin"] = 0
        selected_annotations.loc[selected_annotations.xmax > w, "xmax"] = w
        selected_annotations.loc[selected_annotations.ymin < 0, "ymin"] = 0
        selected_annotations.loc[selected_annotations.ymax > h, "ymax"] = h

    return selected_annotations


def split_raster(path_to_raster,
                 base_dir="images",
                 patch_size=400,
                 patch_overlap=0.05):
    """Divide a large tile into smaller arrays. Each crop will be saved to
    file.

    Args:
        path_to_raster: (str): Path to a tile that can be read by rasterio on disk
        base_dir (str): Where to save the annotations and image
            crops relative to current working dir
        patch_size (int): Maximum dimensions of square window
        patch_overlap (float): Percent of overlap among windows 0->1

    """
    # Load raster as image
    raster = Image.open(path_to_raster)
    numpy_image = np.array(raster)
    numpy_image = numpy_image[:, :, :3]

    # Check that its 3 band
    bands = numpy_image.shape[2]
    if not bands == 3:
        raise IOError("Input file {} has {} bands. DeepForest only accepts 3 band RGB "
                      "rasters in the order (height, width, channels). "
                      "If the image was cropped and saved as a .jpg, "
                      "please ensure that no alpha channel was used.".format(
            path_to_raster, bands))

    # Check that patch size is greater than image size
    height = numpy_image.shape[0]
    width = numpy_image.shape[1]
    if any(np.array([height, width]) < patch_size):
        raise ValueError("Patch size of {} is larger than the image dimensions {}".format(
            patch_size, [height, width]))

    # Compute sliding window index
    windows = compute_windows(numpy_image, patch_size, patch_overlap)

    # Get image name for indexing
    image_name = os.path.basename(path_to_raster)

    for index, window in enumerate(windows):
        # Crop image
        tile_pos = tile_xy(windows[index].indices())
        crop = numpy_image[windows[index].indices()]
        image_path = save_crop(base_dir, image_name, index, tile_pos, crop)

    return


def split_raster_annotations(path_to_raster,
                             annotations_file,
                             base_dir=".",
                             patch_size=400,
                             patch_overlap=0.05,
                             allow_empty=False):
    """Divide a large tile into smaller arrays. Each crop will be saved to
    file.

    Args:
        path_to_raster: (str): Path to a tile that can be read by rasterio on disk
        annotations_file (str): Path to annotations file (with column names)
            data in the format -> image_path, xmin, ymin, xmax, ymax, label
        base_dir (str): Where to save the annotations and image
            crops relative to current working dir
        patch_size (int): Maximum dimensions of square window
        patch_overlap (float): Percent of overlap among windows 0->1
        allow_empty: If True, include images with no annotations
            to be included in the dataset

    Returns:
        A pandas dataframe with annotations file for training.
    """
    # Load raster as image
    raster = Image.open(path_to_raster)
    numpy_image = np.array(raster)
    numpy_image = numpy_image[:, :, :3]

    # Check that its 3 band
    bands = numpy_image.shape[2]
    if not bands == 3:
        raise IOError("Input file {} has {} bands. DeepForest only accepts 3 band RGB "
                      "rasters in the order (height, width, channels). "
                      "If the image was cropped and saved as a .jpg, "
                      "please ensure that no alpha channel was used.".format(
            path_to_raster, bands))

    # Check that patch size is greater than image size
    height = numpy_image.shape[0]
    width = numpy_image.shape[1]
    if any(np.array([height, width]) < patch_size):
        raise ValueError("Patch size of {} is larger than the image dimensions {}".format(
            patch_size, [height, width]))

    # Compute sliding window index
    windows = compute_windows(numpy_image, patch_size, patch_overlap)

    # Get image name for indexing
    image_name = os.path.basename(path_to_raster)

    # Load annotations file and coerce dtype
    annotations = pd.read_csv(annotations_file)

    # open annotations file
    image_annotations = annotations[annotations.image_path == image_name].copy()

    # Sanity checks
    if image_annotations.empty:
        raise ValueError(
            "No image names match between the file:{} and the image_path: {}. "
            "Reminder that image paths should be the relative "
            "path (e.g. 'image_name.tif'), not the full path "
            "(e.g. path/to/dir/image_name.tif)".format(annotations_file, image_name))

    if not annotations.shape[1] == 6:
        raise ValueError("Annotations file has {} columns, should have "
                         "format image_path, xmin, ymin, xmax, ymax, label".format(
            annotations.shape[1]))

    annotations_files = []
    for index, window in enumerate(windows):

        # Crop image
        crop = numpy_image[windows[index].indices()]

        # Find annotations, image_name is the basename of the path
        crop_annotations = select_annotations(image_annotations, windows, index,
                                              allow_empty)

        # If empty images not allowed, select annotations returns None
        if crop_annotations is not None:
            # save annotations
            annotations_files.append(crop_annotations)

            # save image crop
            save_crop_annotations(base_dir, image_name, index, crop)
    if len(annotations_files) == 0:
        raise ValueError(
            "Input file has no overlapping annotations and allow_empty is {}".format(
                allow_empty))

    annotations_files = pd.concat(annotations_files)

    # Checkpoint csv files, useful for parallelization
    # Use filename of the raster path to save the annotations
    image_basename = os.path.splitext(image_name)[0]
    file_path = image_basename + ".csv"
    file_path = os.path.join(base_dir, file_path)
    annotations_files.to_csv(file_path, index=False, header=False)

    return annotations_files


def tile_xy(win):
    """
    Convert the window position to the x,y pixel position of the tile on the image
    The window position take origin on the top-left of the image
    The x,y position takes origin on the top-left corner of the image
    """
    x_min = win[1].start
    x_max = win[1].stop
    y_min = win[0].start
    y_max = win[0].stop
    return (x_min, y_min, x_max, y_max)


# Rescale lat and lon

def get_bounds(df):
    min_lat = df.lat.min()
    max_lat = df.lat.max()
    min_lon = df.lon.min()
    max_lon = df.lon.max()
    bounds = [min_lon, min_lat, max_lon, max_lat]
    return (bounds)


def get_scale(bounds_drone, bounds_ground):
    """
    Computes the scale between drone data and ground data for rescaling
    Args:
        bounds_drone (numpy array): bounded coordinates of drone image
        bounds_ground (numpy array): bounded coordinates of ground site

    Returns:
        scale (float): ratio between distance in drone and ground data (scale = (drone dist/ground dist) >1)
    """
    min_lon, min_lat, max_lon, max_lat = bounds_drone[0], bounds_drone[1], bounds_drone[2], bounds_drone[3]
    g_min_lon, g_min_lat, g_max_lon, g_max_lat = bounds_ground[0], bounds_ground[1], bounds_ground[2], bounds_ground[3]
    r_lat = (max_lat - min_lat) / (g_max_lat - g_min_lat)
    r_lon = (max_lon - min_lon) / (g_max_lon - g_min_lon)
    scale = np.array([r_lat, r_lon])
    return (scale)


def rescale(X_drone, bounds, scale):
    """
    Rescale the positions of drone data (bounding boxes position)
    """
    min_lon, min_lat, max_lon, max_lat = bounds[0], bounds[1], bounds[2], bounds[3]

    # Center of points, defined as the center of the minimal rectangle
    # that contains all points.
    center_lat = (min_lat + max_lat) * .5
    center_lon = (min_lon + max_lon) * .5

    # Points scaled about center.
    X_drone[:, 0] = (X_drone[:, 0] - center_lat) / scale[0] + center_lat
    X_drone[:, 1] = (X_drone[:, 1] - center_lon) / scale[1] + center_lon
    return (X_drone)


# Extract Orthomosaics features

def ratio(size, min, max):
    delta = max - min
    r = delta / float(size)
    return (r)


def create_ortho_data(directory, save_dir):
    """
    Extract data from drone orthomosaics (dimensions (width, height), positions top-left,
    scale on x- and y-axis (ratio_x, ratio_y)).
    """
    ortho_features = read_orthomosaics(directory)

    ortho_dim = []
    for file in os.listdir(directory):
        if file.endswith('.tif'):
            # Open image file for reading (binary mode)
            path_to_raster = os.path.join(directory, file)
            f = open(path_to_raster, 'rb')

            # Return Exif tags
            tags = exifread.process_file(f)
            width = int(str(tags['Image ImageWidth']))
            height = int(str(tags['Image ImageLength']))
            name = file.replace('.tif', '')
            ortho_dim.append([name, width, height])
    ortho_dim = pd.DataFrame(data=ortho_dim, columns=['name', 'width', 'height'])

    ortho_data = pd.merge(ortho_features, ortho_dim, on='name')
    ortho_data['ratio_x_init'] = ortho_data.apply(lambda x: ratio(x.width, x.lon_min, x.lon_max), axis=1)
    ortho_data['ratio_y_init'] = ortho_data.apply(lambda x: ratio(x.height, x.lat_min, x.lat_max), axis=1)
    ortho_data.to_csv(save_dir, index=False)
    return ortho_data


def rescale_bounds(ortho_data, field_data, list_sites):
    for site_name in list_sites:
        # Compute Scale
        # Bounds Drone
        site = ortho_data[ortho_data.name == site_name]
        max_lat = site['lat_max'].values[0]
        min_lat = site['lat_min'].values[0]
        max_lon = site['lon_max'].values[0]
        min_lon = site['lon_min'].values[0]
        bounds_drone = [min_lon, min_lat, max_lon, max_lat]

        # Bounds Ground
        ground_files = field_data[field_data.site == site_name]
        ground_files_site = ground_files.loc[(min_lat < ground_files.lat)
                                        & (ground_files.lat < max_lat)
                                        & (min_lon < ground_files.lon)
                                       & (ground_files.lon < max_lon)]
        bounds_ground = get_bounds(ground_files_site)

        scale = get_scale(bounds_drone, bounds_ground)

        # Store scale for each site in ortho_data
        ortho_data.loc[ortho_data.name == site_name, 'scale_lat'] = scale[0]
        ortho_data.loc[ortho_data.name == site_name, 'scale_lon'] = scale[1]

        # Store scale for each site in ortho_data
        ortho_data.loc[ortho_data.name == site_name, 'center_lat'] = (min_lat + max_lat) * .5
        ortho_data.loc[ortho_data.name == site_name, 'center_lon'] = (min_lon + max_lon) * .5

    # Redefine ratios
    ortho_data['ratio_y'] = ortho_data['ratio_y_init']/ortho_data['scale_lat']
    ortho_data['ratio_x'] = ortho_data['ratio_x_init']/ortho_data['scale_lon']

    # Rename
    ortho_data = ortho_data.rename(columns={'lat_min': 'lat_min_init',
                                            'lat_max': 'lat_max_init',
                                            'lon_min': 'lon_min_init',
                                            'lon_max': 'lon_max_init'})

    # Rescale drone bounds
    ortho_data['lat_min'] = (ortho_data['lat_min_init'] - ortho_data['center_lat'])/ortho_data['scale_lat'] + ortho_data['center_lat']
    ortho_data['lat_max'] = (ortho_data['lat_max_init'] - ortho_data['center_lat'])/ortho_data['scale_lat'] + ortho_data['center_lat']
    ortho_data['lon_min'] = (ortho_data['lon_min_init'] - ortho_data['center_lon'])/ortho_data['scale_lon'] + ortho_data['center_lon']
    ortho_data['lon_max'] = (ortho_data['lon_max_init'] - ortho_data['center_lon'])/ortho_data['scale_lon'] + ortho_data['center_lon']

    return ortho_data



if __name__ == '__main__':
    directory = "data/wwf_ecuador/RGB Orthomosaics"
    save_dir = 'data/tiles'

    ortho_data = create_ortho_data(directory, save_dir)

    # Split images into tiles
    for file in os.listdir(directory):
        if file.endswith('.tif'):
            # Open image file for reading (binary mode)
            path_to_raster = os.path.join(directory, file)
            name = file.replace('.tif', '')

            tiles_dir = os.path.join(save_dir, name)
            if not os.path.exists(tiles_dir):
                os.makedirs(tiles_dir)

            split_raster(path_to_raster, base_dir=tiles_dir, patch_size=4000, patch_overlap=0.05)
