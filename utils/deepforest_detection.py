# Deepforest Preprocessing model
"""The preprocessing module is used to reshape data into format suitable for
training or prediction.

For example cutting large tiles into smaller images.
"""

# Run DeepForest on each tile: it draws bounding boxes (xmin, ymin, xmax, ymax, score, label) around each detected tree
# Gather all bounding boxes in an annotations.csv file.

# Required deepforest==0.3.8 or earlier

import os
from deepforest import deepforest
from deepforest import get_data
import pandas as pd
import pyproj


# Draw Bounding boxes with DeepForest on detected tree on the 4000x4000 tiles

def test_predict_boxes(image_path, test_model, score_threshold):
    """
    Returns the tree detections by DeepForest model.
    Args:
        image_path (str): path to image
        test_model : algorithm to detect trees
        score_threshold (float): minimum probability score to be included in final boxes, ranging from 0 to 1.

    Returns:
        boxes : bounding boxes around each detected tree on the image

    """
    # Predict test image and return boxes
    boxes = test_model.predict_image(image_path=get_data(image_path),
                                     show=False,
                                     return_plot=False,
                                     score_threshold=score_threshold)

    # Returns a 6 column numpy array, xmin, ymin, xmax, ymax, score, label
    assert boxes.shape[1] == 6
    boxes = boxes.drop(boxes[boxes.score < score_threshold].index)
    return (boxes)


def get_annotations(image_path, test_model, score_threshold=0.1):
    """
    Creates pandas DataFrame containing the bounding boxes detected by the test_model of DeepForest.
    Args:
        image_path (str): path to image
        test_model : algorithm to detect trees
        score_threshold (float): minimum probability score to be included in final boxes, ranging from 0 to 1.

    Returns:
        tile_annotations (dataframe): bounding boxes coordinates in tile at path image_path
    """
    column_names = ['img_path', 'xmin', 'ymin', 'xmax', 'ymax', 'score']
    tile_annotations = pd.DataFrame(columns=column_names)

    base = os.path.basename(image_path)

    annotations = test_predict_boxes(image_path, test_model, score_threshold)
    if len(annotations) > 0:
        for index, row in annotations.iterrows():
            new_row = {'img_path': base, 'xmin': row.xmin, 'ymin': row.ymin, 'xmax': row.xmax, 'ymax': row.ymax,
                       'score': row.score}
            tile_annotations = tile_annotations.append(new_row, ignore_index=True)
    return (tile_annotations)


# Process Data

def expand_tile_features(img_path):
    """
    Extracts tile name and position.
    """
    img_features = os.path.splitext(img_path)[0]
    s = img_features.split('_')
    img_name = s[0]
    tile_index = int(s[1])
    tile_xmin = int(s[2])
    tile_ymin = int(s[3])
    tile_xmax = int(s[4])
    tile_ymax = int(s[5])
    return (img_name, tile_index, tile_xmin, tile_ymin, tile_xmax, tile_ymax)


def get_center(min, max):
    return ((min + max) / 2)


def convert_xy_tile_to_lonlat(name, tile_xmin, tile_ymin, x, y, ortho_data):
    """
    Converts x, y position in the tile (origin (0,0) at the top-left of the tile) to lon, lat
    Args:
        name (str): name of the site
        tile_xmin (int): tile position - left
        tile_ymin (int): tile position - top
        x, y (floats): position of a point
        ortho_data (DataFrame): information on individual orthomosaic scale and position (lat, lon)

    Returns:
        longitude and latitude of the point
    """
    row = ortho_data[ortho_data.name == name]
    Y = tile_ymin + y
    X = tile_xmin + x
    lat = row.lat_max.values[0] - Y * row.ratio_y.values[0]
    lon = row.lon_min.values[0] + X * row.ratio_x.values[0]
    return ([lon, lat])


def convert_lonlat_to_xy(name, lon, lat, ortho_data):
    """
        Converts lon, lat to X, Y position (zone UTM 17)
    """
    row = ortho_data[ortho_data.name == name]
    Y = (row.lat_max.values[0] - lat) / row.ratio_y.values[0]
    X = (lon - row.lon_min.values[0]) / row.ratio_x.values[0]
    return ([X, Y])


def convert_lonlat_to_xy_tile(name, tile_xmin, tile_ymin, lon, lat, ortho_data):
    """
        Converts lon, lat to x, y position in the tile (origin (0,0) at the top-left of the tile)
    """
    row = ortho_data[ortho_data.name == name]
    Y = (row.lat_max.values[0] - lat) / row.ratio_y.values[0]
    X = (lon - row.lon_min.values[0]) / row.ratio_x.values[0]
    y = Y - tile_ymin
    x = X - tile_xmin
    return ([x, y])


####################################
### Projection using Pyproj.proj ###
####################################

def convert_xy_to_lonlat_proj(name, tile_xmin, tile_ymin, x, y, ortho_data):
    """
        Converts x, y position in the tile (origin (0,0) at the top-left of the tile) to lon, lat
    """
    row = ortho_data[ortho_data.name == name]

    lon0, lat0 = row.lon_min.values[0], row.lat_max.values[0]
    x0, y0 = lonlat_to_xy(lon0, lat0)

    X = x0 + tile_xmin + x
    Y = y0 + tile_ymin + y

    lon, lat = xy_to_lonlat(X, Y)
    return ([lon, lat])


def convert_lonlat_to_xy_proj(name, lon, lat, ortho_data):
    """
    Converts lon, lat to x, y position in the tile (origin (0,0) at the top-left of the tile)
    """
    row = ortho_data[ortho_data.name == name]
    X, Y = lonlat_to_xy(lon, lat)
    lon0, lat0 = row.lon_min.values[0], row.lat_max.values[0]
    x0, y0 = lonlat_to_xy(lon0, lat0)

    x = X - x0
    y = Y - y0
    return ([x, y])


def xy_to_lonlat(x, y):
    proj_latlon = pyproj.Proj(proj='latlong', datum='WGS84')
    proj_xy = pyproj.Proj(proj="utm", zone=17, datum='WGS84')
    lonlat = pyproj.transform(proj_xy, proj_latlon, x, y)
    return lonlat[0], lonlat[1]


def lonlat_to_xy(lon, lat):
    proj_latlon = pyproj.Proj(proj='latlong', datum='WGS84')
    proj_xy = pyproj.Proj(proj="utm", zone=17, datum='WGS84')
    xy = pyproj.transform(proj_latlon, proj_xy, lon, lat)
    return xy[0], xy[1]


if __name__ == '__main__':

    # Load model
    test_model = deepforest.deepforest()
    test_model.use_release()

    # Split images and predict bounding box - return annotation files
    column_names = ['img_path', 'xmin', 'ymin', 'xmax', 'ymax', 'score']

    dir = os.getcwd()
    for folder in os.listdir("images"):
        annotations_files = pd.DataFrame(columns=column_names)
        for file in os.listdir("images/" + folder):
            tile_annotations = get_annotations(dir + "/images/" + folder + "/" + file, test_model)
            annotations_files = pd.concat([annotations_files, tile_annotations])
        annotations_files = annotations_files.reset_index(drop=True)
        file_path = 'annotations/{}_annotations.csv'.format(folder)
        annotations_files.to_csv(file_path, index=False, header=False)
        print('DeepForest Annotations are saved for site {}'.format(folder))

    # Process annotations

    annot_dir = 'annotations'
    column_names = ['img_path', 'xmin', 'ymin', 'xmax', 'ymax', 'score']

    for file in os.listdir(annot_dir):
        file_path = os.path.join(annot_dir, file)
        df = pd.read_csv(file_path, names=column_names)
        df['img_name'], df['tile_index'], df['tile_xmin'], df['tile_ymin'], df['tile_xmax'], df['tile_ymax'] = zip(
            *df['img_path'].map(expand_tile_features))
        df[['x', 'y']] = df.apply(lambda x: [get_center(x.xmin, x.xmax), get_center(x.ymin, x.ymax)], axis=1,
                                  result_type="expand")
        df[['lon', 'lat']] = df.apply(
            lambda x: convert_xy_tile_to_lonlat(x.img_name, x.tile_xmin, x.tile_ymin, x.x, x.y), axis=1,
            result_type="expand")
        df['Xmin'] = df.xmin + df.tile_xmin
        df['Ymin'] = df.ymin + df.tile_ymin
        df['Xmax'] = df.xmax + df.tile_xmin
        df['Ymax'] = df.ymax + df.tile_ymin
        df['X'] = df.x + df.tile_xmin
        df['Y'] = df.y + df.tile_ymin
        df.to_csv('{}/{}_processed.csv'.format(annot_dir, file.replace('.csv', '')))
