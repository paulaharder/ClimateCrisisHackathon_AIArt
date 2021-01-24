"""
Description : Given a source and a target scalar temperature field, computes least pixelwise
            displacement required for each pixel from source field such that corresponding pixel
            from target field has similar temperature value. Applies latter displacement field
            to RGB satellite view of earth.

    (1): Loads source/reference temperature field and target temperature field
    (2): Computes corresponding displacement field
    (3): Warps RGB view with computed displacement and save distorted image

Usage: run_image_distortion.py --cfg=<path_to_config_file> --input=<path_to_input_img> --o=<image_dumping_path>

Options:
  --cfg=<path_to_config_file>                Path to configuration file specifying execution parameters
  --input=<path_to_input_img>                Path to input satellite RGB image
  --o=<dataset_dumping_path>                 Path to file where image numpy array will be dumped
"""
from docopt import docopt
import yaml
import logging
import xarray as xr
import numpy as np
from tqdm import tqdm
from scipy import interpolate
from PIL import Image


def main(args, cfg):
    # Load temperature fields
    logging.info("Loading temperature fields")
    src_field, tgt_field = load_temperature_fields(cfg)

    # Compute mapping on subgrid
    logging.info("Computing pixelwise mapping")
    mapping = make_temperature_mapping(src_field=src_field,
                                       tgt_field=tgt_field,
                                       exploration_range=cfg['exploration_range'],
                                       grid_step=cfg['grid_step'])

    # Derive interpolated version of mapping on full grid
    pred_lon, pred_lat = interpolate_mapping(mapping=mapping,
                                             height=src_field.shape[0],
                                             width=src_field.shape[1],
                                             grid_step=cfg['grid_step'],
                                             kind=cfg['kind'])

    # Warp full mapping on RGB satellite view of earth and save
    logging.info("Applying displacement field to image")
    remapped_img = warp_transform(img_path=args['--input'],
                                  pred_lon=pred_lon, pred_lat=pred_lat)

    # Dumpy image numpy array
    np.save(args['--o'], remapped_img)
    logging.info(f"Completed - image dumped at {args['--o']}")


def load_temperature_fields(cfg):
    """Reads source and target temperature fields dataarrays and formats them

    Args:
        cfg (dict): input configurations

    Returns:
        type: np.ndarray, np.ndarray

    """
    # Load source and target netcdf files as xarray dataarrays
    src_dataarray = xr.open_dataarray(cfg['src'])
    tgt_dataarray = xr.open_dataarray(cfg['tgt'])

    # Extract fields as numpy arrays and center on 0th meridian
    src_field = np.roll(src_dataarray.values[::-1], 360, axis=1)
    tgt_field = np.roll(tgt_dataarray.values[::-1], 360, axis=1)
    return src_field, tgt_field


def make_temperature_mapping(src_field, tgt_field, exploration_range, grid_step):
    """Computes pixelwise mapping from source temperature field to target temperature
    field

    Args:
        src_field (np.ndarray): source temperature field from which reference temperatures
            are drawn
        tgt_field (np.ndarray): target temperature field in which we look for closest
            temperature pixel
        exploration_range (int): size of search area for each grid position
        grid_step (tuple[int]): step between each grid position

    Returns:
        type: np.ndarray

    """
    height, width = src_field.shape
    grid = np.dstack(np.meshgrid(np.arange(0, width), np.arange(0, height))[::-1])
    mapping = []
    for i in tqdm(range(0, height, grid_step[0])):
        for j in range(0, width, grid_step[1]):
            ref_temperature = src_field[i, j]
            search_area = get_search_area(i, j, exploration_range, grid)
            mapped_position = get_closest_tgt_position(tgt_field, search_area, grid, ref_temperature)
            mapping.append(mapped_position)
    mapping_height = height // grid_step[0]
    mapping_width = width // grid_step[1]
    mapping = np.array(mapping).reshape(mapping_height, mapping_width, 2)
    return mapping


def interpolate_mapping(mapping, height, width, grid_step, kind):
    """Takes mapping defined on 2D grid and interpolates its values
    on larger grid

    Args:
        mapping (np.ndarray): (subgrid_height, subgrid_width, 2) mapping of pixels
        height (int): total height of full grid
        width (int): total width of full grid
        grid_step (tuple[int]): step between each grid position on subgrid
        kind (str): kind of spline interpolation to use in {‘linear’, ‘cubic’, ‘quintic’}

    Returns:
        type: np.ndarray, np.ndarray

    """
    # Compute interpolating functions out of grid
    x = np.arange(0, width, grid_step[0])
    y = np.arange(0, height, grid_step[1])
    lon_transform = interpolate.interp2d(x, y, mapping[..., 0], kind=kind)
    lat_transform = interpolate.interp2d(x, y, mapping[..., 1], kind=kind)

    # Apply interpolating functions to full grid
    x_full = np.arange(0, width)
    y_full = np.arange(0, height)
    pred_lon = lon_transform(x_full, y_full).clip(0, height - 1).round().astype(int)
    pred_lat = lat_transform(x_full, y_full).clip(0, width - 1).round().astype(int)
    return pred_lon, pred_lat


def warp_transform(img_path, pred_lon, pred_lat):
    """Loads RGB satellite earth view and applies displacement field

    Args:
        img_path (str): path to image to load
        pred_lon (np.narray): pixel displacement following rows
        pred_lat (np.narray): pixel displacement following columns

    Returns:
        type: np.ndarray

    """
    # Load RGB satellite image
    if img_path.endswith('jpg') or img_path.endswith('png'):
        img = Image.open(img_path)
        img = np.array(img)

    elif img_path.endswith('npy'):
        img = np.load(img_path)

    else:
        raise ValueError("Unkown input image type")

    # Remap pixels location according to new longitude and latitudes
    full_mapping = np.dstack([pred_lon, pred_lat]).reshape(-1, 2)
    remapped_img = img[(full_mapping[:, 0], full_mapping[:, 1])].reshape(*img.shape)
    return remapped_img


def get_search_area(i, j, pad_size, grid):
    """Compute boolean mask corresponding to pixels within which we must search
    closest pixel with same temperature

    Args:
        i (int): row number of center pixel
        j (int): column number of center pixel
        pad_size (int): size of padding from center pixel defining area size
        grid (np.ndarray): index grid

    Returns:
        type: np.ndarray

    """
    row_mask = np.abs(i - grid[..., 0]) < pad_size
    row_mask_bis = np.abs(i + grid.shape[0] - grid[..., 0]) < pad_size
    col_mask = np.abs(j - grid[..., 1]) < pad_size
    col_mask_bis = np.abs(j + grid.shape[1] - grid[..., 1]) < pad_size
    area_mask = (row_mask | row_mask_bis) & (col_mask | col_mask_bis)
    return area_mask


def get_closest_tgt_position(tgt_field, search_area, grid, ref_temperature):
    """Computes row and column indices of pixel having closest temperature to
    specified reference temperature within allowed search area

    Args:
        tgt_field (np.ndarray): target temperature field in which we look for closest
            temperature pixel
        search_area (np.ndarray): boolean array framing seach area
        grid (np.ndarray): index grid
        ref_temperature (float): value of reference temperature to compare to

    Returns:
        type: np.ndarray

    """
    search_tgt_field = tgt_field[search_area]
    search_positions = grid[search_area]
    best_idx = np.argmin(np.abs(search_tgt_field - ref_temperature))
    best_position = search_positions[best_idx]
    return best_position


if __name__ == "__main__":
    # Read input args
    args = docopt(__doc__)

    # Load config file
    with open(args['--cfg'], "r") as f:
        cfg = yaml.safe_load(f)

    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logging.info(f'Arguments: {args}')

    # Run
    main(args, cfg)
