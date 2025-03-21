# Hayden Feddock
# 1/29/2025

import os
import re
import cv2
import yaml
import numpy as np
from typing import List, Tuple


def load_image_pairs(base_dir: str, camera1: str, camera2: str,
                     image_extensions: Tuple[str, ...] = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
                    ) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    load image pairs for two cameras given a base directory.

    This function searches recursively within the specified subdirectories (one for each camera)
    under `base_dir` for image files. It extracts a common identifier from each filename using a regex
    (expected to be in the format "digits_digits", e.g., "1742563053_510503555"). If images from
    both camera folders share the same identifier, they are loaded via OpenCV and paired.

    Parameters:
        base_dir (str): The base directory containing subdirectories for each camera.
        camera1 (str): Subdirectory name for the first camera (e.g., "firefly_left").
        camera2 (str): Subdirectory name for the second camera (e.g., "ximea").
        image_extensions (tuple): Tuple of supported image file extensions.

    Returns:
        List[Tuple[np.ndarray, np.ndarray]]:
            A list of tuples, where each tuple contains a pair of cv2 images (one from each camera)
            that share the same identifier. Images that fail to load are skipped.
    """

    # Regular expression to extract a timestamp-like identifier from the filename.
    # This pattern matches one or more digits, an underscore, then one or more digits.
    pattern = re.compile(r'(\d+_\d+)')

    def get_images_dict(camera_folder: str) -> dict:
        """Recursively walk through camera_folder and return a dict mapping the identifier to the image file path."""
        images = {}
        cam_path = os.path.join(base_dir, camera_folder)
        for root, _, files in os.walk(cam_path):
            for file in files:
                if file.lower().endswith(image_extensions):
                    match = pattern.search(file)
                    if match:
                        key = match.group(1)
                        full_path = os.path.join(root, file)
                        images[key] = full_path
        return images

    # Get dictionaries of images for each camera
    images_cam1 = get_images_dict(camera1)
    images_cam2 = get_images_dict(camera2)

    # Find common identifiers between the two cameras
    common_keys = set(images_cam1.keys()) & set(images_cam2.keys())

    pairs = []
    for key in sorted(common_keys):
        img_path1 = images_cam1[key]
        img_path2 = images_cam2[key]
        img1 = cv2.imread(img_path1)
        img2 = cv2.imread(img_path2)
        if img1 is not None and img2 is not None:
            pairs.append((img1, img2))
        else:
            # Optionally log or print a warning if an image fails to load.
            print(f"Warning: Failed to load image for key {key}.")

    return pairs

def load_camera_params(yaml_filepath):
    """
    Loads camera parameters from a YAML file.

    The YAML file should contain the following keys:
      - camera_matrix
      - dist_coeffs
      - rectification_matrix (R)
      - projection_matrix (P)
      - width
      - height

    Parameters:
      yaml_filepath: str
          The path to the YAML calibration file.

    Returns:
      camera_params: dict
          A dictionary containing the camera parameters.
    """
    
    with open(yaml_filepath, 'r') as f:
        calib_data = yaml.safe_load(f)

    camera_params = {
        'camera_matrix': np.array(calib_data['camera_matrix']['data'], dtype=np.float32).reshape((3, 3)),
        'dist_coeffs': np.array(calib_data['distortion_coefficients']['data'], dtype=np.float32),
        'rectification_matrix': np.array(calib_data['rectification_matrix']['data'], dtype=np.float32).reshape((3, 3)),
        'projection_matrix': np.array(calib_data['projection_matrix']['data'], dtype=np.float32).reshape((3, 4)),
        'width': int(calib_data['image_width']),
        'height': int(calib_data['image_height'])
    }

    return camera_params

def rectify_image(image, yaml_filepath):
    """
    Loads calibration data from the given YAML file and returns the rectified image.

    The YAML file should contain the following keys:
      - camera_matrix
      - dist_coeffs
      - rectification_matrix (R)
      - projection_matrix (P)
      - width
      - height

    This function computes the undistort/rectify map using cv2.initUndistortRectifyMap and
    then rectifies the image using cv2.remap.

    Parameters:
      image: numpy.ndarray
          The input image to rectify.
      yaml_filepath: str
          The path to the YAML calibration file.

    Returns:
      rectified_image: numpy.ndarray
          The rectified image.
    """
    
    # Load calibration data from YAML
    camera_params = load_camera_params(yaml_filepath)
    camera_matrix = camera_params['camera_matrix']
    dist_coeffs = camera_params['dist_coeffs']
    R = camera_params['rectification_matrix']
    P = camera_params['projection_matrix']
    width = camera_params['width']
    height = camera_params['height']

    # Create undistort/rectify map
    map1, map2 = cv2.initUndistortRectifyMap(
        camera_matrix, dist_coeffs, R, P, (width, height), cv2.CV_16SC2
    )
    # Rectify the image using the computed maps
    rectified_image = cv2.remap(image, map1, map2, cv2.INTER_LINEAR)
    return rectified_image

def demosaic_ximea_5x5(image, sort_bands=True):
    """
    Demosaics a Ximea multispectral NIR camera image with a repeating 
    5x5 mosaic pattern. This image starts in the top left corner at 
    x = 0, y = 3 and ends at x = 2044, y = 1082. The cropped image is 
    2045 x 1080. This image is assumed to be in RAW8 format.

    The spectral bands correspond to the following bandwidths for each mosaic offset:
    
        Row 0: 886, 896, 877, 867, 951
        Row 1: 793, 806, 782, 769, 675
        Row 2: 743, 757, 730, 715, 690
        Row 3: 926, 933, 918, 910, 946
        Row 4: 846, 857, 836, 824, 941

    Parameters:
    - image: numpy.ndarray
        Grayscale image array.
    - sort_bands: bool
        If True, the output dictionary is sorted by bandwidth.

    Returns:
    - hypercube_dict: dict
        A dictionary where each key is a bandwidth (int) and the value is the 
        corresponding 2D numpy.ndarray (of shape (height/5, width/5)) for that spectral band.
    """

    # Crop the image.
    # Rows: 3 to 1082 (inclusive) -> 1080 rows; Columns: 0 to 2044 (inclusive) -> 2045 columns.
    cropped_image = image[3:1083, 0:2045]

    # Get the dimensions of the cropped image.
    cropped_height, cropped_width = cropped_image.shape

    # Ensure that the cropped dimensions are divisible by 5.
    if cropped_height % 5 != 0 or cropped_width % 5 != 0:
        raise ValueError("Cropped image dimensions are not divisible by 5.")

    block_rows = cropped_height // 5  # e.g., 1080 / 5 = 216
    block_cols = cropped_width // 5   # e.g., 2045 / 5 = 409

    # Define the bandwidth keys as a 5x5 array.
    bandwidth_keys = [
        [886, 896, 877, 867, 951],
        [793, 806, 782, 769, 675],
        [743, 757, 730, 715, 690],
        [926, 933, 918, 910, 946],
        [846, 857, 836, 824, 941]
    ]

    # Create a dictionary to store the hypercube channels keyed by bandwidth.
    hypercube_dict = {}

    # Loop over each offset in the 5x5 mosaic pattern.
    for row_offset in range(5):
        for col_offset in range(5):

            # Extract the spectral band for this offset.
            band = cropped_image[row_offset::5, col_offset::5]
            if band.shape != (block_rows, block_cols):
                raise ValueError(
                    f"Unexpected shape for band with offset ({row_offset},{col_offset}): "
                    f"expected ({block_rows}, {block_cols}), got {band.shape}"
                )
            
            # Retrieve the corresponding bandwidth value.
            key = bandwidth_keys[row_offset][col_offset]
            hypercube_dict[key] = band

    if sort_bands:
        # Sort the hypercube channels by bandwidth.
        hypercube_dict = dict(sorted(hypercube_dict.items()))

    return hypercube_dict

def hypercube_dict_to_array(hypercube_dict):
    """
    Converts a dictionary of hypercube channels to a 3D numpy array.

    Parameters:
    - hypercube_dict: dict
        A dictionary where each key is a bandwidth (int) and the value is the 
        corresponding 2D numpy.ndarray (of shape (height/5, width/5)) for that spectral band.

    Returns:
    - hypercube: numpy.ndarray
        A 3D numpy array of shape (num_bands, height/5, width/5) containing the hypercube.
    """

    # Get the dimensions of the first band.
    first_band = next(iter(hypercube_dict.values()))
    num_bands = len(hypercube_dict)
    height, width = first_band.shape

    # Create an empty 3D array to store the hypercube.
    hypercube = np.empty((num_bands, height, width), dtype=first_band.dtype)

    # Populate the hypercube array.
    for i, (key, band) in enumerate(hypercube_dict.items()):
        hypercube[i] = band

    return hypercube

if __name__ == "__main__":
    pass