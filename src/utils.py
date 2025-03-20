# Hayden Feddock
# 1/29/2025

import cv2
import numpy as np

def demosaic_ximea_5x5(image_path, sort_bands=True):
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
    - image_path: str
        Path to the input mosaic image.

    Returns:
    - hypercube_dict: dict
        A dictionary where each key is a bandwidth (int) and the value is the 
        corresponding 2D numpy.ndarray (of shape (height/5, width/5)) for that spectral band.
    """

    # Read the image in grayscale (RAW8 format)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Image not found or unable to read: {image_path}")

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