# Hayden Feddock
# 3/21/2025

import cv2
import numpy as np
import glob
from utils import Camera, MultiCamera, MultiCameraCalibration


def main():

    # Define image folder paths
    firefly_left_path = "image_data/firefly_left"
    firefly_right_path = "image_data/firefly_right"
    ximea_path = "image_data/675/ximea"
    zed_left_path = "image_data/zed_left"
    zed_right_path = "image_data/zed_right"

    # Construct the camera image sets
    firefly_left = Camera("firefly_left")
    firefly_right = Camera("firefly_right")
    ximea = Camera("ximea")
    zed_left = Camera("zed_left")
    zed_right = Camera("zed_right")

    # Construct the multi-camera object
    multi_cam = MultiCamera([firefly_left, firefly_right, ximea, zed_left, zed_right])

    # Add the calibration image captures to the cameras
    firefly_left.add_capture_dir(firefly_left_path)
    firefly_right.add_capture_dir(firefly_right_path)
    ximea.add_capture_dir(ximea_path)
    zed_left.add_capture_dir(zed_left_path)
    zed_right.add_capture_dir(zed_right_path)

    # Define the ChAruCo board parameters
    ARUCO_DICT = cv2.aruco.DICT_5X5_50
    SQUARES_VERTICALLY = 6
    SQUARES_HORIZONTALLY = 4
    SQUARE_LENGTH = 0.04
    MARKER_LENGTH = 0.03

    # Create the ChAruCo board
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    board = cv2.aruco.CharucoBoard((SQUARES_VERTICALLY, SQUARES_HORIZONTALLY), SQUARE_LENGTH, MARKER_LENGTH, dictionary)
    params = cv2.aruco.DetectorParameters()

    # Create the multi-camera calibration object
    multi_cam_calib = MultiCameraCalibration(multi_cam, board, params)
    print("Calibrating cameras...")

    










if __name__ == "__main__":
    main()

