#!/usr/bin/env python3

import cv2
import numpy as np
import argparse
from utils import load_camera_params, load_image_pairs, \
    detect_charuco_board_pose, compute_relative_transform


def main():
    # Define image and calibration file paths
    calib0_path = 'src/image_fusion/calibration_data/firefly_left.yaml'
    calib1_path = 'src/image_fusion/calibration_data/ximea.yaml'

    # Load camera params
    cam0_params = load_camera_params(calib0_path)
    cam1_params = load_camera_params(calib1_path)

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

    # Keep track of the computed relative transformations
    R_rels = []
    t_rels = []

    # Over a sequence of image pairs, compute the average R_rel and t_rel
    base_dir = 'image_data'
    cam0 = 'firefly_left'
    cam1 = 'ximea'
    img_pairs = load_image_pairs(base_dir, cam0, cam1)
    for (img0, img1) in img_pairs:

        # Detect board pose for each image
        rvec0, tvec0 = detect_charuco_board_pose(
            img0,
            board,
            params=params,
            dictionary=dictionary,
            camera_matrix=cam0_params['camera_matrix'],
            dist_coeffs=cam0_params['dist_coeffs'],
            camera_name='Camera 0'
        )
        rvec1, tvec1 = detect_charuco_board_pose(
            img1,
            board,
            params=params,
            dictionary=dictionary,
            camera_matrix=cam1_params['camera_matrix'],
            dist_coeffs=cam1_params['dist_coeffs'],
            camera_name='Camera 1'
        )
        if rvec0 is None or rvec1 is None:
            print("[ERROR] Could not detect board pose in one of the images.")
            return

        # Compute relative transformation from camera 0 to camera 1
        R_rel, t_rel = compute_relative_transform(rvec0, tvec0, rvec1, tvec1)

        print("Relative Rotation Matrix (Camera 0 -> Camera 1):")
        print(R_rel)
        print("\nRelative Translation Vector (Camera 0 -> Camera 1):")
        print(t_rel)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Store the relative transformations
        R_rels.append(R_rel)
        t_rels.append(t_rel)

    # Average the relative transformations
    avg_R_rel = np.mean(R_rels, axis=0)
    avg_t_rel = np.mean(t_rels, axis=0)

    print("Average Relative Rotation Matrix (Camera 0 -> Camera 1):")
    print(avg_R_rel)
    print("\nAverage Relative Translation Vector (Camera 0 -> Camera 1):")
    print(avg_t_rel)


if __name__ == "__main__":
    main()
