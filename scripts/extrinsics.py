#!/usr/bin/env python3

import cv2
import numpy as np
import argparse
from utils import load_camera_params, load_image_pairs


def compute_relative_transform(rvec1, tvec1, rvec2, tvec2):
    """
    Given the poses (rvec, tvec) of the same board in two different camera coordinate systems,
    compute the relative transformation (rotation and translation) from camera 1 to camera 2.

    The transformation is computed using:
        R_rel = R2 * R1^T
        t_rel = t2 - R_rel @ t1
    """
    R1, _ = cv2.Rodrigues(rvec1)
    R2, _ = cv2.Rodrigues(rvec2)

    R_rel = R2 @ R1.T
    t_rel = tvec2 - R_rel @ tvec1

    return R_rel, t_rel


def detect_charuco_board_pose(image, board: cv2.aruco.Board, 
                              params: cv2.aruco.DetectorParameters,
                              dictionary: cv2.aruco.Dictionary,
                              camera_matrix, dist_coeffs, camera_name='Camera 0'):
    """
    Detect an ChAruCo GridBoard in the given image and estimate its pose.

    Parameters:
        image: BGR or grayscale image
        board: A cv2.aruco.GridBoard object describing the marker arrangement
        params: cv2.aruco.DetectorParameters for marker detection
        dictionary: cv2.aruco.Dictionary used for marker detection
        camera_matrix: (3x3) NumPy array with camera intrinsics
        dist_coeffs: 1D or (n,1) array of distortion coefficients

    Returns:
        (rvec, tvec) for the board pose, or (None, None) if detection fails.
    """

    # image_copy = image.copy()

    # Convert to grayscale if the image is in color
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect markers
    marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(image, dictionary, parameters=params)

    if marker_ids is not None and len(marker_ids) > 0:

        # # Sort detected markers by ID so we have a consistent order (important for averaging relative transformations)
        # sorted_indices = np.argsort(marker_ids.ravel())
        # marker_ids = marker_ids[sorted_indices]
        # marker_corners = [marker_corners[i] for i in sorted_indices]
        print(f"[DEBUG] Detected {len(marker_ids)} markers with IDs:", marker_ids.ravel())

        # Interpolate Charuco corners
        charuco_retval, charuco_corners, charuco_ids = \
            cv2.aruco.interpolateCornersCharuco(marker_corners, marker_ids, image, board)

        if charuco_retval > 0:
            print(f"[SUCCESS] Found {charuco_retval} Charuco corners!")
            # Draw the detected markers for debugging
            debug_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            cv2.aruco.drawDetectedMarkers(debug_img, marker_corners, marker_ids)
            # Optionally draw Charuco corners
            for corner in charuco_corners:
                corner_int = (int(corner[0][0]), int(corner[0][1]))
                cv2.circle(debug_img, corner_int, 5, (0, 255, 0), -1)

            # If enough corners are found, estimate the pose
            if charuco_retval:
                retval, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(charuco_corners, charuco_ids, board, camera_matrix, dist_coeffs, None, None)

                # If pose estimation is successful, draw the axis
                if retval:
                    cv2.drawFrameAxes(debug_img, camera_matrix, dist_coeffs, rvec, tvec, length=0.1, thickness=15)
                    cv2.imshow(f"Detected Charuco Board Pose {camera_name}", debug_img)

                return rvec, tvec
        
    return None, None


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
