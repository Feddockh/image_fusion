# Hayden Feddock
# 3/21/2025

import os
import sys
import cv2
import numpy as np
import glob
from typing import List, Dict
from scipy.optimize import least_squares
from utils import Camera, MultiCamCapture


class MultiCamCalibration:
    def __init__(self, cameras: List[Camera], charuco_board: cv2.aruco.CharucoBoard):
        """
        Calibrate a multi-camera system using ChAruCo boards.
        """
        self.cameras = cameras
        self.charuco_board = charuco_board

        charuco_params = cv2.aruco.CharucoParameters()
        charuco_params.tryRefineMarkers = True
        detector_params = cv2.aruco.DetectorParameters()
        detector_params.adaptiveThreshConstant = 9
        detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        refine_params = cv2.aruco.RefineParameters()
        
        self.charuco_detector = cv2.aruco.CharucoDetector(
            board=charuco_board,
            charucoParams=charuco_params,
            detectorParams=detector_params,
            refineParams=refine_params
        )

        self.captures: List[MultiCamCapture] = []

        # 2D image positions of detected ChArUco (chessboard) corners and marker corners
        self.all_charuco_corners: Dict[str, List[np.ndarray]] = {cam.name: [] for cam in cameras}
        self.all_charuco_ids: Dict[str, List[np.ndarray]] = {cam.name: [] for cam in cameras}
        self.all_marker_corners: Dict[str, List[np.ndarray]] = {cam.name: [] for cam in cameras}
        self.all_marker_ids: Dict[str, List[np.ndarray]] = {cam.name: [] for cam in cameras}

        # r_vecs and t_vecs are the rotation and translation vectors for each image
        self.all_r_vecs: Dict[str, List[np.ndarray]] = {cam.name: [] for cam in cameras}
        self.all_t_vecs: Dict[str, List[np.ndarray]] = {cam.name: [] for cam in cameras}

    def add_capture_dir(self, data_dir: str):
        for subdir in glob.glob(os.path.join(data_dir, "*")):
            if os.path.isdir(subdir):
                self.captures.append(MultiCamCapture(self.cameras, subdir))
        
        if not self.captures:
            raise ValueError(f"No valid capture directories found in {data_dir}.")

    def detect_corners(self):
        """
        Detect the corners of the ChAruCo board for each camera in the multi-camera system.
        """

        if self.captures is None or len(self.captures) == 0:
            raise ValueError("No captures have been added to the calibration object.")

        # Loop through each capture set and detect the board and corners
        for capture in self.captures:

            # Load the images for each camera
            capture.load_images()

            # Detect the board and corners for each camera
            for cam in self.cameras:
                img = capture.get_images(cam.name)[0] # Just using the first image if there are multiple

                # Convert to grayscale if necessary
                if len(img.shape) == 3:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                # Set the camera width and height if they haven't been set
                if cam.width == 0 or cam.height == 0:
                    cam.height, cam.width = gray.shape[:2]

                # Get the corners and ids for the ChAruCo board and the marker corners and ids
                """
                charuco_corners   -> ndarray (N×1×2 float)   : 2D image positions of detected ChArUco (chessboard) corners  
                charuco_ids       -> ndarray (N×1 int)       : Unique IDs for each detected ChArUco corner  
                marker_corners    -> list of length M of 4×2 float arrays : Pixel coordinates of all detected ArUco marker corners  
                marker_ids        -> ndarray (M×1 int)       : IDs of detected ArUco markers (in same order as markerCorners)"
                """
                charuco_corners, charuco_ids, marker_corners, marker_ids = self.charuco_detector.detectBoard(gray)

                # Only store detections if enough corners were found
                if charuco_ids is not None and len(charuco_ids) > 0:
                    self.all_charuco_corners[cam.name].append(charuco_corners)
                    self.all_charuco_ids[cam.name].append(charuco_ids)
                    self.all_marker_corners[cam.name].append(marker_corners)
                    self.all_marker_ids[cam.name].append(marker_ids)

    def compute_intrinsics(self, camera: Camera):
        """
        Compute the intrinsics for a single camera.
        """
        if len(self.all_charuco_corners[camera.name]) == 0:
            print(f"Warning: No detections for camera {camera.name}. Skipping calibration.")
            return

        # Calibrate the camera
        err, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
            charucoCorners=self.all_charuco_corners[camera.name],
            charucoIds=self.all_charuco_ids[camera.name],
            board=self.charuco_board,
            imageSize=(camera.width, camera.height),
            cameraMatrix=None,
            distCoeffs=None
        )

        # Save the camera data
        camera.error = err
        camera.camera_matrix = camera_matrix
        camera.dist_coeffs = dist_coeffs
        self.all_r_vecs[camera.name] = rvecs
        self.all_t_vecs[camera.name] = tvecs

        # Set the projection matrix to the camera matrix for now, will be updated later if stereo camera
        camera.projection_matrix[:3, :3] = camera_matrix 

        return err, camera_matrix, dist_coeffs, rvecs, tvecs

    def compute_stereo_rectification(self, camera0: Camera, camera1: Camera):
        """
        Compute the stereo rectification parameters between two cameras.
        """
        # Check for detections in both cameras
        if len(self.all_charuco_corners[camera0.name]) == 0 or len(self.all_charuco_corners[camera1.name]) == 0:
            print(f"Warning: No detections for cameras {camera0.name} and {camera1.name}. Skipping calibration.")
            return
        
        # Match the object points and image points for the two cameras across all captures
        capture_obj_pts = []
        capture_img_pts0 = []
        capture_img_pts1 = []
        for cam0_pts, cam0_ids, cam1_pts, cam1_ids in zip(self.all_charuco_corners[camera0.name], self.all_charuco_ids[camera0.name],
                                                          self.all_charuco_corners[camera1.name], self.all_charuco_ids[camera1.name]):

            # Match the object points (3D coordinates relative to the ChAruCo board) and image points (2D pixel coordinates)
            obj_pts0, img_pts0 = self.charuco_board.matchImagePoints(detectedCorners=cam0_pts, detectedIds=cam0_ids)
            obj_pts1, img_pts1 = self.charuco_board.matchImagePoints(detectedCorners=cam1_pts, detectedIds=cam1_ids)

            # Find the common object points and image points
            _, common_idx0, common_idx1 = np.intersect1d(cam0_ids, cam1_ids, return_indices=True)
            aligned_obj = obj_pts0[common_idx0]
            aligned_img0 = img_pts0[common_idx0]
            aligned_img1 = img_pts1[common_idx1]

            # Store the object points from this capture
            capture_obj_pts.append(aligned_obj)
            capture_img_pts0.append(aligned_img0)
            capture_img_pts1.append(aligned_img1)

        # Compute the stereo rectification parameters
        ret, cm1, dc1, cm2, dc2, R, T, E, F = cv2.stereoCalibrate(
            objectPoints=capture_obj_pts,
            imagePoints1=capture_img_pts0,
            imagePoints2=capture_img_pts1,
            cameraMatrix1=camera0.camera_matrix,
            distCoeffs1=camera0.dist_coeffs,
            cameraMatrix2=camera1.camera_matrix,
            distCoeffs2=camera1.dist_coeffs,
            imageSize=(camera0.width, camera0.height),
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 100, 1e-5),
            flags=cv2.CALIB_FIX_INTRINSIC
        )

        # Stereo‑rectify → get R1, R2, P1, P2, Q
        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
            cameraMatrix1=cm1, distCoeffs1=dc1,
            cameraMatrix2=cm2, distCoeffs2=dc2,
            imageSize=(camera0.width, camera0.height),
            R=R, T=T,
            flags=cv2.CALIB_ZERO_DISPARITY,
            alpha=0
        )

        # Store results
        camera0.rectification_matrix = R1
        camera0.projection_matrix    = P1
        camera1.rectification_matrix = R2
        camera1.projection_matrix    = P2

        return ret, R1, R2, P1, P2, Q

    def compute_extrinsics(self, camera0: Camera, camera1: Camera):
        """
        Compute the extrinsics between two cameras using camera 0 as the reference.
        This function assumes that the cameras are already intrinsically calibrated.
        """

        # Initial guess: identity rotation, zero translation
        x0 = np.zeros(6, dtype=np.float64)

        def residuals(x):
            R = cv2.Rodrigues(x[0:3])[0]
            t = x[3:6].reshape(3, 1)
            residual_list = []
            for r_vec0, t_vec0, r_vec1, t_vec1 in zip(
                    self.all_r_vecs[camera0.name], 
                    self.all_t_vecs[camera0.name], 
                    self.all_r_vecs[camera1.name], 
                    self.all_t_vecs[camera1.name]):
                
                R0 = cv2.Rodrigues(r_vec0)[0]
                R1 = cv2.Rodrigues(r_vec1)[0]

                R1_pred = R @ R0
                t1_pred = R @ t_vec0.reshape(3,1) + t

                # Compute the difference in axis angle representation
                R_delta = R1_pred.T @ R1
                r_resid = cv2.Rodrigues(R_delta)[0].ravel()

                # Compute the difference in translation
                t_resid = (t_vec1 - t1_pred).ravel()

                residual_list.append(r_resid)
                residual_list.append(t_resid)
            
            return np.hstack(residual_list)

        # Run least squares
        res = least_squares(residuals, x0, method='lm')

        # Extract the optimized parameters
        R = cv2.Rodrigues(res.x[0:3])[0]
        t = res.x[3:6].reshape(3, 1)

        # Save the extrinsics
        camera0.transforms[camera1.name] = (R, t)

        return res.cost, R, t

def main():

    # # Retrieve folder containing the capture sets from input arguments
    # if len(sys.argv) < 2:
    #     print("Usage: python multi_cam_calibration.py <data_folder>")
    #     sys.exit(1)

    # data_dir = sys.argv[1]

    # # Expand the data folder path if it contains '~'
    # data_dir = os.path.expanduser(data_dir)
    data_dir = "/home/hayden/cmu/kantor_lab/ros2_ws/calibration_images"

    # Construct the camera objects
    firefly_left = Camera("firefly_left")
    firefly_right = Camera("firefly_right")
    ximea = Camera("ximea")
    zed_left = Camera("zed_left")
    zed_right = Camera("zed_right")
    cameras = [firefly_left, firefly_right, ximea, zed_left, zed_right]

    # Define the ChAruCo board parameters
    ARUCO_DICT = cv2.aruco.DICT_5X5_50
    SQUARES_VERTICALLY = 6
    SQUARES_HORIZONTALLY = 4
    SQUARE_LENGTH = 0.04
    MARKER_LENGTH = 0.03

    # Create the ChAruCo board
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    charuco_board = cv2.aruco.CharucoBoard((SQUARES_VERTICALLY, SQUARES_HORIZONTALLY), SQUARE_LENGTH, MARKER_LENGTH, dictionary)

    # Create the multi-camera calibration object
    multi_cam_calib = MultiCamCalibration(cameras, charuco_board)
    multi_cam_calib.add_capture_dir(data_dir)

    # Calibrate the cameras
    multi_cam_calib.detect_corners()

    # Compute the intrinsics for each camera
    for cam in cameras:
        err = multi_cam_calib.compute_intrinsics(cam)[0]
        print(f"Camera {cam.name} intrinsics computed with error {err:.4f}.")

    # Compute the extrinsics between the ximea and firefly_left cameras
    err = multi_cam_calib.compute_extrinsics(ximea, firefly_left)[0]
    print(f"Extrinsics between ximea and firefly_left computed with error {err:.4f}.")

    # Compute the stereo rectification between the firefly_left and firefly_right cameras
    err = multi_cam_calib.compute_stereo_rectification(firefly_left, firefly_right)[0]
    print(f"Stereo rectification between firefly_left and firefly_right computed with error {err:.4f}.")

    # Compute the stereo rectification between the zed_left and zed_right cameras
    err = multi_cam_calib.compute_stereo_rectification(zed_left, zed_right)[0]
    print(f"Stereo rectification between zed_left and zed_right computed with error {err:.4f}.")

    # Save the camera calibrations
    for cam in cameras:
        cam.save_params()

if __name__ == "__main__":
    main()

