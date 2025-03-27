# Hayden Feddock
# 3/21/2025

import os
import sys
import cv2
import numpy as np
import glob
from typing import List, Dict
from scipy.optimize import least_squares


# Constants
CAMERA_NAMES = ["firefly_left", "firefly_right", "ximea", "zed_left", "zed_right"]
EXTENSION = ".png"
DEMOSAIC_MODE = True
CALIBRATION_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "calibration_files")


class Camera:
    def __init__(self, name: str):
        self.name = name
        self.width: int = 0
        self.height: int = 0
        self.error: float = 0.0
        self.camera_matrix: np.ndarray = np.zeros((3, 3), dtype=np.float32)
        self.dist_coeffs: np.ndarray = np.zeros((5,), dtype=np.float32)
        self.rvecs: List[np.ndarray] = []
        self.tvecs: List[np.ndarray] = []
        self.rectification_matrix: np.ndarray = np.eye(3, dtype=np.float32)
        self.projection_matrix: np.ndarray = np.zeros((3, 4), dtype=np.float32)
        

class MultiCamCapture:
    def __init__(self, cameras: List[Camera], image_set_dir: str):
        """
        Store the paths to the images for each camera used in a single capture.
        """
        self.cameras = cameras
        self.id = os.path.basename(image_set_dir)
        self.image_set_dir = image_set_dir
        self.image_paths: Dict[str, List[str]] = {
            cam.name: self.get_paths(cam.name) for cam in cameras
        }
        self.images: Dict[str, List[np.ndarray]] = {}

    def get_paths(self, camera_name: str) -> List[str]:
        """
        Get the paths to the images for a given camera.
        """
        # Check for the camera name in the image set dir and determine if it's a file or directory
        potential_file = os.path.join(self.image_set_dir, camera_name + EXTENSION)
        potential_dir = os.path.join(self.image_set_dir, camera_name)
        if os.path.isfile(potential_file):
            return [potential_file]
        elif os.path.isdir(potential_dir):
            return sorted(glob.glob(os.path.join(potential_dir, f"*{EXTENSION}")))
        else:
            return []
        
    def load_images(self) -> Dict[str, List[np.ndarray]]:
        """
        Load the images from the image paths.
        """
        for cam_name, paths in self.image_paths.items():
            self.images[cam_name] = [cv2.imread(path) for path in paths]
        return self.images
    
    def get_images(self, camera_name: str) -> List[np.ndarray]:
        """
        Get the images for a given camera.
        """
        return self.images[camera_name]

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

    def add_capture_dir(self, data_dir: str):
        for subdir in glob.glob(os.path.join(data_dir, "*")):
            if os.path.isdir(subdir):
                self.captures.append(MultiCamCapture(self.cameras, subdir))

    def detect_corners(self):
        """
        Detect the corners of the ChAruCo board for each camera in the multi-camera system.
        """

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
        # Skip cameras with no detections
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
        camera.rvecs = rvecs
        camera.tvecs = tvecs

        return err

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
            for r_vec0, t_vec0, r_vec1, t_vec1 in zip(camera0.rvecs, camera0.tvecs, camera1.rvecs, camera1.tvecs):
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

        return R, t, res.cost


def main():

    # # Retrieve folder containing the capture sets from input arguments
    # if len(sys.argv) < 2:
    #     print("Usage: python multi_cam_calibration.py <data_folder>")
    #     sys.exit(1)

    # data_dir = sys.argv[1]

    # # Expand the data folder path if it contains '~'
    # data_dir = os.path.expanduser(data_dir)
    data_dir = "/home/hayden/cmu/kantor_lab/ros2_ws/image_data"

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
        err = multi_cam_calib.compute_intrinsics(cam)
        print(f"Camera {cam.name} intrinsics computed with error {err:.4f}.")

    # Compute the extrinsics between the ximea and firefly_left cameras
    R, t, err = multi_cam_calib.compute_extrinsics(ximea, firefly_left)
    print(f"Extrinsics between ximea and firefly_left computed with error {err:.4f}.")
    print(f"Rotation:\n{R}")
    print(f"Translation:\n{t}")


if __name__ == "__main__":
    main()

