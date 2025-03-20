import cv2
import cv2.aruco as aruco
import numpy as np
import argparse

def load_calibration(calib_file):
    """Load camera calibration from a YAML file using OpenCV FileStorage."""
    fs = cv2.FileStorage(calib_file, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise IOError(f"Failed to open calibration file: {calib_file}")
    # Adjust key names to match your YAML file structure
    camera_matrix = fs.getNode("camera_matrix").mat()
    dist_coeffs = fs.getNode("dist_coeff").mat()  # or "distortion_coefficients"
    fs.release()
    return camera_matrix, dist_coeffs

def get_calibration_for_image(image_path, calib_file):
    """
    Load the image and, if a calibration file is provided, return its calibration.
    Otherwise, create a default camera matrix and zero distortion coefficients.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise IOError(f"Could not load image: {image_path}")
    h, w = image.shape[:2]
    
    if calib_file:
        camera_matrix, dist_coeffs = load_calibration(calib_file)
    else:
        # Default calibration: an example focal length and center based on image size.
        camera_matrix = np.array([[1000, 0, w / 2],
                                  [0, 1000, h / 2],
                                  [0,    0,     1]], dtype=np.float32)
        dist_coeffs = np.zeros((5, 1), dtype=np.float32)
        print(f"Using default calibration for {image_path}")
    
    return image, camera_matrix, dist_coeffs

def detect_board_pose(image, camera_matrix, dist_coeffs, board, aruco_dict):
    """
    Detect the board in the image and estimate its pose relative to the camera.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect ArUco markers in the image
    corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict)
    if ids is None or len(ids) == 0:
        raise ValueError("No markers detected.")
    
    # Interpolate to obtain ChArUco corners
    retval, charucoCorners, charucoIds = aruco.interpolateCornersCharuco(corners, ids, gray, board)
    if retval < 4:
        raise ValueError("Not enough ChArUco corners detected for reliable pose estimation.")
    
    # Estimate the pose of the board relative to the camera
    valid, rvec, tvec = aruco.estimatePoseCharucoBoard(charucoCorners, charucoIds, board, camera_matrix, dist_coeffs)
    if not valid:
        raise ValueError("Pose estimation failed.")
    
    return rvec, tvec

def compute_relative_transform(rvec1, tvec1, rvec2, tvec2):
    """
    Given the poses of the board in two camera coordinate systems (rvec, tvec),
    compute the relative transformation (rotation and translation) from camera 1 to camera 2.
    
    The transformation is computed using:
      R_rel = R2 * R1^T
      t_rel = t2 - R2 * R1^T * t1
    """
    # Convert rotation vectors to rotation matrices
    R1, _ = cv2.Rodrigues(rvec1)
    R2, _ = cv2.Rodrigues(rvec2)
    
    R_rel = R2 @ R1.T
    t_rel = tvec2 - R_rel @ tvec1
    
    return R_rel, t_rel

def main():
    parser = argparse.ArgumentParser(description="Compute extrinsics between two cameras from ArUco/ChArUco board images.")
    parser.add_argument("--image1", required=True, help="Path to the first image")
    parser.add_argument("--calib1", default=None, help="Path to the first camera calibration YAML file (optional)")
    parser.add_argument("--image2", required=True, help="Path to the second image")
    parser.add_argument("--calib2", default=None, help="Path to the second camera calibration YAML file (optional)")
    args = parser.parse_args()
    
    # Define the ArUco dictionary and board parameters.
    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
    
    # Define board parameters (adjust these to match your board)
    squaresX = 5       # number of squares along the X-axis
    squaresY = 7       # number of squares along the Y-axis
    squareLength = 0.04  # square side length (in meters)
    markerLength = 0.02  # marker side length (in meters)
    board = aruco.CharucoBoard_create(squaresX, squaresY, squareLength, markerLength, aruco_dict)
    
    # Get images and calibrations for both cameras.
    image1, camera_matrix1, dist_coeffs1 = get_calibration_for_image(args.image1, args.calib1)
    image2, camera_matrix2, dist_coeffs2 = get_calibration_for_image(args.image2, args.calib2)
    
    # Detect board pose for each image.
    rvec1, tvec1 = detect_board_pose(image1, camera_matrix1, dist_coeffs1, board, aruco_dict)
    rvec2, tvec2 = detect_board_pose(image2, camera_matrix2, dist_coeffs2, board, aruco_dict)
    
    # Compute relative transformation from camera 1 to camera 2.
    R_rel, t_rel = compute_relative_transform(rvec1, tvec1, rvec2, tvec2)
    
    print("Relative Rotation Matrix (Camera1 -> Camera2):")
    print(R_rel)
    print("\nRelative Translation Vector (Camera1 -> Camera2):")
    print(t_rel)
    
    # Optionally, draw axes on each image for visualization.
    axis_length = 0.1  # Adjust as needed (in meters)
    aruco.drawAxis(image1, camera_matrix1, dist_coeffs1, rvec1, tvec1, axis_length)
    aruco.drawAxis(image2, camera_matrix2, dist_coeffs2, rvec2, tvec2, axis_length)
    
    cv2.imshow("Camera 1 Pose", image1)
    cv2.imshow("Camera 2 Pose", image2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
