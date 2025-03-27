# Hayden Feddock
# 1/29/2025

import os
import re
import cv2
import yaml
import glob
import numpy as np
from typing import List, Tuple, Dict, Set



# Constants
EXTENSION = ".png"
CALIBRATION_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "calibration_files")

class FlowList(list):
    """A list that PyYAML will emit in [a, b, c] (flow) style."""
    pass

def _represent_flow_list(dumper, data):
    return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)

yaml.add_representer(FlowList, _represent_flow_list)

class Camera:
    def __init__(self, name: str):
        self.name = name
        self.width: int = 0
        self.height: int = 0
        self.error: float = 0.0
        self.camera_matrix: np.ndarray = np.zeros((3, 3), dtype=np.float32)
        self.dist_coeffs: np.ndarray = np.zeros((5,), dtype=np.float32)
        self.rectification_matrix: np.ndarray = np.eye(3, dtype=np.float32)
        self.projection_matrix: np.ndarray = np.zeros((3, 4), dtype=np.float32)
        self.transforms: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    def load_params(self):
        """
        Load the camera parameters from a YAML file.
        """
        yaml_file = os.path.join(CALIBRATION_DIR, f"{self.name}.yaml")
        with open(yaml_file, 'r') as f:
            calib_data = yaml.safe_load(f)
        self.width = int(calib_data['image_width'])
        self.height = int(calib_data['image_height'])
        self.camera_matrix = np.array(calib_data['camera_matrix']['data'], dtype=np.float32).reshape((3, 3))
        self.dist_coeffs = np.array(calib_data['distortion_coefficients']['data'], dtype=np.float32)
        self.rectification_matrix = np.array(calib_data['rectification_matrix']['data'], dtype=np.float32).reshape((3, 3))
        self.projection_matrix = np.array(calib_data['projection_matrix']['data'], dtype=np.float32).reshape((3, 4))
        # Load transforms if available
        if 'transforms' in calib_data:
            for name, transform in calib_data['transforms'].items():
                R = np.array(transform['R'], dtype=np.float32).reshape((3, 3))
                t = np.array(transform['t'], dtype=np.float32).reshape((3, 1))
                self.transforms[name] = (R, t)
    
    def save_params(self):
        """
        Save the camera parameters to a YAML file, with all `data:` arrays in
        horizontal (flow) style.
        """
        yaml_file = os.path.join(CALIBRATION_DIR, f"{self.name}.yaml")
        calib_data = {
            'image_width': self.width,
            'image_height': self.height,
            'camera_matrix': {
                'rows': 3, 'cols': 3,
                'data': FlowList(self.camera_matrix.ravel().tolist())
            },
            'distortion_coefficients': {
                'rows': 1, 'cols': 5,
                'data': FlowList(self.dist_coeffs.ravel().tolist())
            },
            'rectification_matrix': {
                'rows': 3, 'cols': 3,
                'data': FlowList(self.rectification_matrix.ravel().tolist())
            },
            'projection_matrix': {
                'rows': 3, 'cols': 4,
                'data': FlowList(self.projection_matrix.ravel().tolist())
            },
            'transforms': {
                name: {
                    'R': FlowList(transform[0].ravel().tolist()),
                    't': FlowList(transform[1].ravel().tolist())
                } for name, transform in self.transforms.items()
            }
        }

        os.makedirs(CALIBRATION_DIR, exist_ok=True)
        with open(yaml_file, 'w') as f:
            yaml.dump(calib_data, f, sort_keys=False)
        
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


def project_points(points, colors, intrinsic, extrinsic, width, height):
    """
    Project 3D points to 2D image coordinates using a pinhole camera model.

    Parameters:
        points: Nx3 numpy array of 3D points in world coordinates.
        colors: Nx3 numpy array of RGB colors corresponding to the points.
        intrinsic: 3x3 numpy array representing the camera intrinsic matrix.
        extrinsic: 4x4 numpy array representing the camera extrinsic transform (world->camera).
        width: Width of the output image.
        height: Height of the output image.
    Returns:
        image: 2D numpy array of shape (height, width, 3) representing the projected image.
    """

    # Convert points to homogeneous coordinates
    num_points = points.shape[0]
    homog_points = np.hstack((points, np.ones((num_points, 1))))

    # Transform points from world space to the new camera view
    cam_points = (extrinsic @ homog_points.T).T

    # Keep points in front of the camera (z>0)
    valid = cam_points[:, 2] > 0
    cam_points = cam_points[valid]
    colors = colors[valid]

    # Pin-hole projection
    proj = (intrinsic @ cam_points[:, :3].T).T
    proj /= proj[:, 2:3]  # perspective divide

    # Create empty images for color and depth
    image = np.zeros((height, width, 3), dtype=np.uint8)
    z_buffer = np.full((height, width), np.inf)

    # Draw points based on z-depth
    for pt, color, z in zip(proj, colors, cam_points[:, 2]):
        u, v = int(round(pt[0])), int(round(pt[1]))
        if 0 <= u < width and 0 <= v < height:
            if z < z_buffer[v, u]:
                z_buffer[v, u] = z
                # Scale color to [0..255] if needed
                if np.max(color) <= 1.0:
                    color = (color*255).astype(np.uint8)
                image[v, u] = color
    return image

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

# DEPRECATED
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

# DEPRECATED
def load_image_sets(img_dirs: List[str]) -> List[Dict[str, str]]:
    """
    Load image sets from a list of directories.

    Parameters:
        img_dirs: List[str]
            A list of directories containing matching images.

    Returns:
        List[Dict[str, str]]
            A list of image sets with each element as a path.
    """

    # Create a list of valid image prefixes
    prefixes = []

    # Walk through the first directory and collect image prefixes
    for root, _, files in os.walk(img_dirs[0]):
        for file in files:
            prefix = re.search(r'(\d+_\d+)', file)
            if prefix:
                prefixes.append(prefix.group(1))
    
    # Now for each prefix, load the corresponding images from all directories
    # Only load the corresponding images if they exist in all directories
    images = []
    for prefix in prefixes:
        img_set = []
        for img_dir in img_dirs:

            # Find the file matching the prefix in the current directory
            found = False
            for root, _, files in os.walk(img_dir):
                for file in files:
                    if prefix in file and file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                        img_path = os.path.join(root, file)
                        img_set.append(img_path)
                        found = True
                        break
                if found:
                    break
        if len(img_set) == len(img_dirs):
            images.append(img_set)

    if images:
        images.sort(key=lambda x: len(x), reverse=True)
        return images

    return []

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