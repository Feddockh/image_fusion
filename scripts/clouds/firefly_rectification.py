import cv2
import numpy as np

# ---------------------------
# Define file paths and parameters
# ---------------------------
# Image paths (adjust as needed)
left_image_path = 'data/1739373925_363698530_firefly_left.tif'
right_image_path = 'data/1739373925_363698530_firefly_right.tif'

# Calibration and image parameters
image_width = 1440
image_height = 1080

# Left camera calibration parameters
camera_matrix_left = np.array([[1749.35962,    0.     ,  713.57019],
                               [   0.     , 1749.35962,  507.55742],
                               [   0.     ,    0.     ,    1.     ]])
dist_coeffs_left = np.array([-0.272549, 0.121169, 0.001261, 0.000242, 0.000000])
rectification_matrix_left = np.array([[0.99932901,  0.01597826, -0.03295796],
                                      [-0.0158705 ,  0.99986783,  0.00352846],
                                      [0.03300998, -0.00300304,  0.99945051]])
projection_matrix_left = np.array([[1733.85896,    0.     ,  788.60532,    0.     ],
                                   [   0.     , 1733.85896,  499.08208,    0.     ],
                                   [   0.     ,    0.     ,    1.     ,    0.     ]])

# Right camera calibration parameters
camera_matrix_right = np.array([[1754.87034,    0.     ,  720.00669],
                                [   0.     , 1755.21112,  501.17944],
                                [   0.     ,    0.     ,    1.     ]])
dist_coeffs_right = np.array([-0.287626, 0.170100, 0.001074, -0.000579, 0.000000])
rectification_matrix_right = np.array([[0.9992357 ,  0.01323398, -0.03678146],
                                       [-0.01335407,  0.99990627, -0.00302124],
                                       [0.03673803,  0.00351011,  0.99931877]])
projection_matrix_right = np.array([[1733.85896,    0.     ,  788.60532, -104.72209],
                                    [   0.     , 1733.85896,  499.08208,    0.     ],
                                    [   0.     ,    0.     ,    1.     ,    0.     ]])

# ---------------------------
# Load and rectify images
# ---------------------------
# Load the images
left_img = cv2.imread(left_image_path, cv2.IMREAD_COLOR)
right_img = cv2.imread(right_image_path, cv2.IMREAD_COLOR)
if left_img is None or right_img is None:
    raise IOError("One of the input images was not loaded. Check the file paths.")

# Compute undistort/rectify maps for the left image
left_map1, left_map2 = cv2.initUndistortRectifyMap(
    camera_matrix_left, 
    dist_coeffs_left, 
    rectification_matrix_left, 
    projection_matrix_left, 
    (image_width, image_height), 
    cv2.CV_32FC1
)
left_img_rectified = cv2.remap(left_img, left_map1, left_map2, cv2.INTER_LINEAR)

# Compute undistort/rectify maps for the right image
right_map1, right_map2 = cv2.initUndistortRectifyMap(
    camera_matrix_right, 
    dist_coeffs_right, 
    rectification_matrix_right, 
    projection_matrix_right, 
    (image_width, image_height), 
    cv2.CV_32FC1
)
right_img_rectified = cv2.remap(right_img, right_map1, right_map2, cv2.INTER_LINEAR)

# Save the rectified images
cv2.imwrite('data/left_img_rectified.png', left_img_rectified)
cv2.imwrite('data/right_img_rectified.png', right_img_rectified)