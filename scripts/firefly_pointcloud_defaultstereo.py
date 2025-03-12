# Hayden Feddock
# 3/12/2025

import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def draw_horizontal_lines(img, line_interval=50, color=(0, 255, 0), thickness=1):
    """Draw horizontal lines on an image at every line_interval pixels."""
    img_lines = img.copy()
    h, w = img_lines.shape[:2]
    for y in range(0, h, line_interval):
        cv2.line(img_lines, (0, y), (w, y), color, thickness)
    return img_lines

# Load stereo images
left_image_path = 'data/firefly_left/1739373925_363698530_firefly_left.tif'
right_image_path = 'data/firefly_right/1739373925_363698530_firefly_right.tif'

left_img = cv2.imread(left_image_path, cv2.IMREAD_COLOR)
right_img = cv2.imread(right_image_path, cv2.IMREAD_COLOR)

if left_img is None or right_img is None:
    raise IOError("One of the input images was not loaded. Check the file paths.")

# Load image calibration data
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

# Compute the undistort and rectify maps for the left image
left_map1, left_map2 = cv2.initUndistortRectifyMap(
    camera_matrix_left, 
    dist_coeffs_left, 
    rectification_matrix_left, 
    projection_matrix_left, 
    (image_width, image_height), 
    cv2.CV_32FC1
)
left_img_rectified = cv2.remap(left_img, left_map1, left_map2, cv2.INTER_LINEAR)

# Compute the undistort and rectify maps for the right image
right_map1, right_map2 = cv2.initUndistortRectifyMap(
    camera_matrix_right,
    dist_coeffs_right,
    rectification_matrix_right,
    projection_matrix_right,
    (image_width, image_height),
    cv2.CV_32FC1
)
right_img_rectified = cv2.remap(right_img, right_map1, right_map2, cv2.INTER_LINEAR)

# Display the original and rectified images side by side
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
axs[0, 0].imshow(cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB))
axs[0, 0].set_title('Left Image (Original)')
axs[0, 0].axis('off')

axs[0, 1].imshow(cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB))
axs[0, 1].set_title('Right Image (Original)')
axs[0, 1].axis('off')

axs[1, 0].imshow(cv2.cvtColor(left_img_rectified, cv2.COLOR_BGR2RGB))
axs[1, 0].set_title('Left Image (Rectified)')
axs[1, 0].axis('off')

axs[1, 1].imshow(cv2.cvtColor(right_img_rectified, cv2.COLOR_BGR2RGB))
axs[1, 1].set_title('Right Image (Rectified)')
axs[1, 1].axis('off')

plt.tight_layout()
plt.show()

# Convert rectified images to grayscale
left_gray = cv2.cvtColor(left_img_rectified, cv2.COLOR_BGR2GRAY)
right_gray = cv2.cvtColor(right_img_rectified, cv2.COLOR_BGR2GRAY)

# Set parameters for StereoSGBM
min_disp = 0
num_disp = 16 * 20  # Must be divisible by 16
block_size = 5     # Adjust block size according to the scene
num_channels = 3    # 1 for grayscale images, 3 for color images

stereo = cv2.StereoSGBM_create(
    minDisparity=min_disp,
    numDisparities=num_disp,
    blockSize=block_size,
    P1=8 * num_channels * block_size ** 2,
    P2=32 * num_channels * block_size ** 2,
    disp12MaxDiff=1,
    uniquenessRatio=15,
    speckleWindowSize=300,
    speckleRange=2,
    preFilterCap=63,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)

# Compute the disparity map (dividing by 16.0 to scale down the fixed-point output)
disparity = stereo.compute(left_img_rectified, right_img_rectified).astype(np.float32) / 16.0

# Display the disparity map using matplotlib
plt.figure(figsize=(10, 6))
plt.imshow(disparity, cmap='plasma')
plt.title("Disparity Map")
plt.colorbar(label='Disparity')
plt.axis('off')
plt.show()

# 3D Reprojection
h, w = left_gray.shape
f = camera_matrix_left[0, 0]  # Approximate focal length
T = 104.72209  # Baseline between the two cameras (assumed same units as f)
cx = camera_matrix_left[0, 2]  # Principal point (x-coordinate)
cy = camera_matrix_left[1, 2]  # Principal point (y-coordinate)
Q = np.array([[1, 0, 0, -cx],
              [0, 1, 0, -cy],
              [0, 0, 0,   f],
              [0, 0, 1/T, 0]])

points_3D = cv2.reprojectImageTo3D(disparity, Q)
mask = disparity > disparity.min()

# Extract the valid 3D points.
output_points = points_3D[mask]
colors = cv2.cvtColor(cv2.imread(left_image_path), cv2.COLOR_BGR2RGB)
output_colors = colors[mask]

# --- Filter Points Based on Distance ---
# Define a maximum distance threshold (adjust this value based on your scene's scale)
max_distance = 3000  # For example, remove points further than 2000 units away

# Compute the Euclidean distance from the origin (0,0,0) for each point.
# If you want the distance from a different viewpoint (e.g., the midpoint between cameras),
# you can subtract that point's coordinates.
distances = np.linalg.norm(output_points, axis=1)
distance_mask = distances < max_distance

filtered_points = output_points[distance_mask]
filtered_colors = output_colors[distance_mask]

# --- Display the filtered point cloud ---
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
downsample_rate = 10  # Adjust to display fewer points if necessary

ax.scatter(filtered_points[::downsample_rate, 0],
           filtered_points[::downsample_rate, 1],
           filtered_points[::downsample_rate, 2],
           c=filtered_colors[::downsample_rate] / 255.0,
           s=1, edgecolor='none')
ax.set_title('3D Point Cloud (Filtered)')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# --- Set the Viewpoint ---
# To view from a position roughly between the two cameras, adjust the azimuth and elevation.
# These values can be tweaked until you get a satisfactory view.
ax.view_init(elev=10, azim=-90)

plt.show()
