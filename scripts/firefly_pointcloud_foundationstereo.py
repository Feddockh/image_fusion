# Hayden Feddock
# 3/12/2025

import os
import sys
sys.path.append('.')
import cv2
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from FoundationStereo.core.utils.utils import InputPadder
from FoundationStereo.Utils import *
from FoundationStereo.core.foundation_stereo import *

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

# Display rectified images (optional)
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
axs[0].imshow(cv2.cvtColor(left_img_rectified, cv2.COLOR_BGR2RGB))
axs[0].set_title('Left Image (Rectified, Scaled)')
axs[0].axis('off')
axs[1].imshow(cv2.cvtColor(right_img_rectified, cv2.COLOR_BGR2RGB))
axs[1].set_title('Right Image (Rectified, Scaled)')
axs[1].axis('off')
plt.tight_layout()
plt.show()

# ---------------------------
# Compute disparity using StereoSGBM
# ---------------------------
# Convert to grayscale for SGBM
left_gray = cv2.cvtColor(left_img_rectified, cv2.COLOR_BGR2GRAY)
right_gray = cv2.cvtColor(right_img_rectified, cv2.COLOR_BGR2GRAY)

# Set StereoSGBM parameters
min_disp = 0
num_disp = 16 * 20  # Must be divisible by 16
block_size = 5     # Adjust according to your scene
num_channels = 1   # using grayscale images

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

disparity_sgbm = stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0

plt.figure(figsize=(10, 6))
plt.imshow(disparity_sgbm, cmap='plasma')
plt.title("Disparity Map (StereoSGBM)")
plt.colorbar(label='Disparity')
plt.axis('off')
plt.show()

# ---------------------------
# Compute disparity using FoundationStereo
# ---------------------------
torch.cuda.empty_cache()

# Prepare images for the model (assumes FoundationStereo expects color images)
# Resize images if needed; here we keep the rectified images as-is.
img_left = left_img_rectified.copy()
img_right = right_img_rectified.copy()
H, W = img_left.shape[:2]
img_left_ori = img_left.copy()  # keep a copy for visualization

# Convert images to torch tensors and normalize to float [0, 255]
img_left_tensor = torch.as_tensor(img_left).cuda().float()[None].permute(0, 3, 1, 2)
img_right_tensor = torch.as_tensor(img_right).cuda().float()[None].permute(0, 3, 1, 2)

# Pad images to be divisible by 32 (or as required by the model)
padder = InputPadder(img_left_tensor.shape, divis_by=32, force_square=False)
img_left_tensor, img_right_tensor = padder.pad(img_left_tensor, img_right_tensor)

# Load a pretrained FoundationStereo model
# (Replace 'cfg.yaml' path and 'model_best_bp2.pth' with your actual model files)
ckpt_path = 'FoundationStereo/pretrained_models/model_best_bp2.pth'
cfg_path = 'FoundationStereo/pretrained_models/cfg.yaml'
cfg = OmegaConf.load(cfg_path)
model = FoundationStereo(cfg)
ckpt = torch.load(ckpt_path)
model.load_state_dict(ckpt['model'])
model.cuda()
model.eval()

# Run the model (using non-hierarchical inference for now)
with torch.cuda.amp.autocast(True):
    disp_foundation = model.forward(img_left_tensor, img_right_tensor, iters=32, test_mode=True)

# Remove any padding applied earlier
disp_foundation = padder.unpad(disp_foundation.float())
disp_foundation = disp_foundation.data.cpu().numpy().reshape(H, W)

# Visualize the foundation stereo disparity map
vis_foundation = vis_disparity(disp_foundation)
# Concatenate the original left image and the disparity visualization for side-by-side comparison
vis_combined = np.concatenate([img_left_ori, vis_foundation], axis=1)
imageio.imwrite('output_vis_foundation.png', vis_combined)
plt.figure(figsize=(12, 6))
plt.imshow(cv2.cvtColor(vis_combined, cv2.COLOR_BGR2RGB))
plt.title("FoundationStereo Disparity Visualization")
plt.axis('off')
plt.show()

print("Disparity maps computed and visualized for both StereoSGBM and FoundationStereo.")
