#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# Hayden Feddock
# 3/12/2025

This script computes a disparity map using the StereoSGBM method,
reprojects it to 3D, saves outputs (vis.png, depth_meter.npy, cloud.ply, cloud_denoise.ply)
and opens an Open3D window to visualize the denoised point cloud.
"""

import os
import sys
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d


def load_calibration(intrinsic_file):
    """Load the intrinsic matrix and baseline from a text file.
       The file should have the 9 intrinsic parameters in the first line
       and the baseline in the second line."""
    with open(intrinsic_file, 'r') as f:
        lines = f.readlines()
        K = np.array(list(map(float, lines[0].strip().split()))).reshape(3, 3)
        baseline = float(lines[1].strip())
    return K, baseline

def compute_rectification(img, camera_matrix, dist_coeffs, rect_matrix, proj_matrix, image_size):
    """Compute undistort-rectify maps and rectify the image."""
    map1, map2 = cv2.initUndistortRectifyMap(
        camera_matrix, dist_coeffs, rect_matrix, proj_matrix, image_size, cv2.CV_32FC1)
    img_rectified = cv2.remap(img, map1, map2, cv2.INTER_LINEAR)
    return img_rectified

def create_point_cloud(disparity, Q, left_img, min_disp):
    """Reproject disparity to 3D and extract valid points."""
    points_3D = cv2.reprojectImageTo3D(disparity, Q)
    mask = disparity > min_disp
    output_points = points_3D[mask]
    colors = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
    output_colors = colors[mask]
    return output_points, output_colors

def main():
    parser = argparse.ArgumentParser(
        description="StereoSGBM based disparity and 3D reconstruction demo with optional arguments."
    )
    parser.add_argument('--left_file', type=str, default='demo_data/firefly/left_img_rectified.png',
                        help="Path to the left image")
    parser.add_argument('--right_file', type=str, default='demo_data/firefly/right_img_rectified.png',
                        help="Path to the right image")
    parser.add_argument('--intrinsic_file', type=str, default='demo_data/firefly/K.txt',
                        help="Path to the intrinsic calibration file. "
                             "File should contain 9 intrinsic parameters in first line and baseline in second line.")
    parser.add_argument('--out_dir', type=str, default='output/default_stereo',
                        help="Directory to save output files")
    parser.add_argument('--max_distance', type=float, default=3000,
                        help="Max distance (in depth units) to filter point cloud")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load rectified images
    left_rect = cv2.imread(args.left_file, cv2.IMREAD_COLOR)
    right_rect = cv2.imread(args.right_file, cv2.IMREAD_COLOR)
    if left_rect is None or right_rect is None:
        raise IOError("One of the input images was not loaded. Check the file paths.")
    
    # ---------------------------
    # Compute disparity using StereoSGBM
    # ---------------------------
    # For StereoSGBM, convert images to grayscale
    left_gray = cv2.cvtColor(left_rect, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_rect, cv2.COLOR_BGR2GRAY)

    min_disp = 0
    num_disp = 16 * 20  # Must be divisible by 16
    block_size = 5      # Adjust block size according to the scene

    stereo = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=block_size,
        P1=8 * 1 * block_size ** 2,
        P2=32 * 1 * block_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=15,
        speckleWindowSize=300,
        speckleRange=2,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    disparity = stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0

    # Visualize disparity and save as vis.png
    plt.figure(figsize=(10, 6))
    plt.imshow(disparity, cmap='plasma')
    plt.title("Disparity Map (StereoSGBM)")
    plt.colorbar(label='Disparity')
    plt.axis('off')
    vis_path = os.path.join(args.out_dir, 'vis.png')
    plt.savefig(vis_path, bbox_inches='tight')
    plt.show()
    print(f"Disparity visualization saved to {vis_path}")

    # ---------------------------
    # 3D Reprojection
    # ---------------------------
    # Load intrinsic matrix and baseline from file
    K, baseline = load_calibration(args.intrinsic_file)
    f = K[0, 0]      # focal length
    cx = K[0, 2]     # principal point x
    cy = K[1, 2]     # principal point y

    # Create Q matrix for reprojectImageTo3D:
    Q = np.array([[1, 0, 0, -cx],
                  [0, 1, 0, -cy],
                  [0, 0, 0, f],
                  [0, 0, -1/baseline, 0]])
    
    # Save depth in meters: depth = f * baseline / disparity
    with np.errstate(divide='ignore'):
        depth = f * baseline / disparity
    depth_path = os.path.join(args.out_dir, 'depth_meter.npy')
    np.save(depth_path, depth)
    print(f"Depth map saved to {depth_path}")

    # Create point cloud from disparity
    pts, colors = create_point_cloud(disparity, Q, left_rect, min_disp)

    # Filter points based on distance
    distances = np.linalg.norm(pts, axis=1)
    distance_mask = distances < args.max_distance
    filtered_pts = pts[distance_mask]
    filtered_colors = colors[distance_mask]

    # Save raw point cloud (cloud.ply)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(filtered_pts)
    pcd.colors = o3d.utility.Vector3dVector(filtered_colors.astype(np.float64) / 255.0)
    cloud_path = os.path.join(args.out_dir, 'cloud.ply')
    o3d.io.write_point_cloud(cloud_path, pcd)
    print(f"Raw point cloud saved to {cloud_path}")

    # Denoise point cloud using radius outlier removal
    cl, ind = pcd.remove_radius_outlier(nb_points=30, radius=0.03)
    pcd_denoised = pcd.select_by_index(ind)
    cloud_denoise_path = os.path.join(args.out_dir, 'cloud_denoise.ply')
    o3d.io.write_point_cloud(cloud_denoise_path, pcd_denoised)
    print(f"Denoised point cloud saved to {cloud_denoise_path}")

    # ---------------------------
    # Visualize the point cloud using Open3D
    # ---------------------------
    print("Visualizing point cloud. Close the Open3D window to exit.")
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd_denoised)
    vis.get_render_option().point_size = 1.0
    vis.get_render_option().background_color = np.array([0.5, 0.5, 0.5])
    vis.run()
    vis.destroy_window()

if __name__ == '__main__':
    main()
