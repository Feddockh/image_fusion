#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to project a point cloud (cloud.ply) onto a 2D image plane.
This simulates the view from a nearby camera relative to the left camera of a stereo pair.
The output is a 2D image that can be used to stack additional channels for object detection.
"""

import argparse
import numpy as np
import open3d as o3d
import cv2

def load_point_cloud(ply_file):
    """Load a point cloud from a PLY file."""
    pcd = o3d.io.read_point_cloud(ply_file)
    if not pcd.has_points():
        raise ValueError(f"No points found in {ply_file}")
    return pcd

def project_points(points, colors, intrinsic, extrinsic, width, height):
    """
    Project 3D points to 2D image coordinates using a pinhole camera model.
    
    intrinsic: 3x3 camera intrinsic matrix.
    extrinsic: 4x4 camera extrinsic transformation matrix (from world to camera).
    width, height: Dimensions of the output image.
    """
    # Convert points to homogeneous coordinates (Nx4)
    num_points = points.shape[0]
    homog_points = np.hstack((points, np.ones((num_points, 1))))
    
    # Transform points from world space to camera space
    cam_points = (extrinsic @ homog_points.T).T
    # Only consider points in front of the camera (positive z)
    valid = cam_points[:, 2] > 0
    cam_points = cam_points[valid]
    colors = colors[valid]

    # Project points using the pinhole camera model
    proj_points = (intrinsic @ cam_points[:, :3].T).T
    proj_points = proj_points / proj_points[:, 2:3]  # Normalize by z
    
    # Create an empty image and a depth (z-buffer) image
    image = np.zeros((height, width, 3), dtype=np.uint8)
    z_buffer = np.full((height, width), np.inf)

    # Loop through points and fill the image based on depth
    for pt, color, z in zip(proj_points, colors, cam_points[:, 2]):
        u, v = int(round(pt[0])), int(round(pt[1]))
        if 0 <= u < width and 0 <= v < height:
            # If this point is closer than what's been recorded, update the pixel color
            if z < z_buffer[v, u]:
                z_buffer[v, u] = z
                image[v, u] = (color * 255).astype(np.uint8) if color.max() <= 1.0 else color

    return image

def main():
    parser = argparse.ArgumentParser(
        description="Project a point cloud onto a 2D image plane to simulate a camera view."
    )
    parser.add_argument('--ply_file', type=str, default='output/FoundationStereo/cloud_denoise.ply',
                        help="Path to the input point cloud (.ply) file")
    parser.add_argument('--out_image', type=str, default='projected_view.png',
                        help="Path to save the output image")
    parser.add_argument('--width', type=int, default=1440,
                        help="Width of the output image")
    parser.add_argument('--height', type=int, default=1080,
                        help="Height of the output image")
    # Camera intrinsics parameters (modify as needed)
    parser.add_argument('--focal_length', type=float, default=1749.35962,
                        help="Focal length in pixels")
    parser.add_argument('--cx', type=float, default=713.57019,
                        help="Principal point x-coordinate")
    parser.add_argument('--cy', type=float, default=507.55742,
                        help="Principal point y-coordinate")
    # Extrinsic parameters: translation relative to the original camera (in meters)
    parser.add_argument('--tx', type=float, default=0.0,
                        help="Translation along x-axis")
    parser.add_argument('--ty', type=float, default=-0.1,
                        help="Translation along y-axis")
    parser.add_argument('--tz', type=float, default=0.0,
                        help="Translation along z-axis")
    args = parser.parse_args()

    # Load point cloud
    pcd = load_point_cloud(args.ply_file)
    points = np.asarray(pcd.points)
    # Try to get colors; if none exist, set to white
    if pcd.has_colors():
        colors = np.asarray(pcd.colors)
    else:
        colors = np.ones((points.shape[0], 3))

    # Define intrinsic camera matrix
    intrinsic = np.array([
        [args.focal_length, 0, args.cx],
        [0, args.focal_length, args.cy],
        [0, 0, 1]
    ])

    # Define extrinsic transformation (from world to new camera view)
    # Here we apply a simple translation; you could add rotation if needed.
    extrinsic = np.eye(4)
    extrinsic[0, 3] = -args.tx  # negative translation for camera extrinsics
    extrinsic[1, 3] = -args.ty
    extrinsic[2, 3] = -args.tz

    # Project the points onto the image plane
    projected_image = project_points(points, colors, intrinsic, extrinsic, args.width, args.height)

    # Save the projected image
    cv2.imwrite(args.out_image, cv2.cvtColor(projected_image, cv2.COLOR_RGB2BGR))
    print(f"Projected image saved to {args.out_image}")

if __name__ == '__main__':
    main()
