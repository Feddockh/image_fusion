#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to load a point cloud from a .ply file and display it using Open3D.
"""

import argparse
import open3d as o3d

def load_and_display_cloud(ply_file):
    # Load the point cloud from file
    pcd = o3d.io.read_point_cloud(ply_file)
    if not pcd.has_points():
        print(f"Error: No points found in {ply_file}")
        return

    print(f"Loaded point cloud from {ply_file}")
    
    # Visualize the point cloud
    o3d.visualization.draw_geometries([pcd],
                                      window_name='Point Cloud Viewer',
                                      width=800,
                                      height=600,
                                      left=50,
                                      top=50,
                                      point_show_normal=False)

def main():
    parser = argparse.ArgumentParser(
        description="Load and display a point cloud from a .ply file using Open3D."
    )
    parser.add_argument('--ply_file', type=str, default='output/foundationstereo/cloud_denoise.ply',
                        help="Path to the .ply point cloud file")
    args = parser.parse_args()
    
    load_and_display_cloud(args.ply_file)

if __name__ == '__main__':
    main()
