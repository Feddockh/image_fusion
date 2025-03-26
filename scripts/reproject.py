#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to project a point cloud (cloud.ply) onto a 2D image plane.
This simulates the view from a nearby camera relative to the left camera of a stereo pair.
The output is a 2D image that can be used to stack additional channels for object detection.
"""

import os
import sys
import argparse
import numpy as np
import open3d as o3d
import cv2
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from omegaconf import OmegaConf
from FoundationStereo.core.foundation_stereo import *
from FoundationStereo.core.utils.utils import InputPadder
from FoundationStereo.Utils import *
from FoundationStereo.core.foundation_stereo import *

from utils import load_camera_params, project_points


def main():
    code_dir = os.path.dirname(os.path.realpath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument('--left_file', default=f'{code_dir}/../demo_data/firefly/left_img_rectified.png', type=str)
    parser.add_argument('--right_file', default=f'{code_dir}/../demo_data/firefly/right_img_rectified.png', type=str)
    parser.add_argument('--intrinsic_file', default=f'{code_dir}/../demo_data/firefly/K.txt', type=str, help='camera intrinsic matrix and baseline file')
    parser.add_argument('--ckpt_dir', default=f'{code_dir}/../demo_data/pretrained_models/model_best_bp2.pth', type=str, help='pretrained model path')
    parser.add_argument('--out_dir', default=f'{code_dir}/../output/reprojection', type=str, help='the directory to save results')
    parser.add_argument('--scale', default=1.0, type=float, help='downsize the image by scale, must be <=1')
    parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')
    parser.add_argument('--z_far', default=10, type=float, help='max depth to clip in point cloud')
    parser.add_argument('--denoise_cloud', type=int, default=1, help='whether to denoise the point cloud')
    parser.add_argument('--denoise_nb_points', type=int, default=30, help='number of points to consider for radius outlier removal')
    parser.add_argument('--denoise_radius', type=float, default=0.03, help='radius to use for outlier removal')
    args = parser.parse_args()

    set_logging_format()
    set_seed(0)
    torch.autograd.set_grad_enabled(False)
    os.makedirs(args.out_dir, exist_ok=True)

    # Set up the model
    ckpt_dir = args.ckpt_dir
    cfg = {}
    for k in args.__dict__:
        cfg[k] = args.__dict__[k]
    args = OmegaConf.create(cfg)
    logging.info(f"args:\n{args}")
    logging.info(f"Using pretrained model from {ckpt_dir}")

    cfg = OmegaConf.load(f'{os.path.dirname(ckpt_dir)}/cfg.yaml')
    model = FoundationStereo(cfg)

    ckpt = torch.load(ckpt_dir)
    logging.info(f"ckpt global_step:{ckpt['global_step']}, epoch:{ckpt['epoch']}")
    model.load_state_dict(ckpt['model'])

    model.cuda()
    model.eval()

    # Load and scale the images
    code_dir = os.path.dirname(os.path.realpath(__file__))
    img0 = imageio.imread(args.left_file)
    img1 = imageio.imread(args.right_file)
    scale = args.scale
    assert scale<=1, "scale must be <=1"
    img0 = cv2.resize(img0, fx=scale, fy=scale, dsize=None)
    img1 = cv2.resize(img1, fx=scale, fy=scale, dsize=None)
    H,W = img0.shape[:2]
    img0_ori = img0.copy()
    logging.info(f"img0: {img0.shape}")

    # Send images to GPU and pad them
    img0 = torch.as_tensor(img0).cuda().float()[None].permute(0,3,1,2)
    img1 = torch.as_tensor(img1).cuda().float()[None].permute(0,3,1,2)
    padder = InputPadder(img0.shape, divis_by=32, force_square=False)
    img0, img1 = padder.pad(img0, img1)

    # Start the inference timer
    start_time = time.time()

    # Perform the inference
    with torch.cuda.amp.autocast(True):
        disp = model.forward(img0, img1, iters=args.valid_iters, test_mode=True)
        
    # End the timer
    end_time = time.time()
    logging.info(f"Inference time: {end_time - start_time:.2f} seconds")
    
    # Unpad the disparity map
    disp = padder.unpad(disp.float())
    disp = disp.data.cpu().numpy().reshape(H,W)

    # Remove non-overlapping observations
    yy,xx = np.meshgrid(np.arange(disp.shape[0]), np.arange(disp.shape[1]), indexing='ij')
    us_right = xx-disp
    invalid = us_right<0
    disp[invalid] = np.inf

    # Compute the point cloud
    with open(args.intrinsic_file, 'r') as f:
        lines = f.readlines()
        K = np.array(list(map(float, lines[0].rstrip().split()))).astype(np.float32).reshape(3,3)
        baseline = float(lines[1])
    K[:2] *= scale
    depth = K[0,0]*baseline/disp
    np.save(f'{args.out_dir}/depth_meter.npy', depth)
    xyz_map = depth2xyzmap(depth, K)
    pcd = toOpen3dCloud(xyz_map.reshape(-1,3), img0_ori.reshape(-1,3))
    keep_mask = (np.asarray(pcd.points)[:,2]>0) & (np.asarray(pcd.points)[:,2]<=args.z_far)
    keep_ids = np.arange(len(np.asarray(pcd.points)))[keep_mask]
    pcd = pcd.select_by_index(keep_ids)

    # Denoise the point cloud
    cl, ind = pcd.remove_radius_outlier(nb_points=args.denoise_nb_points, radius=args.denoise_radius)
    inlier_cloud = pcd.select_by_index(ind)
    o3d.io.write_point_cloud(f'{args.out_dir}/cloud_denoise.ply', inlier_cloud)

    # Use the inlier_cloud as our source points/colors
    points = np.asarray(inlier_cloud.points)
    colors = np.asarray(inlier_cloud.colors)

    # Construct the 3x3 instrinsic camera matrix
    K = np.array([
        [594.11134, 0,       198.16808],
        [0,       593.98703, 106.49943],
        [0,         0,         1      ]
    ])
    image_width = 409
    image_height = 217

    # Use the given relative rotation and translation
    R = np.array([
        [9.99791561e-01, -1.07668824e-02, -5.79904528e-04],
        [1.07101035e-02,  9.99621061e-01,  5.79203755e-03],
        [5.38526982e-04, -5.76565471e-03,  9.99544378e-01]
    ], dtype=np.float32)

    t = np.array([
        [-0.02886995],
        [-0.03885978],
        [ 0.01230636]
    ], dtype=np.float32)

    # Build a 4x4 extrinsic matrix (world->camera)
    extrinsic = np.eye(4, dtype=np.float32)
    extrinsic[:3, :3] = R
    extrinsic[:3, 3] = t.flatten()

    # Now call project_points with this extrinsic
    projected_image = project_points(
        points,          # Nx3 array of 3D points
        colors,          # Nx3 array of RGB colors
        K,               # 3x3 intrinsic matrix
        extrinsic,       # 4x4 matrix created above
        image_width,           # output image width
        image_height           # output image height
    )

    # Save the projected image
    cv2.imwrite(f'{args.out_dir}/projected_view.png', cv2.cvtColor(projected_image, cv2.COLOR_RGB2BGR))
    print(f"Projected image saved to {args.out_dir}/projected_view.png")

if __name__ == '__main__':
    main()

