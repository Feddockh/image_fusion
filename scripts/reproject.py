import os
import sys
import torch
import numpy as np
import imageio
import cv2
import open3d as o3d
from utils import Camera, MultiCamCapture, project_points
from omegaconf import OmegaConf

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from FoundationStereo.core.foundation_stereo import *
from FoundationStereo.core.utils.utils import InputPadder
from FoundationStereo.Utils import *


def main():
    """
    Minimal example to load FoundationStereo, run prediction, and create a point cloud.
    Adjust file paths for your environment.
    """

    # ------------------
    # 1. Torch setup
    # ------------------
    torch.set_grad_enabled(False)

    # ------------------
    # 2. Load cameras and images (rectify)
    # ------------------
    firefly_left = Camera("firefly_left")
    firefly_right = Camera("firefly_right")
    ximea = Camera("ximea")
    firefly_left.load_params()
    firefly_right.load_params()
    ximea.load_params()

    capture_path = "/home/hayden/cmu/kantor_lab/ros2_ws/image_data/1739374359_707437312"
    multi_cam_capture = MultiCamCapture([firefly_left, firefly_right, ximea], capture_path)
    multi_cam_capture.load_images()
    multi_cam_capture.undistort_rectify_images()

    # Read the stereo images
    img0 = multi_cam_capture.get_images("firefly_left")[0]
    img1 = multi_cam_capture.get_images("firefly_right")[0]
    img0_ori = img0.copy()
    H,W = img0.shape[:2]

    # Convert to tensor (BxCxHxW), float, move to device
    img0 = torch.as_tensor(img0).cuda().float()[None].permute(0,3,1,2)
    img1 = torch.as_tensor(img1).cuda().float()[None].permute(0,3,1,2)
    padder = InputPadder(img0.shape, divis_by=32, force_square=False)
    img0, img1 = padder.pad(img0, img1)

    # ------------------
    # 3. Load model
    # ------------------
    model_dir = "FoundationStereo/pretrained_models"
    checkpoint_file = model_dir + "/model_best_bp2.pth"
    cfg = OmegaConf.load(model_dir + "/cfg.yaml")
    model = FoundationStereo(cfg)  # create the stereo model
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint["model"])
    model.cuda()
    model.eval()

    # ------------------
    # 4. Inference
    # ------------------
    with torch.cuda.amp.autocast(True):
        disp = model.forward(img0, img1, iters=32, test_mode=True)
    disp = padder.unpad(disp.float())
    disp = disp.data.cpu().numpy().reshape(H,W)

    # Remove non-overlapping observations
    yy,xx = np.meshgrid(np.arange(disp.shape[0]), np.arange(disp.shape[1]), indexing='ij')
    us_right = xx-disp
    invalid = us_right<0
    disp[invalid] = np.inf

    # ------------------
    # 5. Convert disparity -> 3D points
    # ------------------
    K = firefly_right.camera_matrix
    P = firefly_right.projection_matrix
    baseline = abs(P[0, 3] / P[0, 0]) # baseline in meters
    depth = K[0,0]*baseline/disp
    xyz_map = depth2xyzmap(depth, K)
    pcd = toOpen3dCloud(xyz_map.reshape(-1,3), img0_ori.reshape(-1,3))
    keep_mask = (np.asarray(pcd.points)[:,2]>0) & (np.asarray(pcd.points)[:,2]<=10)
    keep_ids = np.arange(len(np.asarray(pcd.points)))[keep_mask]
    pcd = pcd.select_by_index(keep_ids)

    # Denoise the point cloud
    cl, ind = pcd.remove_radius_outlier(nb_points=30, radius=0.03)
    inlier_cloud = pcd.select_by_index(ind)
    pcd = inlier_cloud

    # logging.info("Visualizing point cloud. Press ESC to exit.")
    # vis = o3d.visualization.Visualizer()
    # vis.create_window()
    # vis.add_geometry(pcd)
    # vis.get_render_option().point_size = 1.0
    # vis.get_render_option().background_color = np.array([0.5, 0.5, 0.5])
    # vis.run()
    # vis.destroy_window()
    R, t = ximea.transforms["firefly_left"]
    extrinsic = np.eye(4)
    extrinsic[:3,:3] = R
    extrinsic[:3,3] = t
    extrinsic = np.linalg.inv(extrinsic)

    proj_img, depth_img = project_points(
        points=np.asarray(pcd.points),
        colors=np.asarray(pcd.colors),
        intrinsic=ximea.camera_matrix,
        extrinsic=extrinsic,
        width=ximea.width,
        height=ximea.height
    )
    ximea_img = multi_cam_capture.get_images("ximea")[0]

    cv2.imshow("ximea", ximea_img)
    cv2.imshow("projected firefly", proj_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()