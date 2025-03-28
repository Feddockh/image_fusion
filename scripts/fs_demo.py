import os
import sys
import torch
import numpy as np
import imageio
import cv2
import open3d as o3d
from utils import Camera, MultiCamCapture, project_points
from omegaconf import OmegaConf
import matplotlib.pyplot as plt

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
    model_dir = "/home/hayden/cmu/kantor_lab/ros2_ws/src/image_fusion/FoundationStereo/pretrained_models"
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
    Rt = np.eye(4)
    Rt[:3,:3] = R
    Rt[:3,3] = t

    proj_img, depth_img = project_points(
        points=np.asarray(pcd.points),
        colors=np.asarray(pcd.colors),
        intrinsic=ximea.camera_matrix,
        extrinsic=np.linalg.inv(Rt),
        width=ximea.width,
        height=ximea.height
    )

    # Pop up window to select a point on the image
    def select_point(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"Selected point: ({x}, {y})")
            param["selected_point"] = (x, y)

    params = {"selected_point": None}
    cv2.imshow("Select a point", proj_img)
    cv2.setMouseCallback("Select a point", select_point, params)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    selected_point = params["selected_point"]
    if selected_point:
        print(f"User selected point: {selected_point}")
    else:
        print("No point selected.")
        return
    
    # Get the 3D point corresponding to the selected pixel
    x, y = selected_point
    z = depth_img[y, x]
    if z == 0 or z == np.inf:
        print("Selected point is not valid.")
        return

    # Back-project the pixel using the camera intrinsics.
    K_ximea = ximea.camera_matrix
    p_ximea = np.array([x, y, 1])
    P_ximea = np.linalg.inv(K_ximea) @ p_ximea * z
    P_ximea_homo = np.append(P_ximea, 1)

    # Transform the point to the other camera's coordinate system
    P_firefly = Rt @ P_ximea_homo
    P_firefly /= P_firefly[3]

    # Project the transformed point back onto the target camera image
    K_firefly = firefly_left.camera_matrix
    p_firefly_homo = K_firefly @ P_firefly[:3]
    p_firefly = p_firefly_homo[:2] / p_firefly_homo[2]
    x_proj, y_proj = p_firefly[0], p_firefly[1]

    # Mark the selected point on the projected image (from ximea)
    ximea_img = multi_cam_capture.get_images("ximea")[0].copy()
    sel_pt = (int(selected_point[0]), int(selected_point[1]))

    # Mark the reprojected point on the target camera image (firefly_left)
    firefly_img = multi_cam_capture.get_images("firefly_left")[0].copy()
    proj_pt = (int(x_proj), int(y_proj))

    # Convert images to RGB for matplotlib
    ximea_img_rgb = cv2.cvtColor(ximea_img, cv2.COLOR_BGR2RGB)
    firefly_img_rgb = cv2.cvtColor(firefly_img, cv2.COLOR_BGR2RGB)

    # Create a figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Left subplot: Ximea image
    axes[0].imshow(ximea_img_rgb)
    axes[0].add_patch(plt.Circle(sel_pt, radius=5, color='red', fill=True))
    axes[0].text(sel_pt[0]+10, sel_pt[1]+10, f"({sel_pt[0]}, {sel_pt[1]})",
                fontsize=12, color='red')
    axes[0].set_title("Ximea Projection with Selected Point")
    axes[0].axis('off')

    # Right subplot: Firefly_left image
    axes[1].imshow(firefly_img_rgb)
    axes[1].add_patch(plt.Circle(proj_pt, radius=5, color='green', fill=True))
    axes[1].text(proj_pt[0]+10, proj_pt[1]+10, f"({proj_pt[0]}, {proj_pt[1]})",
                fontsize=12, color='green')
    axes[1].set_title("Firefly_left Image with Projected Point")
    axes[1].axis('off')

    plt.show()





if __name__ == "__main__":
    main()