#!/usr/bin/env python3

"""
director_time_synchronizer.py
Author: Hayden Feddock
Date: 1/14/2025

This script listens for director commands and camera images, and saves images to disk
when a command is received. The images are saved with the timestamp of the director command.
"""

import os
import cv2
import rclpy
import argparse
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration
from rclpy.subscription import Subscription
from std_msgs.msg import String
from sensor_msgs.msg import Image
from utils import demosaic_ximea_5x5, rectify_image
from ament_index_python.packages import get_package_share_directory
import rosbag2_py

# TODO: Give ability to pass folder and then run all rosbags within that folder sequentially

# Be sure to run this code using the sim time parameter
# ros2 run image_fusion bag_to_images.py --ros-args -p use_sim_time:=True
# And be sure to run the ros2 bag play with the clock option
# ros2 bag play /path/to/bag.bag --clock
# If you want to enable demosaicing for the Ximea camera, use the following command:
# ros2 run image_fusion bag_to_images.py --ros-args -p use_sim_time:=True -p demosaic:=True
# To run with the rectification option, use the following command:
# ros2 run image_fusion bag_to_images.py --ros-args -p use_sim_time:=True -p demosaic:=True -p rectify:=True

class DirectorRequest:
    def __init__(self, id: int, validity_s: float, camera_names: list) -> None:
        """
        Holds data for a single 'director command' request.
        """
        self.id = id
        self.request_timestamp = None
        self.request_duration = Duration(seconds=validity_s)
        self.deadline = None
        self.camera_names = camera_names
        self.images_received = dict()  # camera_name -> cv_image

    def add_image(self, camera_name: str, timestamp: Time, cv_image: cv2.Mat) -> bool:
        """
        Store an image if it's >= request_timestamp and we still haven't stored an image 
        for this camera.
        """
        if camera_name not in self.images_received:
            if self.request_timestamp is None:
                self.request_timestamp = timestamp
                self.deadline = self.request_timestamp + self.request_duration
            if timestamp >= self.request_timestamp and timestamp <= self.deadline:
                self.images_received[camera_name] = cv_image
                return True
            else:
                print(f"Image for {camera_name} at {timestamp.to_msg().sec} is out of request window.")
        return False

    def is_fulfilled(self) -> bool:
        """
        True if we have an image for each camera.
        """
        return len(self.images_received) == len(self.camera_names)


class DirectorTimeSynchronizer(Node):
    def __init__(self) -> None:
        """
        Controls the synchronization of camera images with director commands.
        """
        super().__init__('director_time_synchronizer')

        # Define parameters
        self.declare_parameter('validity_window', 0.5)
        self.declare_parameter('director_topic', '/multi_cam_rig/director')
        self.declare_parameter('camera_names', [
            'firefly_left',
            'firefly_right',
            'ximea',
            'zed_left',
            'zed_right'
        ])
        self.declare_parameter('camera_topics', [
            '/flir_node/firefly_left/image_raw',
            '/flir_node/firefly_right/image_raw',
            '/multi_cam_rig/ximea/image',
            '/multi_cam_rig/zed/left_image',
            '/multi_cam_rig/zed/right_image'
        ])
        self.declare_parameter('output_dir', 'image_data')
        self.declare_parameter('demosaic', False)
        self.declare_parameter('rectify', False)
        self.declare_parameter('calibration_dir', get_package_share_directory('image_fusion') + '/calibration_data')
        self.declare_parameter('image_type', 'png')

        # Set up parameters
        self.validity_window = self.get_parameter('validity_window').get_parameter_value().double_value
        self.director_topic = self.get_parameter('director_topic').get_parameter_value().string_value
        self.camera_names = self.get_parameter('camera_names').get_parameter_value().string_array_value
        self.camera_topics = self.get_parameter('camera_topics').get_parameter_value().string_array_value
        raw_output_dir = self.get_parameter('output_dir').get_parameter_value().string_value
        self.demosaic = self.get_parameter('demosaic').get_parameter_value().bool_value
        self.rectify = self.get_parameter('rectify').get_parameter_value().bool_value
        self.image_type = self.get_parameter('image_type').get_parameter_value().string_value

        # Set up the output directory
        self.output_dir = os.path.expanduser(raw_output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"Output directory: {self.output_dir}")

        # Check the demosaic option
        self.get_logger().info(f"Demosaic enabled: {self.demosaic}")

        # Check the rectification option and the calibration directory
        self.get_logger().info(f"Rectify enabled: {self.rectify}")
        if self.rectify:
            self.calibration_dir = self.get_parameter('calibration_dir').get_parameter_value().string_value
            self.get_logger().info(f"Calibration directory: {self.calibration_dir}")

        # Check the image type
        self.get_logger().info(f"Image type: {self.image_type}")
        if self.image_type.lower() not in ['png', 'jpg', 'jpeg', 'tif', 'tiff']:
            self.get_logger().error(f"Invalid image type: {self.image_type}. Must be 'png', 'jpg', 'jpeg', 'tif', or 'tiff'.")
            exit(1)

        # Map topics to camera names
        self.topic_camera_map = {topic: name for topic, name in zip(self.camera_topics, self.camera_names)}

        # request queue from the director (arrivals are in order of timestamps so no need to sort)
        self.active_request: DirectorRequest = None

        # Director command subscription
        self.director_sub = self.create_subscription(
            String,
            self.director_topic,
            self.director_callback,
            10
        )

        # Create cv bridge
        self.bridge = CvBridge()

        # Image subscriptions
        for topic in self.camera_topics:
            self.create_subscription(Image, topic, lambda msg, t=topic: self.image_callback(t, msg), 10)

        self.get_logger().info("director_time_synchronizer initialized.")

    def director_callback(self, msg: String) -> None:
        """
        A new command arrives from the director. If it's a 'capture' command, we'll open a new request.
        """

        # Get the text of the command
        command = msg.data

        # If the command is 'capture #', open a new request
        if command.startswith('Capture'):

            # Get the id number from the command
            id = int(command.split()[1])

            # Destroy the previous request if it exists
            if self.active_request is not None:
                self.get_logger().info(f"Request {self.active_request.id} expired.")
            
            # Create a new request object
            self.active_request = DirectorRequest(
                id=id,
                validity_s=self.validity_window,
                camera_names=self.camera_names
            )
            self.get_logger().info(f"New request opened: {id}")

    def image_callback(self, camera_topic: str, msg: Image) -> bool:
        """
        Each camera publishes an Image. We'll store it in the active request if it's valid.
        """

        # Get the camera name from the topic
        camera_name = self.topic_camera_map[camera_topic]

        # Get the timestamp of the image from the message header
        timestamp = Time.from_msg(msg.header.stamp)
        print(f"Image received for {camera_name} at {timestamp.to_msg().sec}")

        # Convert Image to CV image
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # If demosaicing is enabled for Ximea, convert image to grayscale and process later.
        if self.demosaic and camera_name.lower() == "ximea":
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        # Attempt to add the image to the request
        if self.active_request is None:
            self.get_logger().info(f"No active request for {camera_name} at {timestamp.to_msg().sec}")
            return False
        
        image_added = self.active_request.add_image(
            camera_name=camera_name,
            timestamp=timestamp,
            cv_image=cv_image
        )

        # If the request is fulfilled, save the images to disk
        if image_added:
            self.get_logger().info(f"Added image for {camera_name} at {timestamp.to_msg().sec}")
            if self.active_request.is_fulfilled():
                self.save_request_images(self.active_request)
                self.active_request = None
                self.get_logger().info("Request fulfilled.")
        else:
            self.get_logger().info(f"Image for {camera_name} at {timestamp.to_msg().sec} not added")

        return image_added
        
    def save_request_images(self, request: DirectorRequest):
        """
        Save each camera's image to disk, using the director's command timestamp
        for the filename.
        """

        # Use the time stamp of the director command for the filename
        director_sec = request.request_timestamp.nanoseconds // 1_000_000_000
        director_nsec = request.request_timestamp.nanoseconds % 1_000_000_000
        stamp_str = f"{director_sec}_{director_nsec}"

        # Save each image
        for cam_name, cv_img in request.images_received.items():

            # Process demosaicing and saving for the Ximea image.
            if self.demosaic and cam_name.lower() == "ximea":
                try:
                    hypercube_dict = demosaic_ximea_5x5(cv_img)
                except Exception as e:
                    self.get_logger().error(f"Demosaicing failed for {cam_name}: {e}")
                    continue

                # Save each band in the hypercube.
                for bandwidth, band_img in hypercube_dict.items():

                    # Get the folder to save the image to.
                    cam_dir = os.path.join(self.output_dir, cam_name)
                    band_dir = os.path.join(cam_dir, str(bandwidth))
                    os.makedirs(band_dir, exist_ok=True)

                    # Check if rectification is enabled.
                    if self.rectify:
                        calib_file = os.path.join(self.calibration_dir, f"{cam_name}.yaml")
                        band_img = rectify_image(band_img, calib_file)
                        filename = f"{stamp_str}_rect_{cam_name}_{bandwidth}.{self.image_type}"
                    else:
                        filename = f"{stamp_str}_{cam_name}_{bandwidth}.{self.image_type}"

                    # Save the image at the correct folder and filename and type.
                    out_path = os.path.join(band_dir, filename)
                    cv2.imwrite(out_path, band_img)
                    self.get_logger().info(f"Saved demosaiced band {bandwidth} for {cam_name} to {out_path}")
            
            # Saving all camera images.
            else:

                # Get the folder to save the image to.
                cam_dir = os.path.join(self.output_dir, cam_name)
                os.makedirs(cam_dir, exist_ok=True)

                # Check if rectification is enabled.
                if self.rectify:
                    calib_file = os.path.join(self.calibration_dir, f"{cam_name}.yaml")
                    cv_img = rectify_image(cv_img, calib_file)
                    filename = f"{stamp_str}_rect_{cam_name}.{self.image_type}"
                else:
                    filename = f"{stamp_str}_{cam_name}.{self.image_type}"

                # Save the image at the correct folder and filename and type.
                out_path = os.path.join(cam_dir, filename)
                cv2.imwrite(out_path, cv_img)

        self.get_logger().info(f"Images saved to {self.output_dir}")


def main(args=None):
    rclpy.init(args=args)
    node = DirectorTimeSynchronizer()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
