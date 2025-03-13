#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class DemosaicNode(Node):
    def __init__(self):
        super().__init__('demosaic_node')
        # Subscribe to the input image topic (adjust topic name as needed)
        self.subscription = self.create_subscription(
            Image,
            '/multi_cam_rig/ximea/image',
            self.image_callback,
            10)
        # Publisher for the demosaicked image (adjust topic name as needed)
        self.publisher = self.create_publisher(
            Image,
            '/multi_cam_rig/ximea/image_demosaic',
            10)
        self.bridge = CvBridge()
        self.get_logger().info("Demosaic node started.")

    def image_callback(self, msg: Image):
        try:
            # Convert the ROS Image message to an OpenCV image (assumed mono8)
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")
            return

        # Optionally, if the image dimensions are not divisible by 5, crop the image
        height, width = cv_image.shape
        if height % 5 != 0 or width % 5 != 0:
            new_height = (height // 5) * 5
            new_width = (width // 5) * 5
            self.get_logger().warn(
                f"Image dimensions ({height}x{width}) not divisible by 5, cropping to {new_height}x{new_width}."
            )
            cv_image = cv_image[:new_height, :new_width]

        # Extract the top-left pixel from each 5x5 block
        demosaicked_image = cv_image[0::5, 0::5]

        # Convert the resulting image back to a ROS Image message
        new_msg = self.bridge.cv2_to_imgmsg(demosaicked_image, encoding='mono8')
        self.publisher.publish(new_msg)
        self.get_logger().info("Published demosaicked image.")

def main(args=None):
    rclpy.init(args=args)
    node = DemosaicNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard interrupt received, shutting down node.")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
