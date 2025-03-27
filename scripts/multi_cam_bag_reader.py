# Hayden Feddock

import os
import sys
import cv2
from cv_bridge import CvBridge
from rclpy.serialization import deserialize_message
import rosbag2_py
from std_msgs.msg import String
from sensor_msgs.msg import Image
from typing import Dict
from utils import demosaic_ximea_5x5, rectify_image


# Constants
DIRECTOR_TOPIC = "/multi_cam_rig/director"
CAMERA_TOPICS = {
    "firefly_left":  "/flir_node/firefly_left/image_raw",
    "firefly_right": "/flir_node/firefly_right/image_raw",
    "ximea":         "/multi_cam_rig/ximea/image",
    "zed_left":      "/multi_cam_rig/zed/left_image",
    "zed_right":     "/multi_cam_rig/zed/right_image"
}
VALIDITY_WINDOW = 1.0 * 1e9 # nanoseconds
DEMOSAIC_MODE = True
RECTIFY_MODE = False
CALIBRATION_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "calibration_files")


class CaptureRequest:
    def __init__(self, start_time: int, duration: int) -> None:
        """
        Set of all images captured for a single request.
        """
        self.start_time: int = start_time # nanoseconds
        self.duration: int = duration     # nanoseconds
        self.deadline: int = start_time + duration
        self.images_received: Dict[str, cv2.Mat] = {}

    def add_image(self, camera_name: str, time: int, cv_image: cv2.Mat) -> bool:
        """
        Store an image if it's within the request window.
        """
        if camera_name not in CAMERA_TOPICS:
            print(f"Error: Unknown camera name '{camera_name}'.")
            return False
        if camera_name not in self.images_received:
            if time <= self.deadline:
                self.images_received[camera_name] = cv_image
                return True
        return False

    def is_fulfilled(self) -> bool:
        """
        True if we have an image for each camera.
        """
        return len(self.images_received) == len(CAMERA_TOPICS)
    
    def save_image_set(self, output_dir: str):
        """
        Write the images to the output directory.
        """
        # Create the output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # Create a subdirectory for the request (should not already exist)
        seconds = int(self.start_time // 1e9)
        nano_seconds = int(self.start_time % 1e9)
        output_dir = os.path.join(output_dir, f"{seconds}_{nano_seconds}")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=False)

        # Write each image to the output directory
        for camera_name, image in self.images_received.items():

            # Demosaic the image if necessary
            if camera_name == "ximea":
                if DEMOSAIC_MODE:
                    hypercube_dict = demosaic_ximea_5x5(image, sort_bands=False)

                    # Create a directory for the ximea images
                    band_dir = os.path.join(output_dir, f"{camera_name}")
                    if not os.path.exists(band_dir):
                        os.makedirs(band_dir, exist_ok=False)

                    # Write each band to the output directory
                    for band, band_image in hypercube_dict.items():

                        # Rectify the image if necessary
                        if RECTIFY_MODE:
                            calibration_file = os.path.join(CALIBRATION_DIR, f"{camera_name}.yaml")
                            band_image = rectify_image(band_image, calibration_file)

                        output_file = os.path.join(band_dir, f"{band}.png")
                        cv2.imwrite(output_file, band_image)

                # If not demosaicing, save the grayscale image
                else:
                    output_file = os.path.join(output_dir, f"{camera_name}.png")
                    cv2.imwrite(output_file, image)

            # Save the image as is for other cameras
            else:
                
                # Rectify the image if necessary
                if RECTIFY_MODE:
                    calibration_file = os.path.join(CALIBRATION_DIR, f"{camera_name}.yaml")
                    image = rectify_image(image, calibration_file)
                
                output_file = os.path.join(output_dir, f"{camera_name}.png")
                cv2.imwrite(output_file, image)

        print(f"Saved images to: {output_dir}")
    
class MultiCamBagReader:
    def __init__(self, output_dir: str) -> None:
        """
        Object for reading a multi-camera bag file and storing the capture sets in an output file.
        """
        self.output_dir: str = output_dir
        self.reader: rosbag2_py.SequentialReader = None
        self.type_map: Dict[str, int] = None
        self.active_request: CaptureRequest = None
        self.bridge = CvBridge()

    def read_data(self, reader: rosbag2_py.SequentialReader):
        """
        Begin reading all messages from the bag file and store the capture sets in the output file.
        """
        self.reader = reader
        self.type_map: Dict[str, int] = {meta.name: meta.type for meta in reader.get_all_topics_and_types()}
        while self.reader.has_next():
            topic, data, timestamp = self.reader.read_next()
            self.process_message(topic, data, timestamp)

    def process_message(self, topic: str, data: bytes, timestamp: int):
        """
        Deserialize the message and process based on the topic.
        """
        if topic == DIRECTOR_TOPIC:
            msg = deserialize_message(data, String)
            self.process_director_message(msg, timestamp)
        elif topic in CAMERA_TOPICS.values():
            msg = deserialize_message(data, Image)
            camera_name = [name for name, topic_name in CAMERA_TOPICS.items() if topic_name == topic][0]
            self.process_camera_message(camera_name, msg, timestamp)

    def process_director_message(self, msg: String, timestamp: int):
        """
        Process a message from the director topic.
        """
        # Check for the "Capture" command
        if msg.data.startswith("Capture"):

            # Close the previous request if it exists
            if self.active_request is not None:
                self.close_active_request()

            # Create a new request
            self.active_request = CaptureRequest(
                start_time=timestamp,
                duration=VALIDITY_WINDOW
            )

    def process_camera_message(self, camera_name: str, msg: Image, timestamp: int):
        """
        Process a message from a camera topic.
        """
        # Convert Image to CV image
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # If image is from Ximea convert to grayscale
        if camera_name == "ximea":
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        # Add the image to the active request if it exists
        if self.active_request is not None:
            self.active_request.add_image(camera_name, timestamp, cv_image)

            # Close the request if it is fulfilled
            if self.active_request.is_fulfilled():
                self.close_active_request()

    def close_active_request(self):
        """
        Close the active request and write the images to the output file.
        """
        # Check if the request still exists
        if self.active_request is None:
            return
        
        # Check if the request is fulfilled and save the images if so
        if self.active_request.is_fulfilled():
            self.active_request.save_image_set(self.output_dir)
        else:
            print("Request expired before all images were received.")
        self.active_request = None


def main():

    # Retrieve the bag file path and output file path from the arguments
    if len(sys.argv) < 3:
        print("Usage: python multi_cam_bag_reader.py <bag_file_path> <output_file_path>")
        sys.exit(1)

    bag_file = sys.argv[1]
    output_file = sys.argv[2]

    # Expand the output file path if it contains '~'
    output_file = os.path.expanduser(output_file)

    # Create the output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Configure storage options: specify the file path and storage type (e.g., 'sqlite3')
    storage_options = rosbag2_py.StorageOptions(uri=bag_file, storage_id="sqlite3")

    # Configure converter options for message serialization
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format='cdr',
        output_serialization_format='cdr'
    )

    # Create a reader instance and open the bag file
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    # Create the MultiCamBagReader instance and read the bag file
    multi_cam_reader = MultiCamBagReader(output_file)
    multi_cam_reader.read_data(reader)

    print("Finished reading bag file.")
    
if __name__ == "__main__":
    main()










