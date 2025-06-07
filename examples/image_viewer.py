#!/usr/bin/env python3

import time

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image

# --- Configuration ---
DEFAULT_RAW_IMAGE_TOPIC = "/image_raw"
DEFAULT_COMPRESSED_IMAGE_TOPIC = "/camera/fourd/stereo/raw/compressed"

RAW_WINDOW_NAME = "Raw Image"
COMPRESSED_WINDOW_NAME = "Compressed Image"
WINDOW_BG_COLOR = (30, 30, 30)  # Dark gray for placeholder

# Text properties for overlay
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
FONT_COLOR = (255, 255, 255)  # White
FONT_THICKNESS = 1
LINE_TYPE = cv2.LINE_AA


class ImageVisualizer(Node):
    def __init__(self, initial_raw_topic, initial_compressed_topic):
        super().__init__("image_visualizer_interactive")
        self.bridge = CvBridge()

        # --- State for topic management ---
        self.available_raw_topics = []
        self.available_compressed_topics = []

        self.current_raw_topic_name = initial_raw_topic
        self.current_compressed_topic_name = initial_compressed_topic

        self.current_raw_topic_index = -1
        self.current_compressed_topic_index = -1

        # --- Image storage ---
        self.last_raw_cv_image = None
        self.last_compressed_cv_image = None

        # --- Subscribers ---
        self.raw_image_sub = None  # Initialize raw_image_sub to avoid AttributeError
        self.compressed_image_sub = (
            None  # Initialize compressed_image_sub to avoid AttributeError
        )
        # --- Initialize and discover topics ---
        self.update_available_topics()  # Initial discovery

        # Set initial indices based on provided topic names
        if self.current_raw_topic_name in self.available_raw_topics:
            self.current_raw_topic_index = self.available_raw_topics.index(
                self.current_raw_topic_name
            )
        elif self.available_raw_topics:
            self.current_raw_topic_index = 0
            self.current_raw_topic_name = self.available_raw_topics[0]

        if self.current_compressed_topic_name in self.available_compressed_topics:
            self.current_compressed_topic_index = (
                self.available_compressed_topics.index(
                    self.current_compressed_topic_name
                )
            )
        elif self.available_compressed_topics:
            self.current_compressed_topic_index = 0
            self.current_compressed_topic_name = self.available_compressed_topics[0]

        # --- Subscribers (will be created/recreated) ---
        self.raw_image_sub = None
        self.compressed_image_sub = None
        self._create_raw_subscriber()
        self._create_compressed_subscriber()

        # --- OpenCV Windows & Display Timer ---
        cv2.namedWindow(RAW_WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.namedWindow(COMPRESSED_WINDOW_NAME, cv2.WINDOW_NORMAL)

        self.display_timer_period = 0.05  # 20 FPS
        self.display_timer = self.create_timer(
            self.display_timer_period, self.update_display_and_handle_keys
        )

        self.get_logger().info("Interactive Image Visualizer started.")
        self.get_logger().info(
            "Press 'N' for next Raw Topic, 'M' for next Compressed Topic."
        )
        self.get_logger().info("Press 'R' to Refresh Topic Lists, 'Q' to Quit.")

    def update_available_topics(self):
        self.get_logger().info("Updating available image topics...")
        topic_names_and_types = self.get_topic_names_and_types()

        self.available_raw_topics = sorted(
            [
                name
                for name, types in topic_names_and_types
                if "sensor_msgs/msg/Image" in types
            ]
        )
        self.available_compressed_topics = sorted(
            [
                name
                for name, types in topic_names_and_types
                if "sensor_msgs/msg/CompressedImage" in types
            ]
        )

        self.get_logger().info(
            f"Found {len(self.available_raw_topics)} raw topics: {self.available_raw_topics}"
        )
        self.get_logger().info(
            f"Found {len(self.available_compressed_topics)} compressed topics: {self.available_compressed_topics}"
        )

        # If current topic is no longer available, try to reset (or handle gracefully)
        if (
            self.current_raw_topic_name not in self.available_raw_topics
            and self.available_raw_topics
        ):
            self.current_raw_topic_index = 0
            self.current_raw_topic_name = self.available_raw_topics[0]
            self._recreate_raw_subscriber()
        elif not self.available_raw_topics:
            self.current_raw_topic_index = -1
            self.current_raw_topic_name = "N/A"
            if self.raw_image_sub:
                self.destroy_subscription(self.raw_image_sub)
                self.raw_image_sub = None

        if (
            self.current_compressed_topic_name not in self.available_compressed_topics
            and self.available_compressed_topics
        ):
            self.current_compressed_topic_index = 0
            self.current_compressed_topic_name = self.available_compressed_topics[0]
            self._recreate_compressed_subscriber()
        elif not self.available_compressed_topics:
            self.current_compressed_topic_index = -1
            self.current_compressed_topic_name = "N/A"
            if self.compressed_image_sub:
                self.destroy_subscription(self.compressed_image_sub)
                self.compressed_image_sub = None

    def _create_raw_subscriber(self):
        if self.raw_image_sub:
            self.destroy_subscription(self.raw_image_sub)  # Stop the old subscriber
            self.raw_image_sub = None
            self.last_raw_cv_image = None  # Clear last image

        if self.current_raw_topic_index != -1 and self.available_raw_topics:
            topic_name = self.available_raw_topics[self.current_raw_topic_index]
            self.current_raw_topic_name = topic_name
            self.raw_image_sub = self.create_subscription(
                Image, topic_name, self.raw_image_callback, 10
            )
            self.get_logger().info(f"Subscribing to RAW topic: '{topic_name}'")
        else:
            self.get_logger().warn(
                "No suitable raw topic to subscribe to or index out of bounds."
            )
            self.current_raw_topic_name = "N/A"
            self.last_raw_cv_image = None  # Ensure placeholder image is shown

    def _create_compressed_subscriber(self):
        if self.compressed_image_sub:
            self.destroy_subscription(
                self.compressed_image_sub
            )  # Stop the old subscriber
            self.compressed_image_sub = None
            self.last_compressed_cv_image = None  # Clear last image

        if (
            self.current_compressed_topic_index != -1
            and self.available_compressed_topics
        ):
            topic_name = self.available_compressed_topics[
                self.current_compressed_topic_index
            ]
            self.current_compressed_topic_name = topic_name
            self.compressed_image_sub = self.create_subscription(
                CompressedImage, topic_name, self.compressed_image_callback, 10
            )
            self.get_logger().info(f"Subscribing to COMPRESSED topic: '{topic_name}'")
        else:
            self.get_logger().warn(
                "No suitable compressed topic to subscribe to or index out of bounds."
            )
            self.current_compressed_topic_name = "N/A"
            self.last_compressed_cv_image = None  # Ensure placeholder image is shown

    def _recreate_raw_subscriber(self):
        if self.raw_image_sub:
            self.destroy_subscription(self.raw_image_sub)  # Stop the old subscriber
        self.last_raw_cv_image = None
        self._create_raw_subscriber()

    def _recreate_compressed_subscriber(self):
        if self.compressed_image_sub:
            self.destroy_subscription(
                self.compressed_image_sub
            )  # Stop the old subscriber
        self.last_compressed_cv_image = None
        self._create_compressed_subscriber()

    def raw_image_callback(self, msg: Image):
        try:
            self.last_raw_cv_image = self.bridge.imgmsg_to_cv2(
                msg, desired_encoding="bgr8"
            )
            # self.get_logger().debug(f"Raw image received: {msg.width}x{msg.height}, encoding: {msg.encoding}")
        except Exception as e:
            self.get_logger().error(f"Failed to convert raw image: {e}")
            self.last_raw_cv_image = None

    def compressed_image_callback(self, msg: CompressedImage):
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            decoded_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if decoded_image is not None:
                self.last_compressed_cv_image = decoded_image
            else:
                self.get_logger().warn(
                    f"Failed to decode compressed image (format: {msg.format})"
                )
                self.last_compressed_cv_image = None
            # self.get_logger().debug(f"Compressed image received: format {msg.format}")
        except Exception as e:
            self.get_logger().error(f"Failed to process compressed image: {e}")
            self.last_compressed_cv_image = None

    def _draw_text_on_image(
        self, image, topic_name, key_next, key_refresh="R", key_quit="Q"
    ):
        """Helper to draw informational text on an image."""
        # Create a black bar at the bottom for text
        h, w = image.shape[:2]
        bar_height = 60
        cv2.rectangle(image, (0, h - bar_height), (w, h), (0, 0, 0), -1)  # Black bar

        line1 = f"Topic: {topic_name}"
        line2 = f"[{key_next}]Next [{key_refresh}]Refresh [{key_quit}]Quit"

        cv2.putText(
            image,
            line1,
            (10, h - bar_height + 20),
            FONT,
            FONT_SCALE,
            FONT_COLOR,
            FONT_THICKNESS,
            LINE_TYPE,
        )
        cv2.putText(
            image,
            line2,
            (10, h - bar_height + 45),
            FONT,
            FONT_SCALE,
            FONT_COLOR,
            FONT_THICKNESS,
            LINE_TYPE,
        )
        return image

    def _create_placeholder_image(self, window_name, message, topic_name="N/A"):
        """Creates a blank image with a message."""
        img = np.full((480, 640, 3), WINDOW_BG_COLOR, dtype=np.uint8)

        title_y = 30
        cv2.putText(
            img,
            window_name,
            (10, title_y),
            FONT,
            FONT_SCALE * 1.2,
            FONT_COLOR,
            FONT_THICKNESS,
            LINE_TYPE,
        )

        msg_y = 220
        cv2.putText(
            img,
            message,
            (10, msg_y),
            FONT,
            FONT_SCALE,
            FONT_COLOR,
            FONT_THICKNESS,
            LINE_TYPE,
        )

        h, w = img.shape[:2]
        bar_height = 60
        cv2.rectangle(img, (0, h - bar_height), (w, h), (0, 0, 0), -1)

        line1 = f"Topic: {topic_name}"
        line2 = ""
        if window_name == RAW_WINDOW_NAME:
            line2 = f"[N]Next [R]Refresh [Q]Quit"
        elif window_name == COMPRESSED_WINDOW_NAME:
            line2 = f"[M]Next [R]Refresh [Q]Quit"

        cv2.putText(
            img,
            line1,
            (10, h - bar_height + 20),
            FONT,
            FONT_SCALE,
            FONT_COLOR,
            FONT_THICKNESS,
            LINE_TYPE,
        )
        cv2.putText(
            img,
            line2,
            (10, h - bar_height + 45),
            FONT,
            FONT_SCALE,
            FONT_COLOR,
            FONT_THICKNESS,
            LINE_TYPE,
        )
        return img

    def update_display_and_handle_keys(self):
        # Raw Image Window
        if self.raw_image_sub and self.last_raw_cv_image is not None:
            if self.current_raw_topic_name in self.available_raw_topics:
                display_raw = self.last_raw_cv_image.copy()
                self._draw_text_on_image(display_raw, self.current_raw_topic_name, "N")
                cv2.imshow(RAW_WINDOW_NAME, display_raw)
            else:
                self.last_raw_cv_image = None  # Clear image if topic is invalid
        else:
            status_msg = "Waiting for image..."
            if not self.available_raw_topics:
                status_msg = "No raw topics found."
            elif self.current_raw_topic_index == -1:
                status_msg = "No raw topic selected."
            placeholder_raw = self._create_placeholder_image(
                RAW_WINDOW_NAME, status_msg, self.current_raw_topic_name
            )
            cv2.imshow(RAW_WINDOW_NAME, placeholder_raw)

        # Compressed Image Window
        if self.compressed_image_sub and self.last_compressed_cv_image is not None:
            if self.current_compressed_topic_name in self.available_compressed_topics:
                display_compressed = self.last_compressed_cv_image.copy()
                self._draw_text_on_image(
                    display_compressed, self.current_compressed_topic_name, "M"
                )
                cv2.imshow(COMPRESSED_WINDOW_NAME, display_compressed)
            else:
                self.last_compressed_cv_image = None  # Clear image if topic is invalid
        else:
            status_msg = "Waiting for image..."
            if not self.available_compressed_topics:
                status_msg = "No compressed topics found."
            elif self.current_compressed_topic_index == -1:
                status_msg = "No compressed topic selected."
            placeholder_compressed = self._create_placeholder_image(
                COMPRESSED_WINDOW_NAME, status_msg, self.current_compressed_topic_name
            )
            cv2.imshow(COMPRESSED_WINDOW_NAME, placeholder_compressed)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            self.get_logger().info("Shutdown requested via 'Q' key.")
            self.initiate_shutdown()
        elif key == ord("n"):
            if self.available_raw_topics:
                self.current_raw_topic_index = (self.current_raw_topic_index + 1) % len(
                    self.available_raw_topics
                )
                self._recreate_raw_subscriber()
            else:
                self.get_logger().warn("No raw topics available to switch.")
        elif key == ord("m"):
            if self.available_compressed_topics:
                self.current_compressed_topic_index = (
                    self.current_compressed_topic_index + 1
                ) % len(self.available_compressed_topics)
                self._recreate_compressed_subscriber()
            else:
                self.get_logger().warn("No compressed topics available to switch.")
        elif key == ord("r"):
            self.update_available_topics()
            # Re-check and potentially re-create subscribers if current ones are invalid or lists changed
            if (
                self.current_raw_topic_name not in self.available_raw_topics
                and self.available_raw_topics
            ):
                self.current_raw_topic_index = (
                    0  # Default to first if current disappeared
                )
                self._recreate_raw_subscriber()
            elif (
                not self.available_raw_topics and self.raw_image_sub
            ):  # No topics, but sub exists
                self.destroy_subscription(self.raw_image_sub)
                self.raw_image_sub = None
                self.current_raw_topic_name = "N/A"

            if (
                self.current_compressed_topic_name
                not in self.available_compressed_topics
                and self.available_compressed_topics
            ):
                self.current_compressed_topic_index = 0
                self._recreate_compressed_subscriber()
            elif not self.available_compressed_topics and self.compressed_image_sub:
                self.destroy_subscription(self.compressed_image_sub)
                self.compressed_image_sub = None
                self.current_compressed_topic_name = "N/A"

    def initiate_shutdown(self):
        self.display_timer.cancel()  # Stop the timer
        # destroy_node will be called in main's finally block
        # which also handles cv2.destroyAllWindows()
        if rclpy.ok():
            rclpy.try_shutdown()  # Request shutdown of rclpy context

    def custom_destroy_node(self):
        """Custom cleanup."""
        self.get_logger().info("Custom destroy: Closing OpenCV windows...")
        if self.display_timer and not self.display_timer.canceled:
            self.display_timer.cancel()
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)

    temp_node = rclpy.create_node("param_fetcher_for_visualizer")
    temp_node.declare_parameter("raw_topic", DEFAULT_RAW_IMAGE_TOPIC)
    temp_node.declare_parameter("compressed_topic", DEFAULT_COMPRESSED_IMAGE_TOPIC)

    initial_raw_topic = (
        temp_node.get_parameter("raw_topic").get_parameter_value().string_value
    )
    initial_compressed_topic = (
        temp_node.get_parameter("compressed_topic").get_parameter_value().string_value
    )
    temp_node.destroy_node()

    image_visualizer = ImageVisualizer(initial_raw_topic, initial_compressed_topic)

    try:
        rclpy.spin(image_visualizer)
    except KeyboardInterrupt:
        image_visualizer.get_logger().info(
            "Keyboard interrupt received. Shutting down..."
        )
    except Exception as e:
        image_visualizer.get_logger().error(f"Unhandled exception during spin: {e}")
    finally:
        if rclpy.ok():
            image_visualizer.get_logger().info(
                "Spin ended or interrupted. Cleaning up..."
            )
            image_visualizer.custom_destroy_node()  # Ensure OpenCV windows are closed
        if rclpy.ok():
            rclpy.shutdown()
        image_visualizer.get_logger().info("Visualizer shut down complete.")


if __name__ == "__main__":
    main()
