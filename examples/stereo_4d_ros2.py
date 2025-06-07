import threading
import time
from typing import List
import traceback
import numpy as np
import rclpy
from cv_bridge import CvBridge
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus
from rclpy.node import Node
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import CameraInfo, CompressedImage, Image
from tf2_ros import TransformBroadcaster
import textwrap
from stereo_4d import Stereo4DCameraHandler, Stereo4DCameraInfo, Stereo4DFrame


def measure_execution_time(func):
    """Decorator to measure the execution time of a function."""

    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        args[0].get_logger().info(
            f"Execution time for {func.__name__}: {execution_time * 1000:.2f} ms"
        )
        return result

    return wrapper


class FourDCameraROS2(Node):
    def __init__(self):
        super().__init__("stereo_4d_node")

        self.logger = self.get_logger()

        self.declare_parameter("publish_raw", False)
        self.declare_parameter("publish_raw_compressed", False)
        self.declare_parameter("publish_rectified", False)
        self.declare_parameter("publish_rectified_compressed", True)

        self.should_publish_raw = self.get_parameter("publish_raw").get_parameter_value().bool_value
        self.should_publish_raw_compressed = self.get_parameter("publish_raw_compressed").get_parameter_value().bool_value
        self.should_publish_rectified = self.get_parameter("publish_rectified").get_parameter_value().bool_value
        self.should_publish_rectified_compressed = self.get_parameter("publish_rectified_compressed").get_parameter_value().bool_value

        # stereo maps
        self.map_left_x = None
        self.map_left_y = None
        self.map_right_x = None
        self.map_right_y = None
        self.intrinsics_set = False
        self.rect_intrinsics_set = False

        # Data storage
        # current frame from camera
        self.current_frame_raw: Stereo4DFrame = Stereo4DFrame()
        self.left_raw_intrinsics: Stereo4DCameraInfo = Stereo4DCameraInfo()
        self.right_raw_intrinsics: Stereo4DCameraInfo = Stereo4DCameraInfo()

        # NOTE: no rect Stereo4DCameraInfo, rectified instrinsics are derived from raw intrinsics and extrinsics

        # camera raw intrinsics
        self.left_raw_k: List = None
        self.left_raw_dist: List = None
        self.right_raw_k: List = None
        self.right_raw_dist: List = None

        # camera rectified intrinsics
        self.left_rect_k: np.ndarray = None
        self.left_rect_dist = np.zeros((5,), dtype=np.float32).tolist()
        self.right_rect_k: np.ndarray = None
        self.right_rect_dist = np.zeros((5,), dtype=np.float32).tolist()

        # Raw publishers
        self.stereo_raw_publisher = self.create_publisher(
            Image, "stereo_4d/stereo_raw/image", 2
        )
        self.stereo_raw_compressed_publisher = self.create_publisher(
            CompressedImage, "stereo_4d/stereo_raw/image/compressed", 2
        )
        self.left_raw_info_publisher = self.create_publisher(
            CameraInfo, "stereo_4d/left_raw/camera_info", 2
        )
        self.right_raw_info_publisher = self.create_publisher(
            CameraInfo, "stereo_4d/right_raw/camera_info", 2
        )
        self.left_raw_publisher = self.create_publisher(
            Image, "stereo_4d/left_raw/image", 2
        )
        self.right_raw_publisher = self.create_publisher(
            Image, "stereo_4d/right_raw/image", 2
        )
        self.left_raw_compressed_publisher = self.create_publisher(
            CompressedImage, "stereo_4d/left_raw/image/compressed", 2
        )
        self.right_raw_compressed_publisher = self.create_publisher(
            CompressedImage, "stereo_4d/right_raw/image/compressed", 2
        )

        # Rectified publishers
        self.stereo_rect_publisher = self.create_publisher(
            Image, "stereo_4d/stereo_rect/image", 2
        )
        self.stereo_rect_compressed_publisher = self.create_publisher(
            CompressedImage, "stereo_4d/stereo_rect/image/compressed", 2
        )
        self.left_rect_info_publisher = self.create_publisher(
            CameraInfo, "stereo_4d/left_rect/camera_info", 2
        )
        self.right_rect_info_publisher = self.create_publisher(
            CameraInfo, "stereo_4d/right_rect/camera_info", 2
        )
        self.left_rect_publisher = self.create_publisher(
            Image, "stereo_4d/left_rect/image", 2
        )
        self.right_rect_publisher = self.create_publisher(
            Image, "stereo_4d/right_rect/image", 2
        )
        self.left_rect_compressed_publisher = self.create_publisher(
            CompressedImage, "stereo_4d/left_rect/image/compressed", 2
        )
        self.right_rect_compressed_publisher = self.create_publisher(
            CompressedImage, "stereo_4d/right_rect/image/compressed", 2
        )

        self.diagnostics_publisher = self.create_publisher(
            DiagnosticArray, "diagnostics", 2
        )

        # Transform Broadcaster for extrinsics
        self.tf_broadcaster = TransformBroadcaster(self)

        # CV Bridge
        self.bridge = CvBridge()

        # Events for synchronization
        self.frame_event = threading.Event()
        self.intrinsics_event = threading.Event()

        # Start the camera handler
        self.setup_camera()
        self.start_camera()

        # wait for connection
        self.camera_handler.wait_for_connection(timeout=30.0)

        # wait first frame
        self.camera_handler.wait_for_image(timeout=30.0)

        # clear the frame event
        self.frame_event.clear()

        # Start the publishing loops
        self.frame_publisher_thread = threading.Thread(target=self.publish_frame_loop)
        self.frame_publisher_thread.start()

        # timer for intrinsics
        self.intrinsics_publisher_timer = self.create_timer(
            0.5, self.publish_intrinsics
        )

        # timer for diagnostics
        self.diagnostics_timer = self.create_timer(1.0, self.publish_diagnostics)
        self.print_intro()

    def print_intro(self):
        fourd_intro = """

                       ######     #####
                    ########     ##   ##
                  ###    ##     ##     ##
                ###     ##     ##       ##
              ###      ##     ##         ##
            ###       ##     ##          ##
          ###        ##     ##           ##
        ////////////////// //          ///
                   ##     ##         ###
                  ##     ##       ###
                 ##     ## ######

          Stereo 4D Camera Node for ROS2
          ==============================
        """

        untable_text = textwrap.dedent(fourd_intro)
        self.logger.info(untable_text)

    def setup_camera(self):
        # Initialize FourDCameraHandler
        self.camera_handler = Stereo4DCameraHandler(
            show_stream=False, rectify_internally=False
        )
        self.camera_handler.set_frame_callback(self.handle_frame_event)
        self.camera_handler.set_intrinsics_callback(self.handle_intrinsics_event)

    def start_camera(self):
        self.camera_handler.start()

    def stop_camera(self):
        # Stop the camera handler
        self.camera_handler.stop()
        self.frame_event.clear()
        self.intrinsics_event.clear()
        self.logger.info("Camera handler stopped")

    def handle_frame_event(self, frame):
        self.current_frame_raw = frame
        self.frame_event.set()

    def handle_intrinsics_event(
        self, left_intrinsics: Stereo4DCameraInfo, right_intrinsics: Stereo4DCameraInfo
    ):
        self.logger.info("Intrinsics received", once=True)
        self.left_raw_intrinsics: Stereo4DCameraInfo = left_intrinsics
        self.right_raw_intrinsics: Stereo4DCameraInfo = right_intrinsics

    def publish_raw_compressed_images(self, curr_image, left_image, right_image):
        # Publish left compressed image
        left_compressed_msg = self.bridge.cv2_to_compressed_imgmsg(
            left_image, dst_format="jpeg"
        )
        left_compressed_msg.header.stamp = self.get_clock().now().to_msg()
        self.left_raw_compressed_publisher.publish(left_compressed_msg)

        # Publish right compressed image
        right_compressed_msg = self.bridge.cv2_to_compressed_imgmsg(
            right_image, dst_format="jpeg"
        )
        right_compressed_msg.header.stamp = self.get_clock().now().to_msg()
        self.right_raw_compressed_publisher.publish(right_compressed_msg)

        # Publish combined stereo image
        compressed_msg = self.bridge.cv2_to_compressed_imgmsg(
            curr_image, dst_format="jpeg"
        )
        compressed_msg.header.stamp = self.get_clock().now().to_msg()
        self.stereo_raw_compressed_publisher.publish(compressed_msg)

    def publish_raw_images(self, curr_image, left_image, right_image):
        # Publish left raw image
        left_msg = self.bridge.cv2_to_imgmsg(left_image, encoding="bgr8")
        left_msg.header.stamp = self.get_clock().now().to_msg()
        self.left_raw_publisher.publish(left_msg)

        # Publish right raw image
        right_msg = self.bridge.cv2_to_imgmsg(right_image, encoding="bgr8")
        right_msg.header.stamp = self.get_clock().now().to_msg()
        self.right_raw_publisher.publish(right_msg)

        # Publish combined stereo image
        msg = self.bridge.cv2_to_imgmsg(curr_image, encoding="bgr8")
        msg.header.stamp = self.get_clock().now().to_msg()
        self.stereo_raw_publisher.publish(msg)

    @measure_execution_time
    def publish_rect_compressed_images(self, curr_image, left_image, right_image):
        # Publish left rectified compressed image
        left_rect_compressed_msg = self.bridge.cv2_to_compressed_imgmsg(
            left_image, dst_format="jpeg"
        )
        left_rect_compressed_msg.header.stamp = self.get_clock().now().to_msg()
        self.left_rect_compressed_publisher.publish(left_rect_compressed_msg)

        # Publish right rectified compressed image
        right_rect_compressed_msg = self.bridge.cv2_to_compressed_imgmsg(
            right_image, dst_format="jpeg"
        )
        right_rect_compressed_msg.header.stamp = self.get_clock().now().to_msg()
        self.right_rect_compressed_publisher.publish(right_rect_compressed_msg)

        # Publish combined stereo rectified compressed image
        rect_compressed_msg = self.bridge.cv2_to_compressed_imgmsg(
            curr_image, dst_format="jpeg"
        )
        rect_compressed_msg.header.stamp = self.get_clock().now().to_msg()
        self.stereo_rect_compressed_publisher.publish(rect_compressed_msg)

    def publish_rect_images(self, curr_image, left_image, right_image):
        # Publish left rectified image
        left_rect_msg = self.bridge.cv2_to_imgmsg(left_image, encoding="bgr8")
        left_rect_msg.header.stamp = self.get_clock().now().to_msg()
        self.left_rect_publisher.publish(left_rect_msg)

        # Publish right rectified image
        right_rect_msg = self.bridge.cv2_to_imgmsg(right_image, encoding="bgr8")
        right_rect_msg.header.stamp = self.get_clock().now().to_msg()
        self.right_rect_publisher.publish(right_rect_msg)

        # Publish combined stereo rectified image
        rect_msg = self.bridge.cv2_to_imgmsg(curr_image, encoding="bgr8")
        rect_msg.header.stamp = self.get_clock().now().to_msg()
        self.stereo_rect_publisher.publish(rect_msg)

    def publish_frame_loop(self):

        while rclpy.ok():
            img_received = self.frame_event.wait(timeout=5.0)

            # if no image received for more than 5 seconds, skip
            if not img_received:
                self.logger.info("No image received, skipping")
                continue

            self.frame_event.clear()

            if self.current_frame_raw is None or self.current_frame_raw.image is None:
                continue

            # Extract left and right images from the stereo image
            # Convert stereo images to desired format
            stereo_raw_image = self.current_frame_raw.image

            left_raw_image = stereo_raw_image[
                :, : stereo_raw_image.shape[1] // 2
            ]
            right_raw_image = stereo_raw_image[
                :, stereo_raw_image.shape[1] // 2 :
            ]

            # Convert images to uint8 using numpy
            stereo_raw_image = np.clip(stereo_raw_image, 0, 255).astype(np.uint8)
            left_raw_image = np.clip(left_raw_image, 0, 255).astype(np.uint8)
            right_raw_image = np.clip(right_raw_image, 0, 255).astype(np.uint8)

            # rectify images if stereo maps are set
            if self.camera_handler.stereo_maps_set and (self.should_publish_rectified or self.should_publish_rectified_compressed):
                left_rect_image, right_rect_image, left_k, right_k = self.camera_handler.rectify_stereo_images(left_raw_image, right_raw_image)
                stereo_rect_image = np.concatenate((left_rect_image, right_rect_image), axis=1)

                if not self.rect_intrinsics_set:
                    # Initialize rectified intrinsics if not set
                    self.left_rect_k = left_k
                    self.right_rect_k = right_k
                    self.rect_intrinsics_set = True

                if self.should_publish_rectified_compressed:
                    self.publish_rect_compressed_images(stereo_rect_image, left_rect_image, right_rect_image)
                if self.should_publish_rectified:
                    self.publish_rect_images(stereo_rect_image, left_rect_image, right_rect_image)

            if self.should_publish_raw_compressed:
                self.publish_raw_compressed_images(stereo_raw_image, left_raw_image, right_raw_image)

            if self.should_publish_raw:
                self.publish_raw_images(stereo_raw_image, left_raw_image, right_raw_image)

    def publish_intrinsics(self):
        try:

            if not self.camera_handler.intrinsics_set:
                self.logger.info("Intrinsics not set, waiting for intrinsics")
                return

            # Publish left camera info
            if self.should_publish_raw or self.should_publish_raw_compressed:
                left_raw_info = CameraInfo()
                left_raw_info.header.stamp = self.get_clock().now().to_msg()
                left_raw_info.header.frame_id = "left_camera"
                left_raw_info.k = self.left_raw_intrinsics.k.ravel().tolist()
                left_raw_info.d = self.left_raw_intrinsics.d.ravel().tolist()
                left_raw_info.width = 1920
                left_raw_info.height = 1080
                self.left_raw_info_publisher.publish(left_raw_info)

                # Publish right camera info
                right_raw_info = CameraInfo()
                right_raw_info.header.stamp = self.get_clock().now().to_msg()
                right_raw_info.header.frame_id = "right_camera"
                right_raw_info.k = self.right_raw_intrinsics.k.ravel().tolist()
                right_raw_info.d = self.right_raw_intrinsics.d.ravel().tolist()
                right_raw_info.width = 1920
                right_raw_info.height = 1080
                self.right_raw_info_publisher.publish(right_raw_info)

            if self.should_publish_rectified or self.should_publish_rectified_compressed:

                # check if rectified intrinsics are set
                if not self.rect_intrinsics_set:
                    # wait for intrinsics to be set
                    return

                # Publish left rectified camera info
                left_rect_info = CameraInfo()
                left_rect_info.header.stamp = self.get_clock().now().to_msg()
                left_rect_info.header.frame_id = "left_camera_rect"
                left_rect_info.k = self.left_rect_k.ravel().tolist()
                left_rect_info.d = self.left_rect_dist
                left_rect_info.width = 1920
                left_rect_info.height = 1080
                self.left_rect_info_publisher.publish(left_rect_info)

                # Publish right rectified camera info
                right_rect_info = CameraInfo()
                right_rect_info.header.stamp = self.get_clock().now().to_msg()
                right_rect_info.header.frame_id = "right_camera_rect"
                right_rect_info.k = self.right_rect_k.ravel().tolist()
                right_rect_info.d = self.right_rect_dist
                right_rect_info.width = 1920
                right_rect_info.height = 1080
                self.right_rect_info_publisher.publish(right_rect_info)

        except Exception as e:
            self.get_logger().error(f"Failed to publish intrinsics: {e}")
            traceback.print_exc()

    def publish_diagnostics(self):

        # Create a DiagnosticArray message
        diag_array = DiagnosticArray()
        diag_array.header.stamp = self.get_clock().now().to_msg()

        cam_status = self.camera_handler.get_status()
        status = DiagnosticStatus()
        status.name = "4D Camera"
        status.level = DiagnosticStatus.OK if cam_status else DiagnosticStatus.ERROR
        status.message = f"Camera Status: {cam_status}"

        fps = DiagnosticStatus()
        fps.name = "FPS"
        fps.level = (
            DiagnosticStatus.OK if self.current_frame_raw else DiagnosticStatus.ERROR
        )
        fps.message = f"FPS: {self.camera_handler.get_fps()}"

        cam_resolution = self.camera_handler.get_resolution()
        resolution = DiagnosticStatus()
        resolution.name = "Resolution"
        resolution.level = DiagnosticStatus.OK
        resolution.message = f"Resolution: {cam_resolution[0]}x{cam_resolution[1]}"

        diag_array.status.append(status)
        diag_array.status.append(fps)
        diag_array.status.append(resolution)
        self.diagnostics_publisher.publish(diag_array)

    @staticmethod
    def rotation_matrix_to_quaternion(matrix):
        # Convert a 3x3 rotation matrix to a quaternion using scipy
        rotation = R.from_matrix(matrix)
        quaternion = rotation.as_quat()  # Returns [x, y, z, w]
        return quaternion

    def destroy_node(self):
        self.camera_handler.stop()
        self.frame_publisher_thread.join()
        self.intrinsics_publisher_timer.cancel()
        self.diagnostics_timer.cancel()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = FourDCameraROS2()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_camera()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
