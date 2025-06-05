import threading
import time

import traceback
import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from rclpy.timer import Timer
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import CameraInfo, CompressedImage, Image
from tf2_ros import TransformBroadcaster

from stereo_4d import Stereo4DCameraHandler, Stereo4DCameraInfo


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
        self.logger.info("Initializing 4D Camera Node")

        self.should_publish_raw = False
        self.should_publish_compressed = True

        # stereo maps
        self.map_left_x = None
        self.map_left_y = None
        self.map_right_x = None
        self.map_right_y = None
        self.intrinsics_set = False
        self.stereo_maps_set = False

        # Data storage
        self.current_frame = None
        self.left_intrinsics = None
        self.right_intrinsics = None
        self.left_mtx = None
        self.left_dist = None
        self.right_mtx = None
        self.right_dist = None
        self.optimal_left_mtx = None
        self.optimal_left_roi = None
        self.optimal_right_mtx = None
        self.optimal_right_roi = None

        # Publishers
        self.frame_publisher = self.create_publisher(
            Image, "camera/fourd/stereo/raw", 2
        )
        self.compressed_frame_publisher = self.create_publisher(
            CompressedImage, "camera/fourd/stereo/raw/compressed", 2
        )
        self.left_camera_info_publisher = self.create_publisher(
            CameraInfo, "camera/fourd/left/camera_info", 2
        )
        self.right_camera_info_publisher = self.create_publisher(
            CameraInfo, "camera/fourd/right/camera_info", 2
        )
        self.left_frame_publisher = self.create_publisher(
            Image, "camera/fourd/left/image_raw", 2
        )
        self.right_frame_publisher = self.create_publisher(
            Image, "camera/fourd/right/image_raw", 2
        )
        self.left_compressed_frame_publisher = self.create_publisher(
            CompressedImage, "camera/fourd/left/image_raw/compressed", 2
        )
        self.right_compressed_frame_publisher = self.create_publisher(
            CompressedImage, "camera/fourd/right/image_raw/compressed", 2
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
        self.logger.info("4D Camera Node initialized")

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
        self.camera_handler = None
        self.frame_event.clear()
        self.intrinsics_event.clear()
        self.stereo_maps_set = False
        self.logger.info("Camera handler stopped")

    def handle_frame_event(self, frame):
        self.current_frame = frame
        self.frame_event.set()

    def handle_intrinsics_event(
        self, left_intrinsics: Stereo4DCameraInfo, right_intrinsics: Stereo4DCameraInfo
    ):
        self.logger.info("Intrinsics received", once=True)
        self.left_intrinsics: Stereo4DCameraInfo = left_intrinsics
        self.right_intrinsics: Stereo4DCameraInfo = right_intrinsics

    def publish_compressed_images(self, curr_image, left_image, right_image):
        # Publish left compressed image
        # left_compressed_msg = self.bridge.cv2_to_compressed_imgmsg(
        #     left_image, dst_format="jpeg"
        # )
        # left_compressed_msg.header.stamp = self.get_clock().now().to_msg()
        # self.left_compressed_frame_publisher.publish(left_compressed_msg)

        # # Publish right compressed image
        # right_compressed_msg = self.bridge.cv2_to_compressed_imgmsg(
        #     right_image, dst_format="jpeg"
        # )
        # right_compressed_msg.header.stamp = self.get_clock().now().to_msg()
        # self.right_compressed_frame_publisher.publish(right_compressed_msg)

        # Publish combined stereo image
        compressed_msg = self.bridge.cv2_to_compressed_imgmsg(
            curr_image, dst_format="jpeg"
        )
        compressed_msg.header.stamp = self.get_clock().now().to_msg()
        self.compressed_frame_publisher.publish(compressed_msg)

    def publish_raw_images(self, curr_image, left_image, right_image):
        # Publish left raw image
        left_msg = self.bridge.cv2_to_imgmsg(left_image, encoding="bgr8")
        left_msg.header.stamp = self.get_clock().now().to_msg()
        self.left_frame_publisher.publish(left_msg)

        # Publish right raw image
        right_msg = self.bridge.cv2_to_imgmsg(right_image, encoding="bgr8")
        right_msg.header.stamp = self.get_clock().now().to_msg()
        self.right_frame_publisher.publish(right_msg)

        # Publish combined stereo image
        msg = self.bridge.cv2_to_imgmsg(curr_image, encoding="bgr8")
        msg.header.stamp = self.get_clock().now().to_msg()
        self.frame_publisher.publish(msg)

    def publish_frame_loop(self):

        while rclpy.ok():
            img_received = self.frame_event.wait(timeout=5.0)

            # if no image received for more than 5 seconds, skip
            if not img_received:
                self.logger.info("No image received, skipping")
                continue

            self.frame_event.clear()

            if self.current_frame is None or self.current_frame.image is None:
                continue

            # Extract left and right images from the stereo image
            # Convert stereo images to desired format
            curr_image = self.current_frame.image

            left_image = curr_image[
                :, : curr_image.shape[1] // 2
            ]
            right_image = curr_image[
                :, curr_image.shape[1] // 2 :
            ]

            # rectify images if stereo maps are set
            if self.camera_handler.stereo_maps_set:
                left_image, right_image = self.camera_handler.rectify_stereo_images(left_image, right_image)
                curr_image = np.concatenate((left_image, right_image), axis=1)


            # Convert images to uint8 using numpy
            curr_image = np.clip(curr_image, 0, 255).astype(np.uint8)
            left_image = np.clip(left_image, 0, 255).astype(np.uint8)
            right_image = np.clip(right_image, 0, 255).astype(np.uint8)


            if self.should_publish_compressed:
                self.publish_compressed_images(curr_image, left_image, right_image)

            if self.should_publish_raw:
                self.publish_raw_images(curr_image, left_image, right_image)

    def publish_intrinsics(self):
        try:

            if not self.camera_handler.intrinsics_set:
                self.logger.info("Intrinsics not set, waiting for intrinsics")
                return

            # Publish left camera info
            left_camera_info = CameraInfo()
            left_camera_info.header.stamp = self.get_clock().now().to_msg()
            left_camera_info.header.frame_id = "left_camera"
            left_camera_info.k = self.left_intrinsics.k.ravel().tolist()
            left_camera_info.d = self.left_intrinsics.d.ravel().tolist()
            left_camera_info.width = 1920
            left_camera_info.height = 1080
            self.left_camera_info_publisher.publish(left_camera_info)

            # Publish right camera info
            right_camera_info = CameraInfo()
            right_camera_info.header.stamp = self.get_clock().now().to_msg()
            right_camera_info.header.frame_id = "right_camera"
            right_camera_info.k = self.right_intrinsics.k.ravel().tolist()
            right_camera_info.d = self.right_intrinsics.d.ravel().tolist()
            right_camera_info.width = 1920
            right_camera_info.height = 1080
            self.right_camera_info_publisher.publish(right_camera_info)

            # Publish extrinsics as a transform
            transform = TransformStamped()
            transform.header.stamp = self.get_clock().now().to_msg()
            transform.header.frame_id = "left_camera"
            transform.child_frame_id = "right_camera"

            self.tf_broadcaster.sendTransform(transform)
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
            DiagnosticStatus.OK if self.current_frame else DiagnosticStatus.ERROR
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
