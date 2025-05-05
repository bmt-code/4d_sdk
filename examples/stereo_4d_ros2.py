import threading
import time

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


class FourDCameraROS2(Node):
    def __init__(self):
        super().__init__("stereo_4d_node")

        self.logger = self.get_logger()
        self.logger.info("Initializing 4D Camera Node")

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
        self.camera_handler = Stereo4DCameraHandler(show_stream=False, rectify_internally=False)
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

    def handle_intrinsics_event(self, left_intrinsics: Stereo4DCameraInfo, right_intrinsics: Stereo4DCameraInfo):
        self.logger.info("Intrinsics received", once=True)
        self.left_intrinsics: Stereo4DCameraInfo = left_intrinsics
        self.right_intrinsics: Stereo4DCameraInfo = right_intrinsics

    def publish_frame_loop(self):

        while rclpy.ok():
            img_received = self.frame_event.wait(timeout=5.0)

            # if no image received for mmore than 5 seconds, skip
            if not img_received:
                self.logger.info("No image received, skipping")
                continue

            self.frame_event.clear()

            if self.current_frame is None or self.current_frame.image is None:
                continue

            # Stereo rectify the images based on intrinsics
            # if self.camera_handler.stereo_maps_set:

            #     # Apply rectification to the current frame
            #     left_image = self.current_frame.image[
            #         :, : self.current_frame.image.shape[1] // 2
            #     ]
            #     right_image = self.current_frame.image[
            #         :, self.current_frame.image.shape[1] // 2 :
            #     ]

            #     # Rectify the images
            #     left_rectified, right_rectified = self.camera_handler.rectify_stereo_images(
            #         left_image, right_image
            #     )

            #     curr_image = np.hstack((left_rectified, right_rectified))
            # else:
            #     self.logger.info(
            #         "Stereo maps not set, using original image",
            #         throttle_duration_sec=5.0,
            #     )
            #     curr_image = self.current_frame.image
            curr_image = self.current_frame.image

            # Publish compressed image
            # Publish compressed image
            compressed_msg = self.bridge.cv2_to_compressed_imgmsg(
                curr_image, dst_format="jpeg"
            )
            compressed_msg.header.stamp = self.get_clock().now().to_msg()
            self.compressed_frame_publisher.publish(compressed_msg)

            # Publish raw image
            msg = self.bridge.cv2_to_imgmsg(curr_image)
            msg.header.stamp = self.get_clock().now().to_msg()
            self.frame_publisher.publish(msg)

    def publish_intrinsics(self):
        try:

            if not self.camera_handler.intrinsics_set:
                self.logger.info("Intrinsics not set, waiting for intrinsics")
                return

            if not self.camera_handler.stereo_maps_set:
                self.logger.info("Initializing stereo maps")
                self.init_stereo_rectify_maps()
                self.logger.info("Stereo maps initialized")

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
            # transform.transform.translation.x = self.left_intrinsics.extrinsic_matrix[0, 3]
            # transform.transform.translation.y = self.left_intrinsics.extrinsic_matrix[1, 3]
            # transform.transform.translation.z = self.left_intrinsics.extrinsic_matrix[2, 3]

            # # Convert rotation matrix to quaternion
            # rotation_matrix = self.left_intrinsics.extrinsic_matrix[:3, :3]
            # quaternion = self.rotation_matrix_to_quaternion(rotation_matrix)
            # transform.transform.rotation.x = quaternion[0]
            # transform.transform.rotation.y = quaternion[1]
            # transform.transform.rotation.z = quaternion[2]
            # transform.transform.rotation.w = quaternion[3]

            self.tf_broadcaster.sendTransform(transform)
        except Exception as e:
            self.get_logger().error(f"Failed to publish intrinsics: {e}")

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
