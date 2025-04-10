import threading
import numpy as np
import cv2
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from fourd_handler import FourDCameraHandler
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import CameraInfo, CompressedImage, Image
from tf2_ros import TransformBroadcaster


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
        self.stereo_maps_set = False

        # Publishers
        self.frame_publisher = self.create_publisher(Image, "camera/fourd/stereo/raw", 10)
        self.compressed_frame_publisher = self.create_publisher(
            CompressedImage, "camera/fourd/stereo/raw/compressed", 10
        )
        self.left_camera_info_publisher = self.create_publisher(
            CameraInfo, "camera/fourd/left/camera_info", 10
        )
        self.right_camera_info_publisher = self.create_publisher(
            CameraInfo, "camera/fourd/right/camera_info", 10
        )

        # Transform Broadcaster for extrinsics
        self.tf_broadcaster = TransformBroadcaster(self)

        # CV Bridge
        self.bridge = CvBridge()

        # Initialize FourDCameraHandler
        self.camera_handler = FourDCameraHandler(show_stream=False)
        self.camera_handler.set_frame_callback(self.handle_frame_event)
        self.camera_handler.set_intrinsics_callback(self.handle_intrinsics_event)

        # Events for synchronization
        self.frame_event = threading.Event()
        self.intrinsics_event = threading.Event()

        # Data storage
        self.current_frame = None
        self.current_intrinsics = None

        # Start the camera handler
        self.camera_handler.start()

        # Start the publishing loops
        self.frame_publisher_thread = threading.Thread(target=self.publish_frame_loop)
        self.intrinsics_publisher_thread = threading.Thread(
            target=self.publish_intrinsics_loop
        )
        self.frame_publisher_thread.start()
        self.intrinsics_publisher_thread.start()

    def startCamera(self):
        self.camera_handler.start()

    def initStereoRectifyMaps(self):
        left_camera_matrix = np.array(self.current_intrinsics["left_camera_matrix"])
        right_camera_matrix = np.array(self.current_intrinsics["right_camera_matrix"])
        left_dist_coeffs = np.array(self.current_intrinsics["left_distortion_coefficients"])
        right_dist_coeffs = np.array(self.current_intrinsics["right_distortion_coefficients"])
        extrinsic_matrix = np.array(self.current_intrinsics["extrinsic_matrix"])

        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
            left_camera_matrix,
            left_dist_coeffs,
            right_camera_matrix,
            right_dist_coeffs,
            (1920, 1080),  # Assuming image resolution TODO
            extrinsic_matrix[:3, :3],  # Rotation matrix
            extrinsic_matrix[:3, 3],  # Translation vector
        )

        # Compute rectification maps
        self.map_left_x, self.map_left_y = cv2.initUndistortRectifyMap(
            left_camera_matrix,
            left_dist_coeffs,
            R1,
            P1,
            (1920, 1080),
            cv2.CV_16SC2
        )
        self.map_right_x, self.map_right_y = cv2.initUndistortRectifyMap(
            right_camera_matrix,
            right_dist_coeffs,
            R2,
            P2,
            (1920, 1080),
            cv2.CV_16SC2
        )

        self.stereo_maps_set = True

    def handle_frame_event(self, frame):
        self.current_frame = frame
        self.frame_event.set()

    def handle_intrinsics_event(self, intrinsics):
        self.current_intrinsics = intrinsics
        self.intrinsics_event.set()

    def publish_frame_loop(self):
        while rclpy.ok():
            if self.frame_event.wait(timeout=1.0):
                self.logger.info("Publishing frame", once=True)
                self.publish_frame()
                self.frame_event.clear()

    def publish_intrinsics_loop(self):
        while rclpy.ok():
            if self.intrinsics_event.wait(timeout=1.0):
                self.publish_intrinsics()
                self.intrinsics_event.clear()

    def publish_frame(self):
        if self.current_frame and self.current_frame.image is not None:

            # Stereo rectify the images based on intrinsics
            if self.stereo_maps_set:

                # Apply rectification to the current frame
                left_image = self.current_frame.image[:, self.current_frame.image.shape[1] // 2:]
                right_image = self.current_frame.image[:, :self.current_frame.image.shape[1] // 2]
                left_rectified = cv2.remap(
                    right_image,
                    self.map_left_x,
                    self.map_left_y,
                    interpolation=cv2.INTER_LINEAR
                )
                right_rectified = cv2.remap(
                    left_image,
                    self.map_right_x,
                    self.map_right_y,
                    interpolation=cv2.INTER_LINEAR
                )

                curr_image = np.hstack((left_rectified, right_rectified))
            else:
                self.logger.info("Stereo maps not set, using original image", throttle_duration_sec=5.0)
                curr_image = self.current_frame.image

            # Publish compressed image
            compressed_msg = self.bridge.cv2_to_compressed_imgmsg(
                curr_image
            )
            compressed_msg.header.stamp = self.get_clock().now().to_msg()

            self.compressed_frame_publisher.publish(compressed_msg)

            msg = self.bridge.cv2_to_imgmsg(curr_image, encoding="bgr8")
            msg.header.stamp = self.get_clock().now().to_msg()
            self.frame_publisher.publish(msg)

    def publish_intrinsics(self):
        try:
            intrinsics = self.current_intrinsics

            if not self.stereo_maps_set:
                self.logger.info(f"Intrinsics: {intrinsics}")
                self.logger.info("Initializing stereo maps")
                self.initStereoRectifyMaps()
                self.logger.info("Stereo maps initialized")

            # Publish left camera info
            left_camera_info = CameraInfo()
            left_camera_info.header.stamp = self.get_clock().now().to_msg()
            left_camera_info.header.frame_id = "left_camera"
            left_camera_info.k = (
                np.array(intrinsics["left_camera_matrix"]).ravel().tolist()
            )
            left_camera_info.d = (
                np.array(intrinsics["left_distortion_coefficients"]).ravel().tolist()
            )
            left_camera_info.width = 1920
            left_camera_info.height = 1080
            self.left_camera_info_publisher.publish(left_camera_info)

            # Publish right camera info
            right_camera_info = CameraInfo()
            right_camera_info.header.stamp = self.get_clock().now().to_msg()
            right_camera_info.header.frame_id = "right_camera"
            right_camera_info.k = (
                np.array(intrinsics["right_camera_matrix"]).ravel().tolist()
            )
            right_camera_info.d = (
                np.array(intrinsics["right_distortion_coefficients"]).ravel().tolist()
            )
            right_camera_info.width = 1920
            right_camera_info.height = 1080
            self.right_camera_info_publisher.publish(right_camera_info)

            # Publish extrinsics as a transform
            extrinsic_matrix = np.array(intrinsics["extrinsic_matrix"])
            transform = TransformStamped()
            transform.header.stamp = self.get_clock().now().to_msg()
            transform.header.frame_id = "left_camera"
            transform.child_frame_id = "right_camera"
            transform.transform.translation.x = extrinsic_matrix[0, 3]
            transform.transform.translation.y = extrinsic_matrix[1, 3]
            transform.transform.translation.z = extrinsic_matrix[2, 3]

            # Convert rotation matrix to quaternion
            rotation_matrix = extrinsic_matrix[:3, :3]
            quaternion = self.rotation_matrix_to_quaternion(rotation_matrix)
            transform.transform.rotation.x = quaternion[0]
            transform.transform.rotation.y = quaternion[1]
            transform.transform.rotation.z = quaternion[2]
            transform.transform.rotation.w = quaternion[3]

            self.tf_broadcaster.sendTransform(transform)
        except Exception as e:
            self.get_logger().error(f"Failed to publish intrinsics: {e}")

    @staticmethod
    def rotation_matrix_to_quaternion(matrix):
        # Convert a 3x3 rotation matrix to a quaternion using scipy
        rotation = R.from_matrix(matrix)
        quaternion = rotation.as_quat()  # Returns [x, y, z, w]
        return quaternion

    def destroy_node(self):
        self.camera_handler.stop()
        self.frame_publisher_thread.join()
        self.intrinsics_publisher_thread.join()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = FourDCameraROS2()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
