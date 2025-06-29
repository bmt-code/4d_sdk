import json
import platform
import subprocess
import threading
import time
import traceback
from threading import Timer
from typing import Tuple, Dict

import cv2
import numpy as np
import zmq

from .custom_logger import CustomLogger


class RepeatTimer(Timer):
    def run(self):
        while not self.finished.wait(self.interval):
            self.function(*self.args, **self.kwargs)


class Stereo4DFrame:
    def __init__(self, timestamp=None, frame_id=None, image=None):
        self.timestamp = timestamp
        self.frame_id = frame_id
        self.image = image

    def copy(self):
        """Create a copy of the frame"""
        return Stereo4DFrame(
            timestamp=self.timestamp,
            frame_id=self.frame_id,
            image=self.image.copy() if self.image is not None else None,
        )

    def __repr__(self):
        return (
            f"Frame(\n"
            f"    timestamp={self.timestamp},\n"
            f"    frame_id={self.frame_id}\n"
            f"    frame_shape={self.image.shape if self.image is not None else None}\n"
            f")"
        )


class Stereo4DCameraInfo:
    def __init__(self):
        self.width = None
        self.height = None
        self.distortion_model = None
        self.d = None
        self.k = None
        self.r = None
        self.p = None
        self.extrinsic_matrix = None
        self.rect_matrix = None

    def copy(self):
        """Create a copy of the camera info"""
        __copy = Stereo4DCameraInfo()
        __copy.width = self.width
        __copy.height = self.height
        __copy.distortion_model = self.distortion_model
        __copy.d = self.d.copy() if self.d is not None else None
        __copy.k = self.k.copy() if self.k is not None else None
        __copy.r = self.r.copy() if self.r is not None else None
        __copy.p = self.p.copy() if self.p is not None else None
        __copy.extrinsic_matrix = (
            self.extrinsic_matrix.copy() if self.extrinsic_matrix is not None else None
        )
        __copy.rect_matrix = (
            self.rect_matrix.copy() if self.rect_matrix is not None else None
        )
        return __copy


class Stereo4DCameraHandler:
    def __init__(
        self, ip="172.31.1.77", port=5555, show_stream=False, rectify_internally=False
    ):

        # public attributes
        self.ip = ip
        self.port = port
        self.desired_fps = None  # Frames per second
        self.rectify_internally = rectify_internally  # Rectification flag

        self.max_frame_drop_percent = 95.0  # Maximum allowed frame drop percentage

        # status variables
        self.__handler_status = "stop"  # stop, start
        self.__last_status_time = None  # Track last status time
        self.__status_timeout = 20.0  # Timeout in seconds
        self.__run_interval = 1.0  # Check every second

        # stereo rectification maps
        self.stereo_maps_set = False
        self.map_left_x = None
        self.map_left_y = None
        self.map_right_x = None
        self.map_right_y = None
        self.left_rect_k = None
        self.optimal_left_roi = None
        self.right_rect_k = None
        self.optimal_right_roi = None
        self.intrinsics_set = False
        self.current_intrinsics = None
        self.left_camera_info = None
        self.right_camera_info = None

        # start thread to avoid thead duplication
        self.__start_sequence_thread = None

        # frame drop
        self.__frame_drop_percent = 0.0  # Percentage of frames to drop
        self.__frame_drop_count = 0  # Number of frames dropped
        self.__frame_count_total = 0  # Total number of frames received

        # private attributes
        self.__connected_event = threading.Event()
        self.__frame_event = threading.Event()
        self.__show_stream = show_stream
        self.__logger = CustomLogger("FourDCameraHandler")

        # zmq setup
        self.sub_socket = None
        self.context = zmq.Context()
        self.pub_socket = self.context.socket(zmq.PUB)
        self.pub_socket.connect(f"tcp://{self.ip}:5556")

        # frame and intrinsics
        self.__last_frame = None
        self.__intrinsics_count = 0

        # fps counting
        self.__fps_measurement_interval = 5.0  # Interval in seconds
        self.__prev_fps_measured_time = time.time()
        self.__frame_count_in_interval = 0
        self.__received_fps = 0

        # external callbacks
        self.__left_camera_info_callback = None
        self.__right_camera_info_callback = None
        self.__intrinsics_callback = None
        self.__frame_callback = None

        # should exit flag to stop the system
        self.__should_exit = False

        # threads
        self.__show_stream_thread = None
        self.__receiver_timer = None
        self.__status_checker_timer = None

        # timers
        self.__status_sender_timer = None
        self.__receiver_timer = None

        # start the run thread
        self.__run_thread = threading.Thread(target=self.__run)
        self.__run_thread.start()
        self.__logger.info("FourDCameraHandler initialized")

    def get_fps(self):
        """Get the current frames per second (FPS)"""
        if self.__received_fps == 0:
            return None
        return self.__received_fps

    def get_last_frame(self):
        """Get the last received frame"""
        if self.__last_frame is None:
            return None

        # return a copy of the last frame
        return self.__last_frame.copy()

    def get_resolution(self):
        """Get the camera resolution"""
        if self.__last_frame is None:
            return None

        # get the resolution of the last frame
        height, width = self.__last_frame.image.shape[:2]
        return (width, height)

    def get_status(self):
        """Get the camera status"""
        if self.__last_status_time is None:
            return None

        # check if status message is received within the timeout period
        current_time = time.time()
        if current_time - self.__last_status_time > self.__status_timeout:
            return False
        else:
            return True

    def start(self, wait=True, timeout=None):

        self.__should_exit = False
        self.__logger.info("Starting camera handler...")
        self.__handler_status = "starting"

        # start the camera thread
        if self.__start_sequence_thread is None:
            self.__start_sequence_thread = threading.Thread(
                target=self.__start_sequence
            )
            self.__start_sequence_thread.start()
        elif not self.__start_sequence_thread.is_alive():
            self.__start_sequence_thread = threading.Thread(
                target=self.__start_sequence
            )
            self.__start_sequence_thread.start()
        else:
            self.__logger.warn("Requested to start camera, but it is already starting.")
            return False

        # if wait is True, wait for the camera to start
        if wait:
            self.__logger.info("Waiting for camera to start...")
            if not self.wait_for_connection(timeout=timeout):
                self.__logger.error("Timeout waiting for camera to start.")
                return False

        # show stream
        if self.__show_stream and self.__show_stream_thread is None:
            self.__show_stream_thread = threading.Thread(target=self.__show_stream_loop)
            self.__show_stream_thread.start()

        return True

    def is_connected(self):
        return self.connected

    def wait_for_connection(self, timeout=None):
        t = time.time()
        while not self.__should_exit:
            if self.__handler_status == "started":
                return True
            if timeout is not None:
                if time.time() - t > timeout:
                    return False
            time.sleep(0.1)
        return False

    def wait_for_image(self, timeout=None):
        self.__frame_event.wait(timeout)
        return self.__last_frame is not None

    def stop(self):
        self.__should_exit = True

        self.__cancel_timers()

        # stop sockets
        try:
            if hasattr(self, "pub_socket"):
                self.pub_socket.close()
            if hasattr(self, "sub_socket"):
                self.sub_socket.close()
            if hasattr(self, "context"):
                self.context.term()
        except Exception as e:
            self.__logger.error(f"Error closing sockets: {e}")

        return True

    def rectify_stereo_images(
        self, img_left: np.ndarray, img_right: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Rectify stereo images using the precomputed maps and return new camera matrix and distortion coefficients"""

        if not self.stereo_maps_set:
            self.__logger.warn("Stereo maps not set, cannot rectify images")
            return img_left, img_right, self.left_rect_k, self.right_rect_k

        if img_left is None or img_right is None:
            self.__logger.error("Input images for rectification are None")
            return img_left, img_right, self.left_rect_k, self.right_rect_k

        if not self.stereo_maps_set:
            self.__logger.error("Stereo maps are not set, cannot rectify images")
            return img_left, img_right, self.left_rect_k, self.right_rect_k

        rectified_left = cv2.remap(
            img_left, self.map_left_x, self.map_left_y, cv2.INTER_LINEAR
        )
        rectified_right = cv2.remap(
            img_right, self.map_right_x, self.map_right_y, cv2.INTER_LINEAR
        )
        return (
            rectified_left,
            rectified_right,
            self.left_rect_k,
            self.right_rect_k,
        )

    def set_left_camera_info_callback(self, callback):
        self.__left_camera_info_callback = callback

    def set_right_camera_info_callback(self, callback):
        self.__right_camera_info_callback = callback

    def set_intrinsics_callback(self, callback):
        self.__intrinsics_callback = callback

    def set_frame_callback(self, callback):
        self.__frame_callback = callback

    def __ping(self, host):
        param = "-n" if platform.system().lower() == "windows" else "-c"
        command = ["ping", param, "1", host]
        result = subprocess.call(
            command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        return result == 0

    def __check_ping(self):
        """Pings the camera ip to check if it is reachable"""
        try:
            ping_result = self.__ping(self.ip)
            if ping_result:
                return True
            return False
        except Exception as e:
            self.__logger.error(f"Failed to ping camera: {e}")
            return False

    def __is_frame_drop_ok(self):
        """Check if the frame drop percentage is higher than 50%"""
        if (
            self.__frame_drop_percent > self.max_frame_drop_percent
            and self.__frame_count_total > 100
        ):
            self.__logger.error(
                f"Frame drop percentage is too high: {self.__frame_drop_percent:.2f}%"
            )
            return False
        return True

    def __is_status_within_range(self):
        """Check if the status message is received within the timeout period"""
        if self.__last_status_time is None:
            return False

        current_time = time.time()
        time_wo_status = current_time - self.__last_status_time
        if time_wo_status > self.__status_timeout:
            self.__logger.error(
                f"No status message received for {self.__status_timeout} seconds."
            )
            return False
        return True

    def __run(self):
        """Main thread of the handler that checks the status of the camera"""

        while not self.__should_exit:

            time.sleep(self.__run_interval)

            if self.__handler_status == "starting":
                # check if camera status was received
                if (
                    self.__is_status_within_range()
                    and not self.__connected_event.is_set()
                ):
                    self.__connected_event.set()
                    self.__logger.info("Camera started successfully!")
                    continue

            elif self.__handler_status == "started":
                # restart if frame drop is higher than 50%
                if not self.__is_frame_drop_ok():
                    self.__connected_event.clear()
                    self.start(wait=False)
                    continue

                # check if status message is received within the timeout period
                if not self.__is_status_within_range():
                    self.__connected_event.clear()
                    self.start(wait=False)
                    continue

    def __status_sender(self):
        self.__logger.info(
            "Handler is sending status messages to camera...", log_once=True
        )

        try:
            # send status message to the camera
            msg_content = self.__handler_status
            status_msg = {"action": msg_content}
            status_msg = json.dumps(status_msg).encode()
            self.pub_socket.send_multipart([b"command", status_msg], zmq.NOBLOCK)
        except zmq.error.Again:
            self.__logger.error(
                "Failed to send status message: Socket is not ready. Trying again."
            )
        except zmq.error.ContextTerminated:
            self.__logger.error(
                "Failed to send status message: Context terminated. Trying again."
            )
        except Exception as e:
            self.__logger.error(f"Failed to send status message: {e}")

    def __start_sequence(self):

        # reset frame drop count
        self.__frame_drop_count = 0
        self.__frame_count_total = 0
        self.__frame_drop_percent = 0.0

        # Initialize subscriber socket
        self.__setup_subscriber()

        # wait for ip
        while not self.__should_exit:
            time.sleep(1)
            # check if camera is reachable
            if not self.__check_ping():
                self.__logger.error("Camera is not reachable. Retrying...")
                self.__setup_subscriber()
                continue
            else:
                self.__logger.info("Camera is connected to the network.")
                break

        # wait for the camera to start
        while not self.__should_exit:
            time.sleep(1)

            try:
                # send start message to the camera
                start_msg = {"action": "start"}
                start_msg = json.dumps(start_msg).encode()
                self.pub_socket.send_multipart([b"command", start_msg], zmq.NOBLOCK)
            except zmq.error.Again:
                self.__logger.error(
                    "Failed to send start message: Socket is not ready. Trying again."
                )
                continue
            except zmq.error.ContextTerminated:
                self.__logger.error("Failed to send start message: Context terminated.")
                continue
            except Exception as e:
                self.__logger.error(f"Failed to send start message: {e}")
                continue

            # wait for the status message to be received from the camera
            self.__logger.info("Waiting for camera to report status...")
            if not self.__connected_event.wait(timeout=2):
                self.__logger.error(
                    "Timeout waiting camera to report status. Retrying..."
                )
                continue
            self.__logger.info("Camera reported that it started!")

            # wait for the image to be received from the camera
            self.__logger.info("Waiting for images from camera...")
            if not self.wait_for_image(timeout=10.0):
                self.__logger.error("Failed to receive image from camera. Retrying...")
                continue

            # set status to start
            self.__handler_status = "started"

            self.__logger.info("Camera started to stream images!")
            break

    def __setup_subscriber(self):
        """Setup or reset the subscriber socket with proper topic filters"""

        self.__logger.info("Stopping camera...")

        self.__cancel_timers()

        # send stop message to the camera
        stop_msg = {"action": "stop"}
        stop_msg = json.dumps(stop_msg).encode()
        self.pub_socket.send_multipart([b"command", stop_msg], zmq.NOBLOCK)

        if self.sub_socket is not None:
            self.__logger.debug("Closing existing subscriber socket")
            self.sub_socket.close()
            self.context.term()

        self.context = zmq.Context()
        self.sub_socket = self.context.socket(zmq.SUB)
        # Don't linger on close
        self.sub_socket.setsockopt(zmq.LINGER, 0)
        # Set high water mark to prevent message queue buildup
        self.sub_socket.setsockopt(zmq.RCVHWM, 2)
        # Set a small timeout to prevent blocking
        self.sub_socket.setsockopt(zmq.RCVTIMEO, 100)
        self.sub_socket.connect(f"tcp://{self.ip}:5555")

        # Set up topic filters
        self.sub_socket.setsockopt_string(zmq.SUBSCRIBE, "")

        self.__start_timers()
        self.__logger.debug("Subscriber socket setup complete")

    def __start_timers(self):
        self.__status_sender_timer = RepeatTimer(1, self.__status_sender)
        self.__status_sender_timer.start()
        self.__receiver_timer = RepeatTimer(0.01, self.__receiver)
        self.__receiver_timer.start()

    def __cancel_timers(self):
        # set cancel to the timers
        if self.__status_sender_timer:
            self.__status_sender_timer.cancel()
        if self.__receiver_timer:
            self.__receiver_timer.cancel()

        # wait for the timers to finish
        if self.__status_sender_timer:
            self.__status_sender_timer.join()
        if self.__receiver_timer:
            self.__receiver_timer.join()

    def __decode_frame(self, frame_bytes):
        try:
            decoded_frame = cv2.imdecode(
                np.frombuffer(frame_bytes, dtype=np.uint8), cv2.IMREAD_COLOR
            )
            return decoded_frame
        except Exception as e:
            self.__logger.error(f"Failed to decode frame: {e}")
            return None

    def __decode_string(self, string_bytes):
        try:
            return string_bytes.decode("utf-8")
        except UnicodeDecodeError:
            self.__logger.error("Failed to decode string")
            return None

    def __handle_frame_message(self, msg_json: Dict):

        self.__frame_count_total += 1

        # handle timestamp
        timestamp = msg_json.get("timestamp", None)

        # decode image
        image_bytes = bytes.fromhex(msg_json.get("data", ""))
        image = self.__decode_frame(image_bytes)
        if image is None:
            self.__frame_drop_count += 1
            self.__frame_drop_percent = (
                self.__frame_drop_count / (self.__frame_count_total + 1e-6)
            ) * 100
            return

        self.__frame_drop_percent = (
            self.__frame_drop_count / (self.__frame_count_total + 1e-6)
        ) * 100

        if self.rectify_internally and self.stereo_maps_set:
            # Decode the image into left and right images
            left_image = image[:, : image.shape[1] // 2]
            right_image = image[:, image.shape[1] // 2 :]

            # Rectify the stereo images
            left_image, right_image = self.rectify_stereo_images(
                left_image, right_image
            )

            # Combine the rectified images back into a single image
            image = np.hstack((left_image, right_image))

        frame = Stereo4DFrame(
            timestamp=timestamp,
            image=image,
        )

        self.__last_frame = frame

        current_time = time.time()
        elapsed_time = current_time - self.__prev_fps_measured_time

        # clear and set the frame event
        self.__frame_event.set()

        if elapsed_time >= self.__fps_measurement_interval:
            self.__prev_fps_measured_time = current_time
            self.__received_fps = self.__frame_count_in_interval / elapsed_time
            self.__frame_count_in_interval = 0

        if self.__frame_callback:
            self.__frame_callback(frame.copy())

    def __handle_intrinsics_message(self, msg_json: Dict):
        # handle timestamp
        intrinsics = msg_json.get("data", None)
        if intrinsics is None:
            self.__logger.error("Failed to decode intrinsics")
            return

        self.__intrinsics_count += 1

        w, h = self.get_resolution()
        w = w // 2  # TODO: fix this hardcoded value, since we are using stereo camera

        self.left_camera_info = Stereo4DCameraInfo()
        self.left_camera_info.width = w
        self.left_camera_info.height = h
        self.left_camera_info.k = np.array(intrinsics["left_camera_matrix"])
        self.left_camera_info.d = np.array(intrinsics["left_distortion_coefficients"])
        self.left_camera_info.extrinsic_matrix = np.array(
            intrinsics["extrinsic_matrix"]
        )

        self.right_camera_info = Stereo4DCameraInfo()
        self.right_camera_info.width = w
        self.right_camera_info.height = h
        self.right_camera_info.k = np.array(intrinsics["right_camera_matrix"])
        self.right_camera_info.d = np.array(intrinsics["right_distortion_coefficients"])
        self.right_camera_info.extrinsic_matrix = np.array(
            intrinsics["extrinsic_matrix"]
        )

        if not self.intrinsics_set:
            self.intrinsics_set = True
            self.__logger.info("Intrinsics set successfully")
            self.__init_stereo_rectify_maps()

        if not self.stereo_maps_set:
            self.__logger.info("Stereo maps not set, initializing stereo maps")
            self.__init_stereo_rectify_maps()

        if self.__left_camera_info_callback:
            self.__left_camera_info_callback(self.left_camera_info.copy())
        if self.__right_camera_info_callback:
            self.__right_camera_info_callback(self.right_camera_info.copy())
        if self.__intrinsics_callback:
            self.__intrinsics_callback(
                self.left_camera_info.copy(), self.right_camera_info.copy()
            )

    def __handle_camera_status(self):

        # Update last status time
        self.__last_status_time = time.time()

    def __handle_message(self, msg_json: Dict):
        try:
            # Get topic and validate it
            msg_type = msg_json.get("msg_type", "")
            if msg_type not in ["frame", "intrinsics", "status"]:
                self.__logger.warn(f"Received message with invalid topic: {msg_type}")
                return

            # Handle message based on topic
            if msg_type == "frame":
                self.__handle_frame_message(msg_json)
            elif msg_type == "intrinsics":
                self.__handle_intrinsics_message(msg_json)
            elif msg_type == "status":
                self.__handle_camera_status()
        except Exception as e:
            self.__logger.error(f"Error handling message: {e}")
            traceback.print_exc()

    def __receiver(self):
        try:
            if self.sub_socket is None:
                time.sleep(1)
                return

            # Use a small timeout to prevent blocking
            msg_json = self.sub_socket.recv_json()
            if msg_json and not self.__should_exit:  # Check should_exit again after receive
                self.__handle_message(msg_json)
        except zmq.error.ContextTerminated:
            return
        except zmq.error.Again:
            # No message available, continue
            return
        except Exception as e:
            self.__logger.error(f"Error in receiver loop: {e}")
            traceback.print_exc()
            time.sleep(0.1)  # Sleep a bit longer on error
            return

    def __init_stereo_rectify_maps(self, alpha=0):
        if not self.intrinsics_set:
            self.__logger.info("Intrinsics not set, waiting for intrinsics")
            return

        left_k = self.left_camera_info.k
        right_k = self.right_camera_info.k
        left_dist_coeffs = self.left_camera_info.d
        right_dist_coeffs = self.right_camera_info.d
        extrinsic_matrix = self.left_camera_info.extrinsic_matrix

        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
            left_k,
            left_dist_coeffs,
            right_k,
            right_dist_coeffs,
            (1920, 1080),  # Assuming image resolution TODO
            extrinsic_matrix[:3, :3],  # Rotation matrix
            extrinsic_matrix[:3, 3],  # Translation vector
            alpha=alpha,
            flags=cv2.CALIB_ZERO_DISPARITY,
            newImageSize=(1920, 1080),  # Assuming image resolution TODO
        )

        self.__logger.info("Q matrix:")
        self.__logger.info(f"{Q}")

        myQ = np.array(
            [
                [1, 0, 0, -left_k[0, 2]],
                [0, 1, 0, -left_k[1, 2]],
                [0, 0, 0, left_k[0, 0]],
                [0, 0, 1 / 0.3, 0],
                # Assuming a baseline of 0.3 meters, adjust as needed
            ]
        )

        self.__logger.info("Custom Q matrix:")
        self.__logger.info(f"{myQ}")

        self.map_left_x, self.map_left_y = cv2.initUndistortRectifyMap(
            left_k, left_dist_coeffs, R1, P1, (1920, 1080), cv2.CV_16SC2
        )
        self.map_right_x, self.map_right_y = cv2.initUndistortRectifyMap(
            right_k, right_dist_coeffs, R2, P2, (1920, 1080), cv2.CV_16SC2
        )

        self.left_rect_k = P1[:3, :3]
        self.right_rect_k = P2[:3, :3]
        self.__logger.info(
            "Stereo rectification maps initialized successfully with the following parameters:"
        )
        self.__logger.info(f"Left Raw Matrix:\n{np.array2string(left_k, precision=2, suppress_small=True)}")
        self.__logger.info(f"Right Raw Matrix:\n{np.array2string(right_k, precision=2, suppress_small=True)}")
        self.__logger.info(f"Left Rect Matrix:\n{np.array2string(self.left_rect_k, precision=2, suppress_small=True)}")
        self.__logger.info(f"Right Rect Matrix:\n{np.array2string(self.right_rect_k, precision=2, suppress_small=True)}")
        self.__logger.info(f"ROIs - Left: {self.optimal_left_roi}, Right: {self.optimal_right_roi}")

        # Calculate and log the field of view (FOV) for both cameras
        left_fovx, left_fovy, _, _, _ = cv2.calibrationMatrixValues(
            self.left_rect_k, (1920, 1080), 1920, 1080
        )
        right_fovx, right_fovy, _, _, _ = cv2.calibrationMatrixValues(
            self.right_rect_k, (1920, 1080), 1920, 1080
        )
        self.__logger.info(f"Left FOV: {left_fovx:.1f}x{left_fovy:.1f} deg (HxV), Right FOV: {right_fovx:.1f}x{right_fovy:.1f} deg (HxV)")

        self.stereo_maps_set = True
        self.__logger.info("Stereo rectification maps initialized")

    def __show_stream_loop(self):
        cv2.namedWindow("Stream", cv2.WINDOW_NORMAL)

        while not self.__should_exit:
            if self.__last_frame is not None:
                img = self.__last_frame.image.copy()
                cv2.imshow("Stream", img)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        cv2.destroyAllWindows()

    def __del__(self):
        """Ensure proper cleanup when object is destroyed"""
        try:
            self.stop()
        except Exception as e:
            self.__logger.error(f"Error in destructor: {e}")


if __name__ == "__main__":
    handler = Stereo4DCameraHandler(show_stream=True)
    try:
        handler.start(wait=True)
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopping camera handler...")
        handler.stop()
        print("Camera handler stopped.")
