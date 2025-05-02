import json
import platform
import subprocess
import threading
import time
from threading import Timer

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
        return __copy


class Stereo4DCameraHandler:
    def __init__(self, ip="172.31.1.77", port=5555, show_stream=False):

        # public attributes
        self.ip = ip
        self.port = port
        self.desired_fps = None  # Frames per second

        # status variables
        self.__handler_status = "stop"  # stop, start
        self.__last_status_time = None  # Track last status time
        self.__status_timeout = 10.0  # Timeout in seconds
        self.__run_interval = 1.0  # Check every second

        # start thread to avoid thead duplication
        self.__start_sequence_thread = None

        # frame drop
        self.__frame_drop_percent = 0.0  # Percentage of frames to drop
        self.__frame_drop_count = 0  # Number of frames dropped

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
        self.__frame_count = 0
        self.__received_fps = 0

        # external callbacks
        self.__left_camera_info_callback = None
        self.__right_camera_info_callback = None
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

    def set_left_camera_info_callback(self, callback):
        self.__left_camera_info_callback = callback

    def set_right_camera_info_callback(self, callback):
        self.__right_camera_info_callback = callback

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
        if self.__frame_drop_percent > 50:
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
                if self.__is_status_within_range():
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
        self.__frame_count = 0
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
                self.__logger.error("Timeout waiting camera to report status. Retrying...")
                continue
            self.__logger.info("Camera reported that it started!")

            # wait for the image to be received from the camera
            self.__logger.info("Waiting for images from camera...")
            if not self.wait_for_image(timeout=10.0):
                self.__logger.error(
                    "Failed to receive image from camera. Retrying..."
                )
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
        self.sub_socket.setsockopt_string(zmq.SUBSCRIBE, "frame")
        self.sub_socket.setsockopt_string(zmq.SUBSCRIBE, "intrinsics")
        self.sub_socket.setsockopt_string(zmq.SUBSCRIBE, "status")

        self.__start_timers()
        self.__logger.debug("Subscriber socket setup complete")

    def __start_timers(self):
        self.__status_sender_timer = RepeatTimer(
            1, self.__status_sender
        )
        self.__status_sender_timer.start()
        self.__receiver_timer = RepeatTimer(
            0.01, self.__receiver
        )
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

    def __handle_frame_message(self, parts):

        self.__frame_count += 1

        # handle timestamp
        timestamp = parts[0].decode()

        # decode image
        image_bytes = parts[1]
        image = self.__decode_frame(image_bytes)
        if image is None:
            self.__frame_drop_count += 1
            self.__frame_drop_percent = (
                self.__frame_drop_count / (self.__frame_count + 1e-6)
            ) * 100
            return

        self.__frame_drop_percent = (
            self.__frame_drop_count / (self.__frame_count + 1e-6)
        ) * 100

        frame = Stereo4DFrame(
            timestamp=timestamp,
            image=image,
        )

        self.__frame_count += 1
        self.__last_frame = frame

        current_time = time.time()
        elapsed_time = current_time - self.__prev_fps_measured_time

        # clear and set the frame event
        self.__frame_event.set()

        if elapsed_time >= self.__fps_measurement_interval:
            self.__prev_fps_measured_time = current_time
            self.__received_fps = self.__frame_count / elapsed_time
            self.__frame_count = 0

        if self.__frame_callback:
            self.__frame_callback(frame.copy())

    def __handle_intrinsics_message(self, parts):
        # handle timestamp
        intrinsics = self.__decode_string(parts[0])
        if intrinsics is None:
            self.__logger.error("Failed to decode intrinsics")
            return

        # loads the intrinsics
        intrinsics = json.loads(intrinsics)

        self.__intrinsics_count += 1

        w, h = self.get_resolution()
        w = w // 2  # TODO: fix this hardcoded value, since we are using stereo camera

        left_camera_info = Stereo4DCameraInfo()
        left_camera_info.width = w
        left_camera_info.height = h
        left_camera_info.k = intrinsics["left_camera_matrix"]
        left_camera_info.d = intrinsics["left_distortion_coefficients"]

        right_camera_info = Stereo4DCameraInfo()
        right_camera_info.width = w
        right_camera_info.height = h
        right_camera_info.k = intrinsics["right_camera_matrix"]
        right_camera_info.d = intrinsics["right_distortion_coefficients"]

        if self.__left_camera_info_callback:
            self.__left_camera_info_callback(left_camera_info.copy())
        if self.__right_camera_info_callback:
            self.__right_camera_info_callback(right_camera_info.copy())

    def __handle_camera_status(self):

        # Update last status time
        self.__last_status_time = time.time()

    def __handle_message(self, parts):
        # check if length of parts is more than 0
        if not parts or len(parts) == 0:
            return

        try:
            # Get topic and validate it
            topic = self.__decode_string(parts[0])
            if topic not in ["frame", "intrinsics", "status"]:
                self.__logger.warn(f"Received message with invalid topic: {topic}")
                return

            # Handle message based on topic
            if topic == "frame":
                if len(parts) >= 3:  # Ensure we have enough parts for frame message
                    self.__handle_frame_message(parts[1:])
            elif topic == "intrinsics":
                if (
                    len(parts) >= 2
                ):  # Ensure we have enough parts for intrinsics message
                    self.__handle_intrinsics_message(parts[1:])
            elif topic == "status":
                self.__handle_camera_status()
        except Exception as e:
            self.__logger.error(f"Error handling message: {e}")

    def __receiver(self):
        try:
            if self.sub_socket is None:
                time.sleep(1)
                return

            # Use a small timeout to prevent blocking
            parts = self.sub_socket.recv_multipart(track=True)

            if (
                parts and not self.__should_exit
            ):  # Check should_exit again after receive
                self.__handle_message(parts)
        except zmq.error.ContextTerminated:
            return
        except zmq.error.Again:
            # No message available, continue
            return
        except Exception as e:
            self.__logger.error(f"Error in run loop: {e}")
            time.sleep(0.1)  # Sleep a bit longer on error
            return

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
