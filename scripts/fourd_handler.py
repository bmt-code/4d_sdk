import time

import cv2
import numpy as np
import zmq
import platform
import subprocess
import json
import threading
from threading import Timer
from custom_logger import CustomLogger


class RepeatTimer(Timer):
    def run(self):
        while not self.finished.wait(self.interval):
            self.function(*self.args, **self.kwargs)


class Frame:
    def __init__(self, timestamp=None, frame_id=None, image=None):
        self.timestamp = timestamp
        self.frame_id = frame_id
        self.image = image

    def __repr__(self):
        return (
            f"Frame(\n"
            f"    timestamp={self.timestamp},\n"
            f"    frame_id={self.frame_id}\n"
            f"    frame_shape={self.image.shape if self.image is not None else None}\n"
            f")"
        )


class FourDCameraHandler:
    def __init__(self, ip="172.31.1.77", port=5555, show_stream=False):

        # public attributes
        self.ip = ip
        self.port = port
        self.connected = False
        self.desired_fps = None  # Frames per second

        # status variables
        self.__handler_status = "stopped"  # stopped, started
        self.__last_status_time = time.time()  # Track last status time
        self.__status_timeout = 5.0  # Timeout in seconds
        self.__status_check_interval = 1.0  # Check every second

        # private attributes
        self.__connected_event = threading.Event()
        self.__frame_event = threading.Event()
        self.__show_stream = show_stream
        self.__logger = CustomLogger("FourDCameraHandler")

        # zmq setup
        self.__context = None
        self.__socket = None
        self.__start_socket()

        # frame and intrinsics
        self.__last_frame = None
        self.__frame_count = 0
        self.__intrinsics_count = 0

        # external callbacks
        self.__intrinsics_callback = None
        self.__frame_callback = None

        # should exit flag to stop the system
        self.__should_exit = False

        # threads
        self.__show_stream_thread = None
        self.__run_loop_thread = None
        self.__status_checker_thread = None

        # Start status checker thread
        self.__status_checker_thread = threading.Thread(
            target=self.__status_checker_loop
        )
        self.__status_checker_thread.start()

        # status sender
        self.status_sender = RepeatTimer(1, self.__status_sender)
        self.status_sender.start()

        self.__logger.info("FourDCameraHandler initialized")

    def start(self, wait=True):
        self.__should_exit = False
        self.__logger.info("Starting camera handler...")

        self.__run_loop_thread = threading.Thread(target=self.__run_loop)
        self.__run_loop_thread.start()

        self.__handler_status = "started"

        while not self.__should_exit:
            time.sleep(1)

            # check if socket is ok
            # if not self.__check_connection():
            #     self.__logger.error("Socket is not properly connected. Retrying...")
            #     # self.__restart_socket()
            #     continue

            # check if camera is reachable
            if not self.__check_ping():
                self.__logger.error("Camera is not reachable. Retrying...")
                continue

            try:
                # send start message to the camera
                start_msg = ["start".encode(), "".encode()]
                self.__socket.send_multipart(start_msg, zmq.NOBLOCK)
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

            if wait:
                # wait for the status message to be received from the camera
                if not self.wait_for_connection(timeout=2.0):
                    self.__logger.error("Failed to connect to camera. Retrying...")
                    continue
                self.__logger.info("Camera reported that it started!")

                # wait for the image to be received from the camera
                if not self.wait_for_image(timeout=10.0):
                    self.__logger.error(
                        "Failed to receive image from camera. Retrying..."
                    )
                    continue
                self.__logger.info("Camera started to stream images!")

                break

        # show stream
        if self.__show_stream:
            self.__show_stream_thread = threading.Thread(target=self.__show_stream_loop)
            self.__show_stream_thread.start()

        return True

    def is_connected(self):
        return self.connected

    def wait_for_connection(self, timeout=None):
        # wait the connection event
        if self.__connected_event.wait(timeout):
            return True
        else:
            return False

    def wait_for_image(self, timeout=None):
        self.__frame_event.wait(timeout)
        return self.__last_frame is not None

    def stop(self):
        self.__should_exit = True

        # stop the threads
        if self.__run_loop_thread:
            self.__run_loop_thread.join(timeout=1.0)
        if self.__show_stream_thread:
            self.__show_stream_thread.join(timeout=1.0)
        if self.__status_checker_thread:
            self.__status_checker_thread.join(timeout=1.0)

        # stop socket
        self.__stop_socket()

        return True

    def set_intrinsics_callback(self, callback):
        self.__intrinsics_callback = callback

    def set_frame_callback(self, callback):
        self.__frame_callback = callback

    def __start_socket(self):
        self.__context = zmq.Context()
        self.__socket = self.__context.socket(zmq.PAIR)
        self.__socket.setsockopt(zmq.LINGER, 0)  # Don't linger on close
        self.__socket.connect(f"tcp://{self.ip}:{self.port}")
        self.prev_time = time.time()

    def __stop_socket(self):
        if self.__socket is None:
            return

        if not self.__socket.closed:
            self.__socket.close(linger=0)
        last_context = self.__context
        self.__context = zmq.Context()
        self.__socket = self.__context.socket(zmq.PAIR)
        last_context.term()

    def __restart_socket(self):
        self.__stop_socket()
        self.__start_socket()

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

    def __check_connection(self, timeout_ms=1000):
        """Check if the ZMQ connection is really established by attempting a handshake"""
        if self.__socket is None or self.__socket.closed:
            return False

        try:
            # Set a timeout for this check
            self.__socket.setsockopt(zmq.RCVTIMEO, timeout_ms)

            # Send a test message
            test_msg = ["test".encode(), "".encode()]
            self.__socket.send_multipart(test_msg, zmq.NOBLOCK)

            # Try to receive a response
            try:
                self.__socket.recv_multipart()
                return True
            except zmq.error.Again:
                # No response received within timeout
                return False
        except Exception as e:
            self.__logger.error(f"Connection check failed: {e}")
            return False
        finally:
            # Reset timeout to infinite
            self.__socket.setsockopt(zmq.RCVTIMEO, -1)

    def __status_checker_loop(self):
        """Thread that checks camera status periodically"""
        while not self.__should_exit:
            time.sleep(self.__status_check_interval)

            # continue if camera is not started
            if self.__handler_status != "started":
                continue

            # check if status message is received within the timeout period
            current_time = time.time()
            if current_time - self.__last_status_time > self.__status_timeout:
                self.__logger.error(
                    f"No status message received for {self.__status_timeout} seconds. Restarting camera..."
                )

                # restart socket
                self.__restart_socket()

                # start camera
                self.start()


    def __status_sender(self):
        if self.__socket.closed:
            return

        self.__logger.info(
            "Handler is sending status messages to camera...", log_once=True
        )

        try:
            # send status message to the camera
            msg_content = self.__handler_status.encode()
            status_msg = ["status".encode(), msg_content]
            self.__socket.send_multipart(status_msg, zmq.NOBLOCK)
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

        # handle timestamp
        timestamp = parts[1].decode()

        # decode image
        image_bytes = parts[2]
        image = self.__decode_frame(image_bytes)
        if image is None:
            self.__logger.error("Failed to decode image")
            return

        frame = Frame(
            timestamp=timestamp,
            image=image,
        )

        self.__frame_count += 1
        self.__last_frame = frame

        current_time = time.time()
        elapsed_time = current_time - self.prev_time

        # clear and set the frame event
        self.__frame_event.set()

        if elapsed_time >= 1.0:  # Update FPS every second
            self.prev_time = current_time

        if self.__frame_callback:
            self.__frame_callback(frame)

    def __handle_intrinsics_message(self, parts):
        # handle timestamp
        intrinsics = self.__decode_string(parts[1])
        if intrinsics is None:
            self.__logger.error("Failed to decode intrinsics")
            return

        # loads the intrinsics
        intrinsics = json.loads(intrinsics)

        self.__intrinsics_count += 1

        if self.__intrinsics_callback:
            self.__intrinsics_callback(intrinsics)

    def __handle_camera_status(self, parts):
        if len(parts) < 2:
            return

        status = self.__decode_string(parts[1])
        if status == "started":
            connected = True
        elif status == "stopped":
            connected = False
        else:
            self.__logger.warn(f"Unknown camera status: {status}")
            return

        if self.connected != connected:
            self.connected = connected
            if self.connected:
                self.__connected_event.set()
                self.connected = True
                self.__logger.info("Stereo 4D reported that it is started")
            else:
                self.__connected_event.clear()
                self.connected = False
                self.__logger.info("Stereo 4D reported that it is stopped")

        # Update last status time
        self.__last_status_time = time.time()

    def __handle_message(self, parts):

        # check if length of parts is more than 0
        if len(parts) == 0:
            return

        message_type = self.__decode_string(parts[0])
        if message_type is None:
            self.__logger.error("Failed to decode message type")
            return
        if message_type == "frame":
            self.__handle_frame_message(parts)
        elif message_type == "intrinsics":
            self.__handle_intrinsics_message(parts)
        elif message_type == "camera_status":
            # handle camera status
            self.__handle_camera_status(parts)
        else:
            self.__logger.warn(f"Unknown message type: {message_type}")

    def __run_loop(self):
        while not self.__should_exit:
            try:
                parts = self.__socket.recv_multipart()
            except zmq.error.ContextTerminated:
                break
            except zmq.error.Again:
                continue
            except Exception as e:
                self.__logger.error(f"Error receiving message: {e}")
                continue

            self.__handle_message(parts)

    def __show_stream_loop(self):

        cv2.namedWindow("Stream", cv2.WINDOW_NORMAL)

        while not self.__should_exit:
            if self.__last_frame is not None:
                cv2.imshow("Stream", self.__last_frame.image)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        cv2.destroyAllWindows()


if __name__ == "__main__":
    handler = FourDCameraHandler()
    try:
        handler.start(wait=True)
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopping camera handler...")
        handler.stop()
        print("Camera handler stopped.")
