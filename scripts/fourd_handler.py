import time

import cv2
import numpy as np
import zmq
import json
import threading
from threading import Timer


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

        # private attributes
        self.__connected_event = threading.Event()
        self.__show_stream = show_stream

        # zmq setup
        self.__context = None
        self.__socket = None
        self.__start_socket()

        # frame and intrinsics
        self.__last_frame = None
        self.__frame_count = 0
        self.__last_intrinsics = None
        self.__intrinsics_count = 0
        self.__actual_fps = 0

        # external callbacks
        self.__intrinsics_callback = None
        self.__frame_callback = None

        # should stop
        self.__should_stop = False

        self.__show_stream_thread = None
        self.__run_loop_thread = None

        # status sender
        self.status_sender = RepeatTimer(1, self.__status_sender)
        self.status_sender.start()

        self.__print("FourDCameraHandler initialized")

    def start(self):

        # check if socket is ok
        socket_ok = self.__is_socket_ok()
        if not socket_ok:
            self.__print("Socket was not created correctly. Failed to start.")
            return False

        try:
            # send start message to the camera
            start_msg = ["start".encode(), "".encode()]
            self.__socket.send_multipart(start_msg)
        except zmq.error.ContextTerminated:
            self.__print("Failed to send start message: Context terminated.")
            return False
        except Exception as e:
            self.__print(f"Failed to send start message: {e}")
            return False

        self.__run_loop_thread = threading.Thread(target=self.__run_loop)
        self.__run_loop_thread.start()

        # show stream
        if self.__show_stream:
            self.__show_stream_thread = threading.Thread(target=self.__show_stream_loop)
            self.__show_stream_thread.start()

        return True

    def is_connected(self):
        return self.connected

    def wait_for_connection(self, timeout=None):
        self.__connected_event.wait(timeout)
        return self.connected

    def stop(self):
        self.__should_stop = True

        # stop the threads
        self.__run_loop_thread.join(timeout=0)
        if self.__show_stream:
            self.__show_stream_thread.join(timeout=0)

        # stop socket
        self.__stop_socket()

    def set_intrinsics_callback(self, callback):
        self.__intrinsics_callback = callback

    def set_frame_callback(self, callback):
        self.__frame_callback = callback

    def __start_socket(self):
        self.__context = zmq.Context()
        self.__socket = self.__context.socket(zmq.PAIR)
        self.__socket.connect(f"tcp://{self.ip}:{self.port}")
        self.prev_time = time.time()

    def __is_socket_ok(self):
        if self.__socket is None:
            return False

        if self.__socket.closed:
            return False

        return True

    def __stop_socket(self):
        if self.__socket is None:
            return

        if not self.__socket.closed:
            self.__socket.close(linger=0)
        last_context = self.__context
        self.__context = zmq.Context()
        self.__socket = self.__context.socket(zmq.PAIR)
        last_context.term()

    def __status_sender(self):
        if self.__socket.closed:
            return

        try:
            # send status message to the camera
            status_msg = ["status".encode(), "".encode()]
            self.__socket.send_multipart(status_msg)
        except zmq.error.ContextTerminated:
            self.__print(
                "Failed to send status message: Context terminated. Trying again."
            )
        except zmq.error.Again:
            self.__print(
                "Failed to send status message: Socket is not ready. Trying again."
            )
        except Exception as e:
            self.__print(f"Failed to send status message: {e}")

    def __decode_frame(self, frame_bytes):
        try:
            decoded_frame = cv2.imdecode(
                np.frombuffer(frame_bytes, dtype=np.uint8), cv2.IMREAD_COLOR
            )
            return decoded_frame
        except Exception as e:
            self.__print(f"Failed to decode frame: {e}")
            return None

    def __decode_string(self, string_bytes):
        try:
            return string_bytes.decode("utf-8")
        except UnicodeDecodeError:
            self.__print("Failed to decode string")
            return None

    def __handle_frame_message(self, parts):

        # handle timestamp
        timestamp = parts[1].decode()

        # decode image
        image_bytes = parts[2]
        image = self.__decode_frame(image_bytes)
        if image is None:
            self.__print("Failed to decode image")
            return

        frame = Frame(
            timestamp=timestamp,
            image=image,
        )

        self.__frame_count += 1
        self.__last_frame = frame

        current_time = time.time()
        elapsed_time = current_time - self.prev_time

        if elapsed_time >= 1.0:  # Update FPS every second
            fps = self.__frame_count / elapsed_time
            self.__actual_fps = fps
            self.prev_time = current_time

        if self.__frame_callback:
            self.__frame_callback(frame)

    def __handle_intrinsics_message(self, parts):
        # handle timestamp
        intrinsics = self.__decode_string(parts[1])
        if intrinsics is None:
            self.__print("Failed to decode intrinsics")
            return

        # loads the intrinsics
        intrinsics = json.loads(intrinsics)

        self.__last_intrinsics = intrinsics
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
            self.__print(f"Unknown camera status: {status}")
            return

        if self.connected != connected:
            self.connected = connected
            if self.connected:
                self.__connected_event.set()
            else:
                self.__connected_event.clear()
            self.__print(f"Camera connection status changed: {self.connected}")

    def __handle_message(self, parts):

        # check if length of parts is more than 0
        if len(parts) == 0:
            return

        message_type = self.__decode_string(parts[0])
        if message_type is None:
            self.__print("Failed to decode message type")
            return
        if message_type == "frame":
            self.__handle_frame_message(parts)
        elif message_type == "intrinsics":
            self.__handle_intrinsics_message(parts)
        elif message_type == "camera_status":
            # handle camera status
            self.__handle_camera_status(parts)
        else:
            self.__print(f"Unknown message type: {message_type}")

    def __run_loop(self):
        while not self.__should_stop:
            try:
                parts = self.__socket.recv_multipart()
            except zmq.error.ContextTerminated:
                break
            except zmq.error.Again:
                continue
            except Exception as e:
                self.__print(f"Error receiving message: {e}")
                continue

            self.__handle_message(parts)

    def __show_stream_loop(self):

        cv2.namedWindow("Stream", cv2.WINDOW_NORMAL)

        while not self.__should_stop:
            if self.__last_frame is not None:
                cv2.imshow("Stream", self.__last_frame.image)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        cv2.destroyAllWindows()

    def __print(self, message: str):
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print(f"[FourDCameraHandler] [{current_time}] {message}")


if __name__ == "__main__":
    handler = FourDCameraHandler()
    handler.__run_loop()
    handler.show_stream()
