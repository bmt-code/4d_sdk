import time

import cv2
import numpy as np
import zmq
import threading
import json


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

        self.__ip = ip
        self.__port = port
        self.__show_stream = show_stream

        # zmq setup
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PAIR)
        self.socket.connect(f"tcp://{ip}:{port}")
        self.prev_time = time.time()

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

    def start(self):

        # send start message to the camera
        start_msg = ["start".encode(), "".encode()]
        self.socket.send_multipart(start_msg)

        self.__run_loop_thread = threading.Thread(target=self.__run_loop)
        self.__run_loop_thread.start()

        # show stream
        if self.__show_stream:
            self.__show_stream_thread = threading.Thread(target=self.__show_stream_loop)
            self.__show_stream_thread.start()

    def stop(self):
        self.__should_stop = True
        self.socket.close()
        self.context.term()

        self.__run_loop_thread.join()

        if self.__show_stream:
            self.__show_stream_thread.join()

    def set_intrinsics_callback(self, callback):
        self.__intrinsics_callback = callback

    def set_frame_callback(self, callback):
        self.__frame_callback = callback

    def __decode_frame(self, frame_bytes):
        return cv2.imdecode(
            np.frombuffer(frame_bytes, dtype=np.uint8), cv2.IMREAD_COLOR
        )

    def __handle_frame_message(self, parts):

        # handle timestamp
        timestamp = parts[1].decode()

        # decode image
        image_bytes = parts[2]
        image = self.__decode_frame(image_bytes)

        frame = Frame(
            timestamp=timestamp,
            image=image,
        )

        self.__frame_count += 1
        self.__last_frame = frame

        if self.__frame_callback:
            self.__frame_callback(frame)

    def __handle_intrinsics_message(self, parts):
        # handle timestamp
        intrinsics = parts[1].decode()

        # loads the intrinsics
        intrinsics = json.loads(intrinsics)

        self.__last_intrinsics = intrinsics
        self.__intrinsics_count += 1

        if self.__intrinsics_callback:
            self.__intrinsics_callback(intrinsics)

    def __handle_message(self, parts):

        # check if length of parts is more than 0
        if len(parts) == 0:
            return

        message_type = parts[0].decode()
        if message_type == "frame":
            self.__handle_frame_message(parts)
        elif message_type == "intrinsics":
            self.__handle_intrinsics_message(parts)
        else:
            print(f"Unknown message type: {message_type}")

    def __run_loop(self):
        while not self.__should_stop:
            parts = self.socket.recv_multipart()
            self.__handle_message(parts)

            self.__frame_count += 1
            current_time = time.time()
            elapsed_time = current_time - self.prev_time

            if elapsed_time >= 1.0:  # Update FPS every second
                fps = self.__frame_count / elapsed_time
                self.__actual_fps = fps
                self.prev_time = current_time

    def __show_stream_loop(self):

        cv2.namedWindow("Stream", cv2.WINDOW_NORMAL)

        while not self.__should_stop:
            if self.__last_frame is not None:
                cv2.imshow("Stream", self.__last_frame.image)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        cv2.destroyAllWindows()


if __name__ == "__main__":
    handler = FourDCameraHandler()
    handler.__run_loop()
    handler.show_stream()
