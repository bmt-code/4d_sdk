import os

import cv2
from fourd_handler import FourDCameraHandler


class FourDImageSaver:
    def __init__(self, save_folder="saved_images", ip="172.31.1.77", port=5555):
        self.save_folder = save_folder
        os.makedirs(self.save_folder, exist_ok=True)

        self.camera_handler = FourDCameraHandler(ip=ip, port=port, show_stream=True)
        self.camera_handler.set_frame_callback(self.save_frame)

        self.image_count = 0

    def save_frame(self, frame):
        if frame.image is not None:
            if not hasattr(self, 'last_saved_time'):
                self.last_saved_time = 0

            current_time = cv2.getTickCount() / cv2.getTickFrequency()
            if current_time - self.last_saved_time >= 1.0:  # Save every 1 second
                filename = os.path.join(
                    self.save_folder, f"frame_{self.image_count:04d}.png"
                )
                cv2.imwrite(filename, frame.image)
                print(f"Saved: {filename}")
                self.image_count += 1
                self.last_saved_time = current_time

    def start(self):
        print("Starting ReachImageSaver...")
        self.camera_handler.start()

    def stop(self):
        print("Stopping ReachImageSaver...")
        self.camera_handler.stop()


if __name__ == "__main__":
    image_saver = FourDImageSaver(save_folder="camera_v0.5")
    try:
        image_saver.start()
    except KeyboardInterrupt:
        image_saver.stop()
