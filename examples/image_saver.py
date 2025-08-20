from stereo_4d import Stereo4DCameraHandler, Stereo4DFrame
import time
import cv2
import os
import argparse


def save_frame(frame: Stereo4DFrame, folder_path: str):
    # Ensure the folder exists
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)

    if frame.image is None:
        print("No image data available to save.")
        return

    # Save the frame to a file in the folder
    filename = os.path.join(folder_path, f"frame_{time.time()}.png")
    cv2.imwrite(filename, frame.image)
    print(f"Saved frame to {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stereo4D Image Saver")
    parser.add_argument(
        "--folder", type=str, help="Folder to save images and config", required=True
    )
    args = parser.parse_args()
    folder_path = args.folder

    # Place config in the same folder
    config_path = os.path.join(folder_path, "config.yaml")

    handler = Stereo4DCameraHandler(show_stream=True)
    try:
        handler.start(wait=True)
        while True:
            time.sleep(1)
            frame = handler.get_last_frame()
            if frame is not None:
                save_frame(frame, folder_path)
            else:
                print("No frame available.")
    except KeyboardInterrupt:
        print("\nStopping camera handler...")
        handler.stop()
        print("Camera handler stopped.")
