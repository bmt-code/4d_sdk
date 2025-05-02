from stereo_4d import Stereo4DCameraHandler, Stereo4DFrame
import time
import cv2
import os

folder_path = "cam0.9"  # Replace with your folder path


def save_frame(frame: Stereo4DFrame):
    # Ensure the folder exists
    os.makedirs(folder_path, exist_ok=True)

    # Save the frame to a file in the folder
    filename = os.path.join(folder_path, f"frame_{time.time()}.png")
    cv2.imwrite(filename, frame.image)
    print(f"Saved frame to {filename}")


if __name__ == "__main__":
    handler = Stereo4DCameraHandler(show_stream=True)
    try:
        handler.start(wait=True)
        while True:
            time.sleep(1)
            frame = handler.get_last_frame()
            if frame is not None:
                save_frame(frame)
            else:
                print("No frame available.")
    except KeyboardInterrupt:
        print("\nStopping camera handler...")
        handler.stop()
        print("Camera handler stopped.")
