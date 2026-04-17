from stereo_4d import Stereo4DCameraHandler, Stereo4DFrame
import time
import cv2
import os
import argparse
import numpy as np


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

    handler = Stereo4DCameraHandler()

    # Get initial number of saved images in the folder
    os.makedirs(folder_path, exist_ok=True)
    saved_images = len([f for f in os.listdir(folder_path) if f.endswith(".png")])

    current_time = time.time()
    cv2.namedWindow("Stream", cv2.WINDOW_NORMAL)
    try:
        handler.start(wait=True)
        while True:
            time.sleep(1/30)  # Simulate a frame rate of 30 FPS
            frame = handler.get_last_frame()

            if frame is None:
                print("No frame available.")
                continue

            if frame.image is None:
                print("No image data available to save.")
                continue

            img = np.copy(frame.image)
            # Write the number of saved images at the top left
            cv2.putText(img, f"Saved: {saved_images}", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 3)
            if time.time() - current_time > 1:
                current_time = time.time()
                save_frame(frame, folder_path)
                saved_images += 1
                img = 50 * np.ones_like(img)
                cv2.imshow("Stream", img)
                cv2.waitKey(20)  # Show for 20 ms
            else:
                cv2.imshow("Stream", img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    handler.stop()
                    break

    except KeyboardInterrupt:
        print("\nStopping camera handler...")
        handler.stop()
        print("Camera handler stopped.")
