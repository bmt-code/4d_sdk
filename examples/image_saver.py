from stereo_4d import Stereo4DCameraHandler, Stereo4DFrame
import time
import cv2
import os
import numpy as np

folder_path = "images_stereo_4d"  # Replace with your folder path


def save_frame(frame: Stereo4DFrame):
    # Ensure the folder exists
    os.makedirs(folder_path, exist_ok=True)

    # Save the frame to a file in the folder
    filename = os.path.join(folder_path, f"frame_{time.time()}.png")
    cv2.imwrite(filename, frame.image)
    print(f"Saved frame to {filename}")


if __name__ == "__main__":
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

            img = np.copy(frame.image)
            # Write the number of saved images at the top left
            cv2.putText(img, f"Saved: {saved_images}", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 3)
            if time.time() - current_time > 1:
                current_time = time.time()
                save_frame(frame)
                saved_images += 1
                img = 255 * np.ones_like(img)
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
