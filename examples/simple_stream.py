from stereo_4d import Stereo4DCameraHandler
import time

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
