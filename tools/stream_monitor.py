from stereo_4d import Stereo4DCameraHandler
import time
import logging
from datetime import datetime

# --- Configuration ---
POLL_INTERVAL = 0.025       # seconds between polls
DROPOUT_THRESHOLD = 0.5   # seconds without a new frame to declare dropout
LOG_FILE = "stream_monitor.log"

# --- Logging setup ---
logger = logging.getLogger("stream_monitor")
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

file_handler = logging.FileHandler(LOG_FILE)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


if __name__ == "__main__":
    handler = Stereo4DCameraHandler()

    last_frame_timestamp = None
    last_new_frame_time = None
    in_dropout = False
    dropout_start_time = None
    total_dropouts = 0

    try:
        handler.start(wait=True)
        logger.info("STARTED  | Monitoring camera stream")
        last_new_frame_time = time.time()

        while True:
            time.sleep(POLL_INTERVAL)
            frame = handler.get_last_frame()
            now = time.time()

            if frame is None:
                # No frame at all yet
                if not in_dropout and last_new_frame_time and (now - last_new_frame_time) >= DROPOUT_THRESHOLD:
                    in_dropout = True
                    dropout_start_time = now - (now - last_new_frame_time)
                    total_dropouts += 1
                    logger.warning(
                        "DROPOUT  | No new frames for %.1fs (no frame received) [#%d]",
                        now - last_new_frame_time, total_dropouts
                    )
                continue

            # Got a frame — check if it's new
            if last_frame_timestamp is not None and frame.timestamp == last_frame_timestamp:
                # Same frame as before
                if not in_dropout and (now - last_new_frame_time) >= DROPOUT_THRESHOLD:
                    in_dropout = True
                    dropout_start_time = last_new_frame_time
                    total_dropouts += 1
                    last_ts = datetime.fromtimestamp(last_new_frame_time).strftime("%H:%M:%S")
                    logger.warning(
                        "DROPOUT  | No new frames for %.1fs (last frame at %s) [#%d]",
                        now - last_new_frame_time, last_ts, total_dropouts
                    )
            else:
                # New frame arrived
                if in_dropout:
                    gap = now - dropout_start_time
                    logger.info("RESUMED  | Stream back after %.1fs gap", gap)
                    in_dropout = False
                    dropout_start_time = None

                last_frame_timestamp = frame.timestamp
                last_new_frame_time = now

    except KeyboardInterrupt:
        duration = time.time() - last_new_frame_time if last_new_frame_time else 0
        logger.info("STOPPED  | Monitoring ended. Total dropouts: %d", total_dropouts)
        handler.stop()
