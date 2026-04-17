"""
Live SGBM trackbar tuner for stereo_disparity_node.

Subscribes to /stereo_4d/disparity/image/compressed for live preview.
Trackbar changes are batched and sent at 0.5 Hz to avoid overloading the node.

Buttons (click on the panel below the image):
  [ RESET ] — restore all trackbars to default values, send to node
  [ SAVE  ] — write current params to config/sgbm_params.yaml

Requires stereo_disparity_node.py to be running.

Usage:
    python3 calib/stereo_tuner.py
"""

import os
import threading
import time

import cv2
import numpy as np
import rclpy
import yaml
from rcl_interfaces.msg import Parameter as RclParam
from rcl_interfaces.msg import ParameterType, ParameterValue
from rcl_interfaces.srv import SetParameters
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage

WIN         = "SGBM Tuner  |  q=quit  |  click RESET or SAVE"
TARGET_NODE = "/stereo_disparity_node"
FLUSH_HZ    = 0.5   # max param-set rate
PARAMS_YAML = os.path.join(os.path.dirname(__file__), "..", "config", "sgbm_params.yaml")

# (ros_param_name, trackbar_label, default, trackbar_max, is_x16)
#   is_x16=True  → trackbar stores value//16, actual = trackbar*16
#   is_x16=False → trackbar value IS the actual value
PARAMS = [
    ("num_disparities",     "num_disp (x16)",     21,  64, True),
    ("min_disparity",       "min_disp",            48, 200, False),
    ("block_size",          "block_size (odd)",     9,  21, False),
    ("p1_factor",           "P1 factor",            8,  30, False),
    ("p2_factor",           "P2 factor",           32,  80, False),
    ("disp12_max_diff",     "disp12 max diff",      2,  20, False),
    ("uniqueness_ratio",    "uniqueness ratio",    10,  25, False),
    ("speckle_window_size", "speckle window",     120, 300, False),
    ("speckle_range",       "speckle range",        2,  50, False),
    ("pre_filter_cap",      "prefilter cap",       63,  63, False),
    ("mode",                "mode (0-3)",           2,   3, False),
]

DEFAULTS = {p[0]: p[2] for p in PARAMS}   # ros_name -> default trackbar value

BTN_H    = 60
BTN_RESET = (20,  10, 180, 50)   # x1, y1, x2, y2 inside button panel
BTN_SAVE  = (220, 10, 380, 50)


# ---------------------------------------------------------------------------
# ROS node
# ---------------------------------------------------------------------------

class TunerNode(Node):
    def __init__(self):
        super().__init__("stereo_tuner")
        self._img_lock     = threading.Lock()
        self._pending_lock = threading.Lock()
        self._latest       = None
        self._pending      = {}        # ros_name -> actual int value, dirty queue
        self._last_flush   = 0.0
        self.node_ready    = False     # set True when param service becomes available

        self._param_client = self.create_client(
            SetParameters, f"{TARGET_NODE}/set_parameters"
        )
        self.create_subscription(
            CompressedImage,
            "/stereo_4d/disparity/image/compressed",
            self._img_cb,
            2,
        )
        # Wait for disparity node in background — don't block window creation
        threading.Thread(target=self._wait_for_node, daemon=True).start()

    def _wait_for_node(self):
        if self._param_client.wait_for_service(timeout_sec=30.0):
            self.node_ready = True
            self.get_logger().info(f"Connected to {TARGET_NODE}")
        else:
            self.get_logger().error(f"Could not reach {TARGET_NODE} after 30s. Is it running?")

    # -- image ---------------------------------------------------------------

    def _img_cb(self, msg: CompressedImage):
        buf = np.frombuffer(bytes(msg.data), dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            with self._img_lock:
                self._latest = img

    def get_image(self):
        with self._img_lock:
            return self._latest.copy() if self._latest is not None else None

    # -- param flushing ------------------------------------------------------

    def queue_param(self, name: str, value: int):
        with self._pending_lock:
            self._pending[name] = value

    def try_flush(self):
        """Call from main loop. Sends pending params at most FLUSH_HZ."""
        now = time.time()
        if now - self._last_flush < 1.0 / FLUSH_HZ:
            return
        with self._pending_lock:
            if not self._pending:
                return
            batch = dict(self._pending)
            self._pending.clear()
        self._last_flush = now
        self._send_batch(batch)

    def send_now(self, batch: dict):
        """Force-send a batch immediately (used by RESET)."""
        self._send_batch(batch)

    def _send_batch(self, batch: dict):
        if not self._param_client.service_is_ready():
            return
        req = SetParameters.Request()
        for name, value in batch.items():
            pv            = ParameterValue()
            pv.type       = ParameterType.PARAMETER_INTEGER
            pv.integer_value = int(value)
            p       = RclParam()
            p.name  = name
            p.value = pv
            req.parameters.append(p)
        self._param_client.call_async(req)


# ---------------------------------------------------------------------------
# Trackbar helpers
# ---------------------------------------------------------------------------

def _actual(ros_name: str, trackbar_val: int) -> int:
    for name, _, _, _, is_x16 in PARAMS:
        if name == ros_name:
            return max(16, trackbar_val * 16) if is_x16 else trackbar_val
    return trackbar_val


def _make_cb(node: TunerNode, ros_name: str):
    def cb(val):
        node.queue_param(ros_name, _actual(ros_name, val))
    return cb


def _set_trackbar_defaults():
    for ros_name, label, default, _, is_x16 in PARAMS:
        cv2.setTrackbarPos(label, WIN, default)   # default IS already trackbar units


def _read_all_actuals() -> dict:
    result = {}
    for ros_name, label, _, _, is_x16 in PARAMS:
        raw = cv2.getTrackbarPos(label, WIN)
        result[ros_name] = _actual(ros_name, raw)
    return result


# ---------------------------------------------------------------------------
# Button panel
# ---------------------------------------------------------------------------

def _draw_buttons(clicked_reset=False, clicked_save=False) -> np.ndarray:
    panel = np.zeros((BTN_H, 1280, 3), dtype=np.uint8)
    panel[:] = (40, 40, 40)

    def draw_btn(rect, label, highlight):
        x1, y1, x2, y2 = rect
        color = (0, 200, 80) if highlight else (80, 80, 80)
        cv2.rectangle(panel, (x1, y1), (x2, y2), color, -1)
        cv2.rectangle(panel, (x1, y1), (x2, y2), (200, 200, 200), 1)
        tw, th = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        tx = x1 + (x2 - x1 - tw) // 2
        ty = y1 + (y2 - y1 + th) // 2
        cv2.putText(panel, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    draw_btn(BTN_RESET, "RESET", clicked_reset)
    draw_btn(BTN_SAVE,  "SAVE",  clicked_save)
    cv2.putText(panel, "q = quit   p = print params",
                (440, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (160, 160, 160), 1)
    return panel


def _in_btn(rect, x, y, img_h):
    x1, y1, x2, y2 = rect
    return x1 <= x <= x2 and (img_h + y1) <= y <= (img_h + y2)


# ---------------------------------------------------------------------------
# Mouse callback state
# ---------------------------------------------------------------------------

class MouseState:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.clicked = False

    def callback(self, event, x, y, flags, param):
        self.x, self.y = x, y
        if event == cv2.EVENT_LBUTTONDOWN:
            self.clicked = True


# ---------------------------------------------------------------------------
# Save / load YAML
# ---------------------------------------------------------------------------

def save_yaml(actuals: dict):
    data = {
        TARGET_NODE: {
            "ros__parameters": actuals
        }
    }
    os.makedirs(os.path.dirname(os.path.abspath(PARAMS_YAML)), exist_ok=True)
    with open(PARAMS_YAML, "w") as f:
        yaml.dump(data, f, default_flow_style=False)
    print(f"Saved → {os.path.abspath(PARAMS_YAML)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    rclpy.init()
    node = TunerNode()

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 1280, 760)

    for ros_name, label, default, maxval, is_x16 in PARAMS:
        init = default   # already in trackbar units (defaults are trackbar values)
        mx   = maxval // 16 if is_x16 else maxval
        cv2.createTrackbar(label, WIN, init, mx, _make_cb(node, ros_name))

    mouse = MouseState()
    cv2.setMouseCallback(WIN, mouse.callback)

    placeholder = np.zeros((480, 1280, 3), dtype=np.uint8)
    cv2.putText(placeholder, "Waiting for disparity image...",
                (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (200, 200, 200), 2)

    flash_reset = 0.0
    flash_save  = 0.0

    print(f"Tuner ready. Params flush at {FLUSH_HZ} Hz to avoid overload.")
    print(f"YAML saves to: {os.path.abspath(PARAMS_YAML)}\n")

    while True:
        # --- flush pending params at rate limit ---
        node.try_flush()

        # --- disparity image ---
        img = node.get_image()
        if img is not None:
            display = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            display = cv2.resize(display, (1280, 480))
        else:
            display = placeholder.copy()

        # overlay key params + connection status
        nd = cv2.getTrackbarPos("num_disp (x16)", WIN) * 16
        bs = cv2.getTrackbarPos("block_size (odd)", WIN)
        mn = cv2.getTrackbarPos("min_disp", WIN)
        cv2.putText(display,
                    f"numDisp={nd}  blockSize={bs}  minDisp={mn}  [{FLUSH_HZ}Hz flush]",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 180), 2)
        if not node.node_ready:
            cv2.putText(display, "Waiting for stereo_disparity_node...",
                        (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 100, 255), 2)
        elif node.get_image() is None:
            cv2.putText(display, "Node connected — waiting for bag/camera data",
                        (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)

        # --- button panel ---
        now = time.time()
        panel = _draw_buttons(
            clicked_reset=(now - flash_reset) < 0.3,
            clicked_save =(now - flash_save)  < 0.3,
        )

        # --- handle mouse click ---
        if mouse.clicked:
            mouse.clicked = False
            mx, my = mouse.x, mouse.y
            ih = display.shape[0]

            if _in_btn(BTN_RESET, mx, my, ih):
                flash_reset = now
                _set_trackbar_defaults()
                defaults_actual = {p[0]: _actual(p[0], p[2]) for p in PARAMS}
                node.send_now(defaults_actual)
                print("Reset to defaults.")

            elif _in_btn(BTN_SAVE, mx, my, ih):
                flash_save = now
                actuals = _read_all_actuals()
                save_yaml(actuals)

        # --- compose and show ---
        combined = np.vstack([display, panel])
        cv2.imshow(WIN, combined)

        key = cv2.waitKey(33) & 0xFF
        if key == ord("q"):
            break
        if key == ord("p"):
            print("--- Current params ---")
            for ros_name, label, _, _, _ in PARAMS:
                print(f"  {ros_name:25s} = {_read_all_actuals()[ros_name]}")
            print()

    cv2.destroyAllWindows()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
