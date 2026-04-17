"""
ROS 2 node that subscribes to rectified stereo images and camera_info,
computes disparity, and publishes:
  - /stereo_4d/disparity/image/compressed  (CompressedImage, grayscale: bright=near, dark=far)
  - /stereo_4d/pointcloud                  (sensor_msgs/PointCloud2, XYZRGB)

Built-in preview with SGBM trackbars (no RViz/Foxglove/separate tuner needed):
  Enable in YAML:  show_preview: true
  Or at runtime:   ros2 param set /stereo_disparity_node show_preview true

  Keys in preview window:
    s — save current params to config/sgbm_params.yaml
    r — reset to defaults
    q — quit node

All SGBM parameters are also ROS 2 parameters:
  ros2 param set /stereo_disparity_node num_disparities 256
  ros2 param list /stereo_disparity_node
  ros2 param dump /stereo_disparity_node

Baseline is fixed at 30 cm. Intrinsics from /stereo_4d/left_rect/camera_info.

Usage:
    python3 stereo_disparity_node.py [--ros-args --params-file config/sgbm_params.yaml]
"""

import threading
import time
from pathlib import Path
from threading import Timer

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rcl_interfaces.msg import ParameterDescriptor, SetParametersResult
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import CameraInfo, CompressedImage, PointCloud2, PointField
from sensor_msgs_py import point_cloud2 as pc2
from std_msgs.msg import Header
from tf2_ros import StaticTransformBroadcaster

BASELINE_M = 0.30

_YAML_PATH = Path(__file__).parent.parent / "config" / "sgbm_params.yaml"

_XYZRGB_FIELDS = [
    PointField(name="x",   offset=0,  datatype=PointField.FLOAT32, count=1),
    PointField(name="y",   offset=4,  datatype=PointField.FLOAT32, count=1),
    PointField(name="z",   offset=8,  datatype=PointField.FLOAT32, count=1),
    PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1),
]

# Trackbar definitions: (label, param_name, tb_max, to_param_fn, to_tb_fn)
# Trackbar range is always 0..tb_max. Conversion fns map between tb int and param value.
_TB_DEFS = [
    # label               param_name            tb_max  to_param              to_tb
    ("num_disp (x16)",    "num_disparities",    29,     lambda v: (v+1)*16,   lambda p: max(0, p//16 - 1)),
    ("min_disparity",     "min_disparity",      128,    lambda v: v,          lambda p: max(0, p)),
    ("block_size (odd)",  "block_size",         20,     lambda v: v+1,        lambda p: max(0, p-1)),
    ("P1_factor",         "p1_factor",          63,     lambda v: v+1,        lambda p: max(0, p-1)),
    ("P2_factor",         "p2_factor",          127,    lambda v: v+1,        lambda p: max(0, p-1)),
    ("disp12Diff(0=off)", "disp12_max_diff",    20,     lambda v: v-1,        lambda p: max(0, p+1)),
    ("uniqueness%",       "uniqueness_ratio",   30,     lambda v: v,          lambda p: max(0, p)),
    ("speckle_window",    "speckle_window_size",300,    lambda v: v,          lambda p: max(0, p)),
    ("speckle_range",     "speckle_range",      10,     lambda v: v,          lambda p: max(0, p)),
    ("pre_filter_cap",    "pre_filter_cap",     62,     lambda v: v+1,        lambda p: max(0, p-1)),
    ("mode (0-3)",        "mode",               3,      lambda v: v,          lambda p: max(0, p)),
    ("proc_scale (x10)",  "proc_scale",         9,      lambda v: (v+1)/10.0, lambda p: max(0, round(p*10)-1)),
]

_DEFAULTS = {
    "num_disparities":    336,
    "min_disparity":       48,
    "block_size":           9,
    "p1_factor":            8,
    "p2_factor":           32,
    "disp12_max_diff":      2,
    "uniqueness_ratio":    10,
    "speckle_window_size": 120,
    "speckle_range":        2,
    "pre_filter_cap":      63,
    "mode":                 2,
    "proc_scale":          0.5,
    "min_depth_m":         0.5,
    "max_depth_m":        10.0,
    "show_preview":       False,
    "preview_scale":       0.5,
}


def _desc(text):
    return ParameterDescriptor(description=text)


class StereoDisparityNode(Node):
    def __init__(self):
        super().__init__("stereo_disparity_node")

        self.K        = None
        self.K_scaled = None
        self._K_lock       = threading.Lock()
        self._sgbm_lock    = threading.Lock()
        self._rebuild_lock = threading.Lock()
        self._rebuild_timer: Timer | None = None
        self.sgbm     = None

        # Latest frame for the preview (written by stereo callback, read by main thread)
        self._preview_lock  = threading.Lock()
        self._preview_frame = None

        # ------------------------------------------------------------------
        # Parameters
        # ------------------------------------------------------------------
        self.declare_parameter("proc_scale",         0.5,   _desc("Resize factor before SGBM (0.25–1.0)."))
        self.declare_parameter("min_disparity",       48,    _desc("Min disparity (px). Objects beyond fx*B/min_disp m are ignored."))
        self.declare_parameter("num_disparities",    336,    _desc("Search range (px, must be divisible by 16)."))
        self.declare_parameter("block_size",           9,    _desc("Matching block (px, forced odd). Bigger=smoother."))
        self.declare_parameter("p1_factor",            8,    _desc("P1 = factor * 3 * block_size^2. Penalises small disparity changes."))
        self.declare_parameter("p2_factor",           32,    _desc("P2 = factor * 3 * block_size^2. Penalises large disparity changes."))
        self.declare_parameter("disp12_max_diff",      2,    _desc("LR consistency tolerance (px). Negative = disabled."))
        self.declare_parameter("uniqueness_ratio",    10,    _desc("Reject match if 2nd-best is within ratio%. 0=off."))
        self.declare_parameter("speckle_window_size", 120,   _desc("Remove isolated blobs smaller than N px. 0=off."))
        self.declare_parameter("speckle_range",        2,    _desc("Max disparity variation inside a speckle blob."))
        self.declare_parameter("pre_filter_cap",      63,    _desc("Gradient clamp before matching (1–63)."))
        self.declare_parameter("mode",                 2,    _desc("0=SGBM 1=HH 2=SGBM_3WAY 3=HH4."))
        self.declare_parameter("min_depth_m",          0.5,  _desc("Near clip for point cloud (m)."))
        self.declare_parameter("max_depth_m",         10.0,  _desc("Far clip for point cloud (m)."))
        self.declare_parameter("show_preview",        False,     _desc("Open preview window with trackbars on main thread."))
        self.declare_parameter("preview_scale",        0.5,      _desc("Scale factor for the preview window image."))
        self.declare_parameter("profile",             "default", _desc("Profile name. 's' saves to config/profiles/<name>.yaml."))

        self._rebuild_sgbm()
        self.add_on_set_parameters_callback(self._on_params_changed)

        # Static TF: world → left_camera_rect
        # camera +Z (forward) → world +X
        # camera +Y (down)    → world -Z
        # camera +X (right)   → world -Y
        # q = (w=0.5, x=-0.5, y=0.5, z=-0.5)
        self._static_tf = StaticTransformBroadcaster(self)
        self._publish_world_tf()

        # Subscriptions
        self.create_subscription(
            CameraInfo, "/stereo_4d/left_rect/camera_info", self._camera_info_cb, 10
        )
        left_sub  = Subscriber(self, CompressedImage, "/stereo_4d/left_rect/image/compressed")
        right_sub = Subscriber(self, CompressedImage, "/stereo_4d/right_rect/image/compressed")
        self.sync = ApproximateTimeSynchronizer(
            [left_sub, right_sub], queue_size=5, slop=0.05
        )
        self.sync.registerCallback(self._stereo_cb)

        # Publishers — compressed only (avoids raw Image serialisation overhead)
        self.disp_pub = self.create_publisher(
            CompressedImage, "/stereo_4d/disparity/image/compressed", 2
        )
        self.pcl_pub = self.create_publisher(PointCloud2, "/stereo_4d/pointcloud", 2)

        self.get_logger().info(
            "Stereo disparity node ready. "
            "Preview+trackbars: -p show_preview:=true | "
            "Load profile: --params-file config/profiles/<name>.yaml -p profile:=<name> | "
            "Save profile: press 's' in preview window"
        )

    # ------------------------------------------------------------------
    # Parameter handling
    # ------------------------------------------------------------------

    def _p(self, name):
        return self.get_parameter(name).value

    def _rebuild_sgbm(self):
        scale = self._p("proc_scale")
        bs    = self._p("block_size")
        if bs % 2 == 0:
            bs += 1
        nd = max(16, (self._p("num_disparities") // 16) * 16)

        mode_map = {
            0: cv2.STEREO_SGBM_MODE_SGBM,
            1: cv2.STEREO_SGBM_MODE_HH,
            2: cv2.STEREO_SGBM_MODE_SGBM_3WAY,
            3: cv2.STEREO_SGBM_MODE_HH4,
        }
        new_sgbm = cv2.StereoSGBM_create(
            minDisparity      = self._p("min_disparity"),
            numDisparities    = nd,
            blockSize         = bs,
            P1                = self._p("p1_factor") * 3 * bs ** 2,
            P2                = self._p("p2_factor") * 3 * bs ** 2,
            disp12MaxDiff     = self._p("disp12_max_diff"),
            uniquenessRatio   = self._p("uniqueness_ratio"),
            speckleWindowSize = self._p("speckle_window_size"),
            speckleRange      = self._p("speckle_range"),
            preFilterCap      = self._p("pre_filter_cap"),
            mode              = mode_map.get(self._p("mode"), cv2.STEREO_SGBM_MODE_SGBM_3WAY),
        )
        with self._sgbm_lock:
            self.sgbm   = new_sgbm
            self._min_d = self._p("min_disparity")
            self._max_d = self._min_d + nd
            self._scale = scale

        # Update K_scaled whenever scale changes (K_lock required)
        with self._K_lock:
            if self.K is not None:
                self._update_K_scaled()

        self.get_logger().info(
            f"SGBM rebuilt — scale={scale} numDisp={nd} minDisp={self._min_d} "
            f"blockSize={bs} P1={self._p('p1_factor')*3*bs**2} "
            f"P2={self._p('p2_factor')*3*bs**2} "
            f"unique={self._p('uniqueness_ratio')} "
            f"speckle=({self._p('speckle_window_size')},{self._p('speckle_range')})"
        )

    def _on_params_changed(self, params: list[Parameter]) -> SetParametersResult:
        # Debounce: batch all YAML-load changes into one rebuild 200 ms after last
        with self._rebuild_lock:
            if self._rebuild_timer is not None:
                self._rebuild_timer.cancel()
            self._rebuild_timer = Timer(0.2, self._rebuild_sgbm)
            self._rebuild_timer.start()
        return SetParametersResult(successful=True)

    # ------------------------------------------------------------------
    # Camera info
    # ------------------------------------------------------------------

    def _camera_info_cb(self, msg: CameraInfo):
        with self._K_lock:
            if self.K is None:
                self.K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
                self._update_K_scaled()
                self.get_logger().info(
                    f"Intrinsics — fx={self.K[0,0]:.1f}, "
                    f"cx={self.K[0,2]:.1f}, cy={self.K[1,2]:.1f}"
                )

    def _update_K_scaled(self):
        """Must be called with _K_lock held."""
        s = self._scale
        self.K_scaled    = self.K.copy()
        self.K_scaled[0] *= s
        self.K_scaled[1] *= s

    # ------------------------------------------------------------------
    # Stereo callback
    # ------------------------------------------------------------------

    def _stereo_cb(self, left_msg: CompressedImage, right_msg: CompressedImage):
        with self._sgbm_lock:
            sgbm  = self.sgbm
            min_d = self._min_d
            max_d = self._max_d
            scale = self._scale

        with self._K_lock:
            if self.K is None:
                self.get_logger().warn(
                    "No intrinsics yet, skipping frame", throttle_duration_sec=2.0
                )
                return
            K_scaled = self.K_scaled.copy()

        left  = _decode(left_msg)
        right = _decode(right_msg)
        if left is None or right is None:
            return

        h_orig, w_orig = left.shape[:2]
        left_s  = cv2.resize(left,  None, fx=scale, fy=scale)
        right_s = cv2.resize(right, None, fx=scale, fy=scale)

        gl = cv2.cvtColor(left_s,  cv2.COLOR_BGR2GRAY)
        gr = cv2.cvtColor(right_s, cv2.COLOR_BGR2GRAY)
        disp = sgbm.compute(gl, gr).astype(np.float32) / 16.0

        # Grayscale disparity: bright=near, black=invalid
        valid = disp >= min_d
        norm  = np.zeros(disp.shape, dtype=np.uint8)
        norm[valid] = np.clip(
            (disp[valid] - min_d) / (max_d - min_d) * 255, 0, 255
        ).astype(np.uint8)

        # Upsample to original resolution for publishing
        disp_full = cv2.resize(norm, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)

        stamp  = self.get_clock().now().to_msg()
        header = Header(stamp=stamp, frame_id="left_camera_rect")

        buf = cv2.imencode(".jpg", disp_full)[1].tobytes()
        comp = CompressedImage()
        comp.header = header
        comp.format = "jpeg"
        comp.data   = buf
        self.disp_pub.publish(comp)

        self.pcl_pub.publish(
            self._build_pointcloud(disp, valid,
                                   np.clip(left_s, 0, 255).astype(np.uint8),
                                   K_scaled, header)
        )

        # Write preview frame — read by main thread in run_gui()
        if self._p("show_preview"):
            pscale     = self._p("preview_scale")
            left_small = cv2.resize(left_s, None, fx=pscale, fy=pscale)
            disp_small = cv2.resize(norm,   None, fx=pscale, fy=pscale)
            disp_bgr   = cv2.cvtColor(disp_small, cv2.COLOR_GRAY2BGR)
            for img, label in [(left_small, "Left rect"), (disp_bgr, "Disparity (bright=near)")]:
                cv2.putText(img, label, (8, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 180), 2, cv2.LINE_AA)
            combined = np.hstack([left_small, disp_bgr])
            with self._preview_lock:
                self._preview_frame = combined

    # ------------------------------------------------------------------
    # Point cloud
    # ------------------------------------------------------------------

    def _build_pointcloud(self, disp, valid, color_img, K, header):
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        min_depth = self._p("min_depth_m")
        max_depth = self._p("max_depth_m")

        h, w = disp.shape
        u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))

        d = disp[valid]
        Z = fx * BASELINE_M / d
        X = (u[valid] - cx) * Z / fx
        Y = (v[valid] - cy) * Z / fy

        keep = (Z > min_depth) & (Z < max_depth)
        X, Y, Z = X[keep], Y[keep], Z[keep]

        bgr = color_img[valid][keep]
        rgb_packed = (
            bgr[:, 2].astype(np.uint32) << 16
            | bgr[:, 1].astype(np.uint32) << 8
            | bgr[:, 0].astype(np.uint32)
        ).view(np.float32)

        return pc2.create_cloud(
            header, _XYZRGB_FIELDS, np.column_stack([X, Y, Z, rgb_packed])
        )

    # ------------------------------------------------------------------
    # GUI — must be called from the main thread
    # ------------------------------------------------------------------

    def run_gui(self):
        """Main thread loop: manages OpenCV window + trackbars + key events."""
        WIN = "Stereo Disparity"
        window_open = False

        while rclpy.ok():
            show = self._p("show_preview")

            if show and not window_open:
                cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(WIN, 1280, 480)
                self._create_trackbars(WIN)
                window_open = True
            elif not show and window_open:
                cv2.destroyWindow(WIN)
                window_open = False

            if window_open:
                with self._preview_lock:
                    frame = self._preview_frame
                if frame is not None:
                    cv2.imshow(WIN, frame)
                key = cv2.waitKey(30) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("s"):
                    self._save_yaml()
                elif key == ord("r"):
                    self._reset_defaults(WIN)
            else:
                time.sleep(0.033)

        cv2.destroyAllWindows()

    def _create_trackbars(self, win: str):
        """Create one trackbar per SGBM param; callbacks call set_parameters."""
        for label, param_name, tb_max, to_param, to_tb in _TB_DEFS:
            current = self._p(param_name)
            tb_val  = int(to_tb(current))
            tb_val  = max(0, min(tb_max, tb_val))

            def _make_cb(pname, tp):
                def cb(v):
                    pval = tp(v)
                    if isinstance(pval, float):
                        self.set_parameters([Parameter(pname, Parameter.Type.DOUBLE, pval)])
                    else:
                        self.set_parameters([Parameter(pname, Parameter.Type.INTEGER, int(pval))])
                return cb

            cv2.createTrackbar(label, win, tb_val, tb_max, _make_cb(param_name, to_param))

    def _reset_defaults(self, win: str):
        """Reset all SGBM params to defaults and sync trackbar positions."""
        params = []
        for label, param_name, tb_max, to_param, to_tb in _TB_DEFS:
            default = _DEFAULTS[param_name]
            if isinstance(default, float):
                params.append(Parameter(param_name, Parameter.Type.DOUBLE, default))
            else:
                params.append(Parameter(param_name, Parameter.Type.INTEGER, default))
            tb_val = max(0, min(tb_max, int(to_tb(default))))
            cv2.setTrackbarPos(label, win, tb_val)
        self.set_parameters(params)
        self.get_logger().info("Parameters reset to defaults.")

    def _save_yaml(self):
        """Write current params to config/profiles/<profile>.yaml (--params-file format).

        Also always overwrites config/sgbm_params.yaml so the default run picks up the
        latest save regardless of which profile is active.
        """
        int_params   = ["num_disparities", "min_disparity", "block_size", "p1_factor",
                        "p2_factor", "disp12_max_diff", "uniqueness_ratio",
                        "speckle_window_size", "speckle_range", "pre_filter_cap", "mode"]
        float_params = ["proc_scale", "min_depth_m", "max_depth_m", "preview_scale"]
        bool_params  = ["show_preview"]
        profile      = self._p("profile")

        lines = ["/stereo_disparity_node:\n", "  ros__parameters:\n"]
        lines.append(f"    profile: \"{profile}\"\n")
        for p in int_params:
            lines.append(f"    {p}: {self._p(p)}\n")
        for p in float_params:
            lines.append(f"    {p}: {self._p(p)}\n")
        for p in bool_params:
            lines.append(f"    {p}: {'true' if self._p(p) else 'false'}\n")

        content = "".join(lines)

        # Save to named profile
        profile_path = _YAML_PATH.parent / "profiles" / f"{profile}.yaml"
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        profile_path.write_text(content)

        # Also update the default params file
        _YAML_PATH.write_text(content)

        self.get_logger().info(f"Params saved → {profile_path}  (and {_YAML_PATH.name})")

    def _publish_world_tf(self):
        """Broadcast static world → left_camera_rect transform.

        Mapping:  cam +Z → world +X  (depth = forward in world)
                  cam +Y → world -Z  (down in cam = down in world)
                  cam +X → world -Y  (right in cam = left in world)

        Rotation matrix:
            R = [[ 0,  0,  1],
                 [-1,  0,  0],
                 [ 0, -1,  0]]
        Quaternion: w=0.5  x=-0.5  y=0.5  z=-0.5
        """
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "world"
        t.child_frame_id = "left_camera_rect"
        t.transform.translation.x = 0.0
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.0
        t.transform.rotation.x = -0.5
        t.transform.rotation.y =  0.5
        t.transform.rotation.z = -0.5
        t.transform.rotation.w =  0.5
        self._static_tf.sendTransform(t)

    def destroy_node(self):
        with self._rebuild_lock:
            if self._rebuild_timer is not None:
                self._rebuild_timer.cancel()
        super().destroy_node()


# ---------------------------------------------------------------------------

def _decode(msg: CompressedImage) -> np.ndarray | None:
    buf = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def main(args=None):
    rclpy.init(args=args)
    node = StereoDisparityNode()

    # ROS spin runs in a background thread; main thread is reserved for OpenCV GUI
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    try:
        node.run_gui()          # blocks on main thread until 'q' or rclpy shuts down
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
