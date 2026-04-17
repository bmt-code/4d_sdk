"""
ROS 2 node that subscribes to rectified stereo images and camera_info,
computes disparity, and publishes:
  - /stereo_4d/disparity/image          (sensor_msgs/Image, grayscale: bright=near, dark=far)
  - /stereo_4d/disparity/image/compressed
  - /stereo_4d/pointcloud               (sensor_msgs/PointCloud2, XYZRGB)

All SGBM parameters are exposed as ROS 2 parameters and can be changed
at runtime — the matcher rebuilds automatically on any change.

Tune live:
    ros2 param set /stereo_disparity_node num_disparities 256
    ros2 param set /stereo_disparity_node block_size 11
    ros2 param set /stereo_disparity_node uniqueness_ratio 5

List all params:
    ros2 param list /stereo_disparity_node
    ros2 param dump /stereo_disparity_node

Images are processed at proc_scale for speed; disparity is back-projected
using the scaled intrinsics so metric depth is correct.

Baseline is fixed at 30 cm. Intrinsics from /stereo_4d/left_rect/camera_info.

Usage:
    python3 stereo_disparity_node.py
"""

import threading
from threading import Timer

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rcl_interfaces.msg import ParameterDescriptor, SetParametersResult
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import CameraInfo, CompressedImage, Image, PointCloud2, PointField
from sensor_msgs_py import point_cloud2 as pc2
from std_msgs.msg import Header

BASELINE_M = 0.30

_XYZRGB_FIELDS = [
    PointField(name="x",   offset=0,  datatype=PointField.FLOAT32, count=1),
    PointField(name="y",   offset=4,  datatype=PointField.FLOAT32, count=1),
    PointField(name="z",   offset=8,  datatype=PointField.FLOAT32, count=1),
    PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1),
]


def _desc(text):
    return ParameterDescriptor(description=text)


class StereoDisparityNode(Node):
    def __init__(self):
        super().__init__("stereo_disparity_node")

        self.bridge   = CvBridge()
        self.K        = None
        self.K_scaled = None
        self._K_lock  = threading.Lock()
        self._sgbm_lock    = threading.Lock()
        self._rebuild_lock = threading.Lock()
        self._rebuild_timer: Timer | None = None
        self.sgbm     = None

        # ------------------------------------------------------------------
        # Declare all SGBM params (with defaults tuned for fx=2530, B=0.30m,
        # proc_scale=0.5, target range 1–8 m)
        # ------------------------------------------------------------------
        self.declare_parameter("proc_scale",         0.5,  _desc("Resize factor before SGBM (0.25–1.0). Trades speed vs detail."))
        self.declare_parameter("min_disparity",       48,   _desc("Min disparity (px). Ignores objects beyond fx_eff*B/min_disp."))
        self.declare_parameter("num_disparities",    336,   _desc("Disparity search range (px, must be divisible by 16)."))
        self.declare_parameter("block_size",           9,   _desc("Matching block size (px, odd). Bigger=smoother, less detail."))
        self.declare_parameter("p1_factor",            8,   _desc("P1 = p1_factor * 3 * block_size^2. Penalises small disp changes."))
        self.declare_parameter("p2_factor",           32,   _desc("P2 = p2_factor * 3 * block_size^2. Penalises large disp changes. Must be > p1_factor."))
        self.declare_parameter("disp12_max_diff",      2,   _desc("Left-right consistency check tolerance (px). -1 to disable."))
        self.declare_parameter("uniqueness_ratio",    10,   _desc("Reject match if second-best is within ratio% of best. 0 to disable."))
        self.declare_parameter("speckle_window_size", 120,  _desc("Remove blobs smaller than N px. 0 to disable."))
        self.declare_parameter("speckle_range",        2,   _desc("Max disparity variation within a speckle blob."))
        self.declare_parameter("pre_filter_cap",      63,   _desc("Clamps gradient before matching. Reduces exposure mismatch sensitivity."))
        self.declare_parameter("mode",                 2,   _desc("SGBM mode: 0=SGBM, 1=HH (slow+best), 2=SGBM_3WAY, 3=HH4."))
        self.declare_parameter("max_depth_m",         10.0, _desc("Discard point cloud points beyond this distance (m)."))
        self.declare_parameter("min_depth_m",          0.5, _desc("Discard point cloud points closer than this distance (m)."))

        self._rebuild_sgbm()

        # Rebuild matcher whenever any parameter changes
        self.add_on_set_parameters_callback(self._on_params_changed)

        # Camera info
        self.create_subscription(
            CameraInfo, "/stereo_4d/left_rect/camera_info", self._camera_info_cb, 10
        )

        # Synchronized stereo images
        left_sub  = Subscriber(self, CompressedImage, "/stereo_4d/left_rect/image/compressed")
        right_sub = Subscriber(self, CompressedImage, "/stereo_4d/right_rect/image/compressed")
        self.sync = ApproximateTimeSynchronizer(
            [left_sub, right_sub], queue_size=5, slop=0.05
        )
        self.sync.registerCallback(self._stereo_cb)

        # Publishers
        self.disp_pub            = self.create_publisher(Image,          "/stereo_4d/disparity/image",            2)
        self.disp_compressed_pub = self.create_publisher(CompressedImage, "/stereo_4d/disparity/image/compressed", 2)
        self.pcl_pub             = self.create_publisher(PointCloud2,    "/stereo_4d/pointcloud",                 2)

        self.get_logger().info("Stereo disparity node ready. Tune with: ros2 param set /stereo_disparity_node <param> <value>")

    # ------------------------------------------------------------------
    # Parameter handling
    # ------------------------------------------------------------------

    def _p(self, name):
        return self.get_parameter(name).value

    def _rebuild_sgbm(self):
        scale = self._p("proc_scale")
        bs    = self._p("block_size")
        # Enforce odd block size
        if bs % 2 == 0:
            bs += 1
        # Enforce numDisparities divisible by 16
        nd = self._p("num_disparities")
        nd = max(16, (nd // 16) * 16)

        mode_map = {
            0: cv2.STEREO_SGBM_MODE_SGBM,
            1: cv2.STEREO_SGBM_MODE_HH,
            2: cv2.STEREO_SGBM_MODE_SGBM_3WAY,
            3: cv2.STEREO_SGBM_MODE_HH4,
        }
        mode = mode_map.get(self._p("mode"), cv2.STEREO_SGBM_MODE_SGBM_3WAY)

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
            mode              = mode,
        )

        with self._sgbm_lock:
            self.sgbm       = new_sgbm
            self._min_d     = self._p("min_disparity")
            self._max_d     = self._min_d + nd
            self._scale     = scale

        self.get_logger().info(
            f"SGBM rebuilt — scale={scale} numDisp={nd} minDisp={self._min_d} "
            f"blockSize={bs} P1={self._p('p1_factor')*3*bs**2} P2={self._p('p2_factor')*3*bs**2} "
            f"unique={self._p('uniqueness_ratio')} speckle=({self._p('speckle_window_size')},{self._p('speckle_range')})"
        )

    def _on_params_changed(self, params: list[Parameter]) -> SetParametersResult:
        # Debounce: batch rapid changes (e.g. 14 params from --params-file) into
        # a single rebuild 200 ms after the last change settles.
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
                    f"Intrinsics — fx={self.K[0,0]:.1f}, cx={self.K[0,2]:.1f}, cy={self.K[1,2]:.1f}"
                )

    def _update_K_scaled(self):
        """Call with _K_lock held."""
        s = self._scale
        self.K_scaled = self.K.copy()
        self.K_scaled[0] *= s
        self.K_scaled[1] *= s

    # ------------------------------------------------------------------
    # Stereo callback
    # ------------------------------------------------------------------

    def _stereo_cb(self, left_msg: CompressedImage, right_msg: CompressedImage):
        with self._sgbm_lock:
            sgbm   = self.sgbm
            min_d  = self._min_d
            max_d  = self._max_d
            scale  = self._scale

        with self._K_lock:
            if self.K is None:
                self.get_logger().warn("No intrinsics yet, skipping frame", throttle_duration_sec=2.0)
                return
            # Recompute scaled K if scale changed
            self._update_K_scaled()
            K_scaled = self.K_scaled.copy()

        left  = self._decode(left_msg)
        right = self._decode(right_msg)
        if left is None or right is None:
            return

        h_orig, w_orig = left.shape[:2]
        left_s  = cv2.resize(left,  None, fx=scale, fy=scale)
        right_s = cv2.resize(right, None, fx=scale, fy=scale)

        gl = cv2.cvtColor(left_s,  cv2.COLOR_BGR2GRAY)
        gr = cv2.cvtColor(right_s, cv2.COLOR_BGR2GRAY)
        disp = sgbm.compute(gl, gr).astype(np.float32) / 16.0

        stamp  = self.get_clock().now().to_msg()
        header = Header(stamp=stamp, frame_id="left_camera_rect")

        # Grayscale disparity image: bright = near, black = invalid
        valid = disp >= min_d
        norm  = np.zeros(disp.shape, dtype=np.uint8)
        norm[valid] = np.clip(
            (disp[valid] - min_d) / (max_d - min_d) * 255, 0, 255
        ).astype(np.uint8)
        disp_full = cv2.resize(norm, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)

        self.disp_pub.publish(self.bridge.cv2_to_imgmsg(disp_full, encoding="mono8"))
        compressed = self.bridge.cv2_to_compressed_imgmsg(disp_full, dst_format="jpeg")
        compressed.header = header
        self.disp_compressed_pub.publish(compressed)

        # Point cloud
        pcd_msg = self._build_pointcloud(disp, valid,
                                         np.clip(left_s, 0, 255).astype(np.uint8),
                                         K_scaled, header)
        self.pcl_pub.publish(pcd_msg)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _decode(msg: CompressedImage) -> np.ndarray | None:
        buf = np.frombuffer(bytes(msg.data), dtype=np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)

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

        return pc2.create_cloud(header, _XYZRGB_FIELDS, np.column_stack([X, Y, Z, rgb_packed]))


def main(args=None):
    rclpy.init(args=args)
    node = StereoDisparityNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
