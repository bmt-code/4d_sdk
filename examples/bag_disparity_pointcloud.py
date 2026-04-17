"""
Reads rectified stereo images and camera intrinsics from a ROS 2 bag,
computes a disparity map with colormap overlay, and generates a colored
point cloud. Outputs are saved to an output folder.

Usage:
    python3 bag_disparity_pointcloud.py --bag <path_to_bag> [--output <output_dir>]

Outputs:
    <output_dir>/disparity_color.png   — disparity map with jet colormap
    <output_dir>/left_rect.png         — left rectified image used
    <output_dir>/pointcloud.ply        — colored point cloud (open with MeshLab / CloudCompare)
"""

import argparse
import os

import cv2
import numpy as np
import open3d as o3d
import rosbag2_py
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import CameraInfo, CompressedImage

BASELINE_M = 0.30  # 30 cm


# ---------------------------------------------------------------------------
# Bag reading helpers
# ---------------------------------------------------------------------------

def make_reader(bag_path: str) -> rosbag2_py.SequentialReader:
    reader = rosbag2_py.SequentialReader()
    storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id="sqlite3")
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr",
    )
    reader.open(storage_options, converter_options)
    return reader


def collect_messages(bag_path: str, topics: list[str]) -> dict[str, list]:
    """Return dict topic -> list of deserialized messages (in order)."""
    type_map = {
        "/stereo_4d/left_rect/camera_info": CameraInfo,
        "/stereo_4d/right_rect/camera_info": CameraInfo,
        "/stereo_4d/left_rect/image/compressed": CompressedImage,
        "/stereo_4d/right_rect/image/compressed": CompressedImage,
    }

    reader = make_reader(bag_path)
    filter_ = rosbag2_py.StorageFilter(topics=topics)
    reader.set_filter(filter_)

    collected: dict[str, list] = {t: [] for t in topics}
    while reader.has_next():
        topic, data, _ = reader.read_next()
        if topic in type_map:
            msg = deserialize_message(data, type_map[topic])
            collected[topic].append(msg)

    return collected


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def decode_compressed(msg: CompressedImage) -> np.ndarray:
    buf = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def camera_matrix_from_info(info: CameraInfo) -> np.ndarray:
    return np.array(info.k, dtype=np.float64).reshape(3, 3)


# ---------------------------------------------------------------------------
# Disparity
# ---------------------------------------------------------------------------

def compute_disparity(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """SGBM disparity (float32, pixels). Returns raw disparity."""
    gray_l = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

    bs = 13
    sgbm = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=144,
        blockSize=bs,
        P1=14 * 3 * bs ** 2,
        P2=38 * 3 * bs ** 2,
        disp12MaxDiff=8,
        uniquenessRatio=0,
        speckleWindowSize=99,
        speckleRange=14,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    disp16 = sgbm.compute(gray_l, gray_r)   # fixed-point Q4: divide by 16
    return disp16.astype(np.float32) / 16.0


def disparity_to_colormap(disp: np.ndarray) -> np.ndarray:
    valid = disp > 0
    disp_vis = np.zeros_like(disp)
    disp_vis[valid] = disp[valid]

    # Normalize to 0-255 over valid range
    d_min = disp_vis[valid].min() if valid.any() else 0
    d_max = disp_vis[valid].max() if valid.any() else 1
    norm = np.zeros(disp.shape, dtype=np.uint8)
    norm[valid] = ((disp_vis[valid] - d_min) / (d_max - d_min) * 255).astype(np.uint8)

    color = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    color[~valid] = 0   # black for invalid pixels
    return color


# ---------------------------------------------------------------------------
# Point cloud
# ---------------------------------------------------------------------------

def disparity_to_pointcloud(
    disp: np.ndarray,
    color_img: np.ndarray,
    K: np.ndarray,
    baseline: float,
) -> o3d.geometry.PointCloud:
    """
    Back-project disparity to 3-D using pinhole model.
        Z = fx * baseline / disparity
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy
    """
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    h, w = disp.shape
    u_coords, v_coords = np.meshgrid(np.arange(w), np.arange(h))

    valid = disp > 1.0  # ignore zero / near-zero disparity

    d = disp[valid]
    u = u_coords[valid].astype(np.float64)
    v = v_coords[valid].astype(np.float64)

    Z = fx * baseline / d
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy

    # Filter out far points (> 10 m) to keep the cloud clean
    mask = Z < 10.0
    points = np.stack([X[mask], Y[mask], Z[mask]], axis=1)

    bgr = color_img[valid][mask].astype(np.float64) / 255.0
    colors = bgr[:, ::-1]  # BGR -> RGB

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Stereo disparity + point cloud from ROS 2 bag")
    parser.add_argument("--bag", required=True, help="Path to the ROS 2 bag directory")
    parser.add_argument("--output", default="output_disparity_pcl", help="Output folder")
    parser.add_argument(
        "--frame",
        type=int,
        default=10,
        help="Which frame index to process (default: 10, skips early unstable frames)",
    )
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    topics = [
        "/stereo_4d/left_rect/camera_info",
        "/stereo_4d/right_rect/camera_info",
        "/stereo_4d/left_rect/image/compressed",
        "/stereo_4d/right_rect/image/compressed",
    ]

    print("Reading bag…")
    msgs = collect_messages(args.bag, topics)

    cam_infos_left  = msgs["/stereo_4d/left_rect/camera_info"]
    cam_infos_right = msgs["/stereo_4d/right_rect/camera_info"]
    frames_left     = msgs["/stereo_4d/left_rect/image/compressed"]
    frames_right    = msgs["/stereo_4d/right_rect/image/compressed"]

    if not cam_infos_left:
        raise RuntimeError("No camera_info messages found. Check bag topics.")
    if not frames_left:
        raise RuntimeError("No image messages found. Check bag topics.")

    # Use the first available camera_info
    K_left  = camera_matrix_from_info(cam_infos_left[0])
    K_right = camera_matrix_from_info(cam_infos_right[0])
    print(f"Left  K:\n{K_left}")
    print(f"Right K:\n{K_right}")

    # Pick frame
    idx = min(args.frame, len(frames_left) - 1)
    left  = decode_compressed(frames_left[idx])
    right = decode_compressed(frames_right[idx])

    if left is None or right is None:
        raise RuntimeError(f"Failed to decode frame {idx}.")

    print(f"Processing frame {idx} — resolution {left.shape[1]}x{left.shape[0]}")

    # Disparity
    print("Computing disparity…")
    disp = compute_disparity(left, right)
    disp_color = disparity_to_colormap(disp)

    # Save disparity + left image
    cv2.imwrite(os.path.join(args.output, "left_rect.png"), left)
    cv2.imwrite(os.path.join(args.output, "disparity_color.png"), disp_color)
    print(f"  Saved disparity_color.png and left_rect.png")

    # Side-by-side preview
    preview = np.hstack([left, disp_color])
    cv2.imwrite(os.path.join(args.output, "preview.png"), preview)
    print(f"  Saved preview.png (left | disparity)")

    # Point cloud
    print("Building point cloud…")
    pcd = disparity_to_pointcloud(disp, left, K_left, BASELINE_M)
    ply_path = os.path.join(args.output, "pointcloud.ply")
    o3d.io.write_point_cloud(ply_path, pcd)
    print(f"  Saved {ply_path} ({len(pcd.points):,} points)")

    print("\nDone. Open 'pointcloud.ply' with MeshLab or CloudCompare.")


if __name__ == "__main__":
    main()
