"""Check stereo camera calibration quality using a checkerboard.

Detects checkerboard corners in live stereo frames, triangulates 3D points,
and compares pairwise distances against the known grid geometry.

Usage:
    python check_calibration.py
    python check_calibration.py --chessboard 7,5 --square 30.0
"""

import argparse
import time

import cv2
import numpy as np
from scipy.spatial import distance_matrix

from stereo_4d import Stereo4DCameraHandler


def points_3d_from_stereo(left_points, right_points, K_left, K_right, baseline):
    """Triangulate 3D points from stereo correspondences.

    Args:
        left_points: (N, 2) array of left image points.
        right_points: (N, 2) array of right image points.
        K_left: 3x3 left camera intrinsic matrix.
        K_right: 3x3 right camera intrinsic matrix.
        baseline: Distance between cameras in meters.

    Returns:
        (N, 3) array of 3D points in meters.
    """
    if left_points.shape[0] != right_points.shape[0]:
        return np.empty((0, 3))

    disparity = left_points[:, 0] - right_points[:, 0]
    disparity = np.clip(disparity, 1e-6, None)

    fx = (K_left[0, 0] + K_right[0, 0]) / 2
    cx = K_left[0, 2]
    fy = (K_left[1, 1] + K_right[1, 1]) / 2
    cy = K_left[1, 2]

    depths = fx * baseline / disparity

    pts_3d = np.zeros((left_points.shape[0], 3))
    pts_3d[:, 0] = (left_points[:, 0] - cx) * depths / fx
    pts_3d[:, 1] = (left_points[:, 1] - cy) * depths / fy
    pts_3d[:, 2] = depths

    return pts_3d


def check_grid_quality(points_3d, chessboard_size, square_size_mm):
    """Compare pairwise distances of detected 3D points against ideal grid.

    Args:
        points_3d: (N, 3) array of 3D points in mm.
        chessboard_size: (cols, rows) inner corner count.
        square_size_mm: Checkerboard square size in mm.

    Returns:
        (mean_error, max_error) in mm.
    """
    cols, rows = chessboard_size
    x_vec = np.arange(cols) * square_size_mm
    y_vec = np.arange(rows) * square_size_mm
    model_grid = np.array([[x, y, 0] for y in y_vec for x in x_vec])

    model_D = distance_matrix(model_grid, model_grid)
    D = distance_matrix(points_3d, points_3d)

    idx = np.triu_indices_from(D, k=1)
    diff = np.abs(D[idx] - model_D[idx])

    return np.mean(diff), np.max(diff)


def find_checkerboard(img_left, img_right, chessboard_size):
    """Detect checkerboard corners in both images with sub-pixel refinement."""
    gray_l = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY) if len(img_left.shape) == 3 else img_left
    gray_r = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY) if len(img_right.shape) == 3 else img_right

    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    flags_fast = flags | cv2.CALIB_CB_FAST_CHECK

    ret_l, corners_l = cv2.findChessboardCorners(gray_l, chessboard_size, flags=flags_fast)
    if not ret_l:
        ret_l, corners_l = cv2.findChessboardCorners(gray_l, chessboard_size, flags=flags)

    ret_r, corners_r = cv2.findChessboardCorners(gray_r, chessboard_size, flags=flags_fast)
    if not ret_r:
        ret_r, corners_r = cv2.findChessboardCorners(gray_r, chessboard_size, flags=flags)

    if ret_l and ret_r:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners_l = cv2.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1), criteria)
        corners_r = cv2.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1), criteria)
        cv2.drawChessboardCorners(img_left, chessboard_size, corners_l, ret_l)
        cv2.drawChessboardCorners(img_right, chessboard_size, corners_r, ret_r)
        return corners_l.reshape(-1, 2), corners_r.reshape(-1, 2)

    return None, None


def main():
    parser = argparse.ArgumentParser(description="Check stereo calibration quality with a checkerboard.")
    parser.add_argument("--chessboard", type=str, default="9,6",
                        help="Inner corners as 'cols,rows' (default: 9,6)")
    parser.add_argument("--square", type=float, default=26.5,
                        help="Square size in mm (default: 26.5)")
    parser.add_argument("--ip", type=str, default="172.31.1.77",
                        help="Camera IP address (default: 172.31.1.77)")
    args = parser.parse_args()

    chessboard_size = tuple(int(x) for x in args.chessboard.split(","))
    square_size_mm = args.square

    handler = Stereo4DCameraHandler(ip=args.ip, rectify_internally=True)
    handler.start(wait=True)

    # Wait for intrinsics
    print("Waiting for intrinsics...")
    while not handler.intrinsics_set:
        time.sleep(0.1)

    # Get baseline from extrinsic matrix (translation X component)
    baseline_m = abs(handler.left_camera_info.extrinsic_matrix[0, 3])
    print(f"Baseline: {baseline_m * 1000:.1f} mm")
    print(f"Chessboard: {chessboard_size[0]}x{chessboard_size[1]}, square: {square_size_mm} mm")
    print("Press 'q' to quit.\n")

    cv2.namedWindow("Calibration Check", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Calibration Check", 1920, 540)

    try:
        while True:
            frame = handler.get_last_frame()
            if frame is None:
                time.sleep(0.05)
                continue

            img = frame.image
            h, w = img.shape[:2]
            img_left = img[:, :w // 2].copy()
            img_right = img[:, w // 2:].copy()

            corners_l, corners_r = find_checkerboard(img_left, img_right, chessboard_size)

            if corners_l is not None:
                # Get rectified camera matrix
                K = handler.left_rect_k

                # Triangulate in meters, convert to mm
                pts_3d = points_3d_from_stereo(corners_l, corners_r, K, K, baseline_m) * 1000

                mean_err, max_err = check_grid_quality(pts_3d, chessboard_size, square_size_mm)
                mean_dist = np.mean(np.linalg.norm(pts_3d, axis=1))

                label = f"Mean err: {mean_err:.2f} mm | Max err: {max_err:.2f} mm | Distance: {mean_dist:.0f} mm"
                cv2.putText(img_left, label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            combined = np.hstack((img_left, img_right))
            cv2.imshow("Calibration Check", combined)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        handler.stop()
        print("Done.")


if __name__ == "__main__":
    main()
