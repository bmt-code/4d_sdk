import concurrent.futures
import glob
import logging
import os

import cv2
import numpy as np
import yaml
from tqdm import tqdm


class StereoCalibration:
    def __init__(self, camera_name, imgs_path, chessboard_size, square_size):
        self.camera_name = camera_name
        self.imgs_path = imgs_path
        self.chessboard_size = chessboard_size
        self.square_size = square_size
        self.objpoints = []  # 3D points in real world
        self.imgpoints_left = []  # 2D points in left image
        self.imgpoints_right = []  # 2D points in right image
        self.nested_images_paths = sorted(glob.glob(self.imgs_path))
        self.calibration_file = os.path.join(
            self.camera_name, "stereo_calibration.yaml"
        )

        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )

    def check_existing_calibration(self):
        if os.path.exists(self.calibration_file):
            logging.info(f"Calibration file already exists: {self.calibration_file}")
            redo = (
                input(
                    "Calibration file already exists. Do you want to redo the calibration? (y/n): "
                )
                .strip()
                .lower()
            )
            if redo != "y":
                logging.info("Loading existing calibration data...")
                with open(self.calibration_file, "r") as f:
                    calibration_data = yaml.safe_load(f)
                self.generate_epipolar_lines(calibration_data)
                exit(0)

    def generate_epipolar_lines(self, calibration_data):
        logging.info("Generating epipolar lines using existing calibration data...")

        # Load calibration parameters
        mtxL = np.array(calibration_data["mtxL"])
        distL = np.array(calibration_data["distL"])
        mtxR = np.array(calibration_data["mtxR"])
        distR = np.array(calibration_data["distR"])
        R = np.array(calibration_data["R"])
        T = np.array(calibration_data["T"])

        # Load a sample image pair
        nested_img = cv2.imread(self.nested_images_paths[len(self.nested_images_paths) // 2])
        imgL = nested_img[:, : nested_img.shape[1] // 2]
        imgR = nested_img[:, nested_img.shape[1] // 2 :]

        # Undistort and rectify images
        h, w = imgL.shape[:2]
        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
            mtxL, distL, mtxR, distR, (w, h), R, T, alpha=0
        )
        mapLx, mapLy = cv2.initUndistortRectifyMap(
            mtxL, distL, R1, P1, (w, h), cv2.CV_32FC1
        )
        mapRx, mapRy = cv2.initUndistortRectifyMap(
            mtxR, distR, R2, P2, (w, h), cv2.CV_32FC1
        )
        imgL_rectified = cv2.remap(imgL, mapLx, mapLy, cv2.INTER_LINEAR)
        imgR_rectified = cv2.remap(imgR, mapRx, mapRy, cv2.INTER_LINEAR)

        # Draw epipolar lines with alternating colors
        colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0)]  # Green, Red, Blue
        for i, y in enumerate(range(0, h, 20)):  # Draw lines every 20 pixels
            color = colors[i % len(colors)]  # Alternate colors
            cv2.line(imgL_rectified, (0, y), (w, y), color, 1)
            cv2.line(imgR_rectified, (0, y), (w, y), color, 1)

        # show the not rectified images
        not_rectified = np.hstack((imgL, imgR))
        for i in range(0, not_rectified.shape[0], 20):
            color = colors[i % len(colors)]
            cv2.line(not_rectified, (0, i), (not_rectified.shape[1], i), color, 1)

        cv2.namedWindow("Not Rectified Images", cv2.WINDOW_NORMAL)
        cv2.imshow("Not Rectified Images", not_rectified)

        # Concatenate and display the images
        combined = np.hstack((imgL_rectified, imgR_rectified))
        cv2.namedWindow("Epipolar Lines", cv2.WINDOW_NORMAL)
        cv2.imshow("Epipolar Lines", combined)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        logging.info("Epipolar lines generated and displayed.")

    def prepare_object_points(self):
        objp = np.zeros((np.prod(self.chessboard_size), 3), np.float32)
        objp[:, :2] = np.indices(self.chessboard_size).T.reshape(-1, 2)
        objp *= self.square_size
        return objp

    def process_image(self, nested_img_path, objp):
        nested_img = cv2.imread(nested_img_path)
        imgL = nested_img[:, : nested_img.shape[1] // 2]
        imgR = nested_img[:, nested_img.shape[1] // 2 :]
        imgL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
        imgR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

        retL, cornersL = cv2.findChessboardCorners(
            imgL,
            self.chessboard_size,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE,
        )
        retR, cornersR = cv2.findChessboardCorners(
            imgR,
            self.chessboard_size,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE,
        )

        if retL and retR:
            logging.debug(f"Chessboard corners found in image: {nested_img_path}")
            return objp, cornersL, cornersR
        return None

    def detect_corners(self, objp):
        if not self.nested_images_paths:
            logging.error("No images found in the specified path. Exiting.")
            exit(1)

        logging.info("Starting chessboard corner detection...")
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(self.process_image, path, objp)
                for path in self.nested_images_paths
            ]
            for future in tqdm(
                concurrent.futures.as_completed(futures),
                desc="Processing images",
                total=len(self.nested_images_paths),
            ):
                result = future.result()
                if result:
                    objp, cornersL, cornersR = result
                    self.objpoints.append(objp)
                    self.imgpoints_left.append(cornersL)
                    self.imgpoints_right.append(cornersR)

        logging.info(
            f"Found corners in {len(self.objpoints)} out of {len(self.nested_images_paths)} images."
        )

    def calibrate_cameras(self, image_shape):
        logging.info("Calibrating individual cameras...")
        retL, mtxL, distL, _, _ = cv2.calibrateCamera(
            self.objpoints, self.imgpoints_left, image_shape[:2][::-1], None, None
        )
        retR, mtxR, distR, _, _ = cv2.calibrateCamera(
            self.objpoints, self.imgpoints_right, image_shape[:2][::-1], None, None
        )
        logging.info(f"Left camera calibration complete with RMS error: {retL} pixels.")
        logging.info(
            f"Right camera calibration complete with RMS error: {retR} pixels."
        )
        return mtxL, distL, mtxR, distR

    def stereo_calibrate(self, mtxL, distL, mtxR, distR, image_shape):
        logging.info("Performing stereo calibration...")
        retS, mtxL, distL, mtxR, distR, R, T, E, F = cv2.stereoCalibrate(
            self.objpoints,
            self.imgpoints_left,
            self.imgpoints_right,
            mtxL,
            distL,
            mtxR,
            distR,
            image_shape[:2][::-1],
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 1000, 1e-5),
            flags=cv2.CALIB_SAME_FOCAL_LENGTH,
        )
        logging.info(f"Stereo calibration complete with RMS error: {retS} pixels.")
        logging.info(f"Calibration matrices:\nmtxL: {mtxL}\ndistL: {distL}\nmtxR: {mtxR}\ndistR: {distR}")
        logging.info(f"Rotation matrix R:\n{R}\nTranslation vector T:\n{T}")
        return mtxL, distL, mtxR, distR, R, T, E, F

    def save_calibration(self, mtxL, distL, mtxR, distR, R, T, E, F):
        calibration_data = {
            "mtxL": mtxL.tolist(),
            "distL": distL.tolist(),
            "mtxR": mtxR.tolist(),
            "distR": distR.tolist(),
            "R": R.tolist(),
            "T": T.tolist(),
            "E": E.tolist(),
            "F": F.tolist(),
        }

        os.makedirs(self.camera_name, exist_ok=True)
        with open(self.calibration_file, "w") as f:
            yaml.dump(calibration_data, f, default_flow_style=False)
        logging.info(
            f"Stereo calibration complete. Results saved to {self.calibration_file}"
        )

    def run(self):
        self.check_existing_calibration()
        objp = self.prepare_object_points()
        self.detect_corners(objp)

        image_shape = cv2.imread(self.nested_images_paths[0]).shape
        # divide the width by 2 to get the left or right image shape
        image_shape = (image_shape[0], image_shape[1] // 2, image_shape[2])
        logging.info(f"Image shape detected: {image_shape}")

        mtxL, distL, mtxR, distR = self.calibrate_cameras(image_shape)
        mtxL, distL, mtxR, distR, R, T, E, F = self.stereo_calibrate(mtxL, distL, mtxR, distR, image_shape)
        self.save_calibration(mtxL, distL, mtxR, distR, R, T, E, F)


if __name__ == "__main__":
    camera_name = "stereo_camera_vXX"
    imgs_path = "/path/to/camXX/*.png"
    chessboard_size = (9, 6)
    square_size = 0.0262  # in m or your unit

    calibration = StereoCalibration(
        camera_name, imgs_path, chessboard_size, square_size
    )
    logging.info("--- Welcome to Stereo Calibration ---")
    logging.info(f"Camera name: {camera_name}")
    logging.info(f"Images path: {imgs_path}")
    logging.info(f"Chessboard size: {chessboard_size}")
    logging.info(f"Square size: {square_size}")

    confirm = input("Are these settings correct? (y/n): ").strip().lower()
    if confirm != "y":
        logging.info("Settings not confirmed. Exiting.")
        exit(0)

    calibration.run()

    with open(calibration.calibration_file, "r") as f:
        calibration_data = yaml.safe_load(f)
    calibration.generate_epipolar_lines(calibration_data)
    logging.info("Stereo calibration process completed successfully.")
    cv2.destroyAllWindows()