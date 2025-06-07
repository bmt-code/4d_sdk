import numpy as np
import yaml
import os


def parse_stereo_calib(file_path):
    with open(file_path, "r") as file:
        data = file.read()

    sections = data.split("[image]")
    if len(sections) < 2:
        raise ValueError("Missing [image] section in the input file.")

    left_section = sections[1].split("[narrow_stereo/left]")
    if len(left_section) < 2:
        raise ValueError("Missing [narrow_stereo/left] section in the input file.")

    left_data = left_section[1].split("[narrow_stereo/right]")[0]

    if len(sections) < 3:
        raise ValueError("Missing the right image section in the input file.")

    right_sections = sections[2].split("[narrow_stereo/right]")
    if len(right_sections) < 2:
        raise ValueError("Missing [narrow_stereo/right] section in the input file.")

    right_data = right_sections[1]

    def parse_matrix(matrix_str):
        lines = matrix_str.strip().split("\n")
        valid_lines = [line for line in lines if not line.strip().startswith("#")]
        return np.array([list(map(float, row.split())) for row in valid_lines])

    def parse_distortion(distortion_str):
        lines = distortion_str.strip().split("\n")
        valid_lines = [line for line in lines if not line.strip().startswith("#")]
        return list(map(float, " ".join(valid_lines).split()))

    def parse_self_params(data, param_name):
        for line in data.splitlines():
            if line.startswith(f"self.{param_name}"):
                values = line.split("=")[1].strip().strip("[]").split(",")
                return list(map(float, values))
        return None

    left = {
        "camera_matrix": parse_matrix(
            left_data.split("camera matrix")[1].split("distortion")[0]
        ),
        "distortion": parse_distortion(
            left_data.split("distortion")[1].split("rectification")[0]
        ),
        "rectification": parse_matrix(
            left_data.split("rectification")[1].split("projection")[0]
        ),
        "projection": parse_matrix(left_data.split("projection")[1]),
    }

    right = {
        "camera_matrix": parse_matrix(
            right_data.split("camera matrix")[1].split("distortion")[0]
        ),
        "distortion": parse_distortion(
            right_data.split("distortion")[1].split("rectification")[0]
        ),
        "rectification": parse_matrix(
            right_data.split("rectification")[1].split("projection")[0]
        ),
        "projection": parse_matrix(right_data.split("projection")[1]),
    }

    self_T = np.array(parse_self_params(data, "T")).reshape(3, 1)
    self_R = np.array(parse_self_params(data, "R")).reshape(3, 3)

    return left, right, self_T, self_R


def transform_to_calibration(left, right, self_T, self_R):
    calibration_data = {
        "R": self_R.tolist(),
        "T": self_T.tolist(),
        "distL": left["distortion"],
        "distR": right["distortion"],
        "mtxL": left["camera_matrix"].tolist(),
        "mtxR": right["camera_matrix"].tolist(),
        "rectL": left["rectification"].tolist(),
        "rectR": right["rectification"].tolist(),
    }
    return calibration_data


def save_calibration(file_path, calibration_data):
    with open(file_path, "w") as file:
        yaml.dump(calibration_data, file)


if __name__ == "__main__":
    this_file_path = os.path.abspath(__file__)
    this_dir = os.path.dirname(this_file_path)
    input_path = "stereo_calib.txt"
    output_path = "stereo_calibration.yaml"
    input_path = os.path.join(this_dir, input_path)
    output_path = os.path.join(this_dir, output_path)

    print("Parsing stereo calibration data...")
    left, right, self_T, self_R = parse_stereo_calib(input_path)
    print("Transforming to calibration format...")
    calibration_data = transform_to_calibration(left, right, self_T, self_R)
    print("Saving calibration data to YAML file...")
    save_calibration(output_path, calibration_data)
    print(f"Calibration data saved to {output_path}. Please check the file for details.")
