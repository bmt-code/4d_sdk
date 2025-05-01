# 4D SDK

The **4D SDK** is a software development kit designed for the **4D Stereo Camera**. It provides tools and utilities to work with stereo camera data in a ROS2 environment.

## Installation

Follow these steps to install and set up the 4D SDK:

1. **Install ROS2 dependencies**:
    - Install `compressed_image_transport` and `image_transport` for your ROS2 version:

      ```bash
      sudo apt install ros-<ros2-distro>-compressed-image-transport ros-<ros2-distro>-image-transport
      ```

      Replace `<ros2-distro>` with your ROS2 distribution (e.g., `humble`, `foxy`, etc.).

2. **Install Python dependencies**:
    - Use `pip3` to install the required Python packages:

      ```bash
      pip3 install zmq numpy opencv-python
      ```

3. **Clone the repository**:
    - Clone the 4D SDK repository into your workspace:

      ```bash
      git clone <repository-url>
      ```

      Replace `<repository-url>` with the actual URL of the repository.

4. **Run the SDK**:
    - Navigate to the repository directory and run the `fourd_ros2.py` script:

      ```bash
      cd scripts
      python3 fourd_ros2.py
      ```

## Usage

The 4D SDK enables seamless integration with the 4D Stereo Camera, providing tools for image processing and data handling in ROS2.

For more details, refer to the documentation or contact the development team.
