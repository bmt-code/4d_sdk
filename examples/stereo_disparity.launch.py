"""
Launch file for stereo disparity pipeline.

Starts stereo_disparity_node which computes disparity + point cloud and
optionally shows a built-in preview window with live SGBM trackbars.

Usage:
    # disparity node only (no preview window)
    ros2 launch src/4d_sdk/examples/stereo_disparity.launch.py

    # with built-in preview + trackbar tuner
    ros2 launch src/4d_sdk/examples/stereo_disparity.launch.py preview:=true

Preview keys: s=save params, r=reset defaults, q=quit
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration

SDK_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARAMS_YAML = os.path.join(SDK_DIR, "config", "sgbm_params.yaml")
NODE_SCRIPT = os.path.join(SDK_DIR, "examples", "stereo_disparity_node.py")


def generate_launch_description():
    preview_arg = DeclareLaunchArgument(
        "preview",
        default_value="false",
        description="Enable built-in preview window with SGBM trackbars",
    )

    disparity_node = ExecuteProcess(
        cmd=[
            "python3", "-u", NODE_SCRIPT,
            "--ros-args",
            "--params-file", PARAMS_YAML,
            "-p", ["show_preview:=", LaunchConfiguration("preview")],
        ],
        output="screen",
        emulate_tty=True,
        name="stereo_disparity_node",
    )

    return LaunchDescription([
        preview_arg,
        disparity_node,
    ])
