"""
Launch file for stereo disparity pipeline.

Starts:
  1. stereo_disparity_node  — computes disparity + point cloud, loads sgbm_params.yaml
  2. stereo_tuner           — live trackbar GUI (optional, set TUNER=true)

Usage:
    # disparity node only
    ros2 launch src/4d_sdk/examples/stereo_disparity.launch.py

    # with live tuner GUI
    ros2 launch src/4d_sdk/examples/stereo_disparity.launch.py tuner:=true
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression

SDK_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARAMS_YAML = os.path.join(SDK_DIR, "config", "sgbm_params.yaml")
NODE_SCRIPT = os.path.join(SDK_DIR, "examples", "stereo_disparity_node.py")
TUNER_SCRIPT = os.path.join(SDK_DIR, "calib", "stereo_tuner.py")


def generate_launch_description():
    tuner_arg = DeclareLaunchArgument(
        "tuner",
        default_value="false",
        description="Launch live SGBM tuner GUI alongside the disparity node",
    )

    disparity_node = ExecuteProcess(
        cmd=[
            "python3", "-u", NODE_SCRIPT,
            "--ros-args",
            "--params-file", PARAMS_YAML,
        ],
        output="screen",
        emulate_tty=True,
        name="stereo_disparity_node",
    )

    tuner_node = ExecuteProcess(
        cmd=["python3", "-u", TUNER_SCRIPT],
        output="screen",
        emulate_tty=True,
        name="stereo_tuner",
        condition=IfCondition(LaunchConfiguration("tuner")),
    )

    return LaunchDescription([
        tuner_arg,
        disparity_node,
        tuner_node,
    ])
