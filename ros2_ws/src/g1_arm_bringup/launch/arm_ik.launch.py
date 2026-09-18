"""Bring up the IK and control nodes only. No hardware interface, no GUI.

Use this when the robot interface (bridge / DDS) is already running, or to
exercise the stack against the mock backend.

    ros2 launch g1_arm_bringup arm_ik.launch.py
    ros2 launch g1_arm_bringup arm_ik.launch.py side:=right backend:=mock
    ros2 launch g1_arm_bringup arm_ik.launch.py backend:=dds \\
        dds_network_interface:=eth0 side:=left
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _launch_setup(context, *args, **kwargs):
    bringup_share = get_package_share_directory("g1_arm_bringup")

    side = LaunchConfiguration("side").perform(context)
    backend = LaunchConfiguration("backend").perform(context)
    urdf = LaunchConfiguration("urdf_path").perform(context)
    dds_iface = LaunchConfiguration("dds_network_interface").perform(context)
    start_enabled = LaunchConfiguration("start_enabled").perform(context).lower() == "true"

    ik_params = os.path.join(bringup_share, "config", "ik_params.yaml")
    ctrl_params = os.path.join(bringup_share, "config", "control_params.yaml")

    # One URDF path for every node, so the IK and the control node can never
    # disagree about geometry.
    urdf_override = {}
    if urdf:
        urdf_override = {
            "urdf_path": urdf,
            "joint_state_topic": LaunchConfiguration("joint_state_topic").perform(context),
        }

    ik_node = Node(
        package="g1_arm_ik_node",
        executable="ik_node",
        name="g1_arm_ik_node",
        output="screen",
        parameters=[ik_params, {"side": side}, urdf_override],
    )

    ctrl_node = Node(
        package="g1_arm_control_node",
        executable="control_node",
        name="g1_arm_control_node",
        output="screen",
        parameters=[
            ctrl_params,
            {
                "side": side,
                "backend": backend,
                "start_enabled": start_enabled,
                "dds.network_interface": dds_iface,
            },
            urdf_override,
        ],
    )

    return [ik_node, ctrl_node]


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            DeclareLaunchArgument("side", default_value="left", choices=["left", "right"]),
            DeclareLaunchArgument(
                "backend",
                default_value="mock",
                choices=["mock", "topic", "dds"],
                description="how commands reach the robot",
            ),
            DeclareLaunchArgument("urdf_path", default_value=""),
            DeclareLaunchArgument("joint_state_topic", default_value="/g1/joint_states"),
            DeclareLaunchArgument("dds_network_interface", default_value="eth0"),
            DeclareLaunchArgument("start_enabled", default_value="false"),
            LogInfo(
                msg=[
                    "g1_arm_ik: the arm starts DISABLED. Enable it with:\n",
                    "  ros2 service call /g1/arm_control/set_enabled ",
                    "g1_arm_msgs/srv/SetArmEnabled \"{data: true}\"",
                ]
            ),
            OpaqueFunction(function=_launch_setup),
        ]
    )
