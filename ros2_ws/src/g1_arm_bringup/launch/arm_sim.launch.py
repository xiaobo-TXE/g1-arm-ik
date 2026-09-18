"""Full simulation bring-up: IK + control on the mock backend. No hardware, no GUI.

This is the launch file to use for checking that the whole pipeline works and for
your own integration tests, before anything is connected to the robot.

    ros2 launch g1_arm_bringup arm_sim.launch.py
    ros2 launch g1_arm_bringup arm_sim.launch.py side:=right

Published interfaces (all under /g1):
    g1/arm_ik/joint_target      std_msgs/Float64MultiArray    (IK output)
    g1/arm/command              std_msgs/Float64MultiArray    (shaped command)
    g1/arm_control/status       g1_arm_msgs/ArmIKStatus
    g1/arm_ik/status            g1_arm_msgs/ArmIKStatus
    g1/arm_ik/torso_target_pose geometry_msgs/PoseStamped    (subscribe to command)
    tf                          pelvis -> full URDF tree
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _launch_setup(context, *args, **kwargs):
    share = get_package_share_directory("g1_arm_bringup")

    side = LaunchConfiguration("side").perform(context)

    ik_params = os.path.join(share, "config", "ik_params.yaml")
    ctrl_params = os.path.join(share, "config", "control_params.yaml")

    return [
        Node(
            package="g1_arm_ik_node",
            executable="ik_node",
            name="g1_arm_ik_node",
            output="screen",
            parameters=[ik_params, {"side": side}],
        ),
        Node(
            package="g1_arm_control_node",
            executable="control_node",
            name="g1_arm_control_node",
            output="screen",
            parameters=[
                ctrl_params,
                {"side": side, "backend": "mock", "start_enabled": False},
            ],
        ),
    ]


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            DeclareLaunchArgument("side", default_value="left", choices=["left", "right"]),
            LogInfo(
                msg=[
                    "g1_arm_sim: MOCK backend, arm starts disabled.\n",
                    "Enable:  ros2 service call /g1/arm_control/set_enabled ",
                    "g1_arm_msgs/srv/SetArmEnabled \"{data: true}\"\n",
                    "Move:    ros2 service call /g1/arm_ik_node/torso_target ",
                    "g1_arm_msgs/srv/SolveIK \"{x: 0.05, y: 0.15, z: -0.10}\"\n",
                    "Watch:   ros2 topic echo /g1/arm_control/status",
                ]
            ),
            OpaqueFunction(function=_launch_setup),
        ]
    )
