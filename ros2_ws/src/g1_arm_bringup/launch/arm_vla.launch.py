"""IK -> ZMQ 6002 action frames for the Groot VLA path.

Starts SILENT (zmq.enabled=false, zmq.dry_run=true). Two deliberate gates, so
nothing reaches the robot until you ask for it.

    # dry run: solve and log, transmit nothing
    ros2 launch g1_arm_bringup arm_vla.launch.py

    # go live on the real robot
    ros2 launch g1_arm_bringup arm_vla.launch.py \\
        zmq_enabled:=true zmq_dry_run:=false zmq_robot_ip:=192.168.123.222

Interfaces:
    <node>/torso_target         g1_arm_msgs/SolveIK    target in the TORSO frame
    <node>/pelvis_target        g1_arm_msgs/SolveIK    target in the PELVIS frame
    <node>/torso_target_pose    geometry_msgs/PoseStamped  streaming target
    <node>/set_enabled          g1_arm_msgs/SetArmEnabled  hand port 6002 back
    <node>/status               g1_arm_msgs/ArmIKStatus
    <node>/frame_preview        std_msgs/String        exact bytes we would send
    /g1/vla/velocity            geometry_msgs/Twist     pass-through velocity

IMPORTANT: this node OWNS port 6002. Do not run a second producer against it --
two PUSH peers interleave into one PULL socket and the receiver drops any frame
that does not carry all 14 arm joints.
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
    robot_ip = LaunchConfiguration("zmq_robot_ip").perform(context)
    port = int(LaunchConfiguration("zmq_port").perform(context))
    enabled = LaunchConfiguration("zmq_enabled").perform(context).lower() == "true"
    dry_run = LaunchConfiguration("zmq_dry_run").perform(context).lower() == "true"

    return [
        Node(
            package="g1_arm_ik_node",
            executable="vla_node",
            name="g1_arm_vla_node",
            output="screen",
            parameters=[
                os.path.join(share, "config", "vla_params.yaml"),
                {
                    "side": side,
                    "zmq.robot_ip": robot_ip,
                    "zmq.port": port,
                    "zmq.enabled": enabled,
                    "zmq.dry_run": dry_run,
                },
            ],
        )
    ]


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            DeclareLaunchArgument("side", default_value="left", choices=["left", "right"]),
            DeclareLaunchArgument("zmq_robot_ip", default_value="192.168.123.222"),
            DeclareLaunchArgument("zmq_port", default_value="6002"),
            DeclareLaunchArgument(
                "zmq_enabled",
                default_value="false",
                description="true = the set_enabled state starts engaged",
            ),
            DeclareLaunchArgument(
                "zmq_dry_run",
                default_value="true",
                description="true = validate and log every frame but transmit none",
            ),
            LogInfo(
                msg=[
                    "g1_arm_vla: target -> IK -> ZMQ action frames on port ",
                    LaunchConfiguration("zmq_port"),
                    "\n",
                    "  starts SILENT (enabled=", LaunchConfiguration("zmq_enabled"),
                    ", dry_run=", LaunchConfiguration("zmq_dry_run"), ")\n",
                    "  watch the frame without arming anything:\n",
                    "    ros2 topic echo /g1_arm_vla_node/frame_preview\n",
                    "  enable when ready:\n",
                    "    ros2 service call /g1_arm_vla_node/set_enabled ",
                    "g1_arm_msgs/srv/SetArmEnabled \"{data: true}\"\n",
                    "  NOTE: this node owns port 6002; do not run another producer.",
                ]
            ),
            OpaqueFunction(function=_launch_setup),
        ]
    )
