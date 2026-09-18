"""Real-robot bring-up: /lowstate bridge + IK + control.

THE DEFAULT IS A DRY RUN. `arm_dds_dry_run:=true` builds and validates the whole
stack, publishes everything, but never writes to DDS. Work through this sequence
before allowing real output:

  1. ros2 launch g1_arm_bringup arm_bringup.launch.py
     -> check /g1/joint_states is arriving and the values look like the real pose

  2. ... backend:=topic
     -> the control node publishes /g1/arm/command; verify it by echo, with the
        robot's own arm control still in charge

  3. ... backend:=dds arm_dds_dry_run:=true
     -> the DDS path is constructed and every send is a no-op

  4. ... backend:=dds arm_dds_dry_run:=false
     -> LIVE. Keep velocity_scale low, keep a hand on the e-stop, and have the
        robot supported (hung or lying) for the first moves.

Example:
    ros2 launch g1_arm_bringup arm_bringup.launch.py \\
        side:=left backend:=dds dds_network_interface:=eth0
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
    backend = LaunchConfiguration("backend").perform(context)
    dds_iface = LaunchConfiguration("dds_network_interface").perform(context)
    dds_dry = LaunchConfiguration("arm_dds_dry_run").perform(context).lower() == "true"
    relay_source = LaunchConfiguration("relay_source").perform(context)
    relay_iface = LaunchConfiguration("relay_network_interface").perform(context)
    start_tf = LaunchConfiguration("start_tf").perform(context).lower() == "true"

    ik_params = os.path.join(share, "config", "ik_params.yaml")
    ctrl_params = os.path.join(share, "config", "control_params.yaml")

    nodes = [
        Node(
            package="g1_arm_bringup",
            executable="joint_state_relay",
            name="g1_joint_state_relay",
            output="screen",
            parameters=[
                {
                    "source": relay_source,
                    "network_interface": relay_iface,
                    "output_topic": "/g1/joint_states",
                }
            ],
        ),
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
                {
                    "side": side,
                    "backend": backend,
                    "start_enabled": False,
                    "dds.network_interface": dds_iface,
                    "dds.dry_run": dds_dry,
                },
            ],
        ),
    ]

    if start_tf:
        nodes.append(
            Node(
                package="g1_arm_bringup",
                executable="tf_broadcaster",
                name="g1_tf_broadcaster",
                output="screen",
                parameters=[{"joint_state_topic": "/g1/joint_states"}],
            )
        )

    return nodes


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            DeclareLaunchArgument("side", default_value="left", choices=["left", "right"]),
            DeclareLaunchArgument(
                "backend", default_value="topic", choices=["mock", "topic", "dds"]
            ),
            DeclareLaunchArgument("dds_network_interface", default_value="eth0"),
            DeclareLaunchArgument(
                "arm_dds_dry_run",
                default_value="true",
                description="true = never write to DDS (safe). Set false only when ready.",
            ),
            DeclareLaunchArgument(
                "relay_source",
                default_value="dds",
                choices=["dds", "ros"],
                description="dds = unitree_sdk2py; ros = unitree_ros2 LowState topic",
            ),
            DeclareLaunchArgument("relay_network_interface", default_value="eth0"),
            DeclareLaunchArgument("start_tf", default_value="false"),
            LogInfo(
                msg=[
                    "g1_arm_bringup: arm starts DISABLED, backend=",
                    LaunchConfiguration("backend"),
                    ", dds_dry_run=",
                    LaunchConfiguration("arm_dds_dry_run"),
                ]
            ),
            OpaqueFunction(function=_launch_setup),
        ]
    )
