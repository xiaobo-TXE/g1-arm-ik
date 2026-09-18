"""Unitree /lowstate -> sensor_msgs/JointState.

Two ways to run this:

1. **ROS 2 DDS (unitree_ros2, no bridge needed).** If you build unitree_ros2 in
   the same workspace, the robot's LowState is already visible as a ROS 2 topic
   (`rt/lowstate`). Set `source:=ros` and this node just re-labels the joints
   into a JointState with proper names.

2. **unitree_sdk2py (Python DDS).** Set `source:=dds` and this node opens its
   own DDS subscription. Useful when you do not want the whole unitree_ros2
   workspace.

    ros2 run g1_arm_bringup joint_state_relay --ros-args -p source:=dds \
        -p network_interface:=eth0

Joint naming: the 29-DOF G1 publishes 29 motor states in ascending motor-index
order, which maps 1:1 onto G1_29DOF_JOINT_NAMES from g1_arm_ik_node.ros_common.
The relay verifies the message length and refuses to publish a partial mapping,
because a shifted index would silently produce a wrong arm pose.
"""

from __future__ import annotations

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

from g1_arm_ik_core.ros_common import G1_29DOF_JOINT_NAMES


class JointStateRelay(Node):
    def __init__(self) -> None:
        super().__init__("g1_joint_state_relay")

        self.declare_parameter("source", "dds")  # dds | ros
        self.declare_parameter("input_topic", "/lowstate")
        self.declare_parameter("output_topic", "/g1/joint_states")
        self.declare_parameter("network_interface", "eth0")
        self.declare_parameter("domain_id", 0)
        self.declare_parameter("num_joints", 29)
        self.declare_parameter("qos_depth", 10)

        self.n = int(self.get_parameter("num_joints").value)
        if not 1 <= self.n <= len(G1_29DOF_JOINT_NAMES):
            raise ValueError(
                f"num_joints must be between 1 and {len(G1_29DOF_JOINT_NAMES)}, "
                f"got {self.n}"
            )
        if self.n != len(G1_29DOF_JOINT_NAMES):
            self.get_logger().warn(
                f"num_joints={self.n} but the G1 29-DOF layout has "
                f"{len(G1_29DOF_JOINT_NAMES)}. Publishing only the first "
                f"{self.n} joints -- the joint NAMES must still line up with the "
                "motor order in LowState, otherwise downstream consumers get a "
                "shifted, wrong pose."
            )
        # The mapping below is positional: LowState.motor_state[i] -> name[i].
        # It is only correct if LowState really publishes in ascending motor
        # index order for YOUR firmware. Verify with `ros2 topic echo
        # /g1/joint_states` and move one joint by hand.
        self.names = G1_29DOF_JOINT_NAMES[: self.n]

        self.pub = self.create_publisher(
            JointState, self.get_parameter("output_topic").value, 10
        )

        source = self.get_parameter("source").value
        if source == "ros":
            self._setup_ros_input()
        elif source == "dds":
            self._setup_dds_input()
        else:
            raise ValueError(f"source must be 'ros' or 'dds', got {source!r}")

        self.get_logger().info(
            f"g1_joint_state_relay: source={source}, publishing "
            f"{self.get_parameter('output_topic').value} with {self.n} joints"
        )

    # -- input: a ROS 2 topic carrying the Unitree LowState -------------------

    def _setup_ros_input(self) -> None:
        try:
            from unitree_go.msg import LowState as LowStateGo
        except ImportError:
            LowStateGo = None
        try:
            from unitree_hg.msg import LowState as LowStateHg
        except ImportError:
            LowStateHg = None

        if LowStateHg is None and LowStateGo is None:
            raise ImportError(
                "source:=ros needs the Unitree message package in this workspace. "
                "Build unitreerobotics/unitree_ros2 into ros2_ws/src, or use "
                "source:=dds which only needs unitree_sdk2py."
            )

        topic = self.get_parameter("input_topic").value
        msg_type = LowStateHg if LowStateHg is not None else LowStateGo
        which = "unitree_hg/LowState" if LowStateHg is not None else "unitree_go/LowState"
        self.get_logger().info(f"subscribing {topic} ({which})")
        self.create_subscription(msg_type, topic, self._on_lowstate_ros, 10)

    def _on_lowstate_ros(self, msg) -> None:
        try:
            motor_state = msg.motor_state
        except AttributeError:
            return
        if len(motor_state) < self.n:
            self.get_logger().warn(
                f"LowState has {len(motor_state)} motors, expected >= {self.n}",
                throttle_duration_sec=5.0,
            )
            return
        self._publish([float(motor_state[i].q) for i in range(self.n)])

    # -- input: own DDS subscription via unitree_sdk2py -----------------------

    def _setup_dds_input(self) -> None:
        try:
            from unitree_sdk2py.core.channel import (
                ChannelFactoryInitialize,
                ChannelSubscriber,
            )
            from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
        except ImportError as exc:
            raise ImportError(
                "source:=dds requires unitree_sdk2py:\n"
                "  pip install unitree-sdk2py\n"
                "or build the unitree_sdk2_python repository."
            ) from exc

        iface = self.get_parameter("network_interface").value
        domain = int(self.get_parameter("domain_id").value)
        self.get_logger().info(f"DDS init on iface={iface}, domain={domain}")
        ChannelFactoryInitialize(domain, iface)

        self._sub = ChannelSubscriber("rt/lowstate", LowState_)
        self._sub.Init(self._on_lowstate_dds, 10)

    def _on_lowstate_dds(self, msg) -> None:
        try:
            motor_state = msg.motor_state
        except AttributeError:
            return
        if len(motor_state) < self.n:
            return
        self._publish([float(motor_state[i].q) for i in range(self.n)])

    # -- output ---------------------------------------------------------------

    def _publish(self, q: list) -> None:
        out = JointState()
        out.header.stamp = self.get_clock().now().to_msg()
        # frame_id is intentionally empty: a JointState does not live in a frame,
        # and setting one invites wrong assumptions downstream.
        out.name = list(self.names)
        out.position = q
        self.pub.publish(out)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = JointStateRelay()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
