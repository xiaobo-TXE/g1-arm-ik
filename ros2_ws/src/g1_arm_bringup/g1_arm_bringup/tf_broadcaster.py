"""Publish the full G1 TF tree from the URDF, driven by /g1/joint_states.

Why this exists instead of `robot_state_publisher`
--------------------------------------------------
We already have a verified forward-kinematics implementation, and the official
G1 URDF uses *relative* mesh paths (`filename="meshes/x.STL"`) which
`robot_state_publisher` resolves against the URDF's own directory. Using our own
FK keeps the whole stack consistent with the IK (same parser, same joint order)
and needs no mesh packaging.

This node is OPTIONAL -- nothing in the control path depends on TF. It is here
because a correct TF tree is the quickest way to check that the joint state you
are feeding the IK is actually right, and because downstream tools (recording,
offline analysis, any external viewer you may already use) expect it.

TF frames: `root_frame` -> `base_frame` -> (URDF tree from pelvis).
"""

from __future__ import annotations

import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from sensor_msgs.msg import JointState
from tf2_ros import TransformBroadcaster

from g1_ik.fk import NumpyChainFK

from g1_arm_ik_core.ros_common import find_robot_description


class TFBroadcasterNode(Node):
    def __init__(self) -> None:
        super().__init__("g1_tf_broadcaster")

        self.declare_parameter("urdf_path", "")
        self.declare_parameter("joint_state_topic", "/g1/joint_states")
        self.declare_parameter("root_frame", "odom")
        self.declare_parameter("base_frame", "pelvis")
        self.declare_parameter("publish_rate", 50.0)
        self.declare_parameter("require_complete_state", True)

        urdf = find_robot_description(self.get_parameter("urdf_path").value)
        self.model = NumpyChainFK(urdf, "pelvis")  # tip is unused; we want the whole model
        self.root_frame = self.get_parameter("root_frame").value
        self.base_frame = self.get_parameter("base_frame").value
        if self.root_frame == self.base_frame:
            raise ValueError(
                f"root_frame and base_frame are both {self.root_frame!r}; that "
                "would publish a pelvis->pelvis identity transform"
            )

        # Every actuated joint we must have seen before the tree is trustworthy.
        # A partial joint state (say, only the arm) would otherwise publish a
        # tree with the un-seen joints silently held at 0 -- a wrong TF that
        # looks perfectly valid.
        self._expected = sorted(
            n
            for n in self.model.model.joint_order
            if self.model.model.joints[n].is_actuated
        )
        self._require_complete = bool(
            self.get_parameter("require_complete_state").value
        )
        self.get_logger().info(
            f"g1_tf_broadcaster: {len(self.model.model.joints)} joints "
            f"({len(self._expected)} actuated), "
            f"{self.root_frame} -> {self.base_frame} -> URDF tree"
        )

        self._q: dict = {}
        self._published_once = False
        self.create_subscription(
            JointState, self.get_parameter("joint_state_topic").value,
            self._on_joint_state, 10,
        )
        self.tf = TransformBroadcaster(self)
        self.create_timer(
            1.0 / float(self.get_parameter("publish_rate").value), self._publish
        )

    def _on_joint_state(self, msg: JointState) -> None:
        for name, pos in zip(msg.name, msg.position):
            if name in self.model.model.joints:
                self._q[name] = float(pos)

    def _publish(self) -> None:
        if not self._q:
            return
        missing = [j for j in self._expected if j not in self._q]
        if missing and not self._published_once:
            if self._require_complete:
                self.get_logger().warn(
                    f"waiting for a complete joint state: {len(missing)} joints "
                    f"never seen (e.g. {missing[:4]}). TF not published yet. Set "
                    "require_complete_state:=false to publish anyway.",
                    throttle_duration_sec=5.0,
                )
                return
            self.get_logger().warn(
                f"publishing TF with {len(missing)} joints held at 0 "
                f"(e.g. {missing[:4]})",
                throttle_duration_sec=5.0,
            )
        self._published_once = True
        try:
            poses = self.model.fk_all_links(self._q)
        except Exception as exc:  # noqa: BLE001 - never kill the broadcaster
            self.get_logger().warn(f"FK failed: {exc}", throttle_duration_sec=5.0)
            return

        now = self.get_clock().now().to_msg()
        stamp = TransformStamped()
        stamp.header.stamp = now
        stamp.header.frame_id = self.root_frame
        stamp.child_frame_id = self.base_frame
        stamp.transform.rotation.w = 1.0
        self.tf.sendTransform(stamp)

        for child, T in poses.items():
            if child == self.base_frame:
                continue
            t = TransformStamped()
            t.header.stamp = now
            t.header.frame_id = self.base_frame
            t.child_frame_id = child
            t.transform.translation.x = float(T[0, 3])
            t.transform.translation.y = float(T[1, 3])
            t.transform.translation.z = float(T[2, 3])
            qx, qy, qz, qw = _matrix_to_quat(T[:3, :3])
            t.transform.rotation.x = qx
            t.transform.rotation.y = qy
            t.transform.rotation.z = qz
            t.transform.rotation.w = qw
            self.tf.sendTransform(t)


def _matrix_to_quat(R: np.ndarray):
    """Rotation matrix -> quaternion (Shepperd's method, numerically stable)."""
    m00, m01, m02 = R[0]
    m10, m11, m12 = R[1]
    m20, m21, m22 = R[2]
    tr = m00 + m11 + m22
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * s
        qx = (m21 - m12) / s
        qy = (m02 - m20) / s
        qz = (m10 - m01) / s
    elif m00 > m11 and m00 > m22:
        s = np.sqrt(1.0 + m00 - m11 - m22) * 2.0
        qw = (m21 - m12) / s
        qx = 0.25 * s
        qy = (m01 + m10) / s
        qz = (m02 + m20) / s
    elif m11 > m22:
        s = np.sqrt(1.0 + m11 - m00 - m22) * 2.0
        qw = (m02 - m20) / s
        qx = (m01 + m10) / s
        qy = 0.25 * s
        qz = (m12 + m21) / s
    else:
        s = np.sqrt(1.0 + m22 - m00 - m11) * 2.0
        qw = (m10 - m01) / s
        qx = (m02 + m20) / s
        qy = (m12 + m21) / s
        qz = 0.25 * s
    return float(qx), float(qy), float(qz), float(qw)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TFBroadcasterNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
