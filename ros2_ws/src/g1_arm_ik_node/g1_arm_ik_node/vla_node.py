"""g1_arm_vla_node -- IK solutions published as ZMQ VLA action frames.

This node is the last link in the chain the user asked for:

    end-effector target  ->  inverse kinematics  ->  14 arm joint targets
                         ->  ZMQ 6002 YAML action frame  ->  Groot deploy (C++)

It does NOT touch legs. In VLA mode the lower body is driven by the on-board
Groot ONNX policy; `State_Groot::publish_targets()` merges the leg targets from
the policy with the arm targets from this frame into one LowCmd at 1 kHz. The
merging is the C++ side's job -- this node only produces well-formed arm frames.

Why this node exists next to g1_arm_control_node
------------------------------------------------
`g1_arm_control_node` writes joint targets to a ROS topic / Unitree DDS and owns
its own arm_sdk handover ramp. In a Groot deployment that ramp is wrong: the
C++ side already takes over from the measured pose with a Bezier trajectory
(`ArmBezierTrajectory`, at `arm_transition.velocity_scale` = 2% of the URDF
velocity limit) and applies a per-tick rate limit at 1 kHz. Adding a second ramp
would fight it. So this node has NO ramping and NO rate limiting of its own: it
sends the IK result, and the receiving side shapes it.

Safety posture
--------------
* starts DISABLED and in dry-run: nothing is transmitted until asked
* a failed IK solve transmits NOTHING (the C++ side then holds the last target)
* every frame is validated against the receiver's own rules before sending
* `set_enabled` service lets you hand the port back to another producer
"""

from __future__ import annotations

import threading
from typing import Optional

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, String

from g1_arm_ik_core.ros_common import extract_by_name, find_robot_description
from g1_arm_ik_core.vla_protocol import (
    NUM_ARM_JOINTS,
    FrameError,
    VelocityCommand,
    VlaActionFrame,
    assert_slot_alignment,
    lerobot_key_for_slot,
)
from g1_arm_ik_core.zmq_bridge import ZmqUnavailable, ZmqVlaBridge
from g1_arm_msgs.msg import ArmIKStatus
from g1_arm_msgs.srv import SolveIK, SetArmEnabled
from g1_ik import G1Arm, IKResult, build_arm_ik
from g1_ik.config import build_ik_config


class ArmVlaNode(Node):
    def __init__(self) -> None:
        super().__init__("g1_arm_vla_node")

        # ---------------- parameters -------------------------------------
        self.declare_parameter("side", "left")
        self.declare_parameter("urdf_path", "")
        self.declare_parameter("ik_base_link", "torso_link")
        self.declare_parameter("ee_offset", [0.05, 0.0, 0.0])
        self.declare_parameter("joint_state_topic", "/g1/joint_states")
        self.declare_parameter("velocity_topic", "/g1/vla/velocity")
        self.declare_parameter("velocity_timeout", 0.3)
        self.declare_parameter("control_rate", 50.0)
        self.declare_parameter("default_pos_tol", 5.0e-4)

        # zmq transport
        self.declare_parameter("zmq.robot_ip", "192.168.123.222")
        self.declare_parameter("zmq.port", 6002)
        self.declare_parameter("zmq.enabled", False)
        self.declare_parameter("zmq.dry_run", True)
        self.declare_parameter("zmq.sndhwm", 2)

        # ik solver
        self.declare_parameter("ik.position_weight", 200.0)
        self.declare_parameter("ik.multistart", True)
        self.declare_parameter("ik.multistart_max_extra", 12)
        self.declare_parameter("ik.control_rotation", False)

        self.side = str(self.get_parameter("side").value).lower()
        if self.side not in ("left", "right"):
            raise ValueError(f"side must be 'left' or 'right', got {self.side!r}")

        # ---------------- kinematics -------------------------------------
        urdf = find_robot_description(self.get_parameter("urdf_path").value)
        self.arm: G1Arm = build_arm_ik(
            urdf,
            side=self.side,
            ee_offset=tuple(float(v) for v in self.get_parameter("ee_offset").value),
            base_link=self.get_parameter("ik_base_link").value,
            config=build_ik_config(
                {
                    "position_weight": self.get_parameter("ik.position_weight").value,
                    "multistart": self.get_parameter("ik.multistart").value,
                    "multistart_max_extra": self.get_parameter("ik.multistart_max_extra").value,
                    "control_rotation": self.get_parameter("ik.control_rotation").value,
                    "pos_tol": self.get_parameter("default_pos_tol").value,
                }
            ),
        )
        self.n = len(self.arm.joint_names)

        # The load-bearing assumption: our IK joint order must equal the ZMQ arm
        # slot order, otherwise solved angles drive the wrong joints. Checked at
        # startup, not trusted.
        assert_slot_alignment(self.arm.joint_names, self.side)
        # The receiver demands all 14 arm joints, so one arm must be exactly 7.
        if self.n != NUM_ARM_JOINTS // 2:
            raise ValueError(
                f"side={self.side} has {self.n} joints; the VLA frame format needs "
                f"{NUM_ARM_JOINTS // 2} per side (14 total). Is the URDF a 23-DOF variant?"
            )

        # Which of the 14 slots belong to the arm we control.
        self.my_slots = list(range(0, 7)) if self.side == "left" else list(range(7, 14))
        self.other_side = "right" if self.side == "left" else "left"
        self.other_joint_names = [f"{self.other_side}_{n.split('_', 1)[1]}"
                                  for n in self.arm.joint_names]
        # name -> slot for the arm we control, so we can fill from a JointState
        self.slot_of_joint = {n: s for n, s in zip(self.arm.joint_names, self.my_slots)}

        self.get_logger().info(
            f"g1_arm_vla_node: side={self.side}, {self.n} DOF, "
            f"ee={self.arm.spec.ee_frame} on {self.arm.spec.ee_parent_link}, "
            f"base={self.arm.spec.root_link}, slots {self.my_slots[0]}..{self.my_slots[-1]}"
        )
        self.get_logger().info(
            f"ZMQ action frame -> {lerobot_key_for_slot(self.my_slots[0])} .. "
            f"{lerobot_key_for_slot(self.my_slots[-1])}; legs are NOT touched"
        )

        # ---------------- zmq --------------------------------------------
        self.bridge = ZmqVlaBridge(
            robot_ip=self.get_parameter("zmq.robot_ip").value,
            port=self.get_parameter("zmq.port").value,
            enabled=self.get_parameter("zmq.enabled").value,
            dry_run=self.get_parameter("zmq.dry_run").value,
            sndhwm=self.get_parameter("zmq.sndhwm").value,
        )
        self.bridge.connect()
        self.get_logger().warn(f"transport: {self.bridge.describe()}")
        if not self.bridge.transmitting:
            self.get_logger().warn(
                "NOT transmitting. Enable with the `set_enabled` service (and set "
                "zmq.dry_run:=false) when you are ready to command the arm."
            )

        # ---------------- state ------------------------------------------
        self._waist = np.zeros(3)
        self._waist_valid = False
        self._q_mine = np.zeros(self.n)          # measured, our arm
        self._q_mine_valid = False
        self._q_other = np.zeros(self.n)         # measured, held arm
        self._q_other_valid = False
        self._velocity = VelocityCommand()
        self._velocity_time: Optional[float] = None
        # `_last_result` is the most recent ATTEMPT (for diagnostics);
        # `_last_success` is the pose we are actually commanding, so a failed
        # solve can never blank out a good target.
        self._last_result: Optional[IKResult] = None
        self._last_success: Optional[IKResult] = None
        self._last_frame_payload = b""
        self._solve_lock = threading.Lock()

        self.cb_group = ReentrantCallbackGroup()

        # ---------------- interfaces -------------------------------------
        self.create_subscription(
            JointState,
            self.get_parameter("joint_state_topic").value,
            self._on_joint_state,
            10,
            callback_group=self.cb_group,
        )
        self.create_subscription(
            Twist,
            self.get_parameter("velocity_topic").value,
            self._on_velocity,
            10,
            callback_group=self.cb_group,
        )
        self.sub_pose = self.create_subscription(
            PoseStamped, "torso_target_pose", self._on_pose_target, 10,
            callback_group=self.cb_group,
        )
        self.srv_torso = self.create_service(
            SolveIK, "torso_target", self._on_solve_torso, callback_group=self.cb_group
        )
        self.srv_pelvis = self.create_service(
            SolveIK, "pelvis_target", self._on_solve_pelvis, callback_group=self.cb_group
        )
        self.srv_enable = self.create_service(
            SetArmEnabled, "set_enabled", self._on_set_enabled, callback_group=self.cb_group
        )
        self.pub_status = self.create_publisher(ArmIKStatus, "status", 10)
        self.pub_preview = self.create_publisher(String, "frame_preview", 10)
        self.pub_ik_ok = self.create_publisher(Bool, "ik_ok", 10)

        self.create_timer(
            1.0 / float(self.get_parameter("control_rate").value),
            self._on_tick,
            callback_group=self.cb_group,
        )

    # ------------------------------------------------------------------ inputs

    def _on_joint_state(self, msg: JointState) -> None:
        waist = extract_by_name(
            msg.name, msg.position,
            ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"],
        )
        if waist is not None:
            self._waist = waist
            self._waist_valid = True

        mine = extract_by_name(msg.name, msg.position, self.arm.joint_names)
        if mine is not None:
            self._q_mine = mine
            self._q_mine_valid = True

        other = extract_by_name(msg.name, msg.position, self.other_joint_names)
        if other is not None:
            self._q_other = other
            self._q_other_valid = True

    def _on_velocity(self, msg: Twist) -> None:
        self._velocity = VelocityCommand(
            vx=float(msg.linear.x), vy=float(msg.linear.y), wz=float(msg.angular.z)
        )
        self._velocity_time = self.get_clock().now().nanoseconds * 1e-9

    def _current_velocity(self) -> VelocityCommand:
        """Pass-through velocity, with a staleness timeout.

        Matching the receiver's `vla_timeout_` (0.3 s): once the upstream stops
        publishing, stop claiming a velocity rather than repeating the last one
        forever. Zeros make the on-board policy choose its balance branch.
        """
        timeout = float(self.get_parameter("velocity_timeout").value)
        if self._velocity_time is None:
            return VelocityCommand()
        age = self.get_clock().now().nanoseconds * 1e-9 - self._velocity_time
        if timeout > 0.0 and age > timeout:
            return VelocityCommand()
        return self._velocity

    def _on_pose_target(self, msg: PoseStamped) -> None:
        p = msg.pose.position
        self._set_target([p.x, p.y, p.z])

    def _on_solve_torso(self, req, resp) -> SolveIK.Response:
        return self._solve_request([req.x, req.y, req.z])

    def _on_solve_pelvis(self, req, resp) -> SolveIK.Response:
        if not self._waist_valid:
            self.get_logger().error(
                "pelvis_target requested but no waist angles received yet. Is "
                f"{self.get_parameter('joint_state_topic').value} publishing the "
                "waist joints?"
            )
            resp.success = False
            resp.status = "no_waist_state"
            return resp
        p_torso = self.arm.target_in_torso([req.x, req.y, req.z], self._waist)
        return self._solve_request(p_torso)

    def _on_set_enabled(self, req, resp) -> SetArmEnabled.Response:
        if req.data:
            self.bridge.enable()
            resp.success = True
            resp.message = (
                "VLA frames ENABLED"
                + (" (still dry-run: nothing transmitted)" if self.bridge.dry_run else "")
            )
            self.get_logger().warn(resp.message)
        else:
            self.bridge.disable()
            resp.success = True
            resp.message = "VLA frames DISABLED; port 6002 is free for another producer"
            self.get_logger().warn(resp.message)
        return resp

    # ------------------------------------------------------------------ solve

    def _set_target(self, position) -> IKResult:
        """Solve for one target point and adopt it if it converged.

        Only a converged solve becomes the active target; a failed one is
        recorded for diagnostics but leaves `_last_success` untouched, so
        `_on_tick` keeps holding the previous good pose instead of transmitting
        a bogus one.
        """
        with self._solve_lock:
            res = self.arm.solve_position(position)
            self._last_result = res
            if res.success:
                self._last_success = res
            else:
                self.get_logger().warn(
                    f"IK failed ({res.status}, err={res.pos_error * 1000:.3f} mm). "
                    "Target unchanged; nothing is transmitted for it.",
                    throttle_duration_sec=2.0,
                )
            return res

    def _solve_request(self, position) -> SolveIK.Response:
        res = self._set_target(position)
        resp = SolveIK.Response()
        resp.success = bool(res.success)
        resp.position_error = float(res.pos_error)
        resp.rotation_error = float(res.rot_error)
        resp.q_solution = [float(v) for v in res.q]
        resp.limit_margin = float(res.limit_margin)
        resp.iterations = int(res.iterations)
        resp.solve_time = float(res.solve_time)
        resp.attempts = int(res.attempts)
        resp.status = str(res.status)
        return resp

    # ------------------------------------------------------------------ send

    def _on_tick(self) -> None:
        # Transmit the last CONVERGED pose, not the last attempt. A failed solve
        # must not blank out a target that is still valid -- the arm should keep
        # holding where it was last told to go.
        res = self._last_success
        if res is None:
            # Nothing has ever converged, so there is no safe pose to send.
            self._publish_status()
            return

        if not (self._q_mine_valid and self._q_other_valid):
            self.get_logger().warn(
                "waiting for a complete arm joint state before transmitting; the "
                "receiver requires all 14 arm values in every frame",
                throttle_duration_sec=5.0,
            )
            self._publish_status()
            return

        frame = VlaActionFrame()
        frame.set_arm(self.other_side, self._q_other)   # hold the free arm where it is
        frame.set_arm(self.side, res.q)
        frame.velocity = self._current_velocity()

        now = self.get_clock().now().nanoseconds * 1e-9
        try:
            payload = frame.to_payload(frame.next_timestamp(now))
        except FrameError as exc:
            # Our own guard caught something the receiver would reject. Better to
            # drop one frame than to have the whole stream rejected as invalid.
            self.get_logger().error(f"refusing to send an invalid frame: {exc}",
                                    throttle_duration_sec=2.0)
            self._publish_status()
            return

        self._last_frame_payload = payload
        sent = self.bridge.send(payload)
        if sent:
            self.get_logger().debug(
                f"sent {len(payload)} B (err={res.pos_error*1000:.5f} mm, "
                f"v=({frame.velocity.vx:.2f},{frame.velocity.vy:.2f},{frame.velocity.wz:.2f}))"
            )
        self._publish_status()

    def _publish_status(self) -> None:
        s = ArmIKStatus()
        s.side = self.side
        s.joint_names = list(self.arm.joint_names)
        s.q_current = self._q_mine.tolist() if self._q_mine_valid else []
        # q_target is the pose being COMMANDED (last converged), while
        # position_error/ik_ok describe the last ATTEMPT. Reporting the commanded
        # pose keeps the status consistent with what is actually on the wire.
        s.q_target = self._last_success.q.tolist() if self._last_success else []
        s.q_command = []
        res = self._last_result
        if res is not None:
            s.position_error = float(res.pos_error)
            s.ik_ok = bool(res.success)
        else:
            s.position_error = -1.0
            s.ik_ok = False
        s.state = (
            "VLA_TRANSMITTING" if self.bridge.transmitting
            else ("VLA_DRY_RUN" if self.bridge.dry_run else "VLA_DISABLED")
        )
        s.backend = self.bridge.describe()
        s.cmds_sent = int(self.bridge.sent)
        s.cmds_rejected = int(self.bridge.rejected)
        s.last_reject_reason = self.bridge.last_error
        self.pub_status.publish(s)

        ok = Bool()
        ok.data = bool(res.success) if res is not None else False
        self.pub_ik_ok.publish(ok)

        if self._last_frame_payload:
            prev = String()
            prev.data = self._last_frame_payload.decode("utf-8")
            self.pub_preview.publish(prev)

    def destroy_node(self) -> bool:
        try:
            self.bridge.close()
        except Exception:  # noqa: BLE001 - shutdown must not raise
            pass
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    try:
        node = ArmVlaNode()
    except (ZmqUnavailable, ValueError) as exc:
        print(f"g1_arm_vla_node failed to start: {exc}")
        rclpy.shutdown()
        raise SystemExit(2) from exc
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
