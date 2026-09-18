"""g1_arm_ik_node -- the kinematics brain.

Solves inverse kinematics for ONE G1 arm and publishes the resulting joint
target. It never talks to the robot: g1_arm_control_node consumes the target and
handles ramping, rate limiting and the arm_sdk handover.

Frame handling
--------------
The IK base is `torso_link`, so all solve services work in the **torso** frame.
Because the torso moves with the waist, the node also exposes:

* `torso_target`  -- you give a torso-frame point, it gets solved directly
* `pelvis_target` -- you give a pelvis-frame point; the node applies
  T_pelvis_torso(waist) from /g1/joint_states and then solves
  `waist_state`   -- publishes the waist angles it is currently using, so a
  caller can undo the transform itself if it prefers

Safety posture: a failed solve publishes NOTHING (the control node keeps holding
the last command) and is surfaced on /g1/arm_ik/status with ik_ok=false.
"""

from __future__ import annotations

import threading
from typing import List, Optional

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float64MultiArray

from g1_arm_ik_core import __version__ as core_version
from g1_arm_msgs.msg import ArmIKStatus
from g1_arm_msgs.srv import SolveIK
from g1_ik import G1Arm, IKConfig, build_arm_ik
from g1_ik.config import build_ik_config

from g1_arm_ik_core.ros_common import extract_by_name, find_robot_description


class ArmIKNode(Node):
    def __init__(self) -> None:
        super().__init__("g1_arm_ik_node")

        # ---------------- parameters -------------------------------------
        self.declare_parameter("side", "left")
        self.declare_parameter("urdf_path", "")
        self.declare_parameter("ee_offset", [0.05, 0.0, 0.0])
        self.declare_parameter("ik_base_link", "torso_link")
        self.declare_parameter("joint_state_topic", "/g1/joint_states")
        self.declare_parameter("control_rate", 100.0)
        self.declare_parameter("default_pos_tol", 5.0e-4)
        self.declare_parameter("publish_hz", 50.0)
        # IK config: any field of g1_ik.IKConfig can be overridden with an
        # `ik.` prefix, e.g. -p ik.max_iter:=100
        self.declare_parameter("ik.position_weight", 200.0)
        self.declare_parameter("ik.multistart", True)
        self.declare_parameter("ik.multistart_max_extra", 12)
        self.declare_parameter("ik.control_rotation", False)
        self.declare_parameter("ik.limit_barrier_weight", 0.0)

        self.side: str = self.get_parameter("side").value
        self.control_dt = 1.0 / float(self.get_parameter("control_rate").value)

        # ---------------- kinematics -------------------------------------
        urdf = find_robot_description(self.get_parameter("urdf_path").value)
        ee_offset = tuple(float(v) for v in self.get_parameter("ee_offset").value)
        base_link = self.get_parameter("ik_base_link").value

        cfg_dict = {
            "position_weight": self.get_parameter("ik.position_weight").value,
            "multistart": self.get_parameter("ik.multistart").value,
            "multistart_max_extra": self.get_parameter("ik.multistart_max_extra").value,
            "control_rotation": self.get_parameter("ik.control_rotation").value,
            "limit_barrier_weight": self.get_parameter("ik.limit_barrier_weight").value,
            "pos_tol": self.get_parameter("default_pos_tol").value,
        }
        ik_cfg: IKConfig = build_ik_config(cfg_dict)

        # No reduced_urdf_path: the library defaults to a deterministic path in
        # the system temp directory, which keeps the installed package read-only
        # and does not depend on the current working directory.
        self.arm: G1Arm = build_arm_ik(
            urdf,
            side=self.side,
            ee_offset=ee_offset,
            base_link=base_link,
            config=ik_cfg,
        )
        self.joint_names: List[str] = self.arm.joint_names
        self.n = len(self.joint_names)

        self.get_logger().info(
            f"g1_arm_ik_node {core_version}: side={self.side}, {self.n} DOF, "
            f"ee={self.arm.spec.ee_frame} on {self.arm.spec.ee_parent_link}, "
            f"offset={self.arm.spec.ee_offset}, base={base_link}"
        )
        self.get_logger().info(
            "IK frame is the TORSO frame. Use the pelvis_target topic/service if "
            "your target is in the pelvis frame."
        )
        self.get_logger().info(f"symbolic IK backend: {self.arm.ik.backend}")

        # ---------------- state ------------------------------------------
        self._waist = np.zeros(3)
        self._waist_valid = False
        self._q_current = np.zeros(self.n)
        self._q_current_valid = False
        self._q_target = np.zeros(self.n)
        self._has_target = False
        self._last_result = None
        # The services and the pose topic all funnel into one ArmIK instance,
        # which owns mutable warm-start state. Under a MultiThreadedExecutor two
        # callbacks would otherwise solve concurrently on the same object; this
        # serialises them. Contention is irrelevant here (a solve is ~0.3 ms)
        # and correctness is not.
        self._solve_lock = threading.Lock()

        self.cb_group = ReentrantCallbackGroup()

        # ---------------- interfaces -------------------------------------
        qos = 10
        self.create_subscription(
            JointState,
            self.get_parameter("joint_state_topic").value,
            self._on_joint_state,
            qos,
            callback_group=self.cb_group,
        )

        self.srv_torso = self.create_service(
            SolveIK, "torso_target", self._on_solve_torso, callback_group=self.cb_group
        )
        self.srv_pelvis = self.create_service(
            SolveIK, "pelvis_target", self._on_solve_pelvis, callback_group=self.cb_group
        )

        self.pub_target = self.create_publisher(
            Float64MultiArray, "joint_target", qos
        )
        self.pub_status = self.create_publisher(ArmIKStatus, "status", qos)
        self.pub_waist = self.create_publisher(Float64MultiArray, "waist_state", qos)
        self.pub_ik_ok = self.create_publisher(Bool, "ik_ok", qos)

        # Optional convenience topic: PoseStamped in the torso frame.
        # A point-only message is impossible with PoseStamped, so orientation is
        # only honoured when control_rotation is enabled.
        self.sub_pose = self.create_subscription(
            PoseStamped, "torso_target_pose", self._on_pose_target, qos,
            callback_group=self.cb_group,
        )

        self.create_timer(
            1.0 / float(self.get_parameter("publish_hz").value),
            self._publish_status,
            callback_group=self.cb_group,
        )

    # ------------------------------------------------------------------ state

    def _on_joint_state(self, msg: JointState) -> None:
        waist_names = [
            "waist_yaw_joint",
            "waist_roll_joint",
            "waist_pitch_joint",
        ]
        waist = extract_by_name(msg.name, msg.position, waist_names)
        if waist is not None:
            self._waist = waist
            self._waist_valid = True
            out = Float64MultiArray()
            out.data = waist.tolist()
            self.pub_waist.publish(out)

        q_arm = extract_by_name(msg.name, msg.position, self.joint_names)
        if q_arm is not None:
            self._q_current = q_arm
            self._q_current_valid = True

    # --------------------------------------------------------------- solving

    def _solve(self, target_T, q_seed: Optional[np.ndarray], req) -> SolveIK.Response:
        with self._solve_lock:
            return self._solve_locked(target_T, q_seed, req)

    def _solve_locked(self, target_T, q_seed: Optional[np.ndarray], req) -> SolveIK.Response:
        res = self.arm.solve(target_T, q_seed=q_seed)

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

        if res.success:
            self._q_target = np.asarray(res.q, dtype=float)
            self._has_target = True
            self._publish_target()
            self.get_logger().debug(
                f"solved: err={res.pos_error * 1000:.5f} mm "
                f"margin={res.limit_margin:.4f} rad "
                f"{res.solve_time * 1000:.2f} ms attempts={res.attempts}"
            )
        else:
            self.get_logger().warn(
                f"IK failed ({res.status}, err={res.pos_error * 1000:.3f} mm). "
                "NOT publishing a target -- the control node keeps holding."
            )
        self._last_result = res
        self._publish_status()
        return resp

    def _seed_from_request(self, req) -> Optional[np.ndarray]:
        if req.use_q_seed and len(req.q_seed) == self.n:
            return np.asarray(req.q_seed, dtype=float)
        return None

    def _on_solve_torso(self, req, resp) -> SolveIK.Response:
        target = self.arm.ik.make_target(
            [req.x, req.y, req.z],
            np.asarray(req.orientation_row_major, dtype=float).reshape(3, 3)
            if req.use_orientation
            else None,
        )
        return self._solve(target, self._seed_from_request(req), req)

    def _on_solve_pelvis(self, req, resp) -> SolveIK.Response:
        if not self._waist_valid:
            self.get_logger().error(
                "pelvis_target requested but no waist angles received yet. "
                f"Is {self.get_parameter('joint_state_topic').value} publishing "
                "waist_yaw/roll/pitch_joint?"
            )
            resp.success = False
            resp.status = "no_waist_state"
            return resp
        p_torso = self.arm.target_in_torso([req.x, req.y, req.z], self._waist)
        target = self.arm.ik.make_target(
            p_torso,
            np.asarray(req.orientation_row_major, dtype=float).reshape(3, 3)
            if req.use_orientation
            else None,
        )
        return self._solve(target, self._seed_from_request(req), req)

    def _on_pose_target(self, msg: PoseStamped) -> None:
        p = msg.pose.position
        R = None
        if self.arm.ik.cfg.control_rotation:
            q = msg.pose.orientation
            R = _quat_to_matrix(q.x, q.y, q.z, q.w)
        target = self.arm.ik.make_target([p.x, p.y, p.z], R)
        self._solve(target, None, None)

    # --------------------------------------------------------------- outputs

    def _publish_target(self) -> None:
        out = Float64MultiArray()
        out.data = self._q_target.tolist()
        self.pub_target.publish(out)

    def _publish_status(self) -> None:
        s = ArmIKStatus()
        s.side = self.side
        s.joint_names = list(self.joint_names)
        s.q_current = self._q_current.tolist() if self._q_current_valid else []
        s.q_target = self._q_target.tolist() if self._has_target else []
        s.q_command = []
        res = self._last_result
        if res is not None:
            # Report the residual of the LAST ATTEMPT, even when it failed -- a
            # zero here would read as "perfect" on a failed solve. This node
            # never commands motion, so `state` says what it is doing and
            # `ik_ok` says whether the target can be trusted.
            s.position_error = float(res.pos_error)
            s.ik_ok = bool(res.success)
            s.state = "IK_SUCCESS" if res.success else "IK_FAILED"
        else:
            s.position_error = -1.0  # no solve attempted yet
            s.ik_ok = False
            s.state = "IK_IDLE"
        s.backend = "n/a"
        self.pub_status.publish(s)

        ok = Bool()
        ok.data = bool(res.success) if res is not None else False
        self.pub_ik_ok.publish(ok)


def _quat_to_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    return np.array(
        [
            [1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
            [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
            [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)],
        ]
    )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ArmIKNode()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
