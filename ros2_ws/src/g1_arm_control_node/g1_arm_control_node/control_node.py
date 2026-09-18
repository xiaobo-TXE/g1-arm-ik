"""g1_arm_control_node -- the only node that commands motion.

It consumes the joint target produced by g1_arm_ik_node, runs it through
ArmCommandShaper (watchdog -> validation -> rate limiting -> arm_sdk ramp) and
writes it to a robot backend.

Backends (`backend` parameter):

  mock   -- internal first-order joint model. No hardware, no DDS. Use this to
            verify the whole pipeline and your launch/config on a laptop.
  topic  -- publish the command on `command_topic` as Float64MultiArray. This is
            the bridge point: subscribe and forward to your own DDS writer, the
            Unitree bridge, or a simulator.
  dds    -- write to `rt/arm_sdk` directly with unitree_sdk2py.

Safety defaults, all deliberate:

* the node starts **DISABLED**; call the `set_enabled` service to engage
* the arm_sdk weight ramps 0 -> 1 over `ramp_up` seconds, never instantly
* every command is rate limited to `velocity_scale` x the URDF velocity limit
* a stale state stream trips the watchdog and stops all motion
* targets that are non-finite, outside the joint limits, or more than
  `max_joint_step` away from the current pose are rejected and counted
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, String

from g1_arm_ik_core import ArmCommandShaper, ControlConfig
from g1_arm_ik_core.robot_backends import make_backend
from g1_arm_msgs.msg import ArmIKStatus
from g1_arm_msgs.srv import SetArmEnabled
from g1_ik import build_arm_ik

from g1_arm_ik_core.ros_common import extract_by_name, find_robot_description


class ArmControlNode(Node):
    def __init__(self) -> None:
        super().__init__("g1_arm_control_node")

        # ---------------- parameters -------------------------------------
        self.declare_parameter("side", "left")
        self.declare_parameter("urdf_path", "")
        self.declare_parameter("joint_state_topic", "/g1/joint_states")
        self.declare_parameter("target_topic", "/g1/arm_ik/joint_target")
        self.declare_parameter("command_topic", "/g1/arm/command")
        self.declare_parameter("status_topic", "/g1/arm_control/status")

        self.declare_parameter("backend", "mock")
        self.declare_parameter("control_rate", 100.0)
        self.declare_parameter("velocity_scale", 0.2)
        self.declare_parameter("ramp_up", 1.0)
        self.declare_parameter("ramp_down", 1.0)
        self.declare_parameter("max_joint_step", 1.0)
        self.declare_parameter("state_timeout", 0.5)
        self.declare_parameter("state_alpha", 0.3)
        self.declare_parameter("start_enabled", False)
        # Auto-disable if no target is received for this long (0 = never).
        self.declare_parameter("target_timeout", 0.0)

        # dds backend
        self.declare_parameter("dds.network_interface", "eth0")
        self.declare_parameter("dds.topic", "rt/arm_sdk")
        self.declare_parameter("dds.weight_index", -2)
        self.declare_parameter("dds.domain_id", 0)
        self.declare_parameter("dds.dry_run", False)

        # mock backend
        self.declare_parameter("mock.tau", 0.05)
        self.declare_parameter("mock.noise", 1e-5)

        self.side = self.get_parameter("side").value
        self.control_rate = float(self.get_parameter("control_rate").value)
        self.dt = 1.0 / self.control_rate

        # ---------------- model (limits + velocity budgets) --------------
        urdf = find_robot_description(self.get_parameter("urdf_path").value)
        self.arm = build_arm_ik(urdf, side=self.side)
        self.joint_names = self.arm.joint_names
        self.n = len(self.joint_names)

        self.cfg = ControlConfig(
            control_dt=self.dt,
            velocity_scale=float(self.get_parameter("velocity_scale").value),
            ramp_up=float(self.get_parameter("ramp_up").value),
            ramp_down=float(self.get_parameter("ramp_down").value),
            max_joint_step=float(self.get_parameter("max_joint_step").value),
            state_timeout=float(self.get_parameter("state_timeout").value),
            state_alpha=float(self.get_parameter("state_alpha").value),
        )
        v_limits = self.cfg.velocity_limits(self.arm.max_velocity)

        # ---------------- backend ----------------------------------------
        self.pub_command = self.create_publisher(
            Float64MultiArray, self.get_parameter("command_topic").value, 10
        )
        backend_kind = self.get_parameter("backend").value
        self.backend = make_backend(
            backend_kind,
            publish_fn=self._publish_topic_command,
            topic=self.get_parameter("dds.topic").value,
            weight_index=self.get_parameter("dds.weight_index").value,
            network_interface=self.get_parameter("dds.network_interface").value,
            domain_id=self.get_parameter("dds.domain_id").value,
            dry_run=self.get_parameter("dds.dry_run").value,
            q_init=np.zeros(self.n),
            q_min=self.arm.q_min,
            q_max=self.arm.q_max,
            tau=float(self.get_parameter("mock.tau").value),
            noise=float(self.get_parameter("mock.noise").value),
            dt=self.dt,
        )
        self.backend.connect()

        self.shaper = ArmCommandShaper(
            q_min=self.arm.q_min, q_max=self.arm.q_max,
            velocity_limits=v_limits, config=self.cfg,
        )
        if self.backend.feeds_state:
            # Seed the shaped state from the simulator so the watchdog does not
            # trip before the first tick.
            self.shaper.set_joint_state(self.backend.state())
        if bool(self.get_parameter("start_enabled").value):
            self.shaper.enable()

        self.get_logger().info(
            f"g1_arm_control_node: side={self.side}, backend={self.backend.describe()}, "
            f"{self.control_rate:.0f} Hz, velocity budget "
            f"{np.round(v_limits, 2).tolist()} rad/s"
        )
        if not self.shaper._enabled:  # noqa: SLF001 - intentional startup notice
            self.get_logger().warn(
                "starting DISABLED (safe default). Call the `set_enabled` service "
                "with data=true to engage the arm."
            )

        # ---------------- state ------------------------------------------
        self._last_target_time: Optional[float] = None
        self._status: Optional[object] = None
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
            Float64MultiArray,
            self.get_parameter("target_topic").value,
            self._on_target,
            10,
            callback_group=self.cb_group,
        )

        self.srv_enable = self.create_service(
            SetArmEnabled, "set_enabled", self._on_set_enabled,
            callback_group=self.cb_group,
        )

        self.pub_status = self.create_publisher(
            ArmIKStatus, self.get_parameter("status_topic").value, 10
        )
        self.pub_state = self.create_publisher(String, "state", 10)

        self.create_timer(self.dt, self._on_tick, callback_group=self.cb_group)

    # ------------------------------------------------------------------ inputs

    def _on_joint_state(self, msg: JointState) -> None:
        if self.backend.feeds_state:
            # A simulating backend is its own source of truth; letting an
            # external /g1/joint_states in would fight the simulation.
            return
        q = extract_by_name(msg.name, msg.position, self.joint_names)
        if q is None:
            return
        self.shaper.set_joint_state(q)

    def _on_target(self, msg: Float64MultiArray) -> None:
        q = np.asarray(msg.data, dtype=float)
        if q.size != self.n:
            self.get_logger().warn(
                f"ignoring target with {q.size} values, expected {self.n} "
                f"({self.joint_names})"
            )
            return
        if self.shaper.set_target(q):
            self._last_target_time = self.get_clock().now().nanoseconds * 1e-9
        else:
            self.get_logger().warn(
                f"target REJECTED: {self.shaper._last_reject}"  # noqa: SLF001
            )

    def _on_set_enabled(self, req, resp) -> SetArmEnabled.Response:
        if req.data:
            self.shaper.enable()
            resp.success = True
            resp.message = "arm ENABLED; the arm_sdk weight will ramp up"
            self.get_logger().warn(resp.message)
        else:
            self.shaper.disable()
            resp.success = True
            resp.message = "arm DISABLED; weight ramps down and commands stop"
            self.get_logger().warn(resp.message)
        return resp

    # ----------------------------------------------------------------- output

    def _publish_topic_command(self, cmd, weight: float) -> None:
        out = Float64MultiArray()
        out.data = [float(v) for v in cmd]
        self.pub_command.publish(out)

    def _on_tick(self) -> None:
        # A simulating backend produces state as a side effect of send(), so
        # feed its feedback back in at the top of every tick.
        if self.backend.feeds_state:
            self.shaper.set_joint_state(self.backend.state())

        # Auto-disable on target starvation, if requested.
        timeout = float(self.get_parameter("target_timeout").value)
        if timeout > 0.0 and self._last_target_time is not None:
            now = self.get_clock().now().nanoseconds * 1e-9
            if (now - self._last_target_time) > timeout:
                if self.shaper.is_enabled:
                    self.get_logger().warn(
                        f"no target for {timeout:.1f}s -> disabling the arm"
                    )
                    self.shaper.disable()
                self._last_target_time = None

        status = self.shaper.step()
        self._status = status

        if status.ok:
            self.backend.send(status.q_command, status.arm_sdk_weight)

        self._publish_status(status)

    def _publish_status(self, status) -> None:
        s = ArmIKStatus()
        s.side = self.side
        s.joint_names = list(self.joint_names)
        s.q_current = np.nan_to_num(status.q_current).tolist()
        s.q_command = np.nan_to_num(status.q_command).tolist()
        s.arm_sdk_weight = float(status.arm_sdk_weight)
        s.ramp_progress = float(status.ramp_progress)
        s.target_error = (
            float(status.target_error) if np.isfinite(status.target_error) else -1.0
        )
        s.watchdog_tripped = bool(status.watchdog_tripped)
        s.state = status.state
        s.backend = self.backend.describe()
        s.cmds_sent = int(status.cmds_sent)
        s.cmds_rejected = int(status.cmds_rejected)
        s.last_reject_reason = status.last_reject_reason
        self.pub_status.publish(s)

        st = String()
        st.data = status.state
        self.pub_state.publish(st)

    def destroy_node(self) -> bool:
        try:
            self.backend.close()
        except Exception:  # noqa: BLE001 - shutdown must not raise
            pass
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ArmControlNode()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
