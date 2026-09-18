"""Robot backends: how a joint command actually reaches the G1.

Three implementations, chosen with the `backend` parameter of
g1_arm_control_node:

* ``mock``  -- no hardware. Internal first-order joint model. Use this to
  exercise the whole pipeline (IK -> shaping -> ramp -> feedback) on a laptop.
* ``topic`` -- publish the command on a ROS 2 topic. This is the *bridge point*:
  point a subscriber at it and forward to whatever interface you actually use
  (your own DDS writer, a Unitree bridge node, a simulator).
* ``dds``   -- write directly to the Unitree DDS topic with unitree_sdk2py.

The DDS backend is import-guarded: if `unitree_sdk2py` is missing, selecting it
raises a clear error instead of failing at import time, so the rest of the
stack still builds and runs.
"""

from typing import Sequence

import numpy as np

from ..control import ArmBackend, MockBackend

__all__ = ["ArmBackend", "MockBackend", "TopicBackend", "DDSBackend", "make_backend"]


class TopicBackend(ArmBackend):
    """Publish the command as std_msgs/Float64MultiArray-ish via a callback.

    The ROS publisher is injected as `publish_fn` so this class stays importable
    without rclpy -- which is what lets the offline test suite cover it.
    """

    name = "topic"

    def __init__(self, publish_fn, topic: str, weight_index: int = -2):
        self._publish = publish_fn
        self.topic = topic
        # Unitree's arm_sdk convention: the second-to-last motor command carries
        # the weight. Kept configurable because it has changed between firmware
        # generations -- verify against the official example for your robot.
        self.weight_index = int(weight_index)

    def connect(self) -> None:
        pass

    def send(self, q: Sequence[float], weight: float) -> None:
        cmd = np.asarray(q, dtype=float).tolist()
        if 0 <= self.weight_index < len(cmd):
            cmd[self.weight_index] = float(weight)
        self._publish(cmd, float(weight))

    def describe(self) -> str:
        return f"topic({self.topic}, weight_index={self.weight_index})"


class DDSBackend(ArmBackend):
    """Write directly to Unitree DDS via unitree_sdk2py.

    IMPORTANT: this is the layer where the arm_sdk weight handover happens, and
    the exact motor-command layout is firmware dependent. Before using it on
    hardware:

      * run the official example from unitree_sdk2_python for YOUR G1 model and
        confirm the weight index and the topic name,
      * test with the robot on a stand / holding its own weight,
      * keep `velocity_scale` low and `max_joint_step` tight at first.

    `weight_index=-2` mirrors the widely used convention (last motor command is
    a mode/reserved field, the one before it is the weight).
    """

    name = "dds"

    def __init__(
        self,
        network_interface: str = "eth0",
        topic: str = "rt/arm_sdk",
        weight_index: int = -2,
        mode_machine: int = 0,
        domain_id: int = 0,
        dry_run: bool = False,
    ):
        self.iface = network_interface
        self.topic = topic
        self.weight_index = int(weight_index)
        self.mode_machine = int(mode_machine)
        self.domain_id = int(domain_id)
        self.dry_run = bool(dry_run)
        self._pub = None

    def connect(self) -> None:
        if self.dry_run:
            return
        try:
            from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
            from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
            from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
        except ImportError as exc:  # pragma: no cover - hardware only
            raise ImportError(
                "backend=dds requires unitree_sdk2py.\n"
                "  pip install unitree-sdk2py\n"
                "or install it from the Unitree repository. Use backend=topic or "
                "backend=mock if you only want to test the pipeline."
            ) from exc

        self._LowCmd_ = LowCmd_
        self._default_cmd = unitree_hg_msg_dds__LowCmd_
        ChannelFactoryInitialize(self.domain_id, self.iface)
        self._pub = ChannelPublisher(self.topic, LowCmd_)
        self._pub.Init()
        self._cmd = self._default_cmd()

    def send(self, q: Sequence[float], weight: float) -> None:
        if self.dry_run or self._pub is None:
            return
        q = np.asarray(q, dtype=float)
        n = len(self._cmd.motor_cmd)
        for i in range(min(len(q), n)):
            self._cmd.motor_cmd[i].q = float(q[i])
            self._cmd.motor_cmd[i].dq = 0.0
            self._cmd.motor_cmd[i].tau = 0.0
            self._cmd.motor_cmd[i].kp = 60.0
            self._cmd.motor_cmd[i].kd = 1.5
        idx = self.weight_index if self.weight_index >= 0 else n + self.weight_index
        if 0 <= idx < n:
            self._cmd.motor_cmd[idx].q = float(weight)
        try:
            self._pub.Write(self._cmd)
        except Exception:  # noqa: BLE001 - a dropped DDS write must not kill the loop
            pass

    def describe(self) -> str:
        state = "dry-run" if self.dry_run else "live"
        return f"dds({self.topic}, iface={self.iface}, {state})"


def make_backend(kind: str, **kwargs) -> ArmBackend:
    """Factory used by the control node."""
    kind = (kind or "mock").lower()
    if kind == "mock":
        allowed = {
            "q_init",
            "q_min",
            "q_max",
            "tau",
            "noise",
            "seed",
        }
        return MockBackend(**{k: v for k, v in kwargs.items() if k in allowed})
    if kind == "topic":
        missing = [k for k in ("publish_fn",) if k not in kwargs]
        if missing:
            raise ValueError(f"backend=topic requires {missing}")
        allowed = {"publish_fn", "topic", "weight_index"}
        return TopicBackend(**{k: v for k, v in kwargs.items() if k in allowed})
    if kind == "dds":
        allowed = {
            "network_interface",
            "topic",
            "weight_index",
            "mode_machine",
            "domain_id",
            "dry_run",
        }
        return DDSBackend(**{k: v for k, v in kwargs.items() if k in allowed})
    raise ValueError(f"unknown backend {kind!r}; expected mock, topic or dds")
