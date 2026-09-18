"""Build the ZMQ 6002 VLA action frame for the G1 Groot controller.

This module is pure Python (numpy + PyYAML only) so the wire format can be
verified offline, byte for byte, against a reimplementation of the receiving
C++ parser.  The ZMQ transport lives in `zmq_bridge.py`; the ROS node in
`g1_arm_vla_node/vla_node.py`.

The contract, read from `deploy/include/groot/RemoteCommandReceiver.h` and
`deploy/include/groot/JointNameMap.h` on branch `groot-control`:

    port  6002
    role  robot-side PULL  <-  we PUSH   (tcp://<robot>:6002)
    codec YAML (yaml-cpp), NOT json

      {"cmd": "action",
       "action": {"kLeftShoulderPitch.q": -0.20, ... , "kRightWristYaw.q": -0.01,
                  "remote.lx": 0.0, "remote.ly": 0.0, "remote.rx": 0.0, "remote.ry": 0.0},
       "timestamp": 12345.67}

Validation the receiver performs, all of which this builder guarantees:

  1. `action` and `timestamp` must be present; timestamp finite
  2. timestamp strictly increasing vs. the previous accepted frame ("stale timestamp")
  3. exactly **14** arm joints must be present, matched by name case-insensitively
  4. every joint value finite and |q| <= 3.2
  5. the frame must parse as YAML

Legs are NOT in this frame.  In VLA mode the lower body is driven by the on-board
Groot ONNX policy; `State_Groot::publish_targets()` merges the two into a single
LowCmd at 1 kHz.  Nothing here touches joints 0..14.
"""

from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml

# --------------------------------------------------------------------------- #
#  joint naming: SDK motor order  ->  LeRobot "k.../.q" keys
# --------------------------------------------------------------------------- #

# Motor index order 15..28 (see deploy/docs/joint_naming_and_order.md).
# Left arm first, then right; within each arm:
#   shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll, wrist_pitch, wrist_yaw
#
# This is the *authoritative* slot order. `arm_slot_of_lerobot_name()` on the C++
# side walks exactly this list, and the receiver writes slot i to
# `lowcmd->msg_.motor_cmd()[15 + i]`.
ARM_START_INDEX = 15
NUM_ARM_JOINTS = 14
JOINTS_PER_ARM = 7

# slot 0..13 -> LeRobot key base name (without the ".q" suffix)
ARM_SLOT_LEROBOT_NAMES: Tuple[str, ...] = (
    "kLeftShoulderPitch",
    "kLeftShoulderRoll",
    "kLeftShoulderYaw",
    "kLeftElbow",
    "kLeftWristRoll",
    "kLeftWristPitch",
    "kLeftWristYaw",
    "kRightShoulderPitch",
    "kRightShoulderRoll",
    "kRightShoulderYaw",
    "kRightElbow",
    "kRightWristRoll",
    "kRightWristPitch",
    "kRightWristYaw",
)

# Same slot order, but the SDK/URDF names our IK library uses.  These two lists
# MUST stay index-aligned; `assert_slot_alignment` checks that against the IK
# library's own ordering at import time.
ARM_SLOT_SDK_NAMES: Tuple[str, ...] = (
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

REMOTE_AXIS_KEYS: Tuple[str, ...] = ("remote.lx", "remote.ly", "remote.rx", "remote.ry")

# Hard bound enforced by the receiver: |q| > 3.2 -> whole frame rejected.
MAX_ABS_JOINT_VALUE = 3.2


def arm_slots(side: str) -> List[int]:
    """Slots for one arm: left = 0..6, right = 7..13."""
    side = side.lower().strip()
    if side == "left":
        return list(range(0, JOINTS_PER_ARM))
    if side == "right":
        return list(range(JOINTS_PER_ARM, NUM_ARM_JOINTS))
    raise ValueError(f"side must be 'left' or 'right', got {side!r}")


def sdk_names_for_slots(slots: Sequence[int]) -> List[str]:
    return [ARM_SLOT_SDK_NAMES[s] for s in slots]


def lerobot_key_for_slot(slot: int) -> str:
    return f"{ARM_SLOT_LEROBOT_NAMES[slot]}.q"


def sdk_names_for_arm(side: str) -> List[str]:
    """The 7 SDK joint names of one arm, in ZMQ slot order."""
    return [ARM_SLOT_SDK_NAMES[s] for s in arm_slots(side)]


def assert_slot_alignment(ik_joint_names: Sequence[str], side: str) -> None:
    """Fail loudly if the IK library's arm ordering drifts from the ZMQ slots.

    A silent mismatch here would send the right magnitudes to the wrong joints,
    which on a humanoid means the elbow command drives the wrist. Cheap to check,
    catastrophic to get wrong.

    `side` matters: the check must compare against *that* arm's slot names. An
    earlier version only knew the left arm and so rejected every right-arm setup.
    """
    got = list(ik_joint_names)
    expected = sdk_names_for_arm(side)
    if got != expected:
        raise ValueError(
            f"IK joint order for side={side!r} does not match the ZMQ arm slot order.\n"
            f"  IK     : {got}\n"
            f"  slots  : {expected}"
        )


# --------------------------------------------------------------------------- #
#  frame
# --------------------------------------------------------------------------- #


@dataclasses.dataclass
class VelocityCommand:
    """Commanded base velocity, metres/s and rad/s."""

    vx: float = 0.0
    vy: float = 0.0
    wz: float = 0.0

    def as_remote_axes(self) -> Dict[str, float]:
        """Invert the receiver's mapping `vx=ly, vy=-lx, wz=-rx`."""
        return {
            "remote.lx": -float(self.vy),
            "remote.ly": float(self.vx),
            "remote.rx": -float(self.wz),
            "remote.ry": 0.0,  # unused by the receiver
        }


class FrameError(ValueError):
    """The frame we were about to send would be rejected by the receiver."""


class VlaActionFrame:
    """Accumulates 14 arm targets, then serializes to the 6002 payload.

    One instance represents one outgoing frame.  `arm_q` always holds all 14
    slots because the receiver rejects any frame with a different count.
    """

    def __init__(self) -> None:
        self.arm_q = np.zeros(NUM_ARM_JOINTS, dtype=float)
        self.velocity = VelocityCommand()
        self._filled = np.zeros(NUM_ARM_JOINTS, dtype=bool)
        self._last_timestamp: Optional[float] = None
        self._seq = 0

    # -- filling ------------------------------------------------------------

    def set_arm(self, side: str, q: Sequence[float]) -> "VlaActionFrame":
        """Write the 7 joint values for one arm into its slots."""
        q = np.asarray(q, dtype=float).reshape(-1)
        slots = arm_slots(side)
        if q.size != len(slots):
            raise ValueError(
                f"side={side!r} needs {len(slots)} joints, got {q.size}"
            )
        for slot, value in zip(slots, q):
            self.arm_q[slot] = float(value)
            self._filled[slot] = True
        return self

    def set_slot(self, slot: int, value: float) -> "VlaActionFrame":
        if not 0 <= slot < NUM_ARM_JOINTS:
            raise ValueError(f"slot {slot} out of range 0..{NUM_ARM_JOINTS - 1}")
        self.arm_q[slot] = float(value)
        self._filled[slot] = True
        return self

    def set_all_arms(self, q14: Sequence[float]) -> "VlaActionFrame":
        q14 = np.asarray(q14, dtype=float).reshape(-1)
        if q14.size != NUM_ARM_JOINTS:
            raise ValueError(f"expected {NUM_ARM_JOINTS} values, got {q14.size}")
        self.arm_q = q14.copy()
        self._filled[:] = True
        return self

    # -- validation ---------------------------------------------------------

    def validate(self) -> None:
        problems = []
        missing = [ARM_SLOT_LEROBOT_NAMES[i] for i in range(NUM_ARM_JOINTS) if not self._filled[i]]
        if missing:
            problems.append(
                f"{len(missing)}/{NUM_ARM_JOINTS} arm joints unset "
                f"(receiver requires exactly {NUM_ARM_JOINTS}): {missing[:4]}"
            )
        if not np.all(np.isfinite(self.arm_q)):
            bad = [ARM_SLOT_LEROBOT_NAMES[i] for i in range(NUM_ARM_JOINTS)
                   if not np.isfinite(self.arm_q[i])]
            problems.append(f"non-finite joint values: {bad}")
        over = np.abs(self.arm_q) > MAX_ABS_JOINT_VALUE
        if np.any(over):
            bad = [
                f"{ARM_SLOT_LEROBOT_NAMES[i]}={self.arm_q[i]:.3f}"
                for i in range(NUM_ARM_JOINTS)
                if over[i]
            ]
            problems.append(
                f"|q| > {MAX_ABS_JOINT_VALUE} would be rejected: {bad}"
            )
        vel = self.velocity
        if not all(np.isfinite([vel.vx, vel.vy, vel.wz])):
            problems.append(f"non-finite velocity {vel}")
        if problems:
            raise FrameError("; ".join(problems))

    # -- serialization ------------------------------------------------------

    def next_timestamp(self, now: float) -> float:
        """Return a timestamp strictly greater than the previous one.

        The receiver drops any frame whose timestamp is not strictly greater
        than the last accepted one.  A monotonic clock can still hand out the
        same value twice at high rates, so nudge forward when that happens.
        """
        ts = float(now)
        if not np.isfinite(ts):
            raise FrameError(f"non-finite timestamp {now!r}")
        if self._last_timestamp is not None and ts <= self._last_timestamp:
            ts = self._last_timestamp + 1e-6
        self._last_timestamp = ts
        return ts

    def to_dict(self, timestamp: float) -> Dict[str, object]:
        self.validate()
        action: Dict[str, float] = {}
        for slot in range(NUM_ARM_JOINTS):
            action[lerobot_key_for_slot(slot)] = float(self.arm_q[slot])
        action.update(self.velocity.as_remote_axes())
        return {
            "cmd": "action",
            "action": action,
            "timestamp": float(timestamp),
        }

    def to_payload(self, timestamp: float) -> bytes:
        return serialize_frame(self.to_dict(timestamp))

    def reset_timestamp(self) -> None:
        """Forget the timestamp high-water mark (call when starting a new stream)."""
        self._last_timestamp = None


def serialize_frame(frame: Dict[str, object]) -> bytes:
    """YAML-encode the frame.

    `sort_keys=False` and the default flow style produce the same simple block
    mapping the receiver's yaml-cpp parser expects.  Plain `safe_dump` is used
    deliberately: no anchors, no aliases, no custom tags.
    """
    text = yaml.safe_dump(frame, default_flow_style=False, sort_keys=False)
    return text.encode("utf-8")


def make_frame(
    side: str,
    q_solved: Sequence[float],
    q_other_arm: Sequence[float],
    velocity: Optional[VelocityCommand] = None,
) -> VlaActionFrame:
    """Convenience: solved arm + the other arm held, in one frame.

    The receiver demands all 14 joints, so when we control only one arm the
    other one must still be filled.  Holding it at its *measured* position is
    the honest choice: it neither fights the on-board controller nor teleports
    the free arm.
    """
    frame = VlaActionFrame()
    other = "right" if side.lower() == "left" else "left"
    frame.set_arm(other, q_other_arm)
    frame.set_arm(side, q_solved)
    if velocity is not None:
        frame.velocity = velocity
    return frame
