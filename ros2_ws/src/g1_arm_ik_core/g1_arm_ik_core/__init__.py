"""g1_arm_ik_core -- ROS-free core for G1 single-arm position control.

Exports
-------
* the kinematics library (`g1_ik`)          -- FK/IK, no ROS anywhere
* `ArmCommandShaper` / `SafetyMonitor`      -- command shaping and safety
* `ArmBackend` + implementations            -- how commands reach the robot

The ROS 2 entry points live in the sibling packages so that this one keeps
working (and keeps being testable) on a machine with no ROS installed.
"""

from . import ros_common
from .control import (
    ArmBackend,
    ArmCommandShaper,
    ControlConfig,
    ControlStatus,
    MockBackend,
    SafetyMonitor,
)
from .vla_protocol import (
    ARM_SLOT_LEROBOT_NAMES,
    ARM_SLOT_SDK_NAMES,
    NUM_ARM_JOINTS,
    FrameError,
    VelocityCommand,
    VlaActionFrame,
    arm_slots,
    assert_slot_alignment,
    lerobot_key_for_slot,
    make_frame,
)
from .zmq_bridge import ZmqUnavailable, ZmqVlaBridge
from .ros_common import (
    G1_29DOF_JOINT_NAMES,
    arm_indices,
    extract_by_name,
    find_robot_description,
    waist_indices,
)

__version__ = "0.1.0"

__all__ = [
    "ArmCommandShaper",
    "ControlConfig",
    "ControlStatus",
    "SafetyMonitor",
    "ArmBackend",
    "MockBackend",
    # ZMQ 6002 VLA action frame
    "VlaActionFrame",
    "VelocityCommand",
    "FrameError",
    "arm_slots",
    "lerobot_key_for_slot",
    "make_frame",
    "assert_slot_alignment",
    "ARM_SLOT_LEROBOT_NAMES",
    "ARM_SLOT_SDK_NAMES",
    "NUM_ARM_JOINTS",
    "ZmqVlaBridge",
    "ZmqUnavailable",
    # joint layout / ROS interop helpers
    "G1_29DOF_JOINT_NAMES",
    "arm_indices",
    "waist_indices",
    "extract_by_name",
    "find_robot_description",
    "ros_common",
]
