"""Shared ROS 2 helpers for the g1_arm nodes.

Small on purpose: robot description lookup, the G1 joint-name layout, and the
conversion between the Unitree /lowstate joint ordering and our arm ordering.
Keeping it in one place means the IK node, the control node and the bridge all
agree about what "joint index 3" means.
"""

from __future__ import annotations

import os
from typing import List, Optional, Sequence

import numpy as np

# --------------------------------------------------------------------------- #
#  G1 29-DOF joint layout as published by /lowstate (ascending motor index)
# --------------------------------------------------------------------------- #

# Left leg (0-5), right leg (6-11), waist (12-14), left arm (15-21),
# right arm (22-28). Source: Unitree G1JointIndex.
G1_29DOF_JOINT_NAMES: List[str] = [
    # legs
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    # waist
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    # left arm
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    # right arm
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

_INDEX = {name: i for i, name in enumerate(G1_29DOF_JOINT_NAMES)}

WAIST_INDEX = [_INDEX[n] for n in ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint")]


def arm_indices(side: str) -> List[int]:
    """Indices of the 7 arm joints inside a 29-DOF /lowstate vector."""
    side = side.lower()
    if side not in ("left", "right"):
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")
    prefix = "left_" if side == "left" else "right_"
    names = [
        f"{prefix}shoulder_pitch_joint",
        f"{prefix}shoulder_roll_joint",
        f"{prefix}shoulder_yaw_joint",
        f"{prefix}elbow_joint",
        f"{prefix}wrist_roll_joint",
        f"{prefix}wrist_pitch_joint",
        f"{prefix}wrist_yaw_joint",
    ]
    return [_INDEX[n] for n in names]


def waist_indices() -> List[int]:
    return list(WAIST_INDEX)


def indices_to_names(indices: Sequence[int]) -> List[str]:
    return [G1_29DOF_JOINT_NAMES[i] for i in indices]


# --------------------------------------------------------------------------- #
#  name-based <-> index-based joint vectors
# --------------------------------------------------------------------------- #


def extract_by_name(
    names: Sequence[str], positions: Sequence[float], wanted: Sequence[str]
) -> Optional[np.ndarray]:
    """Pick `wanted` joints out of a JointState, or None if any are missing.

    Returning None rather than zeros matters: silently substituting 0.0 for a
    missing joint would produce a bogus arm pose that looks valid.
    """
    if len(names) != len(positions):
        return None
    index = {n: i for i, n in enumerate(names)}
    if not all(w in index for w in wanted):
        return None
    return np.array([float(positions[index[w]]) for w in wanted], dtype=float)


# --------------------------------------------------------------------------- #
#  robot description
# --------------------------------------------------------------------------- #


def find_robot_description(default_uri: str = "") -> str:
    """Resolve the URDF for the node.

    Order: explicit parameter -> `robot_description` env var -> the file shipped
    in g1_arm_bringup. Raises with an actionable message if none work, because a
    silent fallback to a wrong URDF would produce wrong kinematics.
    """
    candidates: List[str] = []
    if default_uri:
        candidates.append(default_uri)
    env = os.environ.get("G1_ARM_URDF", "")
    if env:
        candidates.append(env)
    try:
        from ament_index_python.packages import get_package_share_directory

        share = get_package_share_directory("g1_arm_bringup")
        candidates.append(os.path.join(share, "urdf", "g1_29dof_rev_1_0.urdf"))
        candidates.append(os.path.join(share, "urdf", "g1_29dof_with_hand_rev_1_0.urdf"))
    except Exception:  # noqa: BLE001 - ament_index may be unavailable in tests
        pass

    for c in candidates:
        path = c
        if path.startswith("file://"):
            path = path[len("file://") :]
        if path and os.path.exists(path):
            return path

    raise FileNotFoundError(
        "could not locate the G1 URDF. Tried:\n  "
        + "\n  ".join(candidates)
        + "\n\nFix by one of:\n"
        "  * ros2 launch g1_arm_bringup arm_ik.launch.py   (sets it automatically)\n"
        "  * export G1_ARM_URDF=/path/to/g1_29dof_rev_1_0.urdf\n"
        "  * pass -p urdf_path:=/path/to/urdf"
    )
