"""Build a reduced, deterministic kinematic model for ONE arm.

Why reduce instead of using the full 29-DOF URDF directly?

1. Determinism.  In the full URDF the configuration vector mixes legs, waist,
   both arms and hands.  The IK would carry 29+ irrelevant variables, and the
   mapping from "arm joint i" to "q[i]" would depend on URDF ordering.
   Here the arm is exactly 7 DOF, in a fixed, explicit order.

2. Hard-fixing the waist.  The user's decision: waist (3 joints) is a *rigid
   base* whose pose comes from /lowstate.  Freezing it in the model makes that
   structural rather than a cost term the optimiser could trade away.

3. Speed.  CasADi/IPOPT on 7 variables instead of 29 is dramatically faster
   and far better conditioned.

The reduced model is emitted as a small standalone URDF so that *both*
pinocchio and the independent NumPy FK read the same ground truth.
"""

from __future__ import annotations

import dataclasses
import os
import tempfile
from typing import List, Optional, Sequence

from .urdf_parser import Joint, parse_urdf


@dataclasses.dataclass
class ReducedArmModel:
    """Everything downstream code needs to know about one arm chain."""

    urdf_path: str  # emitted reduced URDF (torso_link as root)
    root_link: str  # e.g. "torso_link" -- the IK base frame
    ee_frame: str  # e.g. "L_ee"
    ee_parent_link: str  # e.g. "left_wrist_yaw_link"
    ee_offset: tuple  # (x, y, z) of the EE frame in the parent link frame
    joint_names: List[str]  # exactly 7, in root->leaf order == q order
    lower: List[float]
    upper: List[float]
    effort: List[float]
    velocity: List[float]
    axis: List[tuple]  # joint axes, for the NumPy FK cross-check
    origin_xyz: List[tuple]  # joint origins, for the NumPy FK cross-check
    origin_rpy: List[tuple]  # joint origins, for the NumPy FK cross-check
    joint_types: List[str]
    full_urdf_path: str  # the original, unmodified URDF

    def to_summary(self) -> str:
        lines = [
            f"reduced URDF     : {self.urdf_path}",
            f"root (IK base)   : {self.root_link}",
            f"EE frame         : {self.ee_frame}  on {self.ee_parent_link} "
            f"offset {self.ee_offset}",
            f"DOF              : {len(self.joint_names)}",
            "",
            f"{'idx':>3} {'joint':<26} {'lower':>9} {'upper':>9} {'effort':>7} {'vel':>6}",
        ]
        for i, n in enumerate(self.joint_names):
            lines.append(
                f"{i:>3} {n:<26} {self.lower[i]:>9.4f} {self.upper[i]:>9.4f} "
                f"{self.effort[i]:>7.1f} {self.velocity[i]:>6.1f}"
            )
        return "\n".join(lines)


# --- defaults matching the official Unitree G1 29-DOF arm geometry ----------

EE_PARENT_LINK = {
    "left": "left_wrist_yaw_link",
    "right": "right_wrist_yaw_link",
}

ARM_JOINT_SUFFIXES = [
    "shoulder_pitch_joint",
    "shoulder_roll_joint",
    "shoulder_yaw_joint",
    "elbow_joint",
    "wrist_roll_joint",
    "wrist_pitch_joint",
    "wrist_yaw_joint",
]

DEFAULT_IK_BASE_LINK = "torso_link"


def arm_joint_names(side: str) -> List[str]:
    side = _check_side(side)
    return [f"{side}_{s}" for s in ARM_JOINT_SUFFIXES]


def _check_side(side: str) -> str:
    side = side.lower().strip()
    if side not in ("left", "right"):
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")
    return side


def _fmt(vals: Sequence[float]) -> str:
    return " ".join(f"{v:.12g}" for v in vals)


def _joint_xml(j: Joint, override_type: Optional[str] = None) -> str:
    jtype = override_type or j.type
    parts = [f'  <joint name="{j.name}" type="{jtype}">']
    parts.append(f'    <origin xyz="{_fmt(j.origin_xyz)}" rpy="{_fmt(j.origin_rpy)}"/>')
    parts.append(f'    <parent link="{j.parent}"/>')
    parts.append(f'    <child link="{j.child}"/>')
    if jtype in ("revolute", "continuous", "prismatic"):
        parts.append(f'    <axis xyz="{_fmt(j.axis)}"/>')
    if jtype == "revolute":
        lo = -1e16 if j.lower is None else j.lower
        up = 1e16 if j.upper is None else j.upper
        eff = 0.0 if j.effort is None else j.effort
        vel = 0.0 if j.velocity is None else j.velocity
        parts.append(
            f'    <limit lower="{lo:.12g}" upper="{up:.12g}" '
            f'effort="{eff:.12g}" velocity="{vel:.12g}"/>'
        )
    parts.append("  </joint>")
    return "\n".join(parts)


def build_reduced_arm_urdf(
    full_urdf_path: str,
    side: str,
    ee_offset: Sequence[float] = (0.05, 0.0, 0.0),
    base_link: str = DEFAULT_IK_BASE_LINK,
    out_path: Optional[str] = None,
    robot_name: Optional[str] = None,
) -> ReducedArmModel:
    """Extract one arm chain from a full G1 URDF into a standalone URDF.

    The emitted model is: base_link as root, then the 7 arm joints, then a
    fixed joint placing the virtual end-effector frame `ee_frame`.

    Note that `base_link` (torso_link) is the root, so in the reduced model a
    pose is expressed **in the torso frame**.  Converting to/from the pelvis or
    world frame is the caller's job (see fk.py / ik.py `waist` handling).
    """
    side = _check_side(side)
    full = parse_urdf(full_urdf_path)

    ee_parent = EE_PARENT_LINK[side]
    ee_frame = "L_ee" if side == "left" else "R_ee"

    if base_link not in full.links:
        raise ValueError(f"base link {base_link!r} not in {full_urdf_path}")
    if ee_parent not in full.links:
        raise ValueError(
            f"end-effector parent link {ee_parent!r} not in {full_urdf_path}. "
            f"Is this a 29-DOF (7-DOF arm) URDF? The 23-DOF variant has no "
            f"wrist_pitch/wrist_yaw joints."
        )

    chain = full.chain_to_root(ee_parent)

    # Keep only the joints strictly below base_link. The chain runs root->leaf,
    # so the first joint whose *parent link* is base_link is where the sub-chain
    # starts. (Comparing against joint names would be wrong: base_link is a
    # link, and it appears in the chain only as some joint's parent.)
    start = None
    for i, j in enumerate(chain):
        if j.parent == base_link:
            start = i
            break
    if start is None:
        raise ValueError(
            f"{base_link!r} is not an ancestor of {ee_parent!r}; got chain "
            f"{[(j.parent, j.child) for j in chain]}"
        )
    arm_chain = chain[start:]
    if not arm_chain:
        raise ValueError(f"empty arm chain below {base_link!r}")

    actuated = [j for j in arm_chain if j.is_actuated]
    expected = arm_joint_names(side)
    got = [j.name for j in actuated]
    if got != expected:
        raise ValueError(
            f"arm chain mismatch.\n  expected: {expected}\n  got     : {got}\n"
            f"The URDF may be a 23-DOF variant or a different revision."
        )

    # Emit reduced URDF.
    xml = [
        '<?xml version="1.0" ?>',
        f'<robot name="{robot_name or f"g1_{side}_arm_reduced"}">',
        f'  <link name="{base_link}"/>',
    ]
    for j in arm_chain:
        xml.append(f'  <link name="{j.child}"/>')
    xml.append("")
    xml.append(f"  <!-- arm chain: {base_link} -> {ee_parent} -->")
    for j in arm_chain:
        xml.append(_joint_xml(j))
    xml.append("")
    xml.append(f"  <!-- virtual end-effector (TCP) frame -->")
    xml.append(f'  <joint name="{ee_frame}_fixed_joint" type="fixed">')
    xml.append(f'    <origin xyz="{_fmt(ee_offset)}" rpy="0 0 0"/>')
    xml.append(f'    <parent link="{ee_parent}"/>')
    xml.append(f'    <child link="{ee_frame}"/>')
    xml.append("  </joint>")
    xml.append(f'  <link name="{ee_frame}"/>')
    xml.append("</robot>")

    if out_path is None:
        # Default to the system temp directory, NOT next to the source URDF.
        # The reduced model is a derived artifact; writing it beside the input
        # pollutes the source tree (and inside an installed ROS package that
        # directory may be read-only). The name stays deterministic so repeated
        # runs reuse the same file and it is easy to find when debugging.
        base = os.path.splitext(os.path.basename(full_urdf_path))[0]
        out_path = os.path.join(
            tempfile.gettempdir(), f"{base}__{side}_arm_reduced.urdf"
        )
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as fh:
        fh.write("\n".join(xml) + "\n")

    # IMPORTANT: the emitted model's actuated-joint *declaration order* is the
    # q order used by pinocchio and by the NumPy FK. We verify it below rather
    # than assuming it.
    reduced = parse_urdf(out_path)
    q_order = reduced.joint_names_in_chain(ee_frame)
    if q_order != expected:
        raise RuntimeError(
            f"reduced URDF joint order {q_order} != expected {expected}"
        )

    return ReducedArmModel(
        urdf_path=out_path,
        root_link=base_link,
        ee_frame=ee_frame,
        ee_parent_link=ee_parent,
        ee_offset=tuple(float(v) for v in ee_offset),
        joint_names=got,
        lower=[(-3.14159265358979 if j.lower is None else j.lower) for j in actuated],
        upper=[(3.14159265358979 if j.upper is None else j.upper) for j in actuated],
        effort=[(0.0 if j.effort is None else j.effort) for j in actuated],
        velocity=[(0.0 if j.velocity is None else j.velocity) for j in actuated],
        axis=[tuple(float(v) for v in j.axis) for j in actuated],
        origin_xyz=[tuple(float(v) for v in j.origin_xyz) for j in actuated],
        origin_rpy=[tuple(float(v) for v in j.origin_rpy) for j in actuated],
        joint_types=[j.type for j in actuated],
        full_urdf_path=os.path.abspath(full_urdf_path),
    )
