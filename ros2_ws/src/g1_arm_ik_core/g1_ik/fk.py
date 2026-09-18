"""Forward kinematics implemented from scratch with numpy.

Deliberately independent of pinocchio: this file is the *cross-check*.  It only
uses the parsed URDF fields (origin xyz/rpy, axis, type) and a textbook chain of
homogeneous transforms.  If numpy FK and pinocchio FK agree to ~1e-12, both the
parser and the pinocchio model are trustworthy.

Frame conventions used throughout the package
---------------------------------------------
`T_pelvis_torso(q_waist)` maps a point expressed in the **torso frame** to the
**pelvis frame**, where q_waist = [waist_yaw, waist_roll, waist_pitch] and the
torso frame is defined by the URDF (its own link frame, not pybullet's
`torso_link` convention).

For G1 the waist chain happens to be:
    pelvis --waist_yaw(Z)--> waist_yaw_link
           --waist_roll(X)--> waist_roll_link
           --waist_pitch(Y)--> torso_link
but this file never hardcodes that: it walks the URDF chain, so it stays
correct if Unitree revises the geometry.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from .urdf_parser import Joint, UrdfModel, make_transform, parse_urdf

WAIST_JOINT_NAMES = ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]


class NumpyChainFK:
    """Plain-numpy FK for an arbitrary serial chain within a URDF."""

    def __init__(self, urdf_path: str, tip_link: str, base_link: Optional[str] = None):
        self.model: UrdfModel = parse_urdf(urdf_path)
        self.tip_link = tip_link
        self.base_link = base_link

        chain: List[Joint] = self.model.chain_to_root(tip_link)
        self._prefix: List[Joint] = []
        if base_link is not None:
            # The chain is root->leaf, so the first joint whose *parent link*
            # is base_link is where the requested sub-chain starts. (base_link
            # is a link, and appears in the chain only as some joint's parent,
            # never as a joint name.)
            start = None
            for i, j in enumerate(chain):
                if j.parent == base_link:
                    start = i
                    break
            if start is None:
                raise ValueError(
                    f"{base_link!r} is not an ancestor of {tip_link!r}; "
                    f"chain is {[(j.parent, j.child) for j in chain]}"
                )
            self._prefix = chain[:start]
            chain = chain[start:]
        self.chain = chain
        self.joint_names = [j.name for j in chain]
        self.actuated = [j.name for j in chain if j.is_actuated]

    def fk(
        self,
        q: Optional[Dict[str, float]] = None,
        base_transform: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Return T (4x4) of `tip_link` relative to `base_link`.

        q: joint name -> value. Missing joints are treated as 0.
        base_transform: optional extra left-multiplied transform.
        """
        q = q or {}
        T = np.eye(4) if base_transform is None else np.array(base_transform, dtype=float)
        for j in self.chain:
            T = T @ j.origin_transform @ j.joint_transform(float(q.get(j.name, 0.0)))
        return T

    def fk_all_links(self, q: Optional[Dict[str, float]] = None) -> Dict[str, np.ndarray]:
        """Poses (4x4) of *every* link in the model, relative to the root link.

        Used for TF broadcasting. One pass over `joint_order` is enough because a
        URDF's declaration order is a valid topological order: a parent joint
        always appears before its children. If that ever fails to hold, the
        affected subtree falls back to an explicit chain walk.
        """
        q = q or {}
        poses: Dict[str, np.ndarray] = {self.model.root_link: np.eye(4)}
        for name in self.model.joint_order:
            j = self.model.joints[name]
            parent = poses.get(j.parent)
            if parent is None:
                parent = self._pose_by_walk(j.parent, q)
            poses[j.child] = (
                parent @ j.origin_transform @ j.joint_transform(float(q.get(name, 0.0)))
            )
        return poses

    def _pose_by_walk(self, link: str, q: Dict[str, float]) -> np.ndarray:
        """Slow path: compose the chain down to `link` from the root."""
        T = np.eye(4)
        for j in self.model.chain_to_root(link):
            T = T @ j.origin_transform @ j.joint_transform(float(q.get(j.name, 0.0)))
        return T

    def prefix_transform(self, q: Optional[Dict[str, float]] = None) -> np.ndarray:
        """Transform from the URDF root to `base_link` (used for base->pelvis)."""
        q = q or {}
        T = np.eye(4)
        for j in self._prefix:
            T = T @ j.origin_transform @ j.joint_transform(float(q.get(j.name, 0.0)))
        return T


class G1NumpyFK:
    """Convenience wrapper exposing the three FK queries the package needs."""

    def __init__(
        self,
        full_urdf_path: str,
        reduced_urdf_path: str,
        side: str,
        ee_frame: Optional[str] = None,
        ee_offset: Sequence[float] = (0.05, 0.0, 0.0),
        ee_parent_link: Optional[str] = None,
        torso_link: str = "torso_link",
        pelvis_link: str = "pelvis",
    ):
        self.side = side
        self.ee_frame = ee_frame or ("L_ee" if side == "left" else "R_ee")
        self.torso_link = torso_link
        self.pelvis_link = pelvis_link
        self.full_urdf_path = full_urdf_path
        self.ee_offset = tuple(float(v) for v in ee_offset)
        if ee_parent_link is None:
            ee_parent_link = (
                "left_wrist_yaw_link" if side == "left" else "right_wrist_yaw_link"
            )

        # Arm chain in the reduced model. Walk to the *parent* link of the
        # virtual end-effector frame and apply the offset explicitly, rather
        # than walking to the EE frame: the reduced URDF contains a trailing
        # fixed joint for it, and using the full chain would apply the offset
        # twice (measured 3.9e-2 m of error).
        self._arm = NumpyChainFK(reduced_urdf_path, ee_parent_link, base_link=torso_link)
        self.ee_parent_link = ee_parent_link
        # Only the *actuated* joints take a q value.
        self.arm_joint_names = self._arm.actuated
        self.arm_chain_joints = list(self._arm.chain)

        # Waist chain in the full model: pelvis -> torso_link.
        self._waist = NumpyChainFK(full_urdf_path, torso_link, base_link=pelvis_link)
        self.waist_joint_names = self._waist.joint_names

    # -- waist ---------------------------------------------------------------

    def waist_transform(self, waist_q: Sequence[float]) -> np.ndarray:
        """T_pelvis_torso: maps torso-frame points into the pelvis frame."""
        if len(waist_q) != len(self.waist_joint_names):
            raise ValueError(
                f"expected {len(self.waist_joint_names)} waist angles "
                f"({self.waist_joint_names}), got {len(waist_q)}"
            )
        return self._waist.fk(dict(zip(self.waist_joint_names, waist_q)))

    # -- forward kinematics --------------------------------------------------

    def fk_torso(self, q_arm: Sequence[float]) -> np.ndarray:
        """T_torso_ee: end-effector pose expressed in the **torso frame**.

        The chain is walked with the 7 actuated angles; the trailing fixed joint
        that places the virtual end-effector frame is applied explicitly, so
        this returns the *TCP* pose and not the wrist yaw link pose.
        """
        if len(q_arm) != len(self.arm_joint_names):
            raise ValueError(
                f"expected {len(self.arm_joint_names)} arm joints "
                f"({self.arm_joint_names}), got {len(q_arm)}"
            )
        T = self._arm.fk(dict(zip(self.arm_joint_names, q_arm)))
        T = T @ make_transform(np.eye(3), np.asarray(self.ee_offset, dtype=float))
        return T

    def fk_pelvis(self, q_arm: Sequence[float], waist_q: Sequence[float]) -> np.ndarray:
        """T_pelvis_ee = T_pelvis_torso @ T_torso_ee."""
        return self.waist_transform(waist_q) @ self.fk_torso(q_arm)

    def ee_position_torso(self, q_arm: Sequence[float]) -> np.ndarray:
        return self.fk_torso(q_arm)[:3, 3]

    def ee_position_pelvis(self, q_arm: Sequence[float], waist_q: Sequence[float]) -> np.ndarray:
        return self.fk_pelvis(q_arm, waist_q)[:3, 3]
