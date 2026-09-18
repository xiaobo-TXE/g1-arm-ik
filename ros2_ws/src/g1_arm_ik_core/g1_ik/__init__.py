"""g1_ik -- position-only inverse kinematics for one Unitree G1 arm.

Typical use
-----------
    from g1_ik import build_arm_ik

    arm = build_arm_ik("assets/urdf/g1_29dof_rev_1_0.urdf", side="left")

    # what the robot is doing right now
    q_now = arm.fk_position([0, 0, 0, 0.5, 0, 0, 0])      # torso-frame point

    # ask for a nearby point
    res = arm.ik.solve_position(q_now + [0.05, 0.0, 0.05])
    print(res.success, res.pos_error, res.q)

Frame conventions
-----------------
* IK/FK targets are expressed in the **torso frame** (`torso_link`).
* The waist is treated as a rigid base.  To work in the pelvis or world frame,
  convert with `G1Arm.torso_from_world` / `G1Arm.world_from_torso` using the
  waist angles read from /lowstate.
* The end effector is a virtual frame placed on `*_wrist_yaw_link`, offset
  `ee_offset` (default 0.05 m along the wrist yaw axis), matching the convention
  used by Unitree's xr_teleoperate.
"""

import numpy as np

from .fk import G1NumpyFK, NumpyChainFK
from .ik import ArmIK, IKConfig, IKResult, RateLimiter, WeightedMovingFilter
from .model import PinArmModel, build_pin_arm_model
from .reduced_model import ReducedArmModel, arm_joint_names, build_reduced_arm_urdf
from .urdf_parser import Joint, UrdfModel, parse_urdf

__version__ = "0.1.0"

__all__ = [
    "G1Arm",
    "build_arm_ik",
    "ArmIK",
    "IKConfig",
    "IKResult",
    "RateLimiter",
    "WeightedMovingFilter",
    "PinArmModel",
    "build_pin_arm_model",
    "ReducedArmModel",
    "build_reduced_arm_urdf",
    "arm_joint_names",
    "G1NumpyFK",
    "NumpyChainFK",
    "parse_urdf",
    "UrdfModel",
    "Joint",
]

DEFAULT_URDF = "assets/urdf/g1_29dof_rev_1_0.urdf"

# G1 waist joint order as read from /lowstate (Unitree publishes them in
# ascending index order, which is yaw -> roll -> pitch).
WAIST_ORDER = ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint")


class G1Arm:
    """One arm: reduced pinocchio model + numpy cross-check + IK solver."""

    def __init__(
        self,
        full_urdf_path: str = DEFAULT_URDF,
        side: str = "left",
        ee_offset=(0.05, 0.0, 0.0),
        base_link: str = "torso_link",
        config: IKConfig = None,
        reduced_urdf_path: str = None,
    ):
        self.side = side
        self.m = build_pin_arm_model(
            full_urdf_path=full_urdf_path,
            side=side,
            ee_offset=ee_offset,
            base_link=base_link,
            out_path=reduced_urdf_path,
        )
        self.spec: ReducedArmModel = self.m.spec
        self.ik = ArmIK(self.m, config=config)
        self.np_fk = G1NumpyFK(
            full_urdf_path=self.spec.full_urdf_path,
            reduced_urdf_path=self.spec.urdf_path,
            side=side,
            ee_frame=self.spec.ee_frame,
            ee_offset=self.spec.ee_offset,
            ee_parent_link=self.spec.ee_parent_link,
            torso_link=base_link,
        )

    # -- naming --------------------------------------------------------------

    @property
    def joint_names(self):
        return list(self.spec.joint_names)

    @property
    def q_min(self):
        return self.m.q_min

    @property
    def q_max(self):
        return self.m.q_max

    @property
    def max_velocity(self):
        return list(self.spec.velocity)

    # -- kinematics ----------------------------------------------------------

    def fk(self, q_arm):
        """End-effector pose in the torso frame."""
        return self.m.fk(q_arm)

    def fk_position(self, q_arm):
        """End-effector position in the torso frame (the 'URDF 正解' output)."""
        return self.m.fk_position(q_arm)

    def fk_position_from_state(self, q_arm, waist_q):
        """End-effector position in the **pelvis** frame, given /lowstate."""
        return self.np_fk.ee_position_pelvis(q_arm, waist_q)
    def torso_from_pelvis(self, waist_q):
        """T_pelvis_torso from the waist angles (pelvis -> torso)."""
        return self.np_fk.waist_transform(waist_q)

    def target_in_torso(self, position_pelvis, waist_q):
        """Convert a target point from the pelvis frame into the torso frame."""
        T_inv = _invert_rigid(self.np_fk.waist_transform(waist_q))
        p = _homog_point(position_pelvis)
        return (np.asarray(T_inv, dtype=float) @ np.asarray(p, dtype=float))[:3]

    def target_in_pelvis(self, position_torso, waist_q):
        """Convert a torso-frame point into the pelvis frame."""
        T = self.np_fk.waist_transform(waist_q)
        p = _homog_point(position_torso)
        return (np.asarray(T, dtype=float) @ np.asarray(p, dtype=float))[:3]

    # -- inverse kinematics --------------------------------------------------

    def solve(self, target, q_seed=None, **kw):
        """Solve IK for a 4x4 target in the torso frame."""
        return self.ik.solve(target, q_seed=q_seed, **kw)

    def solve_position(self, position_torso, q_seed=None, **kw):
        """Solve IK for a 3D point in the torso frame."""
        return self.ik.solve_position(position_torso, q_seed=q_seed, **kw)

    def solve_position_from_pelvis(self, position_pelvis, waist_q, q_seed=None, **kw):
        """Solve IK for a 3D point given in the pelvis frame."""
        p_torso = self.target_in_torso(position_pelvis, waist_q)
        return self.ik.solve_position(p_torso, q_seed=q_seed, **kw)

    # -- helpers -------------------------------------------------------------

    def q_to_dict(self, q_arm):
        return self.m.to_dict(q_arm)

    def q_from_dict(self, q_map):
        return self.m.to_vector(q_map)

    def summary(self):
        return self.spec.to_summary()

    def __repr__(self):
        return f"G1Arm(side={self.side}, nq={self.m.nq}, ee={self.spec.ee_frame})"


def build_arm_ik(
    full_urdf_path: str = DEFAULT_URDF,
    side: str = "left",
    ee_offset=(0.05, 0.0, 0.0),
    config: IKConfig = None,
    **kw,
) -> G1Arm:
    """One-call factory: returns a ready-to-use G1Arm."""
    return G1Arm(
        full_urdf_path=full_urdf_path,
        side=side,
        ee_offset=ee_offset,
        config=config,
        **kw,
    )


def _homog_point(p):
    p = list(p)
    if len(p) == 4:
        return p
    return [p[0], p[1], p[2], 1.0]


def _invert_rigid(T):
    """Analytic inverse of a rigid transform (R^T, -R^T t)."""
    T = np.asarray(T, dtype=float)
    R = T[:3, :3]
    t = T[:3, 3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out
