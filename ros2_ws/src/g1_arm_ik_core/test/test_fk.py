"""Tests 1 & 2: URDF parsing + forward-kinematics cross-validation.

The point of this file is to make the FK trustworthy *before* any IK result is
believed.  Two independent implementations are compared:

    A) g1_ik.urdf_parser + numpy   -- textbook chain of homogeneous transforms
    B) pinocchio                   -- a widely used, independently written library

Plus a third check against the URDF's own geometry: the EE frame must sit
exactly `ee_offset` beyond the wrist yaw link origin, along the wrist yaw axis.

Run directly:  python tests/test_fk.py
Or with pytest: pytest tests/test_fk.py -v
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from g1_ik import G1NumpyFK, build_pin_arm_model, parse_urdf  # noqa: E402
from g1_ik.urdf_parser import (  # noqa: E402
    axis_angle_to_matrix,
    make_transform,
    rpy_to_matrix,
)

HERE = os.path.dirname(os.path.abspath(__file__))


def _find_src_dir() -> str:
    """Locate the workspace `src/` directory by walking up.

    Robust to being run from the source tree *or* from an installed layout
    (colcon copies tests under build/<pkg>/), so the URDF path never depends on
    a hard-coded number of parent directories.
    """
    d = HERE
    for _ in range(8):
        parent = os.path.dirname(d)
        if os.path.isdir(os.path.join(parent, "g1_arm_bringup")) and os.path.isdir(
            os.path.join(parent, "g1_arm_ik_core")
        ):
            return parent
        d = parent
    raise RuntimeError(f"could not locate the workspace src/ directory from {HERE}")


SRC = _find_src_dir()
ASSETS = os.path.join(SRC, "g1_arm_bringup", "urdf")
URDF = os.path.join(ASSETS, "g1_29dof_rev_1_0.urdf")
URDF_HAND = os.path.join(ASSETS, "g1_29dof_with_hand_rev_1_0.urdf")

EXPECTED_ARM_JOINTS = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
]

# Limits as published in the official Unitree G1 URDF. Hard-coded here on
# purpose: if a future URDF revision changes them, this test should shout.
EXPECTED_LIMITS = {
    "left_shoulder_pitch_joint": (-3.0892, 2.6704, 25.0, 37.0),
    "left_shoulder_roll_joint": (-1.5882, 2.2515, 25.0, 37.0),
    "left_shoulder_yaw_joint": (-2.618, 2.618, 25.0, 37.0),
    "left_elbow_joint": (-1.0472, 2.0944, 25.0, 37.0),
    "left_wrist_roll_joint": (-1.972222054, 1.972222054, 25.0, 37.0),
    "left_wrist_pitch_joint": (-1.614429558, 1.614429558, 5.0, 22.0),
    "left_wrist_yaw_joint": (-1.614429558, 1.614429558, 5.0, 22.0),
}

EXPECTED_AXES = {
    "left_shoulder_pitch_joint": (0, 1, 0),
    "left_shoulder_roll_joint": (1, 0, 0),
    "left_shoulder_yaw_joint": (0, 0, 1),
    "left_elbow_joint": (0, 1, 0),
    "left_wrist_roll_joint": (1, 0, 0),
    "left_wrist_pitch_joint": (0, 1, 0),
    "left_wrist_yaw_joint": (0, 0, 1),
}


def _check(cond, msg):
    if not cond:
        raise AssertionError(msg)


# --------------------------------------------------------------------------- #
# helper maths
# --------------------------------------------------------------------------- #


def test_rpy_convention():
    """URDF rpy must be R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    for rpy in [(0, 0, 0), (0.3, 0, 0), (0, 0.3, 0), (0, 0, 0.3), (0.2, -0.4, 0.7)]:
        R = rpy_to_matrix(*rpy)
        _check(np.allclose(R @ R.T, np.eye(3), atol=1e-12), "rpy not orthonormal")
        _check(abs(np.linalg.det(R) - 1) < 1e-12, "rpy det != 1")

    # A pure yaw of +pi/2 maps x -> y.
    R = rpy_to_matrix(0, 0, np.pi / 2)
    _check(np.allclose(R @ [1, 0, 0], [0, 1, 0], atol=1e-12), "yaw direction wrong")
    # Roll about x leaves x fixed.
    R = rpy_to_matrix(0.5, 0, 0)
    _check(np.allclose(R @ [1, 0, 0], [1, 0, 0], atol=1e-12), "roll axis wrong")
    # Order matters: Rz@Ry@Rx != Rx@Ry@Rz for generic angles.
    from g1_ik.urdf_parser import rpy_to_matrix as f

    Rx = f(0.2, 0, 0)
    Ry = f(0, -0.4, 0)
    Rz = f(0, 0, 0.7)
    _check(np.allclose(f(0.2, -0.4, 0.7), Rz @ Ry @ Rx, atol=1e-12), "rpy order wrong")
    _check(not np.allclose(f(0.2, -0.4, 0.7), Rx @ Ry @ Rz, atol=1e-9), "order ambiguous")


def test_axis_angle():
    """Rodrigues: rotation about z by pi/2 maps x -> y; angle 0 is identity."""
    _check(np.allclose(axis_angle_to_matrix([0, 0, 1], 0.0), np.eye(3)), "zero angle")
    R = axis_angle_to_matrix([0, 0, 1], np.pi / 2)
    _check(np.allclose(R @ [1, 0, 0], [0, 1, 0], atol=1e-12), "rotz wrong")
    R = axis_angle_to_matrix([0, 1, 0], np.pi / 2)
    _check(np.allclose(R @ [1, 0, 0], [0, 0, -1], atol=1e-12), "roty wrong")
    # Non-unit axis must be normalised internally.
    R1 = axis_angle_to_matrix([0, 0, 3.0], 0.7)
    R2 = axis_angle_to_matrix([0, 0, 1.0], 0.7)
    _check(np.allclose(R1, R2, atol=1e-12), "axis normalisation wrong")


# --------------------------------------------------------------------------- #
# test 1: parsing
# --------------------------------------------------------------------------- #


def test_parse_full_urdf():
    m = parse_urdf(URDF)
    _check(m.name == "g1_29dof_rev_1_0", f"robot name {m.name!r}")
    _check(m.root_link == "pelvis", f"root link {m.root_link!r}")

    acts = m.actuated_joints()
    _check(len(acts) == 29, f"expected 29 actuated joints, got {len(acts)}")

    # the arm chain, root -> leaf
    chain = m.joint_names_in_chain("left_wrist_yaw_link")
    _check(chain[-7:] == EXPECTED_ARM_JOINTS, f"left arm chain: {chain[-7:]}")

    # waist chain order (this is the order /lowstate reports them in too)
    waist = m.joint_names_in_chain("torso_link")[-3:]
    _check(
        waist == ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"],
        f"waist order: {waist}",
    )

    # a 29-DOF URDF must have the two extra wrist joints
    for extra in ("left_wrist_pitch_joint", "left_wrist_yaw_joint"):
        _check(extra in m.joints, f"{extra} missing -> not a 29-DOF URDF")
    return m


def test_joint_limits_and_axes():
    m = parse_urdf(URDF)
    for name, (lo, up, eff, vel) in EXPECTED_LIMITS.items():
        j = m.joints[name]
        _check(abs(j.lower - lo) < 1e-9, f"{name} lower {j.lower} != {lo}")
        _check(abs(j.upper - up) < 1e-9, f"{name} upper {j.upper} != {up}")
        _check(abs(j.effort - eff) < 1e-9, f"{name} effort {j.effort} != {eff}")
        _check(abs(j.velocity - vel) < 1e-9, f"{name} velocity {j.velocity} != {vel}")
        ax = tuple(int(round(v)) for v in j.axis)
        _check(ax == EXPECTED_AXES[name], f"{name} axis {ax} != {EXPECTED_AXES[name]}")

    # the weak wrist is a real hardware constraint worth surfacing in the log
    print(
        "  note: wrist pitch/yaw effort = "
        f"{m.joints['left_wrist_pitch_joint'].effort} N*m "
        f"(shoulder/elbow = {m.joints['left_shoulder_pitch_joint'].effort} N*m)"
    )


def test_reduced_model_generation(tmp_path=None):
    """The reduced URDF must expose exactly the 7 arm joints, in order."""
    import tempfile

    # ALWAYS pass an explicit out_path: the default writes the derived URDF next
    # to the source, which pollutes the repository tree.
    out = os.path.join(tempfile.mkdtemp(), "reduced.urdf")
    pin_m = build_pin_arm_model(URDF, side="left", out_path=out)

    _check(pin_m.nq == 7, f"nq = {pin_m.nq}, expected 7")
    got = list(pin_m.spec.joint_names)
    _check(got == EXPECTED_ARM_JOINTS, f"joint order: {got}")
    _check(pin_m.spec.root_link == "torso_link", "root is not torso_link")
    _check(pin_m.spec.ee_frame == "L_ee", "ee frame name")

    # limits must survive the round trip through the generated URDF
    for i, name in enumerate(EXPECTED_ARM_JOINTS):
        lo, up, _, _ = EXPECTED_LIMITS[name]
        _check(abs(pin_m.q_min[i] - lo) < 1e-9, f"{name} q_min drift")
        _check(abs(pin_m.q_max[i] - up) < 1e-9, f"{name} q_max drift")

    # the reduced model must be a *closed* chain: torso -> ... -> L_ee
    r = parse_urdf(out)
    chain = r.joint_names_in_chain("L_ee")
    _check(chain == EXPECTED_ARM_JOINTS, f"reduced chain {chain}")
    return pin_m


def test_ee_frame_offset_geometry():
    """L_ee must be exactly ee_offset beyond the wrist yaw link, on its x axis."""
    import tempfile

    from g1_ik import G1Arm

    arm = G1Arm(
        URDF,
        side="left",
        ee_offset=(0.05, 0.0, 0.0),
        reduced_urdf_path=os.path.join(tempfile.mkdtemp(), "r.urdf"),
    )
    m = parse_urdf(arm.spec.urdf_path)
    j = m.joints["L_ee_fixed_joint"]
    _check(j.type == "fixed", "L_ee joint must be fixed")
    _check(j.parent == "left_wrist_yaw_link", f"L_ee parent {j.parent}")
    _check(np.allclose(j.origin_xyz, [0.05, 0, 0]), f"offset {j.origin_xyz}")

    # the same offset must be present in the pinocchio model (ask the frame
    # itself; pinocchio's Frame has no `.id` attribute)
    assert arm.m.frame_id < len(arm.m.model.frames)
    frame = arm.m.model.frames[arm.m.frame_id]
    _check(frame.name == "L_ee", f"frame name {frame.name}")
    _check(
        frame.parentJoint == arm.m.model.getJointId("left_wrist_yaw_joint"),
        "L_ee is not attached to left_wrist_yaw_joint",
    )
    _check(
        np.allclose(frame.placement.translation, [0.05, 0, 0]),
        f"pinocchio frame offset {frame.placement.translation}",
    )

    # numeric check: distance between wrist_yaw_link origin and EE == 0.05
    for q in ([0] * 7, [0.3, -0.2, 0.4, 0.8, 0.1, -0.2, 0.3]):
        frames = arm.m.fk_all_frames(q)
        d = np.linalg.norm(
            frames["L_ee"][:3, 3] - frames["left_wrist_yaw_link"][:3, 3]
        )
        _check(abs(d - 0.05) < 1e-12, f"|EE - wrist_yaw| = {d}, expected 0.05")

    # and the numpy FK must agree with pinocchio about the TCP position
    for q in ([0] * 7, [0.3, -0.2, 0.4, 0.8, 0.1, -0.2, 0.3]):
        d = np.max(np.abs(arm.np_fk.fk_torso(q) - arm.m.fk(q)))
        _check(d < 1e-12, f"numpy FK mistook the wrist link for the TCP (diff {d:.3e})")


# --------------------------------------------------------------------------- #
# test 2: FK cross-validation
# --------------------------------------------------------------------------- #


def test_fk_numpy_vs_pinocchio_arm():
    """numpy FK and pinocchio FK must agree to ~1e-12 over random configs."""
    import tempfile

    from g1_ik import G1Arm

    arm = G1Arm(
        URDF,
        side="left",
        reduced_urdf_path=os.path.join(tempfile.mkdtemp(), "r.urdf"),
    )

    rng = np.random.default_rng(7)
    worst = 0.0
    n = 300
    for _ in range(n):
        q = rng.uniform(arm.q_min, arm.q_max)
        T_pin = arm.fk(q)
        T_np = arm.np_fk.fk_torso(q)
        worst = max(worst, float(np.max(np.abs(T_pin - T_np))))
    print(f"  left arm: max |pinocchio - numpy| over {n} random q = {worst:.3e}")
    _check(worst < 1e-9, f"FK mismatch {worst:.3e} exceeds tolerance")

    # same for the right arm
    arm_r = G1Arm(
        URDF,
        side="right",
        reduced_urdf_path=os.path.join(tempfile.mkdtemp(), "r.urdf"),
    )
    worst_r = 0.0
    for _ in range(n):
        q = rng.uniform(arm_r.q_min, arm_r.q_max)
        worst_r = max(
            worst_r, float(np.max(np.abs(arm_r.fk(q) - arm_r.np_fk.fk_torso(q))))
        )
    print(f"  right arm: max |pinocchio - numpy| over {n} random q = {worst_r:.3e}")
    _check(worst_r < 1e-9, f"right FK mismatch {worst_r:.3e}")


def test_fk_with_hands_urdf():
    """The with-hand URDF must produce the same arm chain (hands are locked)."""
    import tempfile

    from g1_ik import G1Arm

    arm_hand = G1Arm(
        URDF_HAND,
        side="left",
        reduced_urdf_path=os.path.join(tempfile.mkdtemp(), "r.urdf"),
    )
    _check(arm_hand.joint_names == EXPECTED_ARM_JOINTS, "hand URDF arm chain wrong")

    arm_plain = G1Arm(
        URDF,
        side="left",
        reduced_urdf_path=os.path.join(tempfile.mkdtemp(), "r2.urdf"),
    )
    q = [0.2, -0.3, 0.4, 0.9, 0.1, 0.0, 0.0]
    d = np.max(np.abs(arm_hand.fk(q) - arm_plain.fk(q)))
    _check(d < 1e-12, f"arm FK differs between URDF variants: {d:.3e}")


def test_waist_transform_consistency():
    """T_pelvis_ee composed two ways must agree: this validates the waist chain.

    The reference walks the *whole* full-URDF chain (pelvis -> waist -> arm ->
    wrist yaw link) in one pass and then applies the TCP offset, which is
    completely independent of how G1NumpyFK splits waist and arm. Note that a
    full URDF has no `L_ee` link -- that frame only exists in the reduced model.
    """
    import tempfile

    from g1_ik import G1Arm
    from g1_ik.fk import NumpyChainFK
    from g1_ik.urdf_parser import make_transform

    arm = G1Arm(
        URDF,
        side="left",
        reduced_urdf_path=os.path.join(tempfile.mkdtemp(), "r.urdf"),
    )

    rng = np.random.default_rng(11)
    chain_full = NumpyChainFK(URDF, "left_wrist_yaw_link")  # pelvis -> wrist yaw
    ee_offset_T = make_transform(np.eye(3), np.array(arm.spec.ee_offset))
    worst = 0.0
    for _ in range(60):
        waist = rng.uniform([-2.6, -0.5, -0.5], [2.6, 0.5, 0.5])
        q_arm = rng.uniform(arm.q_min, arm.q_max)
        q_map = dict(zip(arm.joint_names, q_arm))
        q_map.update(dict(zip(arm.np_fk.waist_joint_names, waist)))
        # reference: one single pass over the whole chain + the TCP offset
        T_ref = chain_full.fk(q_map) @ ee_offset_T
        p_ref = T_ref[:3, 3]
        p_comp = arm.fk_position_from_state(q_arm, waist)
        worst = max(worst, float(np.max(np.abs(p_ref - p_comp))))
    print(f"  waist composition: max |two ways| over 60 samples = {worst:.3e}")
    _check(worst < 1e-12, f"waist composition mismatch {worst:.3e}")

    # and the waist transform itself must match a waist-only walk
    w_chain = NumpyChainFK(URDF, "torso_link", base_link="pelvis")
    worst_w = 0.0
    for _ in range(20):
        waist = rng.uniform([-2.6, -0.5, -0.5], [2.6, 0.5, 0.5])
        T_a = w_chain.fk(dict(zip(w_chain.joint_names, waist)))
        T_b = arm.torso_from_pelvis(waist)
        worst_w = max(worst_w, float(np.max(np.abs(T_a - T_b))))
    print(f"  waist transform  : max difference = {worst_w:.3e}")
    _check(worst_w < 1e-12, f"waist transform mismatch {worst_w:.3e}")


def test_torso_pelvis_roundtrip():
    """target_in_torso and target_in_pelvis must be exact inverses."""
    import tempfile

    from g1_ik import G1Arm

    arm = G1Arm(
        URDF,
        side="left",
        reduced_urdf_path=os.path.join(tempfile.mkdtemp(), "r.urdf"),
    )
    rng = np.random.default_rng(3)
    worst = 0.0
    for _ in range(50):
        waist = rng.uniform([-2.6, -0.5, -0.5], [2.6, 0.5, 0.5])
        p_torso = rng.uniform(-0.5, 0.5, size=3)
        p_pelvis = arm.target_in_pelvis(p_torso, waist)
        p_back = arm.target_in_torso(p_pelvis, waist)
        worst = max(worst, float(np.max(np.abs(p_back - p_torso))))
    print(f"  torso<->pelvis round trip: max error = {worst:.3e}")
    _check(worst < 1e-12, f"round trip error {worst:.3e}")


def test_reach_envelope():
    """Report the workspace extent -- useful sanity numbers for the user."""
    import tempfile

    from g1_ik import G1Arm

    arm = G1Arm(
        URDF,
        side="left",
        reduced_urdf_path=os.path.join(tempfile.mkdtemp(), "r.urdf"),
    )
    rng = np.random.default_rng(5)
    pts = np.array([arm.fk_position(rng.uniform(arm.q_min, arm.q_max)) for _ in range(4000)])
    d = np.linalg.norm(pts, axis=1)
    print(
        f"  left-arm workspace (torso frame): "
        f"reach min={d.min():.3f} m, max={d.max():.3f} m, "
        f"x=[{pts[:,0].min():+.3f},{pts[:,0].max():+.3f}] "
        f"y=[{pts[:,1].min():+.3f},{pts[:,1].max():+.3f}] "
        f"z=[{pts[:,2].min():+.3f},{pts[:,2].max():+.3f}]"
    )
    _check(0.2 < d.max() < 1.2, f"implausible max reach {d.max():.3f} m")


# --------------------------------------------------------------------------- #

TESTS = [
    ("math.rpy_convention", test_rpy_convention),
    ("math.axis_angle", test_axis_angle),
    ("1.parse_full_urdf", test_parse_full_urdf),
    ("1.joint_limits_and_axes", test_joint_limits_and_axes),
    ("1.reduced_model_generation", test_reduced_model_generation),
    ("1.ee_frame_offset_geometry", test_ee_frame_offset_geometry),
    ("2.fk_numpy_vs_pinocchio_arm", test_fk_numpy_vs_pinocchio_arm),
    ("2.fk_with_hands_urdf", test_fk_with_hands_urdf),
    ("2.waist_transform_consistency", test_waist_transform_consistency),
    ("2.torso_pelvis_roundtrip", test_torso_pelvis_roundtrip),
    ("2.reach_envelope", test_reach_envelope),
]


def main() -> int:
    print("=" * 78)
    print("FK / PARSER VALIDATION")
    print("=" * 78)
    failed = 0
    for name, fn in TESTS:
        try:
            fn()
            print(f"  [PASS] {name}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  [FAIL] {name}: {type(exc).__name__}: {exc}")
    print()
    print(f"{len(TESTS) - failed}/{len(TESTS)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
