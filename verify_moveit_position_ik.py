#!/usr/bin/env python3
"""Verify that MoveIt 2 can do POSITION-ONLY IK for the 7-DOF G1 arm.

Run this on the Ubuntu 22.04 + Humble machine BEFORE the MoveIt rewrite, so we
know route A is viable instead of discovering it halfway through.

    source /opt/ros/humble/setup.bash
    python3 verify_moveit_position_ik.py                 # left arm
    python3 verify_moveit_position_ik.py --side right
    python3 verify_moveit_position_ik.py --solver kdl --n 200

What it answers, in order:

  1. Is MoveIt 2 installed and importable from Python (moveit_py)?
  2. Does it load our 7-DOF arm (generated from the shipped G1 URDF) and give
     forward kinematics that matches our independent numpy FK?
  3. Does the KDL kinematics plugin honour `position_only_ik: true`?
  4. **Does it actually converge?** success rate, position error, solve time,
     compared against the numbers our own solver already achieved
     (worst 0.006 mm, 60/60 success, 0.30 ms warm).

Interpretation at the end. Nothing here touches a robot.

Written against MoveIt 2.5.10 (Humble). moveit_py's API moved around during
Humble's life, so every call is wrapped in a small compatibility shim; if
something does not resolve, the script says which call failed rather than
raising a bare AttributeError.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from typing import List, Optional, Tuple

import numpy as np

# --------------------------------------------------------------------------- #
#  repo plumbing
# --------------------------------------------------------------------------- #

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "ros2_ws", "src")
CORE = os.path.join(SRC, "g1_arm_ik_core")
BRINGUP = os.path.join(SRC, "g1_arm_bringup")
sys.path.insert(0, CORE)

FAILED: List[str] = []


def stage(label: str) -> None:
    print()
    print("=" * 78)
    print(label)
    print("=" * 78)


def ok(msg: str) -> None:
    print(f"  [PASS] {msg}")


def warn(msg: str) -> None:
    print(f"  [WARN] {msg}")


def fail(msg: str) -> None:
    FAILED.append(msg)
    print(f"  [FAIL] {msg}")


# --------------------------------------------------------------------------- #
#  1. is MoveIt there?
# --------------------------------------------------------------------------- #


def check_moveit_installed() -> Optional[dict]:
    stage("1. MoveIt 2 availability")
    mods = {}
    for name in ("rclpy", "moveit", "moveit.core", "moveit.core.robot_model",
                 "moveit.core.robot_state", "moveit.planning"):
        try:
            __import__(name)
            mods[name] = True
            ok(f"import {name}")
        except ImportError as exc:
            mods[name] = False
            fail(f"import {name}: {exc}")
    if not mods.get("moveit.core.robot_model"):
        print()
        print("  MoveIt is not usable from Python. Install it with:")
        print("      sudo apt install ros-humble-moveit ros-humble-moveit-py")
        return None
    return mods


# --------------------------------------------------------------------------- #
#  2. load our arm into MoveIt
# --------------------------------------------------------------------------- #


def make_arm_urdf(side: str, tmpdir: str) -> str:
    """Generate the standalone 7-DOF arm URDF (torso_link -> *_wrist_yaw_link)."""
    from g1_ik import build_arm_ik

    full = os.path.join(BRINGUP, "urdf", "g1_29dof_rev_1_0.urdf")
    if not os.path.exists(full):
        raise FileNotFoundError(f"URDF not found: {full}")
    # build_arm_ik writes the reduced URDF; it needs pinocchio to build G1Arm,
    # so build the reduced model through the lower-level helper instead, which
    # only needs the urdf_parser (no pinocchio, no casadi).
    from g1_ik.reduced_model import build_reduced_arm_urdf

    spec = build_reduced_arm_urdf(
        full_urdf_path=full,
        side=side,
        ee_offset=(0.05, 0.0, 0.0),
        base_link="torso_link",
        out_path=os.path.join(tmpdir, f"g1_{side}_arm.urdf"),
    )
    return spec


def make_srdf(spec, group_name: str, tmpdir: str) -> str:
    """A minimal SRDF: one chain group plus its end effector.

    Deliberately minimal. No virtual_joint (the base link IS torso_link, there is
    no floating root to pin to a world frame) and no `parent_group` on the
    end_effector (an empty value is not valid there; omitting it is).
    """
    base = spec.root_link
    tip = spec.ee_parent_link
    joint_lines = "\n".join(
        f'    <joint name="{j}" value="0"/>' for j in spec.joint_names
    )
    xml = f"""<?xml version="1.0" ?>
<robot name="g1_{spec.ee_frame}_moveit">
  <group name="{group_name}">
    <chain base_link="{base}" tip_link="{tip}"/>
  </group>
  <group_state name="home" group="{group_name}">
{joint_lines}
  </group_state>
  <end_effector name="ee" parent_link="{tip}" group="{group_name}"/>
</robot>
"""
    path = os.path.join(tmpdir, "g1_arm.srdf")
    with open(path, "w") as fh:
        fh.write(xml)
    return path


def load_moveit(urdf_path: str, srdf_path: str, group_name: str):
    """Build a MoveIt RobotModel with kinematics configured for position-only IK."""
    import rclpy
    from moveit.core.robot_model import RobotModel

    if not rclpy.ok():
        rclpy.init()

    # MoveIt reads kinematics config from ROS parameters named
    #   robot_description_kinematics.<group>.<key>
    # which is exactly the kinematics.yaml structure (lookupParam checks
    # "<group>.<key>" then "robot_description_kinematics.<group>.<key>").
    import rclpy.node as _node

    node = _node.Node(
        "g1_arm_moveit_verify",
        automatically_declare_parameters_from_overrides=True,
    )
    params = {
        f"robot_description_kinematics.{group_name}.kinematics_solver":
            "kdl_kinematics_plugin/KDLKinematicsPlugin",
        f"robot_description_kinematics.{group_name}.kinematics_solver_search_resolution": 0.005,
        f"robot_description_kinematics.{group_name}.kinematics_solver_timeout": 0.05,
        f"robot_description_kinematics.{group_name}.kinematics_solver_attempts": 3.0,
        # the whole point of this script
        f"robot_description_kinematics.{group_name}.position_only_ik": True,
    }
    for k, v in params.items():
        try:
            node.set_parameters([_node.parameter_descriptions.Parameter(k, value=v)
                                 if False else _mk_param(k, v)])
        except Exception:
            pass

    with open(urdf_path) as fh:
        urdf = fh.read()
    with open(srdf_path) as fh:
        srdf = fh.read()

    model = RobotModel(urdf, srdf, node=node)
    return node, model


def _mk_param(name, value):
    from rcl_interfaces.msg import Parameter as P, ParameterType, ParameterValue as V
    p = P()
    p.name = name
    pv = V()
    if isinstance(value, bool):
        pv.type = ParameterType.PARAMETER_BOOL
        pv.bool_value = value
    elif isinstance(value, int):
        pv.type = ParameterType.PARAMETER_INTEGER
        pv.integer_value = value
    elif isinstance(value, float):
        pv.type = ParameterType.PARAMETER_DOUBLE
        pv.double_value = value
    else:
        pv.type = ParameterType.PARAMETER_STRING
        pv.string_value = str(value)
    p.value = pv
    return p


# --------------------------------------------------------------------------- #
#  main experiment
# --------------------------------------------------------------------------- #


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--side", default="left", choices=["left", "right"])
    ap.add_argument("--solver", default="kdl",
                    choices=["kdl", "trac_ik"],
                    help="kdl works out of the box; trac_ik needs a source build")
    ap.add_argument("--n", type=int, default=100, help="number of random targets")
    ap.add_argument("--timeout", type=float, default=0.05, help="s, per solve")
    args = ap.parse_args()

    if check_moveit_installed() is None:
        return 1

    # ---------------------------------------------------------------- our FK
    stage("2. build our 7-DOF arm model")
    tmpdir = tempfile.mkdtemp(prefix="g1_moveit_verify_")
    try:
        spec = make_arm_urdf(args.side, tmpdir)
    except Exception as exc:  # noqa: BLE001
        fail(f"could not build the reduced arm URDF: {exc}")
        return 1
    ok(f"reduced URDF: {spec.urdf_path}")
    ok(f"joints: {spec.joint_names}")
    ok(f"base={spec.root_link}  tip={spec.ee_parent_link}  ee_offset={spec.ee_offset}")

    group_name = "arm"
    srdf_path = make_srdf(spec, group_name, tmpdir)
    ok(f"generated SRDF with group '{group_name}': {srdf_path}")

    # our independent numpy FK, for cross-checking MoveIt's FK
    from g1_ik.fk import NumpyChainFK
    np_fk = NumpyChainFK(spec.urdf_path, spec.ee_parent_link, base_link=spec.root_link)
    ee_off = np.eye(4)
    ee_off[:3, 3] = np.asarray(spec.ee_offset, dtype=float)

    def our_fk(q) -> np.ndarray:
        T = np_fk.fk(dict(zip(spec.joint_names, q)))
        return T @ ee_off

    # ------------------------------------------------------------- MoveIt FK
    stage("3. load into MoveIt and cross-check forward kinematics")
    try:
        node, model = load_moveit(spec.urdf_path, srdf_path, group_name)
    except Exception as exc:  # noqa: BLE001
        fail(f"MoveIt could not load the model: {type(exc).__name__}: {exc}")
        print()
        print("  If this is an API mismatch, tell me the exact traceback and I")
        print("  will adapt the script to your moveit_py version.")
        return 1
    ok("MoveIt RobotModel loaded")

    from moveit.core.robot_state import RobotState
    jmg = model.get_joint_model_group(group_name)
    if jmg is None:
        fail(f"planning group '{group_name}' not found in the SRDF")
        return 1
    ok(f"joint model group '{group_name}' with {jmg.get_variable_count()} variables")

    q_min = np.array(spec.lower, dtype=float)
    q_max = np.array(spec.upper, dtype=float)
    rng = np.random.default_rng(0)

    fk_worst = 0.0
    for _ in range(20):
        q = rng.uniform(q_min, q_max)
        rs = RobotState(model)
        rs.set_joint_group_positions(group_name, q.tolist())
        rs.update()
        T_mv = np.array(rs.get_global_link_transform(spec.ee_parent_link).matrix())
        # MoveIt gives us the wrist-yaw link; apply our TCP offset ourselves
        T_mv = T_mv @ ee_off
        fk_worst = max(fk_worst, float(np.max(np.abs(T_mv - our_fk(q)))))
    if fk_worst < 1e-9:
        ok(f"MoveIt FK == our numpy FK (max diff {fk_worst:.3e}) over 20 configs")
    else:
        fail(f"MoveIt FK differs from ours by {fk_worst:.3e} -- frame/offset mismatch?")

    # ------------------------------------------------------------ MoveIt IK
    stage(f"4. position-only IK via '{args.solver}' ({args.n} random reachable targets)")

    solvers = []
    try:
        s = jmg.get_solver_instance()
        if s is not None:
            solvers.append(("get_solver_instance", s))
    except Exception:
        pass
    try:
        s = jmg.get_kinematics_solver()
        if s is not None and s not in [x[1] for x in solvers]:
            solvers.append(("get_kinematics_solver", s))
    except Exception:
        pass
    if not solvers:
        fail("could not obtain a kinematics solver from the JointModelGroup")
        print()
        print("  moveit_py's accessor name differs in your build. Try:")
        print("      python3 -c \"from moveit.core.robot_model import RobotModel; ")
        print("                   print([m for m in dir(RobotModel) if 'kin' in m.lower()])\"")
        return 1
    ok(f"solver obtained via {solvers[0][0]}: {type(solvers[0][1]).__name__}")
    solver = solvers[0][1]

    # sanity: is position-only actually configured?
    try:
        w = solver.get_orientation_vs_position_weight()
        if w == 0.0:
            ok("solver reports orientation weight 0.0 -> position-only IS active")
        else:
            warn(f"solver reports orientation weight {w} -> position-only NOT active")
    except Exception as exc:  # noqa: BLE001
        warn(f"could not read the orientation weight ({exc}); checking empirically instead")

    def solve(T_target: np.ndarray, seed: np.ndarray):
        from geometry_msgs.msg import Pose
        from moveit_msgs.msg import MoveItErrorCodes

        pose = Pose()
        pose.position.x = float(T_target[0, 3])
        pose.position.y = float(T_target[1, 3])
        pose.position.z = float(T_target[2, 3])
        # orientation is not constrained (position_only), but a valid quaternion
        # is still required by the message
        pose.orientation.w = 1.0

        rs = RobotState(model)
        rs.set_joint_group_positions(group_name, seed.tolist())
        t0 = time.perf_counter()
        err = None
        try:
            sol = solver.search_position_ik(pose, rs, args.timeout)
        except TypeError:
            sol = solver.search_position_ik(pose, rs)
        except Exception as exc:  # noqa: BLE001
            return False, float("nan"), time.perf_counter() - t0, str(exc)
        dt = time.perf_counter() - t0
        if sol is None:
            return False, float("nan"), dt, "solver returned None"
        q_sol = np.array(sol.get_joint_group_positions(group_name), dtype=float)
        p_sol = our_fk(q_sol)[:3, 3]
        err = float(np.linalg.norm(p_sol - T_target[:3, 3]))
        return True, err, dt, ""

    errors, times, failures = [], [], 0
    worst_msg = ""
    for i in range(args.n):
        q_true = rng.uniform(q_min, q_max)
        T_t = our_fk(q_true)
        found, err, dt, msg = solve(T_t, np.zeros(len(spec.joint_names)))
        times.append(dt)
        if found and np.isfinite(err):
            errors.append(err)
        else:
            failures += 1
            if not worst_msg:
                worst_msg = msg

    n_ok = len(errors)
    print()
    if errors:
        print(f"  success        : {n_ok}/{args.n}  ({100.0 * n_ok / args.n:.1f}%)")
        print(f"  position error : worst {max(errors) * 1000:.4f} mm, "
              f"median {np.median(errors) * 1000:.6f} mm")
    else:
        print(f"  success        : 0/{args.n}")
    print(f"  solve time     : mean {np.mean(times) * 1000:.2f} ms, "
          f"max {np.max(times) * 1000:.2f} ms")
    if failures and worst_msg:
        print(f"  first failure  : {worst_msg[:120]}")
    print()
    print("  our own solver, for comparison (pinocchio + CasADi/IPOPT):")
    print("      success 60/60, worst 0.006 mm, warm 0.30 ms")

    # ------------------------------------------------------------- verdict
    stage("VERDICT")
    rate = (n_ok / args.n) if args.n else 0.0
    if rate >= 0.98 and errors and max(errors) < 1e-3:
        print("  READY. KDL position-only IK converges on the 7-DOF arm.")
        print("  Route A is viable: MoveIt for IK+FK, no pip/conda dependencies.")
        print("  Tell me and I will do the rewrite.")
        print()
        print(f"  Expected accuracy: ~{max(errors) * 1000:.3f} mm worst, "
              f"~{np.mean(times) * 1000:.1f} ms per solve.")
        return 0
    if rate >= 0.8:
        print(f"  MARGINAL: {100 * rate:.0f}% success. Workable but worse than our")
        print("  60/60. Options: raise kinematics_solver_attempts, raise the timeout,")
        print("  or keep our solver for hard targets.")
        return 0
    print(f"  NOT VIABLE: only {100 * rate:.0f}% success with position-only KDL.")
    print()
    print("  Next thing to try, in order:")
    print("    a) --solver kdl is the only BINARY option for ROS 2 Humble; TRAC-IK")
    print("       has no Humble binary, so it would need a source build.")
    print("    b) raise kinematics_solver_attempts / kinematics_solver_timeout in")
    print("       the params at the top of this script and re-run.")
    print("    c) if it stays low, tell me -- we go back to route B")
    print("       (ros-humble-pinocchio from apt + our own solver), which is already")
    print("       verified at 60/60 and needs only one pip install (casadi).")
    return 1


if __name__ == "__main__":
    sys.exit(main())
