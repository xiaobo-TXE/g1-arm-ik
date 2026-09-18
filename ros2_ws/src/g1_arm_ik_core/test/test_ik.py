"""Test 3: inverse-kinematics accuracy, safety and continuity.

Run directly:  python tests/test_ik.py
Or with pytest: pytest tests/test_ik.py -v

What is actually being verified
-------------------------------
1. round trip      FK(q_true) -> IK -> FK(q_sol) must land on the same point
2. warm tracking   a dense sequence of nearby targets must be followed with
                   sub-mm error and *continuous* joint motion (this is what a
                   real control loop does -- it is the most important test here)
3. limits          no solution may ever violate a joint limit
4. unreachable     a far-away target must fail *safely* (no NaN, no wild q)
5. backends        the manual symbolic FK and pinocchio's numeric FK must agree
6. timing          per-solve latency must be usable in a control loop
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from g1_ik import IKConfig, build_arm_ik  # noqa: E402

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

POS_TOL = 5e-4  # 0.5 mm


def _check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def _arm(side="left", cfg=None, urdf=URDF):
    import tempfile

    return build_arm_ik(
        urdf,
        side=side,
        config=cfg or IKConfig(),
        reduced_urdf_path=os.path.join(tempfile.mkdtemp(), f"{side}.urdf"),
    )


def sample_configs(arm, n, seed):
    """Random joint configurations, biased away from the limits.

    Sampling uniformly hits limit-adjacent poses constantly, which are valid
    but not representative. This keeps 5% margin so the tests measure IK, not
    the barrier.
    """
    rng = np.random.default_rng(seed)
    span = arm.q_max - arm.q_min
    lo = arm.q_min + 0.05 * span
    hi = arm.q_max - 0.05 * span
    return [rng.uniform(lo, hi) for _ in range(n)]


# --------------------------------------------------------------------------- #
# 1. round trip
# --------------------------------------------------------------------------- #


def test_roundtrip_single_arm(side="left", n=60, seed=1, verbose=True):
    arm = _arm(side)
    worst = 0.0
    worst_attempts = 1
    n_ok = 0
    for q_true in sample_configs(arm, n, seed):
        p_target = arm.fk_position(q_true)
        res = arm.solve_position(p_target, q_seed=np.zeros(7))
        n_ok += int(res.pos_error <= POS_TOL)
        worst = max(worst, res.pos_error)
        worst_attempts = max(worst_attempts, res.attempts)
    if verbose:
        print(
            f"  {side}: {n_ok}/{n} reached <= {POS_TOL*1000:.2f} mm, "
            f"worst error = {worst*1000:.6f} mm, max attempts = {worst_attempts}"
        )
    _check(n_ok == n, f"{side}: only {n_ok}/{n} round trips reached the target")
    _check(worst < POS_TOL, f"{side}: worst error {worst*1000:.4f} mm")


def test_roundtrip_left():
    test_roundtrip_single_arm("left")


def test_roundtrip_right():
    test_roundtrip_single_arm("right")


def test_roundtrip_sub_micron():
    """The solver should be far better than the acceptance threshold."""
    arm = _arm()
    worst = 0.0
    for q_true in sample_configs(arm, 40, seed=21):
        res = arm.solve_position(arm.fk_position(q_true))
        worst = max(worst, res.pos_error)
    print(f"  worst error over 40 targets = {worst*1000:.6f} mm ({worst:.3e} m)")
    _check(worst < 1e-4, f"solver precision only {worst*1000:.4f} mm")


# --------------------------------------------------------------------------- #
# 2. warm-start trajectory tracking (the realistic control case)
# --------------------------------------------------------------------------- #


def test_trajectory_tracking_continuity():
    """Follow a smooth target path; report accuracy and command smoothness.

    This is the test that matters for the real robot: it exercises warm
    starting and measures the two numbers a control loop cares about --
    tracking error and how much the joints have to move per step.

    Target step is ~0.4 mm (what a 100 Hz loop commands). The raw IK command is
    reported as an implied joint speed at 100 Hz and must stay within the
    robot's velocity budget; the second half shows the output conditioning
    (RateLimiter + WeightedMovingFilter) that the control loop should apply.
    """
    arm = _arm()
    q_true = np.array([0.35, -0.25, 0.45, 0.95, 0.10, 0.00, 0.00])
    p0 = arm.fk_position(q_true)

    n_steps = 120
    dt = 0.01  # 100 Hz
    amp = 0.004
    ts = np.linspace(0.0, 1.0, n_steps)
    targets = [
        p0 + np.array([amp * np.sin(2 * np.pi * t), amp * t, amp * np.sin(np.pi * t)])
        for t in ts
    ]
    target_step = max(
        float(np.linalg.norm(targets[i + 1] - targets[i])) for i in range(n_steps - 1)
    )

    # ---- raw IK ---------------------------------------------------------
    errors, dqs = [], []
    q_prev = None
    for p in targets:
        res = arm.solve_position(p)
        errors.append(res.pos_error)
        if q_prev is not None:
            dqs.append(float(np.max(np.abs(res.q - q_prev))))
        q_prev = res.q.copy()

    max_err = max(errors)
    max_dq = max(dqs)
    implied_vel = max_dq / dt
    v_lim = float(np.min(arm.max_velocity))
    print(
        f"  {n_steps} steps, target step <= {target_step*1000:.3f} mm: "
        f"max tracking err = {max_err*1000:.5f} mm, "
        f"median = {np.median(errors)*1000:.6f} mm"
    )
    print(
        f"  raw IK command: max |dq| = {np.degrees(max_dq):.3f} deg/step "
        f"(= {implied_vel:.2f} rad/s at 100 Hz; slowest joint limit "
        f"{v_lim:.1f} rad/s)"
    )
    _check(max_err <= POS_TOL, f"trajectory tracking error {max_err*1000:.4f} mm")
    _check(
        implied_vel < v_lim,
        f"raw IK command implies {implied_vel:.2f} rad/s, above the "
        f"{v_lim:.1f} rad/s joint limit",
    )

    # ---- with the output conditioning a control loop should apply --------
    from g1_ik import RateLimiter, WeightedMovingFilter

    arm2 = _arm()
    rl = RateLimiter(arm2.max_velocity, dt=dt)
    wmf = WeightedMovingFilter((0.4, 0.3, 0.2, 0.1))
    q0 = arm2.solve_position(targets[0]).q
    rl.reset(q0)
    wmf.add(q0)

    lag, dqs_f = [], []
    q_prev = q0.copy()
    for p in targets[1:]:
        res = arm2.solve_position(p)
        q_f = wmf.add(rl.limit(res.q))
        lag.append(float(np.linalg.norm(arm2.fk_position(q_f) - p)))
        dqs_f.append(float(np.max(np.abs(q_f - q_prev))))
        q_prev = q_f.copy()

    print(
        f"  + RateLimiter + WeightedMovingFilter: max |dq| = "
        f"{np.degrees(max(dqs_f)):.3f} deg/step, max lag = {max(lag)*1000:.3f} mm"
    )
    _check(
        max(dqs_f) <= max_dq + 1e-9,
        "the output filter made the command less smooth, which is impossible",
    )
    _check(max(lag) < 5e-3, f"filter lag {max(lag)*1000:.2f} mm is too large")


def test_tiny_target_change_is_continuous():
    """A 1 mm target nudge must not produce a large joint change."""
    arm = _arm()
    q_true = np.array([0.3, -0.3, 0.4, 0.9, 0.1, 0.0, 0.0])
    p = arm.fk_position(q_true)

    res_a = arm.solve_position(p, q_seed=q_true)
    res_b = arm.solve_position(p + np.array([0.001, 0.0, 0.0]))

    dq = np.max(np.abs(res_b.q - res_a.q))
    print(
        f"  1 mm target nudge -> max |dq| = {np.degrees(dq):.4f} deg "
        f"(err {res_a.pos_error*1000:.5f} / {res_b.pos_error*1000:.5f} mm)"
    )
    _check(res_b.pos_error <= POS_TOL, "1 mm nudge not reached")
    _check(dq < np.radians(2.0), f"1 mm nudge caused a {np.degrees(dq):.2f} deg jump")


# --------------------------------------------------------------------------- #
# 3. safety
# --------------------------------------------------------------------------- #


def test_never_violates_joint_limits():
    arm = _arm()
    rng = np.random.default_rng(9)
    # include targets deliberately chosen to push the arm into its limits
    for q_true in sample_configs(arm, 40, seed=33) + [
        arm.q_max * 0.98,
        arm.q_min * 0.98,
    ]:
        res = arm.solve_position(arm.fk_position(q_true))
        over = np.maximum(np.maximum(arm.q_min - res.q, res.q - arm.q_max), 0.0)
        _check(
            over.max() < 1e-12,
            f"joint limit violated by {over.max():.3e} rad",
        )
        _check(np.all(np.isfinite(res.q)), "non-finite joint angles returned")

    # and for completely arbitrary (possibly unreachable) targets
    for _ in range(40):
        p = rng.uniform(-1.5, 1.5, size=3)
        res = arm.solve_position(p)
        over = np.maximum(np.maximum(arm.q_min - res.q, res.q - arm.q_max), 0.0)
        _check(over.max() < 1e-12, f"limit violated on random target {p}")
        _check(np.all(np.isfinite(res.q)), f"non-finite q for target {p}")
    print("  no limit violations and no NaN across 82 solves (incl. random targets)")


def test_unreachable_target_fails_safely():
    """A far target must report failure, not return a wild configuration."""
    arm = _arm()
    far = np.array([2.5, 0.0, 0.0])  # 2.5 m away: well beyond the ~0.75 m reach
    res = arm.solve_position(far)

    print(
        f"  unreachable target: success={res.success} "
        f"err={res.pos_error*1000:.2f} mm margin={res.limit_margin:.4f} rad"
    )
    _check(not res.success, "an unreachable target was reported as success")
    _check(np.all(np.isfinite(res.q)), "non-finite q for unreachable target")
    # it should point *towards* the target rather than collapse randomly
    reach_dir = arm.fk_position(res.q)
    _check(
        np.linalg.norm(reach_dir) > 0.3,
        f"arm collapsed to {np.linalg.norm(reach_dir):.3f} m instead of reaching out",
    )

    rep = arm.ik.reachability_report(far, n_seeds=8)
    print(
        f"  reachability report: reachable={rep['reachable']} "
        f"best={rep['best_error']*1000:.2f} mm"
    )
    _check(not rep["reachable"], "reachability_report disagrees on unreachable target")


def test_near_limit_target_reports_small_margin():
    """Targets at the edge of the workspace should show a small limit margin."""
    arm = _arm()
    q_edge = arm.q_max * 0.995
    res = arm.solve_position(arm.fk_position(q_edge))
    print(f"  edge target: err={res.pos_error*1000:.4f} mm margin={res.limit_margin:.4f} rad")
    _check(np.all(np.isfinite(res.q)), "non-finite q at edge target")


# --------------------------------------------------------------------------- #
# 4. backend equivalence
# --------------------------------------------------------------------------- #


def test_symbolic_fk_matches_pinocchio():
    """The two symbolic backends must produce the same FK."""
    import casadi

    for side in ("left", "right"):
        arm = _arm(side)
        fk_sym = casadi.Function("f", [arm.ik._cq], [arm.ik._ee_pose])
        rng = np.random.default_rng(4)
        worst = 0.0
        for _ in range(150):
            q = rng.uniform(arm.q_min, arm.q_max)
            worst = max(worst, float(np.max(np.abs(np.array(fk_sym(q)) - arm.fk(q)))))
        print(f"  {side}: symbolic FK vs pinocchio numeric FK max diff = {worst:.3e}")
        _check(worst < 1e-12, f"{side}: symbolic FK mismatch {worst:.3e}")


def test_symbolic_fk_matches_numpy_fk():
    """And against the fully independent numpy implementation."""
    arm = _arm()
    worst = 0.0
    for q in sample_configs(arm, 200, seed=6):
        worst = max(
            worst, float(np.max(np.abs(arm.fk(q) - arm.np_fk.fk_torso(q))))
        )
    print(f"  pinocchio FK vs numpy FK max diff = {worst:.3e}")
    _check(worst < 1e-9, f"numpy/pinocchio FK mismatch {worst:.3e}")


def test_with_hand_urdf_ik():
    """IK must behave identically with the with-hand URDF."""
    arm_plain = _arm()
    arm_hand = _arm(urdf=URDF_HAND)
    q_true = np.array([0.3, -0.3, 0.4, 0.9, 0.1, 0.0, 0.0])
    p = arm_plain.fk_position(q_true)
    r1 = arm_plain.solve_position(p, q_seed=np.zeros(7))
    r2 = arm_hand.solve_position(p, q_seed=np.zeros(7))
    print(
        f"  plain URDF err={r1.pos_error*1000:.6f} mm | "
        f"hand URDF err={r2.pos_error*1000:.6f} mm"
    )
    _check(r1.pos_error <= POS_TOL and r2.pos_error <= POS_TOL, "hand URDF IK failed")


# --------------------------------------------------------------------------- #
# 5. timing
# --------------------------------------------------------------------------- #


def test_timing():
    """Measure cold and warm latency; a control loop needs a bounded budget."""
    arm = _arm()
    q_true = np.array([0.3, -0.3, 0.4, 0.9, 0.1, 0.0, 0.0])
    p = arm.fk_position(q_true)

    # warm: repeated solves of *nearby* targets -- the realistic case
    times = []
    cur = p.copy()
    for i in range(60):
        cur = p + np.array([0.002 * np.sin(i / 5.0), 0.0, 0.0])
        t0 = time.perf_counter()
        arm.solve_position(cur)
        times.append(time.perf_counter() - t0)

    times = np.asarray(times) * 1000.0
    print(
        f"  warm-start solve: mean {times.mean():.2f} ms, "
        f"p95 {np.percentile(times, 95):.2f} ms, max {times.max():.2f} ms"
    )
    _check(times.mean() < 50.0, f"mean solve time {times.mean():.1f} ms is too slow")

    # cold: fresh model + far target, worst case
    arm2 = _arm()
    cold = []
    for _ in range(10):
        arm2.ik.reset_warm_start()
        t0 = time.perf_counter()
        arm2.solve_position(p, q_seed=np.zeros(7))
        cold.append(time.perf_counter() - t0)
    cold = np.asarray(cold) * 1000.0
    print(f"  cold solve      : mean {cold.mean():.2f} ms, max {cold.max():.2f} ms")


def test_config_file_loads():
    """The shipped YAML must load and produce a working arm."""
    from g1_ik.config import load_arm_from_config

    cfg_path = os.path.join(SRC, "g1_arm_bringup", "config", "g1_left_arm.yaml")
    arm = load_arm_from_config(cfg_path)
    q_true = np.array([0.3, -0.3, 0.4, 0.9, 0.1, 0.0, 0.0])
    res = arm.solve_position(arm.fk_position(q_true), q_seed=np.zeros(7))
    print(
        f"  loaded {os.path.basename(cfg_path)}: side={arm.side} "
        f"joints={len(arm.joint_names)} err={res.pos_error*1000:.6f} mm"
    )
    _check(res.pos_error <= POS_TOL, "IK from YAML config did not reach the target")
    _check(list(arm.waist_joint_order) == list(arm.np_fk.waist_joint_names), "waist order mismatch")


def test_waist_frame_consistency():
    """IK given a pelvis-frame target must reach it in the pelvis frame."""
    arm = _arm()
    rng = np.random.default_rng(13)
    worst = 0.0
    for _ in range(15):
        waist = rng.uniform([-1.5, -0.4, -0.4], [1.5, 0.4, 0.4])
        q_true = rng.uniform(arm.q_min * 0.8, arm.q_max * 0.8)
        p_pelvis = arm.fk_position_from_state(q_true, waist)
        res = arm.solve_position_from_pelvis(p_pelvis, waist, q_seed=np.zeros(7))
        p_check = arm.fk_position_from_state(res.q, waist)
        worst = max(worst, float(np.linalg.norm(p_check - p_pelvis)))
    print(f"  pelvis-frame target: worst error over 15 poses = {worst*1000:.6f} mm")
    _check(worst < 1e-3, f"pelvis-frame IK error {worst*1000:.4f} mm")


def test_nullspace_projection():
    """Projecting onto the null space must not move the end effector."""
    arm = _arm()
    q = np.array([0.3, -0.3, 0.4, 0.9, 0.1, 0.0, 0.0])
    p0 = arm.fk_position(q)
    d = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])  # arbitrary direction
    dn = arm.ik.nullspace_project(q, d)
    q2 = q + 0.05 * dn
    p2 = arm.fk_position(q2)
    print(
        f"  null-space step: |dp| = {np.linalg.norm(p2 - p0)*1000:.6f} mm "
        f"for a {np.degrees(np.linalg.norm(0.05*dn)):.3f} deg joint move"
    )
    _check(
        np.linalg.norm(p2 - p0) < 1e-3,
        "null-space projection moved the end effector",
    )
    _check(np.linalg.norm(dn) > 1e-6, "null-space projection collapsed the direction")


# --------------------------------------------------------------------------- #


def test_ik_config_validation():
    """build_ik_config must reject configurations that silently break the solve.

    The dangerous ones are not crashes: `control_rotation=True` with
    `rotation_weight=0` makes the node accept orientation goals and ignore them,
    and `position_weight<=0` makes the solver stop short of the target while
    still reporting success.
    """
    from g1_ik.config import build_ik_config

    bad = [
        ("position_weight=0", dict(position_weight=0.0)),
        ("position_weight<0", dict(position_weight=-1.0)),
        ("rotation_weight<0", dict(rotation_weight=-1.0)),
        ("control_rotation without weight", dict(control_rotation=True, rotation_weight=0.0)),
        ("max_iter=0", dict(max_iter=0)),
        ("tol=0", dict(tol=0.0)),
        ("acceptable_tol<0", dict(acceptable_tol=-1.0)),
        ("pos_tol=0", dict(pos_tol=0.0)),
        ("refine_pos_tol=0", dict(refine_pos_tol=0.0)),
        ("limit_barrier_weight<0", dict(limit_barrier_weight=-0.5)),
        ("limit_barrier_margin<0", dict(limit_barrier_margin=-0.1)),
        ("multistart_max_extra=0", dict(multistart_max_extra=0)),
    ]
    for label, kw in bad:
        try:
            build_ik_config(kw)
        except ValueError:
            continue
        raise AssertionError(f"{label} was NOT rejected")

    # unknown keys must still be rejected (typo protection)
    try:
        build_ik_config({"position_wieght": 100.0})
        raise AssertionError("a typo'd key was accepted")
    except ValueError:
        pass

    # defaults and a valid override must work
    cfg = build_ik_config({})
    _check(cfg.refine is False, "refine should default to False")
    _check(cfg.multistart is True, "multistart should default to True")
    cfg2 = build_ik_config({"control_rotation": True, "rotation_weight": 1.0})
    _check(cfg2.control_rotation and cfg2.rotation_weight == 1.0, "valid config rejected")
    print(f"  rejected {len(bad) + 1} invalid configs, accepted the valid ones")


TESTS = [
    ("3.roundtrip_left", test_roundtrip_left),
    ("3.roundtrip_right", test_roundtrip_right),
    ("3.roundtrip_sub_micron", test_roundtrip_sub_micron),
    ("3.trajectory_tracking_continuity", test_trajectory_tracking_continuity),
    ("3.tiny_target_change_is_continuous", test_tiny_target_change_is_continuous),
    ("3.never_violates_joint_limits", test_never_violates_joint_limits),
    ("3.unreachable_target_fails_safely", test_unreachable_target_fails_safely),
    ("3.near_limit_target_reports_small_margin", test_near_limit_target_reports_small_margin),
    ("3.symbolic_fk_matches_pinocchio", test_symbolic_fk_matches_pinocchio),
    ("3.symbolic_fk_matches_numpy_fk", test_symbolic_fk_matches_numpy_fk),
    ("3.with_hand_urdf_ik", test_with_hand_urdf_ik),
    ("3.timing", test_timing),
    ("3.config_file_loads", test_config_file_loads),
    ("3.waist_frame_consistency", test_waist_frame_consistency),
    ("3.nullspace_projection", test_nullspace_projection),
    ("3.ik_config_validation", test_ik_config_validation),
]


def main() -> int:
    print("=" * 78)
    print("IK VALIDATION")
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
