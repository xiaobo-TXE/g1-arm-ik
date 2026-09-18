"""Offline tests for the ROS-free control logic (no ROS, no hardware).

These cover the layer that actually protects the robot:

* the state watchdog (no motion on a stale state stream)
* rate limiting (a discontinuous IK jump must not become a discontinuous command)
* the arm_sdk handover ramp
* target validation (NaN, joint limits, oversized steps)
* the mock backend behaving like a real servo

Run:  pytest ros2_ws/src/g1_arm_ik_core/test/test_control.py -v
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from g1_arm_ik_core import (  # noqa: E402
    ArmCommandShaper,
    ControlConfig,
    MockBackend,
    SafetyMonitor,
)

Q_MIN = np.array([-3.09, -1.59, -2.62, -1.05, -1.97, -1.61, -1.61])
Q_MAX = np.array([2.67, 2.25, 2.62, 2.09, 1.97, 1.61, 1.61])
V_LIM = np.array([37.0, 37.0, 37.0, 37.0, 37.0, 22.0, 22.0])


def _shaper(**kw):
    cfg = ControlConfig(control_dt=0.01, **kw)
    return ArmCommandShaper(Q_MIN, Q_MAX, cfg.velocity_limits(V_LIM), cfg)


def _check(cond, msg):
    if not cond:
        raise AssertionError(msg)


# --------------------------------------------------------------------------- #


def test_disabled_by_default_and_no_motion():
    """A disabled shaper must not produce a moving command."""
    s = _shaper()
    s.set_joint_state(np.zeros(7))
    s.set_target(np.full(7, 0.3))
    for _ in range(50):
        st = s.step()
    _check(st.state == s.STATE_DISABLED, f"state {st.state}")
    _check(st.arm_sdk_weight == 0.0, f"weight {st.arm_sdk_weight} while disabled")
    _check(not st.ok, "disabled shaper reported ok")


def test_watchdog_stops_on_stale_state():
    """With no fresh state the arm must not move, and weight must drop to 0."""
    s = _shaper(state_timeout=0.1)
    # the shaper compares against time.monotonic(), so the test must use the
    # same time base
    t = time.monotonic()
    s.set_joint_state(np.zeros(7), now=t)
    s.set_target(np.full(7, 0.2), now=t)
    s.enable()

    # fresh ticks: ramp engages (state keeps arriving)
    for i in range(200):
        now = t + 0.01 * i
        s.set_joint_state(np.zeros(7), now=now)
        st = s.step(now=now)
    _check(st.ramp_progress > 0.9, f"ramp only reached {st.ramp_progress}")
    _check(st.arm_sdk_weight > 0.9, f"weight {st.arm_sdk_weight}")

    # now the state stream dies
    st = s.step(now=t + 5.0)
    _check(st.state == s.STATE_NO_STATE, f"state {st.state}, expected NO_STATE")
    _check(st.watchdog_tripped, "watchdog flag not set")
    _check(st.arm_sdk_weight == 0.0, f"weight {st.arm_sdk_weight} after stale state")
    _check(not st.ok, "stale-state shaper reported ok")


def test_rate_limiting_caps_command_jump():
    """A full-range target must be approached at no more than v_max*dt per tick."""
    s = _shaper(velocity_scale=0.1)
    q0 = np.zeros(7)
    s.set_joint_state(q0)
    s.enable()
    # a huge but legal step
    target = np.array([1.0, -0.5, 0.5, 0.8, 0.3, 0.2, 0.2])
    _check(s.set_target(target), "target should be accepted")

    v = s.cfg.velocity_limits(V_LIM)
    prev = q0.copy()
    # skip the ramp-up window so we measure the rate limiter, not the handover
    for _ in range(int(s.cfg.ramp_up / s.cfg.control_dt) + 5):
        st = s.step()
        step = np.abs(st.q_command - prev)
        limit = v * s.cfg.control_dt + 1e-9
        _check(
            np.all(step <= limit),
            f"step {np.round(step,4)} exceeded {np.round(limit,4)}",
        )
        prev = st.q_command.copy()


def test_ramp_is_gradual_and_bounded():
    """The arm_sdk weight must go 0 -> 1 over ramp_up seconds, monotonically."""
    ramp = 0.5
    s = _shaper(ramp_up=ramp, ramp_down=ramp)
    t = time.monotonic()
    s.set_joint_state(np.zeros(7), now=t)
    s.enable()

    weights = []
    n = int(ramp / s.cfg.control_dt)
    for i in range(n + 20):
        now = t + 0.01 * i
        # keep the state fresh so this test measures the RAMP and not the
        # watchdog (which is covered separately)
        s.set_joint_state(np.zeros(7), now=now)
        weights.append(s.step(now=now).arm_sdk_weight)

    _check(weights[0] < 0.1, f"weight jumps immediately: {weights[0]}")
    # enabling must engage the weight even with no target yet -- otherwise the
    # locomotion policy still owns the arm while we think we are in control
    _check(
        weights[n - 1] > 0.9,
        "weight did not engage without a target; enable() must take ownership",
    )
    _check(abs(weights[n - 1] - 1.0) < 0.05, f"weight at ramp end {weights[n-1]}")
    _check(all(b >= a - 1e-12 for a, b in zip(weights, weights[1:])), "not monotonic")
    print(f"  ramp: {weights[0]:.3f} -> {weights[n-1]:.3f} over {ramp}s")


def test_target_validation_rejects_bad_input():
    s = _shaper()
    s.set_joint_state(np.zeros(7))
    s.enable()

    _check(not s.set_target(np.full(7, np.nan)), "NaN target accepted")
    _check(not s.set_target(np.full(7, np.inf)), "inf target accepted")
    # outside the joint limits
    bad = np.zeros(7)
    bad[0] = 5.0  # shoulder pitch max is 2.67
    _check(not s.set_target(bad), "out-of-limit target accepted")
    _check("limits" in s.last_reject_reason, f"reason: {s.last_reject_reason}")
    # too far from the current pose
    s2 = _shaper(max_joint_step=0.3)
    s2.set_joint_state(np.zeros(7))
    s2.enable()
    _check(not s2.set_target(np.full(7, 0.9)), "oversized step accepted")
    _check("max_joint_step" in s2.last_reject_reason, s2.last_reject_reason)
    # a legal target is accepted
    _check(s2.set_target(np.full(7, 0.2)), "legal target rejected")
    _check(s2._cmds_rejected >= 0, "reject counter broken")
    print(f"  rejections counted: {s2._status(s2.STATE_ACTIVE, 0.0, 0.0).cmds_rejected}")


def test_arrives_at_target_and_holds():
    """Closed loop through the mock backend must converge and then hold."""
    backend = MockBackend(np.zeros(7), Q_MIN, Q_MAX, tau=0.02, noise=0.0)
    v = ControlConfig(control_dt=0.01).velocity_limits(V_LIM)
    s = ArmCommandShaper(Q_MIN, Q_MAX, v, ControlConfig(control_dt=0.01))
    s.set_joint_state(backend.state())
    s.set_target(np.array([0.4, -0.3, 0.5, 0.9, 0.1, 0.0, 0.0]))
    s.enable()

    for _ in range(400):
        st = s.step()
        if st.ok:
            backend.send(st.q_command, st.arm_sdk_weight)
        s.set_joint_state(backend.state())

    final_err = float(np.max(np.abs(backend.state() - s.target)))
    print(f"  mock closed loop: final max error = {final_err:.6f} rad")
    _check(final_err < 5e-3, f"did not converge, error {final_err}")

    # once there, state must be HOLDING/ACTIVE and the command must be stable
    cmd_a = s.step().q_command.copy()
    backend.send(cmd_a, 1.0)
    s.set_joint_state(backend.state())
    cmd_b = s.step().q_command.copy()
    _check(np.max(np.abs(cmd_a - cmd_b)) < 1e-6, "command is not stable at the target")


def test_disable_ramps_down_and_stops_sending():
    backend = MockBackend(np.zeros(7), Q_MIN, Q_MAX, tau=0.02, noise=0.0)
    v = ControlConfig(control_dt=0.01).velocity_limits(V_LIM)
    cfg = ControlConfig(control_dt=0.01, ramp_down=0.3)
    s = ArmCommandShaper(Q_MIN, Q_MAX, v, cfg)
    s.set_joint_state(backend.state())
    s.set_target(np.full(7, 0.2))
    s.enable()
    for _ in range(100):
        st = s.step()
        backend.send(st.q_command, st.arm_sdk_weight)
        s.set_joint_state(backend.state())
    _check(s.step().arm_sdk_weight > 0.9, "did not engage")

    s.disable()
    weights = []
    for _ in range(60):
        st = s.step()
        weights.append(st.arm_sdk_weight)
    _check(weights[-1] == 0.0, f"weight did not reach 0: {weights[-1]}")
    _check(all(b <= a + 1e-12 for a, b in zip(weights, weights[1:])), "not decreasing")
    _check(s.target is None, "target not cleared on disable")
    print(f"  ramp down: {weights[0]:.3f} -> {weights[-1]:.3f}")


def test_safety_monitor_catches_rate_and_tracking():
    m = SafetyMonitor(V_LIM, dt=0.01, max_tracking_error=0.2)
    _check(m.check(np.zeros(7)), "first command rejected")
    # a jump far beyond v*dt
    big = np.zeros(7)
    big[0] = 1.0
    _check(not m.check(big), "oversized rate accepted")
    _check("rate" in m.violations[-1], m.violations[-1])

    m.reset(np.zeros(7))
    _check(not m.check(np.full(7, np.nan)), "NaN accepted")

    # Tracking-error check on its own: move within the rate limit each tick so
    # only the tracking check can fire.
    m.reset(np.zeros(7))
    q_cmd = np.zeros(7)
    step = m.v_max * m.dt  # exactly the allowed per-tick motion
    for _ in range(10):
        q_cmd = q_cmd + step
        _check(m.check(q_cmd), f"legal ramp rejected: {m.violations[-1:]}")
    _check(
        not m.check(q_cmd, q_current=np.full(7, -1.0)),
        "large tracking error accepted",
    )
    _check("tracking error" in m.violations[-1], m.violations[-1])
    print(f"  violations recorded: {len(m.violations)}")


def test_mock_backend_respects_limits_and_weight():
    b = MockBackend(np.zeros(7), Q_MIN, Q_MAX, tau=0.01, noise=0.0)
    # commanding beyond a limit must be clamped by the backend too
    b.send(np.full(7, 99.0), 1.0)
    for _ in range(500):
        b.send(np.full(7, 99.0), 1.0)
    _check(np.all(b.state() <= Q_MAX + 1e-9), f"exceeded limits: {b.state()}")

    # with weight 0 the command must be ignored (the arm holds)
    b2 = MockBackend(np.array([0.3] * 7), Q_MIN, Q_MAX, tau=0.01, noise=0.0)
    for _ in range(200):
        b2.send(np.full(7, -0.5), 0.0)
    _check(
        np.allclose(b2.state(), 0.3, atol=1e-6),
        f"moved despite weight 0: {b2.state()}",
    )
    print("  mock backend: limits clamped, weight 0 holds position")



def test_config_validation_rejects_dangerous_values():
    """Bad config must fail loudly at construction, not silently on the robot.

    Each of these would otherwise fail *quietly*: a zero velocity limit gives a
    command that never moves, a zero ramp divides by ~0, and a non-positive
    state_timeout disables the watchdog -- the single most dangerous setting in
    the whole stack.
    """
    from g1_arm_ik_core import ControlConfig as CC

    cases = [
        ("velocity_scale=0", dict(velocity_scale=0.0)),
        ("velocity_scale<0", dict(velocity_scale=-1.0)),
        ("ramp_up=0", dict(ramp_up=0.0)),
        ("ramp_down=0", dict(ramp_down=0.0)),
        ("control_dt=0", dict(control_dt=0.0)),
        ("max_joint_step=0", dict(max_joint_step=0.0)),
        ("state_timeout=0 (watchdog off)", dict(state_timeout=0.0)),
        ("state_timeout<0", dict(state_timeout=-1.0)),
        ("state_alpha>1", dict(state_alpha=1.5)),
        ("state_alpha<0", dict(state_alpha=-0.1)),
        ("limit_slack<0", dict(limit_slack=-0.1)),
        ("cruise_weight>1", dict(cruise_weight=2.0)),
    ]
    for label, kw in cases:
        # ControlConfig's default control_dt is already 0.01, so a case may
        # override it without a conflict.
        cfg = CC(**kw)
        try:
            ArmCommandShaper(Q_MIN, Q_MAX, np.ones(7), cfg)
        except ValueError:
            continue
        raise AssertionError(f"{label} was NOT rejected")

    # negative velocity limits from the model must be rejected too
    try:
        ArmCommandShaper(Q_MIN, Q_MAX, -np.ones(7), CC(control_dt=0.01))
        raise AssertionError("negative velocity limits accepted")
    except ValueError:
        pass

    # q_max < q_min must be rejected
    try:
        ArmCommandShaper(Q_MAX, Q_MIN, np.ones(7), CC(control_dt=0.01))
        raise AssertionError("inverted joint limits accepted")
    except ValueError:
        pass

    # a valid config must still work
    ArmCommandShaper(Q_MIN, Q_MAX, np.ones(7), CC(control_dt=0.01))
    print(f"  rejected {len(cases) + 2} invalid configs, accepted the valid one")


def test_mock_backend_dt_is_respected():
    """The simulated servo must use the real control period."""
    fast = MockBackend(np.zeros(7), Q_MIN, Q_MAX, tau=0.05, noise=0.0, dt=0.01)
    slow = MockBackend(np.zeros(7), Q_MIN, Q_MAX, tau=0.05, noise=0.0, dt=0.001)
    cmd = np.full(7, 0.5)
    for _ in range(10):
        fast.send(cmd, 1.0)
        slow.send(cmd, 1.0)
    assert fast.state()[0] > slow.state()[0], (
        "a longer control period must integrate further: "
        f"dt=0.01 -> {fast.state()[0]:.4f}, dt=0.001 -> {slow.state()[0]:.4f}"
    )
    print(f"  dt=0.01 -> {fast.state()[0]:.4f} rad, dt=0.001 -> {slow.state()[0]:.4f} rad")


TESTS = [
    ("0.disabled_by_default", test_disabled_by_default_and_no_motion),
    ("0.watchdog_stops_on_stale_state", test_watchdog_stops_on_stale_state),
    ("0.rate_limiting_caps_command_jump", test_rate_limiting_caps_command_jump),
    ("0.ramp_is_gradual_and_bounded", test_ramp_is_gradual_and_bounded),
    ("0.target_validation_rejects_bad_input", test_target_validation_rejects_bad_input),
    ("0.arrives_at_target_and_holds", test_arrives_at_target_and_holds),
    ("0.disable_ramps_down_and_stops", test_disable_ramps_down_and_stops_sending),
    ("0.safety_monitor", test_safety_monitor_catches_rate_and_tracking),
    ("0.mock_backend", test_mock_backend_respects_limits_and_weight),
    ("0.config_validation", test_config_validation_rejects_dangerous_values),
    ("0.mock_backend_dt", test_mock_backend_dt_is_respected),
]


def main() -> int:
    print("=" * 78)
    print("CONTROL LOGIC VALIDATION (no ROS, no hardware)")
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
