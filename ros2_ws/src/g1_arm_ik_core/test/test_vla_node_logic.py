"""Tests for the VLA node's decision logic compared against a replica.

rclpy cannot be imported on this machine, so the tests cannot instantiate the
node. They can, however, pin down the two rules in the node that would be
dangerous to get wrong, by running the *same* algorithm over a table of cases:

  1. a failed solve must never replace the pose that is being commanded
  2. velocity pass-through must expire after the timeout, not latch forever

`AlgoReplica` below is a line-by-line copy of the corresponding node methods
(`_set_target` / `_current_velocity` state machine). If someone changes the node
without changing this file, `test_replica_still_matches_node` fails -- that is
the point: it keeps the two in sync instead of letting the tests drift into
testing nothing.

Run:  python test_vla_node_logic.py
"""

from __future__ import annotations

import ast
import os
import pathlib
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT = os.path.dirname(HERE)          # .../g1_arm_ik_core
SRC_DIR = os.path.dirname(PKG_ROOT)       # .../src  (sibling packages live here)
sys.path.insert(0, PKG_ROOT)

from g1_arm_ik_core.vla_protocol import VelocityCommand  # noqa: E402


def _check(cond, msg):
    if not cond:
        raise AssertionError(msg)


# --------------------------------------------------------------------------- #
#  replica of the node's state machine
# --------------------------------------------------------------------------- #


class AlgoReplica:
    """Mirrors g1_arm_vla_node's target/velocity bookkeeping."""

    def __init__(self, velocity_timeout: float = 0.3, timestamp_source: str = "monotonic"):
        self.velocity_timeout = velocity_timeout
        self.timestamp_source = timestamp_source
        self.last_result = None   # last ATTEMPT
        self.last_success = None  # last CONVERGED pose (what gets sent)
        self.velocity = VelocityCommand()
        self.velocity_time = None

    def set_target(self, res):
        """Mirrors _set_target: only a converged solve becomes the target."""
        self.last_result = res
        if res.success:
            self.last_success = res

    def commanded_q(self):
        """Mirrors _on_tick's source of truth."""
        return None if self.last_success is None else self.last_success.q

    def on_velocity(self, msg_vx, msg_vy, msg_wz, now):
        self.velocity = VelocityCommand(vx=msg_vx, vy=msg_vy, wz=msg_wz)
        self.velocity_time = now

    def frame_timestamp(self, ros_now: float = 0.0, mono_now: float = 0.0) -> float:
        """Mirrors _frame_timestamp."""
        if self.timestamp_source == "ros":
            return ros_now
        return mono_now

    def current_velocity(self, now):
        """Mirrors _current_velocity: pass through, or zeros once stale."""
        if self.velocity_time is None:
            return VelocityCommand()
        if self.velocity_timeout > 0.0 and (now - self.velocity_time) > self.velocity_timeout:
            return VelocityCommand()
        return self.velocity


class FakeResult:
    def __init__(self, success, q, pos_error=0.0, status="Solve_Succeeded"):
        self.success = success
        self.q = np.asarray(q, dtype=float)
        self.pos_error = pos_error
        self.status = status


# --------------------------------------------------------------------------- #
#  tests
# --------------------------------------------------------------------------- #


def test_failed_solve_keeps_the_commanded_pose():
    """A failed solve must not lose the target we are already commanding.

    This is the dangerous case: an operator asks for an unreachable point while
    the arm is holding a good pose. The arm must keep holding, not stop.
    """
    rep = AlgoReplica()
    _check(rep.commanded_q() is None, "nothing commanded before the first solve")

    good = FakeResult(True, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    rep.set_target(good)
    _check(np.allclose(rep.commanded_q(), good.q), "good pose not adopted")
    held = rep.commanded_q().copy()

    # now a failing attempt
    bad = FakeResult(False, [9, 9, 9, 9, 9, 9, 9], pos_error=2.5, status="Maximum_Iterations_Exceeded")
    rep.set_target(bad)
    _check(
        np.allclose(rep.commanded_q(), held),
        f"a failed solve replaced the commanded pose: {rep.commanded_q()}",
    )
    _check(rep.last_result is bad, "the failed attempt was not recorded for diagnostics")
    print(f"  failed solve kept the commanded pose {np.round(held, 2).tolist()}")

    # and a later good solve still takes effect
    good2 = FakeResult(True, [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9])
    rep.set_target(good2)
    _check(np.allclose(rep.commanded_q(), good2.q), "recovery after failure did not work")
    print("  recovery after a failure works")


def test_never_commands_without_any_success():
    """Before the first converged solve, nothing may be commanded."""
    rep = AlgoReplica()
    for _ in range(5):
        rep.set_target(FakeResult(False, np.zeros(7), pos_error=1.0))
    _check(rep.commanded_q() is None, "commanded a pose with no successful solve")
    print("  5 failed solves, still nothing commanded")


def test_velocity_is_passed_through_then_expires():
    """Pass-through must expire: a latched velocity would drive the legs forever."""
    rep = AlgoReplica(velocity_timeout=0.3)
    _check(rep.current_velocity(0.0) == VelocityCommand(), "velocity before any message")

    rep.on_velocity(0.4, -0.2, 0.1, now=100.0)
    v = rep.current_velocity(100.0)
    _check((v.vx, v.vy, v.wz) == (0.4, -0.2, 0.1), f"not passed through: {v}")

    # still fresh just under the timeout
    v = rep.current_velocity(100.29)
    _check(v.vx == 0.4, f"expired too early: {v}")

    # expired beyond it -> zeros, which makes the on-board policy balance
    v = rep.current_velocity(100.31)
    _check(v == VelocityCommand(), f"stale velocity latched: {v}")

    # a new message revives it
    rep.on_velocity(0.1, 0.0, 0.0, now=101.0)
    _check(rep.current_velocity(101.05).vx == 0.1, "did not recover after refresh")
    print("  velocity passes through, expires at 0.3 s, recovers on refresh")


def test_velocity_timeout_zero_means_never_expire():
    """Documented edge case: timeout 0 disables the expiry."""
    rep = AlgoReplica(velocity_timeout=0.0)
    rep.on_velocity(0.5, 0.0, 0.0, now=1.0)
    v = rep.current_velocity(1e6)
    _check(v.vx == 0.5, f"timeout=0 should never expire, got {v}")
    print("  timeout=0 keeps the last velocity (documented behaviour)")


def test_replica_still_matches_node():
    """Guard against the replica drifting away from the real node.

    Checks that the node still contains the two constructs this file mirrors. If
    the node is refactored, this fails and forces the tests to be updated rather
    than silently testing dead logic.
    """
    node_path = os.path.join(
        SRC_DIR, "g1_arm_ik_node", "g1_arm_ik_node", "vla_node.py"
    )
    _check(os.path.exists(node_path), f"node not found: {node_path}")
    src = pathlib.Path(node_path).read_text()
    tree = ast.parse(src)

    # 1. _on_tick must read the CONVERGED pose, not the last attempt
    tick = next(
        (n for n in ast.walk(tree)
         if isinstance(n, ast.FunctionDef) and n.name == "_on_tick"),
        None,
    )
    _check(tick is not None, "_on_tick disappeared")
    tick_src = ast.get_source_segment(src, tick)
    _check(
        "self._last_success" in tick_src,
        "_on_tick no longer reads _last_success; a failed solve could blank out "
        "the commanded pose",
    )
    _check(
        "self._last_result" not in tick_src,
        "_on_tick reads _last_result (the last ATTEMPT) instead of the last "
        "converged pose",
    )

    # 2. _set_target must only adopt a converged result
    st = next(
        (n for n in ast.walk(tree)
         if isinstance(n, ast.FunctionDef) and n.name == "_set_target"),
        None,
    )
    _check(st is not None, "_set_target disappeared")
    st_src = ast.get_source_segment(src, st)
    _check(
        "if res.success:" in st_src and "_last_success = res" in st_src,
        "_set_target no longer gates adoption on res.success",
    )

    # 3. the velocity pass-through must still expire
    cv = next(
        (n for n in ast.walk(tree)
         if isinstance(n, ast.FunctionDef) and n.name == "_current_velocity"),
        None,
    )
    _check(cv is not None, "_current_velocity disappeared")
    cv_src = ast.get_source_segment(src, cv)
    _check(
        "velocity_timeout" in cv_src and "VelocityCommand()" in cv_src,
        "_current_velocity no longer falls back to zeros when stale",
    )
    # 4. the frame timestamp must come from the monotonic clock by default, and
    #    _on_tick must go through the helper rather than the ROS clock
    ft = next(
        (n for n in ast.walk(tree)
         if isinstance(n, ast.FunctionDef) and n.name == "_frame_timestamp"),
        None,
    )
    _check(ft is not None, "_frame_timestamp disappeared")
    ft_src = ast.get_source_segment(src, ft)
    _check(
        "time.monotonic()" in ft_src,
        "_frame_timestamp no longer defaults to a monotonic clock",
    )
    _check(
        "self._frame_timestamp()" in tick_src,
        "_on_tick no longer uses _frame_timestamp; it may be back on the ROS clock",
    )
    print("  node still contains the four constructs this file mirrors")



def test_node_decision_chain_with_real_ik():
    """Node logic driven by the real solver, through to an accepted frame.

    The replica above is fed hand-made `FakeResult`s. This drives it with the
    actual IK solver and the actual frame builder, then pushes the result
    through the mock receiver. So it covers the seam between three modules that
    are otherwise tested separately: solver -> node decision -> wire format.
    """
    import tempfile

    from g1_ik import G1Arm
    from g1_arm_ik_core.vla_protocol import make_frame

    urdf = os.path.join(
        SRC_DIR, "g1_arm_bringup", "urdf", "g1_29dof_rev_1_0.urdf"
    )
    _check(os.path.exists(urdf), f"URDF not found: {urdf}")

    # import the receiver from the protocol suite (single source of the mock)
    sys.path.insert(0, HERE)
    from test_vla_protocol import MockGrootReceiver

    side = "left"
    arm = G1Arm(
        urdf, side=side,
        reduced_urdf_path=os.path.join(tempfile.mkdtemp(), "r.urdf"),
    )
    rep = AlgoReplica(velocity_timeout=0.3)
    recv = MockGrootReceiver()
    q_other = np.array([0.05] * 7)   # the free arm, held at its measured pose

    def tick(now):
        """Mirror _on_tick: build, validate and 'send' one frame."""
        res = rep.last_success
        if res is None:
            return None, "no converged pose yet"
        frame = make_frame(side, res.q, q_other, rep.current_velocity(now))
        return frame.to_payload(frame.next_timestamp(now)), None

    # before any solve: nothing to send
    payload, why = tick(1.0)
    _check(payload is None, "sent a frame with no converged solve")

    # solve for a reachable point
    p_target = arm.fk_position(np.array([0.30, -0.25, 0.40, 0.90, 0.10, 0.0, 0.0]))
    rep.on_velocity(vx := 0.25, -0.10, 0.05, now=1.0)
    rep.set_target(arm.solve_position(p_target))

    payload, why = tick(1.05)
    _check(payload is not None, f"no frame after a good solve: {why}")
    ok, info = recv.parse(payload)
    _check(ok, f"real frame rejected by the mock receiver: {info}")
    _check(
        np.allclose(info["arm_q"][:7], rep.last_success.q, atol=1e-12),
        "solved pose did not survive into the frame",
    )
    _check(
        np.allclose(info["arm_q"][7:], q_other),
        "the free arm was not held at its measured pose",
    )
    vx_out, vy_out, wz_out = info["velocity"]
    _check(abs(vx_out - 0.25) < 1e-9, f"vx pass-through broken: {vx_out}")
    print(f"  real IK -> node -> frame -> receiver: accepted, err "
          f"{rep.last_success.pos_error * 1000:.5f} mm")

    # an unreachable target must NOT replace the commanded pose
    held = rep.last_success.q.copy()
    rep.set_target(arm.solve_position(np.array([3.0, 0.0, 0.0])))
    _check(not rep.last_result.success, "3 m away should not be reachable")
    _check(np.allclose(rep.commanded_q(), held), "a failed solve changed the command")
    payload2, _ = tick(1.10)
    ok2, info2 = recv.parse(payload2)
    _check(ok2, f"held frame rejected: {info2}")
    _check(np.allclose(info2["arm_q"][:7], held), "held frame does not carry the held pose")
    print("  unreachable target: command held, frame still valid")

    # velocity expiry flows through to the wire as zeros
    payload3, _ = tick(5.0)  # way past the 0.3 s timeout
    ok3, info3 = recv.parse(payload3)
    _check(ok3, f"expired-velocity frame rejected: {info3}")
    _check(
        all(abs(v) < 1e-12 for v in info3["velocity"]),
        f"stale velocity did not become zeros: {info3['velocity']}",
    )
    print("  velocity expired to zeros on the wire")


def test_frame_timestamp_is_a_monotonic_gate():
    """The frame timestamp only has to strictly increase -- nothing else.

    Read from the C++ receiver: the timestamp is used for exactly one check
    (`timestamp <= previous->timestamp` -> "stale timestamp"), while freshness
    comes from the receiver's own arrival time. So the value may be any
    monotonic sequence, and the node must never let it step backwards.
    """
    rep = AlgoReplica(timestamp_source="monotonic")

    # a monotonic source that stands still (coarse clock) must be repaired
    stamps = []
    last = None
    for i in range(6):
        ts = rep.frame_timestamp(mono_now=42.0)  # frozen clock
        if last is not None and ts <= last:
            ts = last + 1e-6
        stamps.append(ts)
        last = ts
    _check(all(b > a for a, b in zip(stamps, stamps[1:])), f"not increasing: {stamps}")

    # the ROS clock stepping BACKWARDS is what we are defending against
    ros_backwards = [100.0, 99.5, 98.0]
    rep_ros = AlgoReplica(timestamp_source="ros")
    got = [rep_ros.frame_timestamp(ros_now=t) for t in ros_backwards]
    _check(got == ros_backwards, "ros source should pass the value through unchanged")
    _check(not all(b > a for a, b in zip(got, got[1:])), "ros clock did step backwards")

    # ...and the monotonic source is immune to it
    rep_mono = AlgoReplica(timestamp_source="monotonic")
    real_ros = [100.0, 99.5, 98.0]
    real_mono = [10.0, 10.02, 10.04]
    got2 = [rep_mono.frame_timestamp(ros_now=r, mono_now=m)
            for r, m in zip(real_ros, real_mono)]
    _check(all(b > a for a, b in zip(got2, got2[1:])),
           f"monotonic source followed the ROS clock backwards: {got2}")
    print(f"  monotonic immune to a backwards ROS clock: {got2}")
    print(f"  frozen clock repaired: {stamps}")


TESTS = [
    ("4.failed_solve_keeps_pose", test_failed_solve_keeps_the_commanded_pose),
    ("4.no_command_without_success", test_never_commands_without_any_success),
    ("4.velocity_passthrough_expires", test_velocity_is_passed_through_then_expires),
    ("4.velocity_timeout_zero", test_velocity_timeout_zero_means_never_expire),
    ("4.replica_matches_node", test_replica_still_matches_node),
    ("4.real_ik_through_node_logic", test_node_decision_chain_with_real_ik),
    ("4.frame_timestamp_monotonic", test_frame_timestamp_is_a_monotonic_gate),
]


def main() -> int:
    print("=" * 78)
    print("VLA NODE LOGIC VALIDATION (no ROS, no robot)")
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
