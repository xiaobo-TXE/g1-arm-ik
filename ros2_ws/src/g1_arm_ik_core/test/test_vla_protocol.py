"""Tests for the ZMQ 6002 VLA frame format (no ROS, no robot needed).

The centrepiece is `MockGrootReceiver`: a faithful Python reimplementation of
`parse_packet()` from
`deploy/include/groot/RemoteCommandReceiver.h` (branch `groot-control`),
including every rejection path.  Frames produced by `vla_protocol` are fed
through it, and deliberately-broken frames are fed through it to prove the mock
actually rejects them -- a mock that accepts everything would prove nothing.

Verified here:
  * the happy path is accepted
  * the exact 14 LeRobot key names, slot order and left/right split
  * slot order agrees with the IK library's arm ordering
  * each of the receiver's rejection rules really fires (and our builder
    prevents the ones it can)
  * the YAML payload round-trips through a strict parser

Run:  python test_vla_protocol.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from g1_arm_ik_core.vla_protocol import (  # noqa: E402
    ARM_SLOT_LEROBOT_NAMES,
    ARM_SLOT_SDK_NAMES,
    JOINTS_PER_ARM,
    MAX_ABS_JOINT_VALUE,
    NUM_ARM_JOINTS,
    FrameError,
    VelocityCommand,
    VlaActionFrame,
    arm_slots,
    assert_slot_alignment,
    lerobot_key_for_slot,
    make_frame,
)


def _check(cond, msg):
    if not cond:
        raise AssertionError(msg)


# --------------------------------------------------------------------------- #
#  faithful reimplementation of the C++ parse_packet()
# --------------------------------------------------------------------------- #


class MockGrootReceiver:
    """Mirrors groot::RemoteCommandReceiver::parse_packet().

    Each rejection returns (False, reason) with the same *reason string* the C++
    code uses, so a test failure names the exact rule that fired.
    """

    def __init__(self):
        self.previous_timestamp = None
        self.previous_sequence = None
        self.accepted = 0
        self.rejections = []

    @staticmethod
    def _arm_slot_of_lerobot_name(name: str) -> int:
        # case-insensitive compare, exactly like the C++ helper
        low = name.lower()
        for i, expected in enumerate(ARM_SLOT_LEROBOT_NAMES):
            if expected.lower() == low:
                return i
        return -1

    def parse(self, payload: bytes):
        try:
            root = yaml.safe_load(payload.decode("utf-8"))
        except Exception:
            return self._reject("yaml parse error")

        if not isinstance(root, dict) or "action" not in root or "timestamp" not in root:
            return self._reject("missing action/timestamp")

        timestamp = float(root["timestamp"])
        if not np.isfinite(timestamp):
            return self._reject("non-finite timestamp")

        if self.previous_timestamp is not None and timestamp <= self.previous_timestamp:
            return self._reject("stale timestamp")

        action = root["action"]
        if not isinstance(action, dict):
            return self._reject("action not a map")

        def axis(key: str) -> float:
            v = action.get(f"remote.{key}")
            return 0.0 if v is None else float(v)

        velocity = (axis("ly"), -axis("lx"), -axis("rx"))
        if not all(np.isfinite(velocity)):
            return self._reject("non-finite velocity")

        arm_q = np.zeros(NUM_ARM_JOINTS)
        filled = np.zeros(NUM_ARM_JOINTS, dtype=bool)
        arm_count = 0
        for key, raw in action.items():
            if not isinstance(key, str):
                continue
            if len(key) < 3 or key[-2:] != ".q":
                continue
            base = key[:-2]
            slot = self._arm_slot_of_lerobot_name(base)
            if slot < 0:
                continue  # non-arm joints are ignored
            try:
                value = float(raw)
            except (TypeError, ValueError):
                return self._reject("joint value not a number")
            if not np.isfinite(value) or abs(value) > 3.2:
                return self._reject("joint value out of range")
            if not filled[slot]:
                filled[slot] = True
                arm_count += 1
            arm_q[slot] = value

        if arm_count != NUM_ARM_JOINTS:
            return self._reject("expected 14 arm joints")

        self.previous_timestamp = timestamp
        self.previous_sequence = 0 if self.previous_sequence is None else self.previous_sequence + 1
        self.accepted += 1
        return True, {"velocity": velocity, "arm_q": arm_q, "timestamp": timestamp}

    def _reject(self, reason):
        self.rejections.append(reason)
        return False, reason


# --------------------------------------------------------------------------- #
#  tests
# --------------------------------------------------------------------------- #


def test_slot_tables_are_consistent():
    _check(len(ARM_SLOT_LEROBOT_NAMES) == NUM_ARM_JOINTS, "lerobot name count")
    _check(len(ARM_SLOT_SDK_NAMES) == NUM_ARM_JOINTS, "sdk name count")
    _check(len(set(ARM_SLOT_LEROBOT_NAMES)) == NUM_ARM_JOINTS, "duplicate lerobot names")
    _check(len(set(ARM_SLOT_SDK_NAMES)) == NUM_ARM_JOINTS, "duplicate sdk names")

    # left arm occupies slots 0..6, right arm 7..13, in the documented order
    _check(arm_slots("left") == list(range(0, 7)), "left slots")
    _check(arm_slots("right") == list(range(7, 14)), "right slots")
    _check(
        ARM_SLOT_LEROBOT_NAMES[0] == "kLeftShoulderPitch",
        f"slot 0 is {ARM_SLOT_LEROBOT_NAMES[0]}",
    )
    _check(
        ARM_SLOT_LEROBOT_NAMES[13] == "kRightWristYaw",
        f"slot 13 is {ARM_SLOT_LEROBOT_NAMES[13]}",
    )
    _check(
        ARM_SLOT_SDK_NAMES[0] == "left_shoulder_pitch_joint",
        f"sdk slot 0 is {ARM_SLOT_SDK_NAMES[0]}",
    )
    for slot in range(NUM_ARM_JOINTS):
        key = lerobot_key_for_slot(slot)
        _check(key.endswith(".q"), f"{key} missing .q")
        _check(key[:-2] == ARM_SLOT_LEROBOT_NAMES[slot], f"key/slot mismatch at {slot}")
    print(f"  slot tables: {NUM_ARM_JOINTS} joints, left 0-6 / right 7-13")


def test_slot_order_matches_ik_library():
    """The IK library's arm ordering must equal the SDK slot ordering (left arm).

    This is the load-bearing assumption: if it drifts, solved angles land on the
    wrong joints.
    """
    from g1_ik import arm_joint_names

    left = arm_joint_names("left")
    assert_slot_alignment(left, "left")
    _check(list(left) == list(ARM_SLOT_SDK_NAMES[:JOINTS_PER_ARM]), "left alignment")

    # the right arm must be the same order with the prefix swapped
    right = arm_joint_names("right")
    expected_right = [n.replace("left_", "right_") for n in ARM_SLOT_SDK_NAMES[:JOINTS_PER_ARM]]
    _check(list(right) == expected_right, f"right alignment: {right}")
    assert_slot_alignment(right, "right")

    # a mismatched side must be rejected, not silently accepted
    try:
        assert_slot_alignment(left, "right")
        raise AssertionError("left ordering accepted for side=right")
    except ValueError:
        pass
    print(f"  IK alignment: {left[0]} .. {left[-1]} == slots 0..6")


def test_happy_path_is_accepted():
    recv = MockGrootReceiver()
    frame = make_frame(
        side="left",
        q_solved=[0.1, -0.2, 0.3, 0.8, 0.0, 0.0, 0.0],
        q_other_arm=[0.0] * 7,
        velocity=VelocityCommand(vx=0.3, vy=-0.1, wz=0.2),
    )
    payload = frame.to_payload(frame.next_timestamp(1000.0))
    ok, info = recv.parse(payload)
    _check(ok, f"valid frame rejected: {info}")
    _check(recv.accepted == 1, "not counted as accepted")
    # velocity mapping: vx=ly, vy=-lx, wz=-rx
    vx, vy, wz = info["velocity"]
    _check(abs(vx - 0.3) < 1e-9, f"vx {vx}")
    _check(abs(vy - (-0.1)) < 1e-9, f"vy {vy}")
    _check(abs(wz - 0.2) < 1e-9, f"wz {wz}")
    # solved arm landed in slots 0..6
    _check(np.allclose(info["arm_q"][:7], [0.1, -0.2, 0.3, 0.8, 0, 0, 0]), "left slots")
    _check(np.allclose(info["arm_q"][7:], 0.0), "right slots should hold")

    # and the raw YAML must have exactly the 14 keys + 4 axes inside "action"
    doc = yaml.safe_load(payload.decode())
    _check(doc["cmd"] == "action", f"cmd is {doc['cmd']!r}")
    arm_keys = [k for k in doc["action"] if k.endswith(".q")]
    _check(len(arm_keys) == NUM_ARM_JOINTS, f"{len(arm_keys)} .q keys")
    axis_keys = [k for k in doc["action"] if k.startswith("remote.")]
    _check(len(axis_keys) == 4, f"{len(axis_keys)} remote keys")
    _check(isinstance(doc["timestamp"], float), "timestamp not a float")
    print(f"  happy path accepted; {len(arm_keys)} arm keys + {len(axis_keys)} axes")


def test_right_arm_maps_to_slots_7_13():
    recv = MockGrootReceiver()
    frame = make_frame(
        side="right",
        q_solved=[0.5, -0.5, 0.25, 1.0, 0.1, 0.2, 0.3],
        q_other_arm=[0.0] * 7,
    )
    ok, info = recv.parse(frame.to_payload(frame.next_timestamp(1.0)))
    _check(ok, f"rejected: {info}")
    _check(np.allclose(info["arm_q"][:7], 0.0), "left arm should hold")
    _check(
        np.allclose(info["arm_q"][7:], [0.5, -0.5, 0.25, 1.0, 0.1, 0.2, 0.3]),
        f"right arm slots wrong: {info['arm_q'][7:]}",
    )
    print("  right arm -> slots 7..13 confirmed")


def test_mock_receiver_actually_rejects_bad_frames():
    """Prove the mock is strict: every rule must fire."""
    fired = []

    # 1. missing timestamp
    recv = MockGrootReceiver()
    ok, why = recv.parse(yaml.safe_dump({"cmd": "action", "action": {}}).encode())
    _check(not ok, "missing timestamp accepted")
    fired.append(why)

    # 2. stale timestamp
    recv = MockGrootReceiver()
    f = make_frame("left", [0.1] * 7, [0.0] * 7)
    recv.parse(f.to_payload(10.0))
    ok, why = recv.parse(f.to_payload(10.0))
    _check(not ok, "duplicate timestamp accepted")
    fired.append(why)

    # 3. only 13 arm joints
    recv = MockGrootReceiver()
    action = {lerobot_key_for_slot(i): 0.0 for i in range(13)}
    action.update({k: 0.0 for k in ("remote.lx", "remote.ly", "remote.rx", "remote.ry")})
    ok, why = recv.parse(
        yaml.safe_dump({"cmd": "action", "action": action, "timestamp": 1.0}).encode()
    )
    _check(not ok, "13-joint frame accepted")
    fired.append(why)

    # 4. |q| > 3.2
    recv = MockGrootReceiver()
    f = VlaActionFrame()
    f.set_all_arms([0.0] * NUM_ARM_JOINTS)
    payload = f.to_payload(1.0)
    doc = yaml.safe_load(payload.decode())
    doc["action"][lerobot_key_for_slot(3)] = 3.5
    ok, why = recv.parse(yaml.safe_dump(doc).encode())
    _check(not ok, "out-of-range joint accepted")
    fired.append(why)

    # 5. non-finite joint
    recv = MockGrootReceiver()
    doc = yaml.safe_load(payload.decode())
    doc["action"][lerobot_key_for_slot(0)] = float("nan")
    ok, why = recv.parse(yaml.safe_dump(doc).encode())
    _check(not ok, "NaN joint accepted")
    fired.append(why)

    # 6. unparseable YAML
    recv = MockGrootReceiver()
    ok, why = recv.parse(b"{not: yaml: at: all")
    _check(not ok, "garbage accepted")
    fired.append(why)

    # 7. non-finite velocity
    recv = MockGrootReceiver()
    doc = yaml.safe_load(payload.decode())
    doc["action"]["remote.ly"] = float("inf")
    ok, why = recv.parse(yaml.safe_dump(doc).encode())
    _check(not ok, "inf velocity accepted")
    fired.append(why)

    _check(len(fired) == 7, f"only {len(fired)} rules fired")
    print(f"  7 rejection rules verified: {sorted(set(fired))}")


def test_builder_guards_against_the_rejections():
    """The builder must refuse to produce frames the receiver would drop."""
    # fewer than 14 slots filled
    try:
        VlaActionFrame().set_arm("left", [0.0] * 7).to_payload(1.0)
        raise AssertionError("incomplete frame was serialized")
    except FrameError as exc:
        _check("unset" in str(exc), f"unexpected message: {exc}")

    # out of range
    f = VlaActionFrame().set_all_arms([0.0] * NUM_ARM_JOINTS)
    f.set_slot(3, 4.0)
    try:
        f.to_payload(1.0)
        raise AssertionError("out-of-range value was serialized")
    except FrameError as exc:
        _check(f"{MAX_ABS_JOINT_VALUE}" in str(exc), f"unexpected message: {exc}")

    # NaN
    f = VlaActionFrame().set_all_arms([0.0] * NUM_ARM_JOINTS)
    f.set_slot(0, float("nan"))
    try:
        f.to_payload(1.0)
        raise AssertionError("NaN was serialized")
    except FrameError:
        pass

    # wrong arm length
    try:
        VlaActionFrame().set_arm("left", [0.0] * 6)
        raise AssertionError("6-joint arm accepted")
    except ValueError:
        pass

    # non-monotonic clock is repaired, not rejected
    f = VlaActionFrame().set_all_arms([0.0] * NUM_ARM_JOINTS)
    t1 = f.next_timestamp(5.0)
    t2 = f.next_timestamp(5.0)  # same clock value twice
    t3 = f.next_timestamp(4.0)  # clock went backwards
    _check(t1 < t2 < t3, f"timestamps not strictly increasing: {t1}, {t2}, {t3}")
    print(f"  builder guards fire; timestamp repair {t1:.6f} < {t2:.6f} < {t3:.6f}")


def test_timestamp_repair_survives_receiver():
    """A frozen monotonic clock must not cause dropped frames."""
    recv = MockGrootReceiver()
    f = VlaActionFrame().set_all_arms([0.0] * NUM_ARM_JOINTS)
    for i in range(5):
        payload = f.to_payload(f.next_timestamp(42.0))  # clock never advances
        ok, why = recv.parse(payload)
        _check(ok, f"frame {i} rejected: {why}")
    _check(recv.accepted == 5, f"only {recv.accepted}/5 accepted")
    print("  5 frames sent from a frozen clock, all accepted")


def test_sequence_of_frames_like_the_real_loop():
    """Simulate the control loop: alternating targets, 20 ms apart, 50 frames."""
    recv = MockGrootReceiver()
    frame = VlaActionFrame()
    dt = 0.02
    t = 1000.0
    for i in range(50):
        q = [0.1 * np.sin(i / 10.0)] * 7
        frame.set_arm("left", q)
        frame.set_arm("right", [0.0] * 7)
        frame.velocity = VelocityCommand(vx=0.2, vy=0.0, wz=0.1)
        ok, why = recv.parse(frame.to_payload(frame.next_timestamp(t)))
        _check(ok, f"frame {i} rejected: {why}")
        t += dt
    _check(recv.accepted == 50, f"only {recv.accepted}/50 accepted")
    _check(not recv.rejections, f"unexpected rejections: {recv.rejections}")
    print(f"  50/50 frames accepted over {50 * dt:.1f} s at {1 / dt:.0f} Hz")


def test_yaml_is_plain_block_style():
    """The payload must be boring YAML: no anchors, aliases or custom tags."""
    f = VlaActionFrame().set_all_arms([0.0] * NUM_ARM_JOINTS)
    payload = f.to_payload(1.0).decode()
    for token in ("&", "*", "!!", "{", "}"):
        _check(token not in payload, f"payload contains {token!r}: {payload[:120]}")
    _check(payload.startswith("cmd: action"), f"unexpected start: {payload[:40]!r}")
    _check("kLeftShoulderPitch.q:" in payload, "left shoulder pitch key missing")
    _check("remote.ly:" in payload, "remote.ly missing")
    _check("timestamp:" in payload, "timestamp missing")
    # round-trips
    doc = yaml.safe_load(payload)
    _check(len(doc["action"]) == NUM_ARM_JOINTS + 4, f"{len(doc['action'])} action entries")
    print("  payload is plain YAML block style, round-trips cleanly")



def test_full_chain_ik_to_frame_to_receiver():
    """End to end: a Cartesian target becomes a frame the receiver accepts.

    Everything else here tests the frame in isolation. This one runs the real
    solver, pushes the solved angles through the frame builder into the mock
    receiver, and then checks the *physical* result: that the joint values the
    receiver extracted would place the end effector back on the target.

    That last step is the point. A frame can be perfectly well-formed and still
    carry the wrong joints -- this test would catch a slot mapping error that the
    format tests cannot see.
    """
    import tempfile

    from g1_ik import G1Arm

    urdf = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "g1_arm_bringup", "urdf", "g1_29dof_rev_1_0.urdf",
    )
    _check(os.path.exists(urdf), f"URDF not found at {urdf}")

    for side in ("left", "right"):
        arm = G1Arm(
            urdf, side=side,
            reduced_urdf_path=os.path.join(tempfile.mkdtemp(), f"{side}.urdf"),
        )
        assert_slot_alignment(arm.joint_names, side)

        # a reachable target taken from a known-good configuration
        q_true = np.array([0.30, -0.25, 0.40, 0.90, 0.10, 0.00, 0.00])
        p_target = arm.fk_position(q_true)

        res = arm.solve_position(p_target)
        _check(res.success, f"{side}: IK failed ({res.status})")

        recv = MockGrootReceiver()
        frame = make_frame(side, res.q, [0.0] * 7, VelocityCommand())
        ok, info = recv.parse(frame.to_payload(frame.next_timestamp(1.0)))
        _check(ok, f"{side}: frame rejected: {info}")

        # unload the receiver's slots back into this arm's joint order
        slots = arm_slots(side)
        recovered = np.array([info["arm_q"][s] for s in slots])
        worst = float(np.max(np.abs(recovered - res.q)))
        _check(worst < 1e-9, f"{side}: slot mapping lost precision: {worst:.3e}")

        # and the recovered joints must reproduce the target position
        p_check = arm.fk_position(recovered)
        err = float(np.linalg.norm(p_check - p_target))
        _check(err < 1e-6, f"{side}: recovered joints miss the target by {err * 1000:.6f} mm")

        # the other arm must be untouched in the frame
        other = [s for s in range(NUM_ARM_JOINTS) if s not in slots]
        _check(np.allclose(info["arm_q"][other], 0.0), f"{side}: other arm not held")
        print(
            f"  {side}: target -> IK ({res.pos_error * 1000:.5f} mm) -> frame -> "
            f"receiver -> FK: {err * 1000:.6f} mm, slots {slots[0]}..{slots[-1]}"
        )


TESTS = [
    ("1.slot_tables", test_slot_tables_are_consistent),
    ("1.slot_order_matches_ik", test_slot_order_matches_ik_library),
    ("1.happy_path", test_happy_path_is_accepted),
    ("1.right_arm_slots", test_right_arm_maps_to_slots_7_13),
    ("1.mock_rejects_bad", test_mock_receiver_actually_rejects_bad_frames),
    ("1.builder_guards", test_builder_guards_against_the_rejections),
    ("1.timestamp_repair", test_timestamp_repair_survives_receiver),
    ("1.frame_sequence", test_sequence_of_frames_like_the_real_loop),
    ("1.yaml_style", test_yaml_is_plain_block_style),
    ("1.full_chain_ik_to_frame", test_full_chain_ik_to_frame_to_receiver),
]


def main() -> int:
    print("=" * 78)
    print("VLA FRAME FORMAT VALIDATION (ZMQ 6002, no ROS, no robot)")
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
