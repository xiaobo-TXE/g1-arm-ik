"""End-to-end ZMQ test: real PUSH socket -> real PULL socket.

`test_vla_protocol.py` validates the frame *format* against a reimplementation of
the C++ parser.  This file validates the *transport*: an actual pyzmq PUSH socket
on the node's side reaching an actual PULL socket bound the way
`RemoteCommandReceiver` binds it, with the receiver's parse rules applied to what
comes off the wire.

The socket topology is copied from the C++ side:
    robot:   zmq::socket_type::pull,  bind "tcp://*:6002"
    client:  zmq::socket_type::push,  connect "tcp://<robot>:6002"

Run:  python test_zmq_bridge.py
"""

from __future__ import annotations

import os
import sys
import threading
import time

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from g1_arm_ik_core.vla_protocol import (  # noqa: E402
    NUM_ARM_JOINTS,
    VelocityCommand,
    VlaActionFrame,
    make_frame,
)
from g1_arm_ik_core.zmq_bridge import ZmqVlaBridge  # noqa: E402

try:
    import zmq  # noqa: F401
    HAVE_ZMQ = True
except ImportError:  # pragma: no cover
    HAVE_ZMQ = False


def _check(cond, msg):
    if not cond:
        raise AssertionError(msg)


class FakeGrootReceiver:
    """A real PULL socket bound like the C++ receiver, with its parse rules."""

    def __init__(self, port: int):
        import zmq

        self.ctx = zmq.Context.instance()
        self.sock = self.ctx.socket(zmq.PULL)
        self.sock.setsockopt(zmq.RCVTIMEO, 500)
        self.sock.bind(f"tcp://127.0.0.1:{port}")
        self.received = []
        self.rejections = []
        self._prev_ts = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self._thread.start()

    def _loop(self):
        while not self._stop.is_set():
            try:
                payload = self.sock.recv()
            except Exception:
                continue
            self.received.append(payload)
            self._parse(payload)

    def _parse(self, payload: bytes):
        """Same rules as groot::RemoteCommandReceiver::parse_packet()."""
        try:
            root = yaml.safe_load(payload.decode())
        except Exception:
            self.rejections.append("yaml parse error")
            return
        if not isinstance(root, dict) or "action" not in root or "timestamp" not in root:
            self.rejections.append("missing action/timestamp")
            return
        ts = float(root["timestamp"])
        if not np.isfinite(ts):
            self.rejections.append("non-finite timestamp")
            return
        if self._prev_ts is not None and ts <= self._prev_ts:
            self.rejections.append("stale timestamp")
            return
        action = root["action"]
        if not isinstance(action, dict):
            self.rejections.append("action not a map")
            return
        arm_keys = [k for k in action if isinstance(k, str) and k.endswith(".q")]
        if len(arm_keys) != NUM_ARM_JOINTS:
            self.rejections.append("expected 14 arm joints")
            return
        for k in arm_keys:
            v = float(action[k])
            if not np.isfinite(v) or abs(v) > 3.2:
                self.rejections.append("joint value out of range")
                return
        self._prev_ts = ts

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=1.0)
        self.sock.close(linger=0)


def _free_port() -> int:
    import socket

    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_real_socket_delivers_frames():
    """A real PUSH->PULL hop must deliver valid frames, in order."""
    port = _free_port()
    recv = FakeGrootReceiver(port)
    recv.start()
    time.sleep(0.1)  # let the bind settle before connecting

    bridge = ZmqVlaBridge(robot_ip="127.0.0.1", port=port, enabled=True, dry_run=False)
    bridge.connect()

    frame = VlaActionFrame()
    n = 20
    t = 1000.0
    for i in range(n):
        frame.set_all_arms([0.01 * i] * NUM_ARM_JOINTS)
        frame.velocity = VelocityCommand(vx=0.1, vy=-0.05, wz=0.02)
        sent = bridge.send(frame.to_payload(frame.next_timestamp(t)))
        _check(sent, f"frame {i} not sent")
        t += 0.02
        time.sleep(0.005)

    time.sleep(0.3)
    bridge.close()
    recv.stop()

    _check(len(recv.received) == n, f"received {len(recv.received)}/{n} frames")
    _check(not recv.rejections, f"receiver rejected: {recv.rejections}")
    _check(bridge.sent == n, f"bridge counted {bridge.sent} sends")
    print(f"  {n}/{n} frames crossed a real ZMQ hop, 0 rejections")


def test_real_socket_roundtrip_values():
    """Values must survive the wire unchanged (float formatting check)."""
    port = _free_port()
    recv = FakeGrootReceiver(port)
    recv.start()
    time.sleep(0.1)

    bridge = ZmqVlaBridge(robot_ip="127.0.0.1", port=port, enabled=True, dry_run=False)
    bridge.connect()

    q = [0.123456789, -0.987654321, 1.5, -1.5, 2.9, -2.9, 0.0]
    frame = make_frame("left", q, [0.0] * 7, VelocityCommand(vx=0.4, vy=-0.3, wz=0.25))
    bridge.send(frame.to_payload(frame.next_timestamp(1.0)))
    time.sleep(0.3)
    bridge.close()
    recv.stop()

    _check(len(recv.received) == 1, f"got {len(recv.received)} frames")
    doc = yaml.safe_load(recv.received[0].decode())
    got = [doc["action"][f"kLeft{n}.q"] for n in
           ("ShoulderPitch", "ShoulderRoll", "ShoulderYaw", "Elbow",
            "WristRoll", "WristPitch", "WristYaw")]
    worst = max(abs(a - b) for a, b in zip(q, got))
    print(f"  round-trip: max |value error| = {worst:.3e}")
    _check(worst < 1e-9, f"values drifted on the wire: {got} vs {q}")
    # velocity mapping vx=ly, vy=-lx, wz=-rx must survive too
    _check(abs(doc["action"]["remote.ly"] - 0.4) < 1e-9, "vx not on remote.ly")
    _check(abs(doc["action"]["remote.lx"] - 0.3) < 1e-9, "vy not mapped to -lx")
    _check(abs(doc["action"]["remote.rx"] - (-0.25)) < 1e-9, "wz not mapped to -rx")


def test_disabled_and_dry_run_send_nothing():
    """The safety gates must actually gate."""
    port = _free_port()
    recv = FakeGrootReceiver(port)
    recv.start()
    time.sleep(0.1)

    frame = VlaActionFrame().set_all_arms([0.0] * NUM_ARM_JOINTS)
    payload = frame.to_payload(1.0)

    for label, kwargs in (
        ("disabled", dict(enabled=False, dry_run=False)),
        ("dry-run", dict(enabled=True, dry_run=True)),
        ("both off", dict(enabled=False, dry_run=True)),
    ):
        bridge = ZmqVlaBridge(robot_ip="127.0.0.1", port=port, **kwargs)
        bridge.connect()
        _check(not bridge.transmitting, f"{label} reported transmitting")
        for _ in range(5):
            _check(not bridge.send(payload), f"{label} actually sent a frame")
        _check(bridge.sent == 0, f"{label} incremented the sent counter")
        bridge.close()

    time.sleep(0.3)
    recv.stop()
    _check(not recv.received, f"{len(recv.received)} frames leaked while gated")
    print("  disabled / dry-run / both: 0 frames transmitted")


def test_enable_then_disable_mid_stream():
    """Toggling must take effect immediately and cleanly."""
    port = _free_port()
    recv = FakeGrootReceiver(port)
    recv.start()
    time.sleep(0.1)

    bridge = ZmqVlaBridge(robot_ip="127.0.0.1", port=port, enabled=True, dry_run=False)
    bridge.connect()
    # fill all 14 slots: the receiver rejects any frame that does not carry them
    frame = VlaActionFrame().set_all_arms([0.0] * NUM_ARM_JOINTS)

    t = 1.0
    for _ in range(5):
        bridge.send(frame.to_payload(frame.next_timestamp(t)))
        t += 0.02
    bridge.disable()
    for _ in range(5):
        _check(not bridge.send(frame.to_payload(frame.next_timestamp(t))),
               "frame sent while disabled")
        t += 0.02
    bridge.enable()
    for _ in range(5):
        _check(bridge.send(frame.to_payload(frame.next_timestamp(t))),
               "frame not sent after re-enabling")
        t += 0.02

    time.sleep(0.3)
    bridge.close()
    recv.stop()
    _check(len(recv.received) == 10, f"expected 10 frames, got {len(recv.received)}")
    _check(not recv.rejections, f"rejections: {recv.rejections}")
    print("  5 sent, 5 suppressed while disabled, 5 sent again: 10/10 delivered")


def test_no_receiver_still_returns_quickly():
    """A missing receiver must not block the control loop.

    ZMQ PUSH silently queues when nobody is bound; `send` is non-blocking for a
    PUSH socket, so the loop must keep its timing even with no peer.
    """
    port = _free_port()  # nothing binds this
    bridge = ZmqVlaBridge(robot_ip="127.0.0.1", port=port, enabled=True, dry_run=False)
    bridge.connect()
    frame = VlaActionFrame().set_all_arms([0.0] * NUM_ARM_JOINTS)

    t0 = time.perf_counter()
    n = 200
    for i in range(n):
        bridge.send(frame.to_payload(frame.next_timestamp(1.0 + i * 0.02)))
    dt = time.perf_counter() - t0
    bridge.close()

    per_send = dt / n * 1000.0
    print(f"  {n} sends with no peer: {per_send:.4f} ms each")
    _check(per_send < 1.0, f"send blocked for {per_send:.3f} ms")


TESTS = [
    ("2.real_socket_delivery", test_real_socket_delivers_frames),
    ("2.real_socket_values", test_real_socket_roundtrip_values),
    ("2.gates_block_sending", test_disabled_and_dry_run_send_nothing),
    ("2.enable_disable_midstream", test_enable_then_disable_mid_stream),
    ("2.no_peer_non_blocking", test_no_receiver_still_returns_quickly),
]


def main() -> int:
    print("=" * 78)
    print("ZMQ TRANSPORT VALIDATION (real PUSH -> real PULL)")
    print("=" * 78)
    if not HAVE_ZMQ:
        print("  SKIP: pyzmq is not installed (pip install pyzmq)")
        return 0
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
