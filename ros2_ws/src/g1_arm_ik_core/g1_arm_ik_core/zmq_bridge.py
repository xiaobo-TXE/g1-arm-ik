"""ZMQ transport for the VLA action frames (PUSH -> robot's PULL on port 6002).

Kept separate from `vla_protocol.py` so the frame format can be tested without
pyzmq installed, and separate from the ROS node so the transport can be tested
without ROS.

Client-side socket setup, matching what the robot-side receiver expects:

* socket type  PUSH   (the robot binds PULL on tcp://*:6002)
* connect      tcp://<robot_ip>:6002
* CONFLATE     on, and this is load-bearing. A PUSH socket with no peer CONNECTED
               does not drop messages: it queues them until the send buffer is
               full and then BLOCKS `send()`. Measured with SNDHWM=2 that still
               cost 21 ms per send once the buffer filled -- i.e. the control
               loop would stall whenever the robot-side receiver went away.
               With ZMQ_CONFLATE the outgoing queue keeps only the newest
               message, so the buffer can never fill and `send()` stays
               non-blocking.
* SNDHWM       small. Mostly a latency bound; CONFLATE is what keeps the buffer
               from filling.
* SNDTIMEO     a backstop so a wedged peer can never block the loop indefinitely.
* CONFLATE     off on purpose: conflating would also drop the *state* we might
               later read, and the receiver's timestamp gate already makes
               duplicates harmless.

`enabled` and `dry_run` exist so the node can be brought up, validated and
observed (via `describe`/counters) without writing to the robot at all.
"""

from __future__ import annotations

import time
from typing import Optional

__all__ = ["ZmqVlaBridge", "ZmqUnavailable"]


class ZmqUnavailable(RuntimeError):
    """Raised when pyzmq is missing but the bridge is asked to transmit."""


class ZmqVlaBridge:
    """Owns the PUSH socket and the safety gates around it."""

    def __init__(
        self,
        robot_ip: str = "192.168.123.222",
        port: int = 6002,
        enabled: bool = False,
        dry_run: bool = True,
        sndhwm: int = 2,
        sndtimeo_ms: int = 5,
        conflate: bool = True,
    ):
        self.robot_ip = robot_ip
        self.port = int(port)
        self.sndhwm = int(sndhwm)
        self.sndtimeo_ms = int(sndtimeo_ms)
        self.conflate = bool(conflate)
        self._enabled = bool(enabled)
        self._dry_run = bool(dry_run)
        self._ctx = None
        self._sock = None
        self.sent = 0
        self.rejected = 0
        self.last_error = ""
        self.last_send_time: Optional[float] = None

    # -- lifecycle ----------------------------------------------------------

    @property
    def endpoint(self) -> str:
        return f"tcp://{self.robot_ip}:{self.port}"

    def connect(self) -> None:
        """Create the socket. Safe to call when disabled/dry-run: nothing is sent."""
        try:
            import zmq
        except ImportError as exc:  # pragma: no cover - depends on the host
            raise ZmqUnavailable(
                "pyzmq is required for the VLA bridge.\n"
                "  pip install pyzmq"
            ) from exc

        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.PUSH)
        self._sock.setsockopt(zmq.SNDHWM, self.sndhwm)
        self._sock.setsockopt(zmq.SNDTIMEO, self.sndtimeo_ms)
        self._sock.setsockopt(zmq.LINGER, 0)
        if self.conflate:
            # keeps only the newest frame queued, so send() never blocks on a
            # full buffer when no peer is connected
            self._sock.setsockopt(zmq.CONFLATE, 1)
        self._sock.connect(self.endpoint)

    def close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close(linger=0)
            except Exception:  # noqa: BLE001 - shutdown must not raise
                pass
            self._sock = None

    # -- gates --------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        return self._enabled

    def enable(self) -> None:
        self._enabled = True

    def disable(self) -> None:
        self._enabled = False

    @property
    def dry_run(self) -> bool:
        return self._dry_run

    @property
    def transmitting(self) -> bool:
        return self._enabled and not self._dry_run

    # -- send ---------------------------------------------------------------

    def send(self, payload: bytes, now: Optional[float] = None) -> bool:
        """Send one frame. Returns True only if it actually went out.

        Returns False (without raising) when disabled or dry-running, so the
        caller can keep its loop running and still report status.
        """
        if not self.transmitting:
            return False
        if self._sock is None:
            raise ZmqUnavailable("connect() was not called")
        try:
            self._sock.send(payload)
        except Exception as exc:  # noqa: BLE001 - a dropped frame must not kill the loop
            # zmq.Again means the high-water mark is reached (peer gone or slow).
            # One dropped frame is the correct outcome: the receiver's timestamp
            # gate would discard a late one anyway.
            self.rejected += 1
            self.last_error = f"{type(exc).__name__}: {exc}"
            return False
        self.sent += 1
        self.last_send_time = time.monotonic() if now is None else float(now)
        return True

    def describe(self) -> str:
        state = "TRANSMITTING" if self.transmitting else (
            "dry-run" if self._dry_run else ("disabled" if not self._enabled else "?")
        )
        return (
            f"zmq {self.endpoint} [{state}] sent={self.sent} rejected={self.rejected}"
        )
