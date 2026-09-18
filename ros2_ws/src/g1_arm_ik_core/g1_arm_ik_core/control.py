"""ROS-free control logic: command shaping, rate limiting and safety.

Everything in this module is plain Python + numpy on purpose.  The ROS 2 node
(the `g1_arm_control_node` entry point) is a thin shell that feeds this class
with state and ships its output to a robot backend.  Keeping the logic here
means it is covered by the offline test suite and by `colcon test` without
needing a live DDS connection or hardware.
"""

from __future__ import annotations

import dataclasses
import time
from typing import List, Optional, Sequence, Tuple

import numpy as np


@dataclasses.dataclass
class ControlConfig:
    """Tunables for ArmCommandShaper / SafetyMonitor."""

    # --- timing -----------------------------------------------------------
    control_dt: float = 0.01  # s, loop period (100 Hz)

    # --- velocity / ramp --------------------------------------------------
    # Fraction of the URDF velocity limit we allow. 0.2 = 20%.
    velocity_scale: float = 0.2
    # Hard overrides (rad/s). If empty, derived from the model velocity limits.
    max_velocity: Tuple[float, ...] = ()

    # --- arm_sdk handover -------------------------------------------------
    # Ramp duration for the SDK weight. This is what stops the low-level
    # locomotion policy and our command from fighting each other.
    ramp_up: float = 1.0  # s
    ramp_down: float = 1.0  # s
    # Weight applied at standstill once fully engaged.
    cruise_weight: float = 1.0

    # --- safety -----------------------------------------------------------
    # Refuse a target further than this from the current position (rad).
    max_joint_step: float = 1.0
    # Command is considered stale (and dropped) if we see q_current from that
    # long ago. This is the "state watchdog".
    state_timeout: float = 0.5
    # Reject targets outside the joint limits by more than this (rad).
    limit_slack: float = 0.0

    # --- output -----------------------------------------------------------
    # Low-pass on the measured joint positions (0 = no smoothing).
    state_alpha: float = 0.3
    # Reject NaN/Inf and non-finite targets outright.
    strict_finite: bool = True

    def velocity_limits(self, model_limits: Sequence[float]) -> np.ndarray:
        if self.max_velocity:
            v = np.asarray(self.max_velocity, dtype=float)
            if v.size != len(model_limits):
                raise ValueError(
                    f"max_velocity has {v.size} entries, model has {len(model_limits)}"
                )
            return v
        return np.asarray(model_limits, dtype=float) * self.velocity_scale

    def validate(self, n_joints: int) -> None:
        """Reject nonsense values before they reach the robot.

        Every one of these would fail silently rather than loudly: a zero or
        negative velocity scale produces a command that never moves, a zero ramp
        divides by ~0, and a negative timeout disables the watchdog entirely --
        which is the single most dangerous misconfiguration here.
        """
        problems = []
        if n_joints < 1:
            problems.append(f"n_joints must be >= 1, got {n_joints}")
        if self.control_dt <= 0.0:
            problems.append(f"control_dt must be > 0, got {self.control_dt}")
        if self.velocity_scale <= 0.0:
            problems.append(
                f"velocity_scale must be > 0, got {self.velocity_scale}"
            )
        if self.ramp_up <= 0.0 or self.ramp_down <= 0.0:
            problems.append(
                f"ramp_up/ramp_down must be > 0, got {self.ramp_up}/{self.ramp_down}"
            )
        if self.max_joint_step <= 0.0:
            problems.append(
                f"max_joint_step must be > 0, got {self.max_joint_step}"
            )
        if self.state_timeout <= 0.0:
            problems.append(
                "state_timeout must be > 0: 0 would disable the state watchdog"
            )
        if self.limit_slack < 0.0:
            problems.append("limit_slack must be >= 0")
        if not (0.0 <= self.state_alpha <= 1.0):
            problems.append(f"state_alpha must be in [0, 1], got {self.state_alpha}")
        if not (0.0 <= self.cruise_weight <= 1.0):
            problems.append(
                f"cruise_weight must be in [0, 1], got {self.cruise_weight}"
            )
        if self.max_velocity and len(self.max_velocity) != n_joints:
            problems.append(
                f"max_velocity has {len(self.max_velocity)} entries, "
                f"expected {n_joints}"
            )
        if problems:
            raise ValueError("invalid control config:\n  " + "\n  ".join(problems))


@dataclasses.dataclass
class ControlStatus:
    """Diagnostics published alongside the command."""

    state: str  # DISABLED | ENGAGING | ACTIVE | HOLDING | REJECTED | NO_STATE
    arm_sdk_weight: float
    ramp_progress: float  # 0..1
    q_command: np.ndarray
    q_current: np.ndarray
    target_error: float  # max |q_command - q_current|
    cmds_sent: int
    cmds_rejected: int
    last_reject_reason: str = ""
    watchdog_tripped: bool = False

    @property
    def ok(self) -> bool:
        return self.state in ("ACTIVE", "ENGAGING", "HOLDING")


class ArmCommandShaper:
    """Turns a stream of IK solutions into safe, ramped joint commands.

    Responsibilities, in order of importance:

    1. **Watchdog.** Never command motion when the state stream is stale. If a
       node dies mid-motion, this is what stops the arm instead of the robot
       continuing on a stale target.
    2. **Rate limiting.** Clip q_command to q_current + v_max*dt per joint, so a
       discontinuous IK jump cannot be turned into an instantaneous joint jump.
    3. **Validation.** Reject non-finite targets and anything outside the joint
       limits, and count the rejections so the operator can see them.
    4. **Handover ramp.** Scale the arm_sdk weight in/out smoothly, which is how
       control is taken over from the robot's low-level locomotion policy.
    """

    STATE_DISABLED = "DISABLED"
    STATE_ENGAGING = "ENGAGING"
    STATE_ACTIVE = "ACTIVE"
    STATE_HOLDING = "HOLDING"
    STATE_REJECTED = "REJECTED"
    STATE_NO_STATE = "NO_STATE"

    def __init__(
        self,
        q_min: Sequence[float],
        q_max: Sequence[float],
        velocity_limits: Sequence[float],
        config: Optional[ControlConfig] = None,
    ):
        self.cfg = config or ControlConfig()
        self.q_min = np.asarray(q_min, dtype=float)
        self.q_max = np.asarray(q_max, dtype=float)
        self.v_max = np.asarray(velocity_limits, dtype=float)
        self.n = self.q_min.size

        for name, arr in (
            ("q_min", self.q_min),
            ("q_max", self.q_max),
            ("velocity_limits", self.v_max),
        ):
            if arr.size != self.n:
                raise ValueError(f"{name} size {arr.size} != {self.n}")
        if np.any(self.q_max < self.q_min):
            raise ValueError("q_max is below q_min for at least one joint")
        if np.any(self.v_max <= 0.0):
            raise ValueError(
                f"velocity limits must be > 0, got {self.v_max.tolist()}"
            )
        self.cfg.validate(self.n)

        # runtime state
        self._enabled = False
        self._ramp = 0.0  # 0..1
        self._q_current: Optional[np.ndarray] = None
        self._q_smoothed: Optional[np.ndarray] = None
        self._state_time: Optional[float] = None
        self._q_command: Optional[np.ndarray] = None
        self._target: Optional[np.ndarray] = None
        self._cmds_sent = 0
        self._cmds_rejected = 0
        self._last_reject = ""
        self._watchdog_tripped = False

    # -- input ---------------------------------------------------------------

    def set_joint_state(self, q: Sequence[float], now: Optional[float] = None) -> None:
        """Feed measured joint positions (from /lowstate via the bridge)."""
        q = np.asarray(q, dtype=float).reshape(self.n)
        if not np.all(np.isfinite(q)):
            return  # ignore a corrupt sample rather than poisoning the filter
        now = time.monotonic() if now is None else now
        if self.cfg.state_alpha > 0.0 and self._q_smoothed is not None:
            a = float(np.clip(self.cfg.state_alpha, 0.0, 1.0))
            self._q_smoothed = a * q + (1.0 - a) * self._q_smoothed
        else:
            self._q_smoothed = q.copy()
        self._q_current = self._q_smoothed.copy()
        self._state_time = now
        if self._watchdog_tripped:
            self._watchdog_tripped = False

    def set_target(self, q_target: Sequence[float], now: Optional[float] = None) -> bool:
        """Set the desired joint positions. Returns False if rejected."""
        target = np.asarray(q_target, dtype=float).reshape(self.n)

        if self.cfg.strict_finite and not np.all(np.isfinite(target)):
            self._reject("target contains NaN/Inf")
            return False

        below = self.q_min - self.cfg.limit_slack - target
        above = target - (self.q_max + self.cfg.limit_slack)
        worst = float(max(below.max(), above.max()))
        if worst > 0.0:
            self._reject(f"target outside joint limits by {worst:.4f} rad")
            return False

        if self._q_current is not None:
            step = float(np.max(np.abs(target - self._q_current)))
            if step > self.cfg.max_joint_step:
                self._reject(
                    f"target {step:.3f} rad from current pose, "
                    f"max_joint_step is {self.cfg.max_joint_step:.3f}"
                )
                return False

        self._target = np.clip(target, self.q_min, self.q_max)
        return True

    def clear_target(self) -> None:
        self._target = None

    def enable(self) -> None:
        self._enabled = True

    def disable(self) -> None:
        self._enabled = False
        self._target = None

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    @property
    def target(self) -> Optional[np.ndarray]:
        return None if self._target is None else self._target.copy()

    @property
    def last_reject_reason(self) -> str:
        return self._last_reject

    # -- output --------------------------------------------------------------

    def step(self, now: Optional[float] = None) -> ControlStatus:
        """Advance one control tick and return the command to send."""
        now = time.monotonic() if now is None else now
        dt = self.cfg.control_dt

        if not self._enabled:
            self._ramp = max(0.0, self._ramp - dt / max(self.cfg.ramp_down, 1e-6))
            return self._status(self.STATE_DISABLED, 0.0, now)

        if self._state_time is None or (now - self._state_time) > self.cfg.state_timeout:
            # No fresh state: do not command motion.
            self._watchdog_tripped = True
            self._ramp = 0.0
            return self._status(self.STATE_NO_STATE, 0.0, now)

        if self._q_current is None:
            return self._status(self.STATE_NO_STATE, 0.0, now)

        if self._q_command is None:
            # First tick after enable: start from where the arm actually is.
            self._q_command = self._q_current.copy()

        engaging = self._ramp < 1.0
        # Advance the handover ramp whenever we are enabled and have fresh
        # state -- *not* only when a target exists.  Otherwise there is a window
        # after `enable()` where we are nominally in control but the weight is
        # still 0, i.e. the locomotion policy still owns the arm.  "Enabled but
        # no target yet" must mean "we own the arm, holding its current pose".
        if engaging:
            self._ramp = min(1.0, self._ramp + dt / max(self.cfg.ramp_up, 1e-6))
            engaging = self._ramp < 1.0

        if self._target is None:
            # Nothing to track: hold the pose we already have.
            step_limit = self.v_max * dt
            self._q_command = np.clip(
                self._q_command,
                self._q_current - step_limit,
                self._q_current + step_limit,
            )
            state = self.STATE_ENGAGING if engaging else self.STATE_HOLDING
            weight = self._arm_sdk_weight()
            self._cmds_sent += 1
            return self._status(state, weight, now)

        # rate limit towards the target
        step_limit = self.v_max * dt
        delta = np.clip(self._target - self._q_command, -step_limit, step_limit)
        self._q_command = self._q_command + delta

        # never let the rate limiter walk us out of the joint limits
        self._q_command = np.clip(self._q_command, self.q_min, self.q_max)

        self._cmds_sent += 1
        weight = self._arm_sdk_weight()
        state = self.STATE_ENGAGING if engaging else self.STATE_ACTIVE
        return self._status(state, weight, now)

    def _arm_sdk_weight(self) -> float:
        return float(np.clip(self._ramp * self.cfg.cruise_weight, 0.0, 1.0))

    def _reject(self, reason: str) -> None:
        self._cmds_rejected += 1
        self._last_reject = reason

    def _status(self, state: str, weight: float, now: float) -> ControlStatus:
        q_cmd = (
            self._q_command.copy()
            if self._q_command is not None
            else np.full(self.n, np.nan)
        )
        q_cur = (
            self._q_current.copy()
            if self._q_current is not None
            else np.full(self.n, np.nan)
        )
        err = (
            float(np.max(np.abs(q_cmd - q_cur)))
            if np.all(np.isfinite(q_cmd)) and np.all(np.isfinite(q_cur))
            else float("nan")
        )
        return ControlStatus(
            state=state,
            arm_sdk_weight=weight,
            ramp_progress=float(self._ramp),
            q_command=q_cmd,
            q_current=q_cur,
            target_error=err,
            cmds_sent=self._cmds_sent,
            cmds_rejected=self._cmds_rejected,
            last_reject_reason=self._last_reject,
            watchdog_tripped=self._watchdog_tripped,
        )


class SafetyMonitor:
    """Rate-of-change and tracking-error watchdog on the *outgoing* command.

    Separate from ArmCommandShaper because it also has to work when the
    commands come from somewhere else (e.g. a teleop stream).
    """

    def __init__(
        self,
        max_velocity: Sequence[float],
        dt: float,
        max_tracking_error: float = 0.5,
        rate_scale: float = 1.0,
    ):
        self.v_max = np.asarray(max_velocity, dtype=float)
        self.dt = float(dt)
        self.max_tracking_error = float(max_tracking_error)
        # Producer and consumer rarely agree on the exact tick period, so a
        # command that sits exactly on the limit must not be flagged. 1.05
        # tolerates 5% of timing jitter.
        self.rate_scale = float(rate_scale)
        self._q_prev: Optional[np.ndarray] = None
        self.violations: List[str] = []

    def reset(self, q: Optional[Sequence[float]] = None) -> None:
        self._q_prev = None if q is None else np.asarray(q, dtype=float).copy()
        self.violations.clear()

    def check(self, q_command: Sequence[float], q_current: Optional[Sequence[float]] = None) -> bool:
        """True if the command is acceptable. Records the reason if not."""
        q = np.asarray(q_command, dtype=float)
        if not np.all(np.isfinite(q)):
            self.violations.append("non-finite command")
            return False

        if self._q_prev is not None:
            rate = np.abs(q - self._q_prev) / max(self.dt, 1e-9)
            allowed = self.v_max * self.rate_scale
            over = rate - allowed
            if np.any(over > 1e-9):
                i = int(np.argmax(over))
                self.violations.append(
                    f"joint {i} rate {rate[i]:.2f} rad/s > {allowed[i]:.2f} rad/s"
                )
                return False

        if q_current is not None:
            err = float(np.max(np.abs(q - np.asarray(q_current, dtype=float))))
            if err > self.max_tracking_error:
                self.violations.append(
                    f"tracking error {err:.3f} rad > {self.max_tracking_error:.3f}"
                )
                return False

        self._q_prev = q.copy()
        return True


# --------------------------------------------------------------------------- #
#  backend protocol + mock
# --------------------------------------------------------------------------- #


class ArmBackend:
    """Interface every robot backend implements.

    Implementations must be cheap and non-blocking: `send` runs inside the
    control loop.
    """

    name = "abstract"
    # True when the backend *is* the source of joint state (a simulator), so the
    # control node uses its feedback instead of an external /joint_states.
    feeds_state = False

    def connect(self) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def send(self, q: Sequence[float], weight: float) -> None:  # pragma: no cover
        raise NotImplementedError

    def close(self) -> None:  # pragma: no cover - interface
        pass

    def state(self) -> Optional[np.ndarray]:  # pragma: no cover - interface
        """Current joint state, or None if the backend does not provide one."""
        return None

    def describe(self) -> str:
        return self.name


class MockBackend(ArmBackend):
    """First-order joint model -- lets the whole stack run with no hardware.

    Each joint moves towards the command with time constant `tau`, jittered a
    little so tests can tell a real feedback loop from a straight copy.
    """

    name = "mock"
    feeds_state = True

    def __init__(
        self,
        q_init: Sequence[float],
        q_min: Sequence[float],
        q_max: Sequence[float],
        tau: float = 0.05,
        noise: float = 1e-5,
        dt: float = 0.01,
        seed: int = 0,
    ):
        self.q = np.asarray(q_init, dtype=float).copy()
        self.q_min = np.asarray(q_min, dtype=float)
        self.q_max = np.asarray(q_max, dtype=float)
        self.tau = float(tau)
        self.noise = float(noise)
        # The servo time constant must be integrated with the *real* control
        # period, otherwise the simulated dynamics change when the control rate
        # changes and the tests stop meaning anything.
        self.dt = float(dt)
        self._rng = np.random.default_rng(seed)
        self._last_weight = 0.0
        self.sends = 0

    def connect(self) -> None:
        pass

    def send(self, q: Sequence[float], weight: float) -> None:
        q = np.asarray(q, dtype=float)
        # only the engaged fraction of the command is tracked, mirroring how the
        # arm_sdk weight blends our command with the locomotion policy
        eff = np.clip(weight, 0.0, 1.0) * q + (1.0 - np.clip(weight, 0.0, 1.0)) * self.q
        alpha = 1.0 - np.exp(-self.dt / max(self.tau, 1e-6))
        self.q = self.q + alpha * (eff - self.q)
        if self.noise > 0:
            self.q = self.q + self._rng.normal(0.0, self.noise, size=self.q.shape)
        self.q = np.clip(self.q, self.q_min, self.q_max)
        self._last_weight = float(weight)
        self.sends += 1

    def state(self) -> np.ndarray:
        return self.q.copy()

    def describe(self) -> str:
        return f"mock (tau={self.tau}s, dt={self.dt}s, sends={self.sends})"


__all__ = [
    "ArmBackend",
    "ArmCommandShaper",
    "ControlConfig",
    "ControlStatus",
    "MockBackend",
    "SafetyMonitor",
]
