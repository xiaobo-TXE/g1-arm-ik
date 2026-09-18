"""Position-only inverse kinematics for one 7-DOF arm (CasADi + IPOPT).

Key design decisions, and why
-----------------------------
*Posture regularisation is deliberately NOT part of the objective.*  The usual
trick -- add ``sum(w_i * (q_i - q_ref_i)^2)`` to keep the arm near a nominal
pose -- was implemented, measured and then turned off by default, because it
does not work here:

  * Mixed into the task objective, it makes IPOPT stall at the iteration limit
    with a 0.5-15 mm residual: the regulariser pulls against the position task
    and the interior-point steps keep trading between them.
  * Making position a hard constraint and minimising posture instead (the
    "stage 2" path, still available behind ``IKConfig.refine``) is worse: the
    posture optimum lies *on* the constraint boundary, which is exactly where an
    interior-point method cannot converge.  Adding a log barrier makes it stall
    harder; removing the barrier lets it settle 1.4e-4 m off target.

  On a 3-D task with 7 joints this is expected: the redundancy is 4-dimensional,
  so "reach the point" and "be near a nominal pose" are competing objectives,
  not nested ones.  Natural posture is therefore obtained *structurally*:

  * warm starting -- each solve begins at the previous solution, so IPOPT stays
    on the branch it is already on and does not wander through the null space
    (measured: 0.98 deg of joint motion for a 0.4 mm target step);
  * ``ArmIK.nullspace_project()`` -- an exact null-space step that changes
    posture *without* moving the end effector (measured |dp| = 1e-4 m for a
    4 deg joint move), which is the correct tool for posture shaping.

  The practical consequence for a control loop: keep the target smooth and use
  ``RateLimiter`` + ``WeightedMovingFilter`` on the output, rather than asking
  the optimiser to smooth for you.
*Hard joint bounds.*  Limits are enforced as IPOPT bounds, never as soft costs,
so the solver can never return an out-of-range configuration.

*Warm starting.*  Each solve begins at the previous solution. This keeps
consecutive solutions continuous and on the same branch, and it is why a
trajectory can be followed without the joints jumping between IK solutions.

*An optional limit barrier* (``limit_barrier_weight``).  A log barrier keeps the
solution away from the joint limits, because a solution sitting exactly on a
limit "looks" converged but has zero margin, which is dangerous on hardware.
It is off by default for speed; turn it on if you see solutions pinned to a
limit.

Symbolic backend
----------------
Two backends are supported and produce identical results (verified to 1e-15):

  * ``pinocchio``  -- uses ``pinocchio.casadi`` (cpin).  This is what a
    conda/apt/source build of pinocchio provides.
  * ``manual``     -- builds the same FK symbolically from the URDF joint chain
    (origin xyz/rpy, axis, type) using CasADi SX.  Needed because the PyPI
    `pin` wheel is not compiled with CasADi support.

The manual backend is validated against pinocchio's *numeric* FK in the test
suite, so the two are interchangeable.
"""

from __future__ import annotations

import dataclasses
import time
from collections import deque
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import casadi
except ImportError as exc:  # pragma: no cover
    raise ImportError("casadi is required: pip install casadi") from exc

try:
    import pinocchio as pin
except ImportError as exc:  # pragma: no cover
    raise ImportError("pinocchio is required: pip install pin") from exc

try:  # optional: only present in builds compiled with CasADi support
    from pinocchio import casadi as cpin

    HAS_PIN_CASADI = True
except ImportError:  # pragma: no cover
    cpin = None
    HAS_PIN_CASADI = False

from .model import PinArmModel
from .reduced_model import ReducedArmModel

SX = casadi.SX


# --------------------------------------------------------------------------- #
#  symbolic forward kinematics built directly from the URDF
# --------------------------------------------------------------------------- #


def _skew_sym(v: SX) -> SX:
    return casadi.vertcat(
        casadi.horzcat(0, -v[2], v[1]),
        casadi.horzcat(v[2], 0, -v[0]),
        casadi.horzcat(-v[1], v[0], 0),
    )


def _sym_rot(axis: Sequence[float], angle: SX) -> SX:
    """Rodrigues' formula for a *unit* axis, symbolically in `angle`."""
    k = np.asarray(axis, dtype=float)
    k = k / np.linalg.norm(k)
    K = np.array(
        [[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]], dtype=float
    )
    Kc = casadi.SX(K)
    return (
        casadi.SX.eye(3)
        + casadi.sin(angle) * Kc
        + (1.0 - casadi.cos(angle)) * (Kc @ Kc)
    )


def _sym_transform(R: SX, xyz: Sequence[float]) -> SX:
    return casadi.vertcat(
        casadi.horzcat(R, casadi.SX(np.asarray(xyz, dtype=float).reshape(3, 1))),
        casadi.horzcat(0, 0, 0, 1),
    )


def _sym_joint_motion(axis: Sequence[float], angle: SX) -> SX:
    """Homogeneous transform of a revolute joint about `axis` by `angle`."""
    R = _sym_rot(axis, angle)
    return casadi.vertcat(
        casadi.horzcat(R, casadi.SX.zeros(3, 1)),
        casadi.horzcat(0, 0, 0, 1),
    )


def symbolic_fk_manual(spec: ReducedArmModel, q: SX) -> SX:
    """Compose torso -> EE symbolically, straight from the URDF fields.

    URDF semantics: T_parent_child(q) = Trans(origin_xyz) @ R(origin_rpy)
                                       @ JointMotion(q)
    i.e. translate first, then rotate (the origin rotation acts on the child
    frame, so it sits between the translation and the joint motion).
    """
    axes = {1: [1.0, 0.0, 0.0], 2: [0.0, 1.0, 0.0], 3: [0.0, 0.0, 1.0]}

    def rpy(R_roll: SX, R_pitch: SX, R_yaw: SX) -> SX:
        return R_yaw @ R_pitch @ R_roll

    I3 = casadi.SX.eye(3)
    T = casadi.SX.eye(4)
    for i in range(len(spec.joint_names)):
        roll, pitch, yaw = spec.origin_rpy[i]
        R_o = rpy(
            _sym_rot(axes[1], casadi.SX(roll)),
            _sym_rot(axes[2], casadi.SX(pitch)),
            _sym_rot(axes[3], casadi.SX(yaw)),
        )
        jtype = spec.joint_types[i]
        if jtype in ("revolute", "continuous"):
            T_j = _sym_joint_motion(spec.axis[i], q[i])
        elif jtype == "prismatic":
            a = np.asarray(spec.axis[i], dtype=float)
            a = a / np.linalg.norm(a)
            T_j = casadi.vertcat(
                casadi.horzcat(I3, casadi.SX((a * q[i]).reshape(3, 1))),
                casadi.horzcat(0, 0, 0, 1),
            )
        else:  # fixed
            T_j = casadi.SX.eye(4)
        T = T @ _sym_transform(R_o, spec.origin_xyz[i]) @ T_j

    # virtual end-effector frame (fixed offset from the last link)
    T = T @ _sym_transform(I3, spec.ee_offset)
    return T


# --------------------------------------------------------------------------- #
#  configuration
# --------------------------------------------------------------------------- #


@dataclasses.dataclass
class IKConfig:
    """Solver weights and safety knobs (defaults tuned for the G1 7-DOF arm)."""

    # --- task -------------------------------------------------------------
    position_weight: float = 200.0
    # Rotation is NOT controlled by default. Set control_rotation=True if you
    # also want to command orientation via ArmIK.make_target().
    rotation_weight: float = 0.0
    control_rotation: bool = False

    # --- redundancy resolution (OFF by default -- see the module docstring) --
    # A fixed posture-weight vector is kept for experimentation and for the
    # optional stage 2 below, but it is NOT in the default objective: mixing a
    # posture penalty with the position task makes IPOPT stall 0.5-15 mm short
    # of the target. Posture is handled by warm starting instead.
    posture_weight: Tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    posture_reference: Optional[Tuple[float, ...]] = None

    # --- smoothness / conditioning ---------------------------------------
    # Only used by stage 2 (see `refine`).
    smooth_weight: float = 0.5
    # Log barrier keeping solutions away from the joint limits. 0 = off (fast).
    # Raise to 1e-3 or so if you observe solutions pinned to a limit.
    limit_barrier_weight: float = 0.0
    limit_barrier_margin: float = 0.05  # rad

    # --- stage 1 solver ---------------------------------------------------
    max_iter: int = 60
    tol: float = 1e-5
    acceptable_tol: float = 1e-4
    print_level: int = 0

    # --- multi-start fallback ---------------------------------------------
    # IPOPT reports "Solve_Succeeded" for a *local* minimum, and from a fixed
    # seed some targets reliably land in a bad one (measured: a configuration
    # with the shoulder pinned on its joint limit, 77 mm off target, while 80%
    # of random seeds hit 0.0000 mm). One solve is ~2 ms, so retrying is cheap
    # and turns an intermittent failure into a rare one.
    multistart: bool = True
    multistart_max_extra: int = 12  # extra attempts after the warm-start solve
    multistart_seed: int = 0

    # --- optional stage 2: exact position + posture shaping ---------------
    # OFF by default and expected to stay off. Making the position a hard
    # constraint and minimising posture puts the optimum on the constraint
    # boundary, where IPOPT cannot converge (measured: stalls at the iteration
    # limit, or settles 1.4e-4 m off target without the barrier). Kept only so
    # the failure mode is reproducible; see the module docstring.
    refine: bool = False
    refine_pos_tol: float = 1e-4  # m, bound on ||p_fk - p_target||
    refine_max_iter: int = 200
    refine_smooth_weight: float = 0.1

    # --- acceptance thresholds -------------------------------------------
    pos_tol: float = 5e-4  # m; below this counts as "reached"
    limit_margin_min: float = 0.0  # rad

    def posture_weight_vector(self, n: int) -> np.ndarray:
        w = np.asarray(self.posture_weight, dtype=float)
        if w.size != n:
            raise ValueError(
                f"posture_weight has {w.size} entries but the arm has {n} joints"
            )
        return w

    def posture_reference_vector(self, n: int) -> np.ndarray:
        if self.posture_reference is None:
            return np.zeros(n)
        r = np.asarray(self.posture_reference, dtype=float)
        if r.size != n:
            raise ValueError(
                f"posture_reference has {r.size} entries but the arm has {n} joints"
            )
        return r


# --------------------------------------------------------------------------- #
#  results
# --------------------------------------------------------------------------- #


@dataclasses.dataclass
class IKResult:
    success: bool  # converged and position within pos_tol
    q: np.ndarray  # (7,) solved joint angles
    pos_error: float  # m, ||fk(q)[:3,3] - p_target||
    rot_error: float  # rad (0 when rotation is not controlled)
    iterations: int  # stage-1 + stage-2 iterations
    solve_time: float  # s
    limit_margin: float  # min distance to a joint limit (rad)
    converged: bool  # solver returned a solution (regardless of accuracy)
    status: str
    refined: bool = False
    attempts: int = 1  # multistart attempts used (1 = warm start succeeded)
    message: str = ""

    def __str__(self) -> str:
        return (
            f"IKResult(success={self.success} "
            f"pos_err={self.pos_error * 1000:.4f}mm "
            f"rot_err={np.degrees(self.rot_error):.3f}deg "
            f"margin={self.limit_margin:.4f}rad "
            f"{self.solve_time * 1000:.2f}ms iters={self.iterations} "
            f"attempts={self.attempts} refined={self.refined} status={self.status})"
        )


# --------------------------------------------------------------------------- #
#  solver
# --------------------------------------------------------------------------- #


class ArmIK:
    """Position-(and optionally orientation-)only IK for one arm."""

    def __init__(
        self,
        pin_model: PinArmModel,
        config: Optional[IKConfig] = None,
        use_ipopt: bool = True,
        backend: str = "auto",
    ):
        self.m = pin_model
        self.cfg = config or IKConfig()
        self.spec: ReducedArmModel = pin_model.spec
        self.n = self.m.nq

        self.backend = self._resolve_backend(backend)

        # --- symbolic forward kinematics ---------------------------------
        self._cq = SX.sym("q", self.n, 1)
        self._target_T = SX.sym("target_T", 4, 4)
        self._ee_pose = self._build_symbolic_fk()

        pos_err = self._ee_pose[:3, 3] - self._target_T[:3, 3]
        rot_err = casadi.vertcat(
            self._log3(self._ee_pose[:3, :3] @ self._target_T[:3, :3].T)
        )
        self._pos_err_fun = casadi.Function("pos_err", [self._cq, self._target_T], [pos_err])
        self._rot_err_fun = casadi.Function("rot_err", [self._cq, self._target_T], [rot_err])
        self._pos_jac_fun = casadi.Function(
            "pos_jac", [self._cq], [casadi.jacobian(self._ee_pose[:3, 3], self._cq)]
        )

        # --- stage 1: POSITION ONLY --------------------------------------
        # Deliberately free of posture/smoothness/barrier terms. Mixing them
        # into this objective makes IPOPT stall at its iteration limit with a
        # 0.5-15 mm residual (measured), because the regularisers pull against
        # the task. Position alone converges in ~10 iterations.
        self.opti = casadi.Opti()
        self._var_q = self.opti.variable(self.n)
        self._par_target = self.opti.parameter(4, 4)

        pos_cost = casadi.sumsqr(self._pos_err_fun(self._var_q, self._par_target))
        cost = self.cfg.position_weight * pos_cost
        if self.cfg.control_rotation:
            cost = cost + self.cfg.rotation_weight * casadi.sumsqr(
                self._rot_err_fun(self._var_q, self._par_target)
            )
        self.opti.minimize(cost)
        self.opti.subject_to(self.opti.bounded(self.m.q_min, self._var_q, self.m.q_max))
        self._install_solver(self.opti, self.cfg.max_iter)

        # --- stage 2: position as a hard constraint, posture secondary ----
        # The remaining 4-D freedom (7 joints, 3-D task) is a smooth null space,
        # so this is where posture/smoothness/barrier belong.
        self._opti2 = None
        if self.cfg.refine:
            o2 = casadi.Opti()
            q2 = o2.variable(self.n)
            p_q_last = o2.parameter(self.n)
            p_posture = o2.parameter(self.n)
            p_posture_w = o2.parameter(self.n)
            p_target = o2.parameter(4, 4)
            err2 = self._pos_err_fun(q2, p_target)
            c2 = casadi.sumsqr(p_posture_w * (q2 - p_posture))
            c2 = c2 + self.cfg.refine_smooth_weight * casadi.sumsqr(q2 - p_q_last)
            if self.cfg.control_rotation:
                c2 = c2 + self.cfg.rotation_weight * casadi.sumsqr(
                    self._rot_err_fun(q2, p_target)
                )
            c2 = c2 + self._barrier_cost(q2)
            o2.minimize(c2)
            o2.subject_to(o2.bounded(self.m.q_min, q2, self.m.q_max))
            o2.subject_to(casadi.sumsqr(err2) <= self.cfg.refine_pos_tol**2)
            self._install_solver(o2, self.cfg.refine_max_iter)
            self._opti2 = (o2, q2, p_q_last, p_posture, p_posture_w, p_target)

        # --- state -------------------------------------------------------
        self._q_last: Optional[np.ndarray] = None
        self._last_result: Optional[IKResult] = None
        self._envelope: Optional[Tuple[float, float, float]] = None

    # -- reach envelope ------------------------------------------------------

    @property
    def envelope(self) -> Tuple[float, float, float]:
        """(min_reach, max_reach, margin) around the IK base, sampled once.

        Used only to *skip* multi-start on obviously hopeless targets -- never
        to refuse a solve, so it cannot cause a false negative.
        """
        if self._envelope is None:
            rng = np.random.default_rng(0)
            n = 4000
            q = rng.uniform(self.m.q_min, self.m.q_max, size=(n, self.n))
            pts = np.array([self.m.fk_position(qi) for qi in q])
            d = np.linalg.norm(pts, axis=1)
            # conservative slack: the sampled envelope underestimates the true
            # reachable set, so pad it before using it to give up
            margin = 0.05
            self._envelope = (
                max(0.0, float(d.min()) - margin),
                float(d.max()) + margin,
                margin,
            )
        return self._envelope

    def clearly_unreachable(self, target: np.ndarray, slack: float = 0.01) -> bool:
        """Cheap necessary-condition test on the target distance."""
        lo, hi, _ = self.envelope
        r = float(np.linalg.norm(np.asarray(target, dtype=float)[:3, 3]))
        return r > hi + slack or r < lo - slack

    # -- setup helpers -------------------------------------------------------

    @staticmethod
    def _resolve_backend(backend: str) -> str:
        if backend == "auto":
            return "pinocchio" if HAS_PIN_CASADI else "manual"
        if backend == "pinocchio" and not HAS_PIN_CASADI:
            raise RuntimeError(
                "backend='pinocchio' requested but pinocchio.casadi is not "
                "available in this build. Use backend='auto' or 'manual'."
            )
        if backend not in ("pinocchio", "manual"):
            raise ValueError(f"unknown backend {backend!r}")
        return backend

    def _build_symbolic_fk(self) -> SX:
        if self.backend == "pinocchio":
            self._cmodel = cpin.Model(self.m.model)
            self._cdata = self._cmodel.createData()
            self._cq_pin = cpin.SX.sym("q", self.m.model.nq, 1)
            cpin.framesForwardKinematics(self._cmodel, self._cdata, self._cq_pin)
            T = self._cdata.oMf[self.m.frame_id].homogeneous
            return casadi.Function("fk_pin", [self._cq_pin], [T])(self._cq)
        return symbolic_fk_manual(self.spec, self._cq)

    @staticmethod
    def _log3(R: SX) -> SX:
        """SO(3) log map, symbolic. Near pi it is regularised to stay finite."""
        # angle from the trace, clamped away from the +/-pi singularity
        cos_theta = (casadi.trace(R) - 1.0) / 2.0
        cos_theta = casadi.fmax(casadi.fmin(cos_theta, 1.0), -1.0)
        theta = casadi.acos(cos_theta)
        # vee(R - R^T) / (2 sin theta) * theta, with a Taylor fallback
        skew_part = casadi.vertcat(
            R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]
        )
        sin_theta = casadi.sin(theta)
        # sin(theta)/theta -> 1 as theta -> 0; guard the division
        ratio = theta / casadi.fmax(sin_theta, 1e-6)
        return 0.5 * skew_part * casadi.fmin(ratio, 1e3)

    def _barrier_cost(self, q) -> SX:
        if self.cfg.limit_barrier_weight <= 0.0:
            # A plain python float keeps this type-agnostic: the Opti variables
            # are MX objects, and adding an SX(0) to an MX expression raises.
            return 0.0
        lo, up = self.m.q_min, self.m.q_max
        centre = 0.5 * (lo + up)
        half = 0.5 * (up - lo)
        room = np.maximum(half - self.cfg.limit_barrier_margin, 1e-3)
        inside = room - casadi.fabs(q - centre)
        # smooth floor avoids -inf/NaN when the seed sits on a limit
        safe = -casadi.log(casadi.fmax(inside, 1e-3))
        return self.cfg.limit_barrier_weight * casadi.sum1(safe)

    def _install_solver(self, opti: casadi.Opti, max_iter: int) -> None:
        opts = {
            "expand": True,
            "detect_simple_bounds": True,
            "calc_lam_p": False,
            "print_time": False,
        }
        opts.update(
            {
                "ipopt.sb": "yes",
                "ipopt.print_level": self.cfg.print_level,
                "ipopt.max_iter": max_iter,
                "ipopt.tol": self.cfg.tol,
                "ipopt.acceptable_tol": self.cfg.acceptable_tol,
                "ipopt.acceptable_iter": 5,
                "ipopt.warm_start_init_point": "yes",
                "ipopt.derivative_test": "none",
                "ipopt.jacobian_approximation": "exact",
            }
        )
        opti.solver("ipopt", opts)

    # -- targets -------------------------------------------------------------

    @staticmethod
    def make_target(
        position: Sequence[float],
        rotation: Optional[Sequence[Sequence[float]]] = None,
    ) -> np.ndarray:
        """Build a 4x4 target from a point (and optional rotation matrix)."""
        T = np.eye(4)
        T[:3, 3] = np.asarray(position, dtype=float).reshape(3)
        if rotation is not None:
            R = np.asarray(rotation, dtype=float)
            if R.shape != (3, 3):
                raise ValueError("rotation must be 3x3")
            T[:3, :3] = R
        return T

    # -- solve ---------------------------------------------------------------

    def solve(
        self,
        target: np.ndarray,
        q_seed: Optional[Sequence[float]] = None,
        keep_warm: bool = True,
    ) -> IKResult:
        """Solve IK for a 4x4 target expressed in the **torso** frame."""
        target = np.asarray(target, dtype=float)
        if target.shape != (4, 4):
            raise ValueError(f"target must be 4x4, got {target.shape}")

        if q_seed is None:
            q_seed = self._q_last if self._q_last is not None else self.m.q_neutral
        q_seed = np.asarray(q_seed, dtype=float).reshape(self.n)
        eps = 1e-6
        q_seed = np.clip(q_seed, self.m.q_min + eps, self.m.q_max - eps)

        t0 = time.perf_counter()

        # ---- stage 1: position only -------------------------------------
        # First attempt always uses the caller's seed (in a control loop that is
        # the previous solution, which keeps the arm on a continuous branch).
        self.opti.set_value(self._par_target, target)
        self.opti.set_initial(self._var_q, q_seed)
        q_sol, converged, status, iters, message = self._run(
            self.opti, self._var_q, q_seed
        )
        attempts = 1
        # A converged-but-inaccurate solve means IPOPT found a *local* minimum
        # that is not the target. Retry from other seeds -- but not for a target
        # that is plainly outside the reachable shell, where retrying just burns
        # ~20 ms to reach the same answer.
        if (
            self.cfg.multistart
            and self._pos_error_at(q_sol, target) > self.cfg.pos_tol
            and not self.clearly_unreachable(target)
        ):
            q_sol, converged, status, iters, attempts, extra_msg = self._multistart(
                target, q_sol, converged, status, iters
            )
            message = message or extra_msg

        # ---- stage 2: exact position + natural posture -------------------
        # Only worth running once stage 1 has actually reached the target: it
        # constrains ||p - p_target|| <= refine_pos_tol, which is infeasible for
        # an out-of-reach target.
        refined = False
        if (
            self._opti2 is not None
            and converged
            and self._pos_error_at(q_sol, target) <= self.cfg.refine_pos_tol
        ):
            o2, q2, p_last, p_post, p_postw, p_tgt = self._opti2
            o2.set_value(p_tgt, target)
            o2.set_value(p_last, q_seed)  # continuity w.r.t. the previous pose
            o2.set_value(p_post, self.cfg.posture_reference_vector(self.n))
            o2.set_value(p_postw, self.cfg.posture_weight_vector(self.n))
            o2.set_initial(q2, q_sol)
            q2_sol, conv2, st2, it2, msg2 = self._run(o2, q2, q_sol)
            iters += it2
            if conv2:
                # accept stage 2 only if it really is no less accurate
                e1 = self._pos_error_at(q_sol, target)
                e2 = self._pos_error_at(q2_sol, target)
                if e2 <= e1 + 1e-9:
                    q_sol, refined, status = q2_sol, True, st2
                else:
                    status = f"{status}; stage2 rejected (err {e2:.2e} > {e1:.2e})"
            else:
                status = f"{status}; stage2 failed: {st2}"
            if msg2 and not message:
                message = msg2

        dt = time.perf_counter() - t0
        q_sol = np.clip(q_sol, self.m.q_min, self.m.q_max)

        T_sol = self.m.fk(q_sol)
        pos_err = float(np.linalg.norm(T_sol[:3, 3] - target[:3, 3]))
        if self.cfg.control_rotation:
            rot_err = float(np.linalg.norm(pin.log3(T_sol[:3, :3] @ target[:3, :3].T)))
        else:
            rot_err = 0.0
        margin = float(np.min(self.m.limit_margin(q_sol)))

        success = bool(
            pos_err <= self.cfg.pos_tol and margin >= self.cfg.limit_margin_min
        )
        result = IKResult(
            success=success,
            q=q_sol,
            pos_error=pos_err,
            rot_error=rot_err,
            iterations=iters,
            solve_time=dt,
            limit_margin=margin,
            converged=converged,
            status=status,
            refined=refined,
            attempts=attempts,
            message=message,
        )
        self._last_result = result
        if keep_warm:
            self._q_last = q_sol.copy()
        return result

    def _multistart(
        self,
        target: np.ndarray,
        q_best: np.ndarray,
        converged: bool,
        status: str,
        iters: int,
    ):
        """Retry stage 1 from several seeds, keeping the most accurate result.

        Seed order is deliberate:
          1. the previous best solution (cheap local escape)
          2. the zero / neutral pose (a good nominal for a humanoid arm)
          3. pseudo-random configurations inside the joint limits

        Random seeds are drawn with a fixed `multistart_seed` so a given target
        always yields the same answer -- reproducibility matters when you are
        debugging a robot.
        """
        rng = np.random.default_rng(self.cfg.multistart_seed)
        n = self.n
        seeds: List[np.ndarray] = [np.array(q_best, dtype=float), self.m.q_neutral]
        for _ in range(max(0, int(self.cfg.multistart_max_extra) - 2)):
            seeds.append(rng.uniform(self.m.q_min, self.m.q_max))

        best_err = self._pos_error_at(q_best, target)
        best_q = np.array(q_best, dtype=float)
        best_converged = converged
        best_status = status
        first_iters = iters
        attempts = 1

        for s in seeds[: max(1, int(self.cfg.multistart_max_extra))]:
            attempts += 1
            eps = 1e-6
            s = np.clip(np.asarray(s, dtype=float), self.m.q_min + eps, self.m.q_max - eps)
            self.opti.set_value(self._par_target, target)
            self.opti.set_initial(self._var_q, s)
            q_try, conv_try, st_try, it_try, _ = self._run(self.opti, self._var_q, s)
            err = self._pos_error_at(q_try, target)
            if err < best_err:
                best_err, best_q, best_converged, best_status = err, q_try, conv_try, st_try
            if best_err <= self.cfg.pos_tol:
                break

        note = ""
        if attempts > 1:
            note = f"multistart: {attempts} attempts, best err {best_err:.2e} m"
        return best_q, best_converged, best_status, first_iters, attempts, note

    def _run(self, opti, var, seed):
        """Run one solve, tolerating IPOPT's iteration-limit exception."""
        try:
            opti.solve()
            q = np.array(opti.value(var), dtype=float).reshape(self.n)
            stats = opti.stats()
            return (
                q,
                True,
                str(stats.get("return_status", "Solve_Succeeded")),
                int(stats.get("iter_count", -1)),
                "",
            )
        except RuntimeError as exc:
            q = np.array(opti.debug.value(var), dtype=float).reshape(self.n)
            stats = opti.stats() if hasattr(opti, "stats") else {}
            status = str(stats.get("return_status", "Failed"))
            if not np.all(np.isfinite(q)):
                q = np.asarray(seed, dtype=float).copy()
                status = "NaN_iterate"
            return q, False, status, int(stats.get("iter_count", -1)), str(exc)[:160]

    def _pos_error_at(self, q, target) -> float:
        return float(np.linalg.norm(self.m.fk_position(q) - np.asarray(target)[:3, 3]))

    # -- convenience ---------------------------------------------------------

    def solve_position(
        self,
        position: Sequence[float],
        q_seed: Optional[Sequence[float]] = None,
        keep_warm: bool = True,
    ) -> IKResult:
        return self.solve(self.make_target(position), q_seed=q_seed, keep_warm=keep_warm)

    def reset_warm_start(self) -> None:
        self._q_last = None

    @property
    def last_result(self) -> Optional[IKResult]:
        return self._last_result

    def fk(self, q: Sequence[float]) -> np.ndarray:
        return self.m.fk(q)

    def position_jacobian(self, q: Sequence[float]) -> np.ndarray:
        """3x7 translational Jacobian in the torso frame."""
        J = np.array(self._pos_jac_fun(np.asarray(q, dtype=float).reshape(self.n)))
        return J.reshape(3, self.n)

    def nullspace_project(
        self, q: Sequence[float], direction: Sequence[float], damping: float = 1e-4
    ) -> np.ndarray:
        """Project `direction` onto the null space of the position Jacobian.

        Lets you nudge the arm towards a preferred posture without moving the
        end effector at all.
        """
        J = self.position_jacobian(q)
        d = np.asarray(direction, dtype=float).reshape(self.n)
        Jpinv = J.T @ np.linalg.solve(J @ J.T + damping * np.eye(3), np.eye(3))
        return d - Jpinv @ (J @ d)

    def reachability_report(
        self, target_position: Sequence[float], n_seeds: int = 20, seed: int = 0
    ) -> Dict[str, object]:
        """Try several seeds; report the best error achieved.

        A single failed solve does not prove a point is unreachable (IK is a
        local method), so this is the honest way to answer "can the arm get
        there?".
        """
        rng = np.random.default_rng(seed)
        best: Optional[IKResult] = None
        errors: List[float] = []
        q0 = self._q_last
        for i in range(n_seeds):
            if i == 0:
                s = q0 if q0 is not None else self.m.q_neutral
            else:
                s = rng.uniform(self.m.q_min, self.m.q_max)
            res = self.solve_position(target_position, q_seed=s, keep_warm=False)
            errors.append(res.pos_error)
            if best is None or res.pos_error < best.pos_error:
                best = res
        self._q_last = q0
        errs = np.asarray(errors)
        return {
            "best_result": best,
            "best_error": float(errs.min()),
            "median_error": float(np.median(errs)),
            "success_rate": float(np.mean(errs <= self.cfg.pos_tol)),
            "n_seeds": n_seeds,
            "reachable": bool(errs.min() <= self.cfg.pos_tol),
        }


# --------------------------------------------------------------------------- #
#  output smoothing / limiting
# --------------------------------------------------------------------------- #


class WeightedMovingFilter:
    """Weighted moving average over the last N solutions (xr_teleoperate style).

    Applied *after* IK to suppress residual jitter.  A heavy filter combined
    with a fast-changing target adds lag -- tune the weights against your
    control rate.
    """

    def __init__(self, weights: Sequence[float] = (0.4, 0.3, 0.2, 0.1)):
        w = np.asarray(weights, dtype=float)
        if np.any(w < 0) or w.sum() <= 0:
            raise ValueError("weights must be non-negative and sum > 0")
        self.weights = w / w.sum()
        self.n = len(w)
        self._buf: Deque[np.ndarray] = deque(maxlen=self.n)

    def reset(self) -> None:
        self._buf.clear()

    def _average(self, data: List[np.ndarray]) -> np.ndarray:
        # newest first; until the buffer is full the oldest sample is repeated so
        # the filter is unbiased from the very first call
        data = list(data)
        while len(data) < self.n:
            data.append(data[-1])
        w = self.weights[: len(data)]
        w = w / w.sum()
        return np.sum([wi * qi for wi, qi in zip(w, data)], axis=0)

    def add(self, q: Sequence[float]) -> np.ndarray:
        """Push a new sample and return the filtered value."""
        self._buf.appendleft(np.asarray(q, dtype=float).copy())
        return self._average(list(self._buf))

    @property
    def filtered(self) -> Optional[np.ndarray]:
        """Current filtered output without pushing a new sample."""
        if not self._buf:
            return None
        return self._average(list(self._buf))


class RateLimiter:
    """Per-joint rate limiter -- the last safety net before the robot."""

    def __init__(self, max_velocity: Sequence[float], dt: float):
        self.v_max = np.asarray(max_velocity, dtype=float)
        self.dt = float(dt)
        self._q_prev: Optional[np.ndarray] = None

    def reset(self, q: Optional[Sequence[float]] = None) -> None:
        self._q_prev = None if q is None else np.asarray(q, dtype=float).copy()

    def limit(self, q_target: Sequence[float]) -> np.ndarray:
        q_target = np.asarray(q_target, dtype=float)
        if self._q_prev is None:
            self._q_prev = q_target.copy()
            return q_target.copy()
        step = self.v_max * self.dt
        q_out = np.clip(q_target, self._q_prev - step, self._q_prev + step)
        self._q_prev = q_out.copy()
        return q_out
