"""Pinocchio-backed model for one arm (7 DOF) + virtual end-effector frame.

Responsibilities
----------------
* load the *reduced* URDF produced by reduced_model.py
* expose q <-> named joints mapping (deterministic, index-verified)
* numeric forward kinematics
* the exact frame/link ids that the CasADi IK builds its symbolic model from

This wrapper is intentionally thin: all the frame/freezing decisions were made
in reduced_model.build_reduced_arm_urdf, so pinocchio just consumes them.  That
keeps the numpy cross-check and the optimiser on the same ground truth.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

try:
    import pinocchio as pin
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "pinocchio is required. Install with `pip install pin` (python>=3.10 is "
        "needed on macOS arm64 because the coal/cmeel-assimp wheels require it)."
    ) from exc

from .reduced_model import ReducedArmModel, build_reduced_arm_urdf


class PinArmModel:
    """A 7-DOF pinocchio model with a virtual TCP frame."""

    def __init__(self, spec: ReducedArmModel, verbose: bool = False):
        self.spec = spec
        self.model = pin.buildModelFromUrdf(spec.urdf_path)
        self.data = self.model.createData()
        self.verbose = verbose

        # --- verify the q ordering assumption explicitly -------------------
        # pinocchio orders q by joint declaration order in the URDF (skipping
        # the root). The reduced URDF was generated so that this equals the arm
        # joint order, but never trust it silently.
        # `model.names` is 1-based (it starts with "universe" at index 0), so
        # iterate over joint ids directly instead of zipping the two lists --
        # zipping pairs each name with the *previous* joint, which silently
        # mis-scans the model.
        pin_names: List[str] = []
        self.joint_idx_q: Dict[str, int] = {}
        for jid in range(1, self.model.njoints):
            jmodel = self.model.joints[jid]
            name = self.model.names[jid]
            # nq == 0 means the joint contributes nothing to q (fixed joints).
            if jmodel.nq == 0:
                continue
            if jmodel.nq != 1:
                raise NotImplementedError(
                    f"joint {name!r} has nq={jmodel.nq}; only 1-DOF joints are "
                    f"supported in the reduced model"
                )
            pin_names.append(name)
            self.joint_idx_q[name] = jmodel.idx_q

        if pin_names != list(spec.joint_names):
            raise RuntimeError(
                "pinocchio joint order mismatch!\n"
                f"  pinocchio: {pin_names}\n"
                f"  spec     : {list(spec.joint_names)}"
            )

        self.nq = self.model.nq
        if self.nq != len(spec.joint_names):
            raise RuntimeError(f"nq={self.nq} but {len(spec.joint_names)} joints")

        self.frame_id = self.model.getFrameId(spec.ee_frame)
        if self.frame_id >= len(self.model.frames):
            raise RuntimeError(f"EE frame {spec.ee_frame!r} not found in model")

        self.q_min = np.array(spec.lower, dtype=float)
        self.q_max = np.array(spec.upper, dtype=float)
        self.q_neutral = np.clip(np.zeros(self.nq), self.q_min, self.q_max)

        # Sanity: pinocchio's own limits should match the URDF we parsed.
        lo = np.array(self.model.lowerPositionLimit, dtype=float)
        up = np.array(self.model.upperPositionLimit, dtype=float)
        if not (np.allclose(lo, self.q_min, atol=1e-9) and np.allclose(up, self.q_max, atol=1e-9)):
            raise RuntimeError(
                "joint limits differ between the parser and pinocchio:\n"
                f"  parsed   : {self.q_min}\n             {self.q_max}\n"
                f"  pinocchio: {lo}\n             {up}"
            )

    # -- helpers -------------------------------------------------------------

    def to_vector(self, q_map: Dict[str, float]) -> np.ndarray:
        q = self.q_neutral.copy()
        for name, val in q_map.items():
            if name not in self.joint_idx_q:
                raise KeyError(f"unknown arm joint {name!r}")
            q[self.joint_idx_q[name]] = val
        return q

    def to_dict(self, q: Sequence[float]) -> Dict[str, float]:
        q = np.asarray(q, dtype=float)
        return {n: float(q[self.joint_idx_q[n]]) for n in self.spec.joint_names}

    def clamp(self, q: Sequence[float]) -> np.ndarray:
        return np.clip(np.asarray(q, dtype=float), self.q_min, self.q_max)

    # -- kinematics ----------------------------------------------------------

    def fk(self, q: Sequence[float]) -> np.ndarray:
        """T (4x4) of the EE frame expressed in the torso frame."""
        q = np.asarray(q, dtype=float).reshape(self.nq)
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        return np.array(self.data.oMf[self.frame_id].homogeneous, dtype=float)

    def fk_position(self, q: Sequence[float]) -> np.ndarray:
        return self.fk(q)[:3, 3]

    def fk_all_frames(self, q: Sequence[float]) -> Dict[str, np.ndarray]:
        """All link poses in the torso frame -- handy for collision checking.

        Note: in pinocchio 4.x a Frame object has no `.id` attribute, so the
        index into `data.oMf` must be tracked explicitly.
        """
        q = np.asarray(q, dtype=float).reshape(self.nq)
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        out = {}
        for i in range(self.model.nframes):
            f = self.model.frames[i]
            if f.name in ("universe",):
                continue
            out[f.name] = np.array(self.data.oMf[i].homogeneous, dtype=float)
        return out

    def joint_positions(self, q: Sequence[float]) -> Dict[str, np.ndarray]:
        """World(torso)-frame origins of every joint -- for plotting/debug."""
        q = np.asarray(q, dtype=float).reshape(self.nq)
        pin.forwardKinematics(self.model, self.data, q)
        out = {}
        for name, jmodel in zip(self.model.names, self.model.joints):
            if jmodel.id == 0:
                continue
            out[name] = np.array(self.data.oMi[jmodel.id].translation, dtype=float)
        return out

    def limit_margin(self, q: Sequence[float]) -> np.ndarray:
        """Distance to the nearest limit, per joint (positive = inside)."""
        q = np.asarray(q, dtype=float)
        return np.minimum(q - self.q_min, self.q_max - q)

    def __repr__(self) -> str:
        return (
            f"PinArmModel(side={self.spec.ee_frame}, nq={self.nq}, "
            f"root={self.spec.root_link}, ee={self.spec.ee_frame})"
        )


def build_pin_arm_model(
    full_urdf_path: str,
    side: str,
    ee_offset: Sequence[float] = (0.05, 0.0, 0.0),
    base_link: str = "torso_link",
    out_path: Optional[str] = None,
    verbose: bool = False,
) -> PinArmModel:
    spec = build_reduced_arm_urdf(
        full_urdf_path=full_urdf_path,
        side=side,
        ee_offset=ee_offset,
        base_link=base_link,
        out_path=out_path,
    )
    return PinArmModel(spec, verbose=verbose)
