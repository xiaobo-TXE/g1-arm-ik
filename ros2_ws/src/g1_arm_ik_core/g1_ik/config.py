"""Load g1_ik settings from a YAML file.

Kept deliberately small and explicit: a missing key falls back to the dataclass
default rather than silently becoming None, and unknown keys are rejected so a
typo in the YAML does not silently do nothing.
"""

from __future__ import annotations

import dataclasses
import os
from typing import Any, Dict, Optional

import yaml

from .ik import IKConfig


def load_config(path: str) -> Dict[str, Any]:
    """Parse the YAML config into a plain dict with resolved paths."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"config file not found: {path}")
    with open(path) as fh:
        raw = yaml.safe_load(fh) or {}

    known = {
        "urdf",
        "side",
        "ik_base_link",
        "ee_offset",
        "waist_joint_order",
        "waist_default",
        "ik",
        "output",
    }
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")

    cfg_dir = os.path.dirname(os.path.abspath(path))
    urdf = raw.get("urdf", "urdf/g1_29dof_rev_1_0.urdf")
    if not os.path.isabs(urdf) and not os.path.exists(urdf):
        # The config lives in <bringup>/config and the URDF in <bringup>/urdf, so
        # a path like "urdf/x.urdf" is resolved against the package directory.
        # Under colcon the two end up in the same install share/ directory too,
        # so this also covers the installed layout.
        tried = [urdf]
        pkg_dir = os.path.dirname(cfg_dir)
        candidates = [os.path.join(pkg_dir, urdf), os.path.join(cfg_dir, urdf)]
        try:  # installed layout: share/g1_arm_bringup/{config,urdf}
            from ament_index_python.packages import get_package_share_directory

            share = get_package_share_directory("g1_arm_bringup")
            candidates.append(os.path.join(share, urdf))
        except Exception:  # noqa: BLE001 - ament_index may be unavailable
            pass

        for candidate in candidates:
            tried.append(candidate)
            if os.path.exists(candidate):
                urdf = candidate
                break
        else:
            raise FileNotFoundError(
                "URDF not found. Tried:\n  " + "\n  ".join(tried)
            )

    out: Dict[str, Any] = {
        "urdf": urdf,
        "side": raw.get("side", "left"),
        "ik_base_link": raw.get("ik_base_link", "torso_link"),
        "ee_offset": tuple(raw.get("ee_offset", (0.05, 0.0, 0.0))),
        "waist_joint_order": tuple(
            raw.get(
                "waist_joint_order",
                ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"),
            )
        ),
        "waist_default": tuple(raw.get("waist_default", (0.0, 0.0, 0.0))),
    }

    out["ik"] = build_ik_config(raw.get("ik") or {})

    # output conditioning settings (used by the caller's control loop)
    out_raw = raw.get("output") or {}
    known_out = {
        "control_dt",
        "moving_filter_weights",
        "max_velocity_override",
    }
    unknown = set(out_raw) - known_out
    if unknown:
        raise ValueError(f"unknown output config keys: {sorted(unknown)}")
    out["output"] = {
        "control_dt": out_raw.get("control_dt", 0.01),
        "moving_filter_weights": tuple(
            out_raw.get("moving_filter_weights", (0.4, 0.3, 0.2, 0.1))
        ),
        "max_velocity_override": out_raw.get("max_velocity_override"),
    }
    return out


def build_ik_config(values: Optional[Dict[str, Any]] = None) -> IKConfig:
    """Instantiate IKConfig from a dict, rejecting unknown keys."""
    values = dict(values or {})
    fields = {f.name: f for f in dataclasses.fields(IKConfig)}
    unknown = set(values) - set(fields)
    if unknown:
        raise ValueError(
            f"unknown ik config keys: {sorted(unknown)}\n"
            f"available: {sorted(fields)}"
        )
    kwargs = {}
    for key, val in values.items():
        if key in ("posture_weight", "posture_reference") and val is not None:
            val = tuple(val)
        kwargs[key] = val
    cfg = IKConfig(**kwargs)
    _validate_ik_config(cfg)
    return cfg


def _validate_ik_config(cfg: IKConfig) -> None:
    """Reject nonsense numbers before they reach the solver.

    A silently-wrong weight does not crash -- it produces a solver that quietly
    stops short of the target, which is exactly the failure mode that is hardest
    to notice on a robot.
    """
    problems = []
    if cfg.position_weight <= 0.0:
        problems.append(f"position_weight must be > 0, got {cfg.position_weight}")
    if cfg.rotation_weight < 0.0:
        problems.append(f"rotation_weight must be >= 0, got {cfg.rotation_weight}")
    if cfg.control_rotation and cfg.rotation_weight <= 0.0:
        problems.append(
            "control_rotation is on but rotation_weight is 0 -- orientation would "
            "be ignored; set rotation_weight > 0 or turn control_rotation off"
        )
    if cfg.max_iter < 1:
        problems.append(f"max_iter must be >= 1, got {cfg.max_iter}")
    if cfg.tol <= 0.0 or cfg.acceptable_tol <= 0.0:
        problems.append("tol/acceptable_tol must be > 0")
    if cfg.pos_tol <= 0.0:
        problems.append(f"pos_tol must be > 0, got {cfg.pos_tol}")
    if cfg.refine_pos_tol <= 0.0:
        problems.append(f"refine_pos_tol must be > 0, got {cfg.refine_pos_tol}")
    if cfg.limit_barrier_weight < 0.0:
        problems.append("limit_barrier_weight must be >= 0 (0 disables it)")
    if cfg.limit_barrier_margin < 0.0:
        problems.append("limit_barrier_margin must be >= 0")
    if cfg.multistart_max_extra < 1:
        problems.append(
            f"multistart_max_extra must be >= 1, got {cfg.multistart_max_extra}"
        )
    if problems:
        raise ValueError("invalid IK config:\n  " + "\n  ".join(problems))


def load_arm_from_config(path: str, **overrides):
    """Build a ready G1Arm straight from a YAML config file."""
    from . import G1Arm  # local import to avoid a cycle

    cfg = load_config(path)
    kwargs = dict(
        full_urdf_path=cfg["urdf"],
        side=cfg["side"],
        ee_offset=cfg["ee_offset"],
        base_link=cfg["ik_base_link"],
        config=cfg["ik"],
    )
    kwargs.update(overrides)
    arm = G1Arm(**kwargs)
    arm.waist_joint_order = cfg["waist_joint_order"]
    arm.waist_default = cfg["waist_default"]
    arm.output_config = cfg["output"]
    return arm
