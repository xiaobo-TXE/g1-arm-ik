#!/usr/bin/env python3
"""Run every offline check and print one combined report.

    .venv312/bin/python run_all_checks.py

Covers:
  * FK / parser validation          (no ROS)
  * IK accuracy, safety, timing     (no ROS)
  * control logic: watchdog, ramp, rate limiting, mock backend (no ROS)
  * ROS 2 workspace static validation (packages, msg, entry points, layering)

Exit code 0 only if everything passes.

This machine cannot run `colcon build`; see README.md ("已知限制与待办") for what still has
to be verified on the Ubuntu 22.04 + Humble target.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from typing import Callable, List, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
CORE = os.path.join(HERE, "ros2_ws", "src", "g1_arm_ik_core")
CORE_TEST = os.path.join(CORE, "test")


def _load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    for p in (CORE, CORE_TEST):
        if p not in sys.path:
            sys.path.insert(0, p)

    suites: List[Tuple[str, Callable[[], int]]] = []

    fk = _load("check_fk", os.path.join(CORE_TEST, "test_fk.py"))
    ik = _load("check_ik", os.path.join(CORE_TEST, "test_ik.py"))
    ctl = _load("check_control", os.path.join(CORE_TEST, "test_control.py"))
    vla = _load("check_vla_protocol", os.path.join(CORE_TEST, "test_vla_protocol.py"))
    zmq = _load("check_zmq_bridge", os.path.join(CORE_TEST, "test_zmq_bridge.py"))
    vlogic = _load("check_vla_node_logic", os.path.join(CORE_TEST, "test_vla_node_logic.py"))
    val = _load("check_workspace", os.path.join(HERE, "ros2_ws", "validate_workspace.py"))

    suites.append(("FK / URDF parser", fk.main))
    suites.append(("IK accuracy & safety", ik.main))
    suites.append(("Control logic (no ROS)", ctl.main))
    suites.append(("VLA frame format (ZMQ 6002)", vla.main))
    suites.append(("ZMQ transport (real sockets)", zmq.main))
    suites.append(("VLA node decision logic", vlogic.main))
    suites.append(("ROS 2 workspace (static)", val.main))

    results = []
    for name, fn in suites:
        print()
        print("#" * 78)
        print(f"# {name}")
        print("#" * 78)
        try:
            rc = fn()
        except Exception as exc:  # noqa: BLE001
            print(f"  SUITE CRASHED: {type(exc).__name__}: {exc}")
            rc = 1
        results.append((name, rc))

    print()
    print("=" * 78)
    print("COMBINED RESULT")
    print("=" * 78)
    for name, rc in results:
        print(f"  {'PASS' if rc == 0 else 'FAIL'}  {name}")
    failed = sum(1 for _, rc in results if rc != 0)
    print()
    print("all suites passed" if not failed else f"{failed} suite(s) failed")
    if not failed:
        print()
        print("Next: run `colcon build` on Ubuntu 22.04 + Humble.")
        print("Build/run steps: README.md (Build / Usage)")
        print("ROS-specific notes: ros2_ws/README.md")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
