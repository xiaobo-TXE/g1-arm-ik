"""Pytest configuration for the g1_arm_ik_core test suite.

Everything here runs with **no ROS 2 installed**, so the kinematics and the
control logic stay testable in isolation. It intentionally does not import
rclpy.

Path shim: colcon installs the packages, but these tests also have to run
straight from the source tree (and on a machine that has no ROS at all), so the
package root is put on sys.path here rather than relying on an install step.
"""

import os
import sys

_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)
