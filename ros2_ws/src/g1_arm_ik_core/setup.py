from setuptools import find_packages, setup

package_name = "g1_arm_ik_core"

setup(
    name=package_name,
    version="0.1.0",
    # Two top-level packages are shipped:
    #   g1_ik              -- the pure kinematics library (no ROS anywhere)
    #   g1_arm_ik_core     -- command shaping / safety / robot backends
    packages=find_packages(exclude=["test", "test.*"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=[
        "setuptools",
        "numpy",
        # pinocchio + casadi are NOT installed by colcon/rosdep here: the
        # dependency chain (coal -> cmeel-assimp) is not packaged for ROS.
        # Install them with pip or conda first -- see the repository README.md.
        "pinocchio",
        "casadi",
        "PyYAML",
    ],
    zip_safe=True,
    maintainer="g1_arm maintainer",
    maintainer_email="user@example.com",
    description="ROS-free kinematics core for one Unitree G1 arm (7 DOF)",
    license="BSD-3-Clause",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [],
    },
)
