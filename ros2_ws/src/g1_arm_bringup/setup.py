from setuptools import setup

package_name = "g1_arm_bringup"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        # NOTE: launch files are installed explicitly here rather than via a
        # glob, so a missing/renamed file fails loudly at build time.
        ("share/" + package_name + "/launch", [
            "launch/arm_ik.launch.py",
            "launch/arm_sim.launch.py",
            "launch/arm_bringup.launch.py",
            "launch/arm_vla.launch.py",
        ]),
        ("share/" + package_name + "/config", [
            "config/ik_params.yaml",
            "config/control_params.yaml",
            "config/g1_left_arm.yaml",
            "config/g1_right_arm.yaml",
            "config/vla_params.yaml",
        ]),
        ("share/" + package_name + "/urdf", [
            "urdf/g1_29dof_rev_1_0.urdf",
            "urdf/g1_29dof_with_hand_rev_1_0.urdf",
            "urdf/g1_dual_arm.urdf",
        ]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="g1_arm maintainer",
    maintainer_email="user@example.com",
    description="Bridges, TF and launch files for Unitree G1 single-arm control",
    license="BSD-3-Clause",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "joint_state_relay = g1_arm_bringup.joint_state_relay:main",
            "tf_broadcaster = g1_arm_bringup.tf_broadcaster:main",
            "configure_urdf = g1_arm_bringup.configure_urdf:main",
        ],
    },
)
