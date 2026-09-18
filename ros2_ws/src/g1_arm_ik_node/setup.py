from setuptools import find_packages, setup

package_name = "g1_arm_ik_node"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="g1_arm maintainer",
    maintainer_email="user@example.com",
    description="Inverse kinematics node for one Unitree G1 arm",
    license="BSD-3-Clause",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "ik_node = g1_arm_ik_node.ik_node:main",
            "vla_node = g1_arm_ik_node.vla_node:main",
        ],
    },
)
