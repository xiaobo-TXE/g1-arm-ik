"""Minimal but faithful URDF parser for kinematic chains.

Only the fields needed for kinematics are parsed: links, joints
(name / type / parent / child / origin xyz+rpy / axis / limit).
Meshes and inertias are treated as opaque blobs and dropped.

No third-party dependency: stdlib xml.etree only. This matters because the
parser is the *independent* reference used to cross-check pinocchio.
"""

from __future__ import annotations

import dataclasses
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple

import numpy as np


def rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """URDF convention: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=float)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=float)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=float)
    return rz @ ry @ rx


def axis_angle_to_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues formula. `axis` must be a unit vector."""
    axis = np.asarray(axis, dtype=float)
    n = np.linalg.norm(axis)
    if n < 1e-12:
        return np.eye(3)
    k = axis / n
    K = np.array(
        [[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]], dtype=float
    )
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


def make_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = rotation
    T[:3, 3] = translation
    return T


@dataclasses.dataclass
class Joint:
    name: str
    type: str  # revolute | continuous | prismatic | fixed | floating | planar
    parent: str
    child: str
    origin_xyz: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    origin_rpy: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    axis: Tuple[float, float, float] = (1.0, 0.0, 0.0)
    lower: Optional[float] = None
    upper: Optional[float] = None
    effort: Optional[float] = None
    velocity: Optional[float] = None

    @property
    def is_actuated(self) -> bool:
        return self.type in ("revolute", "continuous", "prismatic")

    @property
    def origin_transform(self) -> np.ndarray:
        return make_transform(
            rpy_to_matrix(*self.origin_rpy), np.array(self.origin_xyz, dtype=float)
        )

    def joint_transform(self, q: float) -> np.ndarray:
        """Transform contributed by this joint's own motion (origins excluded)."""
        if self.type in ("revolute", "continuous"):
            return make_transform(
                axis_angle_to_matrix(np.array(self.axis, dtype=float), q),
                np.zeros(3),
            )
        if self.type == "prismatic":
            a = np.array(self.axis, dtype=float)
            n = np.linalg.norm(a)
            a = a / n if n > 1e-12 else a
            return make_transform(np.eye(3), a * q)
        return np.eye(4)


@dataclasses.dataclass
class UrdfModel:
    name: str
    links: List[str]
    joints: Dict[str, Joint]
    child_to_joint: Dict[str, str]  # child link name -> joint name
    joint_order: List[str]  # declaration order
    root_link: str

    def parent_joint(self, link: str) -> Optional[Joint]:
        jn = self.child_to_joint.get(link)
        return self.joints[jn] if jn else None

    def chain_to_root(self, link: str) -> List[Joint]:
        """Joints from the root link down to `link`, in root->leaf order."""
        chain: List[Joint] = []
        cur = link
        while True:
            j = self.parent_joint(cur)
            if j is None:
                break
            chain.append(j)
            cur = j.parent
        chain.reverse()
        return chain

    def actuated_joints(self) -> List[str]:
        return [n for n in self.joint_order if self.joints[n].is_actuated]

    def joint_names_in_chain(self, link: str) -> List[str]:
        return [j.name for j in self.chain_to_root(link) if j.is_actuated]


def parse_urdf(path: str) -> UrdfModel:
    tree = ET.parse(path)
    robot = tree.getroot()
    robot_name = robot.get("name", "robot")

    links: List[str] = []
    joints: Dict[str, Joint] = {}
    child_to_joint: Dict[str, str] = {}
    joint_order: List[str] = []

    for el in robot:
        tag = el.tag.split("}")[-1]  # tolerate namespaces
        if tag == "link":
            links.append(el.get("name"))
        elif tag == "joint":
            name = el.get("name")
            jtype = el.get("type")
            parent_el = el.find("parent")
            child_el = el.find("child")
            if parent_el is None or child_el is None:
                raise ValueError(f"joint '{name}' missing parent/child")
            parent = parent_el.get("link")
            child = child_el.get("link")

            xyz = (0.0, 0.0, 0.0)
            rpy = (0.0, 0.0, 0.0)
            origin_el = el.find("origin")
            if origin_el is not None:
                if origin_el.get("xyz"):
                    xyz = tuple(float(v) for v in origin_el.get("xyz").split())
                if origin_el.get("rpy"):
                    rpy = tuple(float(v) for v in origin_el.get("rpy").split())

            axis = (1.0, 0.0, 0.0)
            axis_el = el.find("axis")
            if axis_el is not None and axis_el.get("xyz"):
                axis = tuple(float(v) for v in axis_el.get("xyz").split())

            lower = upper = effort = velocity = None
            limit_el = el.find("limit")
            if limit_el is not None:
                if limit_el.get("lower") is not None:
                    lower = float(limit_el.get("lower"))
                if limit_el.get("upper") is not None:
                    upper = float(limit_el.get("upper"))
                if limit_el.get("effort") is not None:
                    effort = float(limit_el.get("effort"))
                if limit_el.get("velocity") is not None:
                    velocity = float(limit_el.get("velocity"))

            joints[name] = Joint(
                name=name,
                type=jtype,
                parent=parent,
                child=child,
                origin_xyz=xyz,
                origin_rpy=rpy,
                axis=axis,
                lower=lower,
                upper=upper,
                effort=effort,
                velocity=velocity,
            )
            child_to_joint[child] = name
            joint_order.append(name)

    child_links = set(child_to_joint.keys())
    roots = [l for l in links if l not in child_links]
    if not roots:
        raise ValueError("no root link found (cycle in URDF)")
    if len(roots) > 1:
        # `world`-style helper roots are common; prefer the one with the most
        # actuated joints underneath it.
        def _count(link: str) -> int:
            return sum(1 for j in joints.values() if _under(link, j, joints))

        def _under(anc: str, j: Joint, allj: Dict[str, Joint]) -> bool:
            cur = j.child
            while True:
                pj = allj.get(child_to_joint.get(cur, ""))
                if pj is None:
                    return False
                if pj.parent == anc:
                    return True
                cur = pj.parent

        roots.sort(key=_count, reverse=True)

    return UrdfModel(
        name=robot_name,
        links=links,
        joints=joints,
        child_to_joint=child_to_joint,
        joint_order=joint_order,
        root_link=roots[0],
    )
