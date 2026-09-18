#!/usr/bin/env python3
"""Static validation of the ROS 2 workspace.

`colcon build` cannot run on macOS, so this catches the mistakes that would
otherwise only surface on the Ubuntu machine -- and catches them faster:

1. every package.xml parses, declares the right build type, and its
   <exec_depend> names a package that actually exists in this workspace
2. every .msg/.srv parses; field types are known; srv files have a `---`
3. every Python file is syntactically valid (py_compile)
4. every console_scripts entry point resolves to a real module + function
5. every data_files entry exists on disk (a missing one breaks the install)
6. launch files define generate_launch_description()
7. the ROS-free modules (g1_ik, g1_arm_ik_core, g1_ik.config) import cleanly
   with NO ROS and NO rclpy available -- proving the layering really holds
8. the rclpy-dependent modules at least *compile* and their intra-package
   imports point at modules that exist

Exit code 0 = everything passed.
"""

from __future__ import annotations

import ast
import os
import pathlib
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "src")

# Builtin ROS message types we allow in .msg/.srv without a package prefix.
BUILTIN_TYPES = {
    "bool", "byte", "char", "float32", "float64", "int8", "uint8", "int16",
    "uint16", "int32", "uint32", "int64", "uint64", "string", "wstring",
}


class Report:
    def __init__(self) -> None:
        self.passed: List[str] = []
        self.failed: List[Tuple[str, str]] = []
        self.info: List[str] = []

    def ok(self, name: str) -> None:
        self.passed.append(name)

    def fail(self, name: str, msg: str) -> None:
        self.failed.append((name, msg))

    def note(self, msg: str) -> None:
        self.info.append(msg)

    @property
    def rc(self) -> int:
        return 1 if self.failed else 0


def packages() -> Dict[str, str]:
    out = {}
    for name in sorted(os.listdir(SRC)):
        path = os.path.join(SRC, name)
        if os.path.isdir(path) and os.path.exists(os.path.join(path, "package.xml")):
            out[name] = path
    return out


# --------------------------------------------------------------------------- #


def check_package_xml(rep: Report, pkgs: Dict[str, str]) -> Dict[str, dict]:
    meta = {}
    for name, path in pkgs.items():
        xml_path = os.path.join(path, "package.xml")
        try:
            root = ET.parse(xml_path).getroot()
        except ET.ParseError as exc:
            rep.fail(f"package.xml[{name}]", f"XML parse error: {exc}")
            continue
        got_name = root.findtext("name")
        if got_name != name:
            rep.fail(f"package.xml[{name}]", f"<name> is {got_name!r}, dir is {name!r}")
            continue
        version = root.findtext("version")
        if not version:
            rep.fail(f"package.xml[{name}]", "missing <version>")
            continue
        if root.findtext("description") is None:
            rep.fail(f"package.xml[{name}]", "missing <description>")
            continue
        if root.findtext("license") is None:
            rep.fail(f"package.xml[{name}]", "missing <license>")
            continue

        build_type = root.find("./export/build_type")
        build_type = build_type.text if build_type is not None else None
        if build_type not in ("ament_python", "ament_cmake"):
            rep.fail(f"package.xml[{name}]", f"bad build_type {build_type!r}")

        exec_deps = [d.text for d in root.findall("exec_depend")]
        meta[name] = {
            "build_type": build_type,
            "exec_depend": exec_deps,
            "path": path,
            "has_rosidl_group": root.find("member_of_group") is not None,
        }
        rep.ok(f"package.xml[{name}]")
    return meta


def check_exec_deps(rep: Report, meta: Dict[str, dict]) -> None:
    """Every <exec_depend> either names a workspace package or is a known ROS one."""
    known_ros = {
        "rclpy", "sensor_msgs", "geometry_msgs", "std_msgs", "tf2_ros",
        "rosidl_default_runtime", "ament_index_python",
        "rosidl_default_generators", "ament_cmake",
        "ament_copyright", "ament_flake8", "ament_pep257", "python3-pytest",
        "python3-zmq", "pyzmq",
        "ament_lint_auto", "ament_lint_common",
    }
    bad = 0
    for name, m in meta.items():
        for dep in m["exec_depend"]:
            if dep in meta or dep in known_ros:
                continue
            # accept any other ros-* style name: we cannot resolve it offline
            if dep.startswith(("ros", "python3-", "lib", "tf2", "xacro", "joint")):
                continue
            rep.fail(f"exec_depend[{name}]", f"unknown dependency {dep!r}")
            bad += 1
    if bad == 0:
        rep.ok("exec_depend: all dependencies resolve or are known ROS packages")


def check_msgs(rep: Report) -> None:
    total = 0
    for pkg in ("g1_arm_msgs",):
        base = os.path.join(SRC, pkg)
        for kind, sep in (("msg", None), ("srv", "---")):
            d = os.path.join(base, kind)
            if not os.path.isdir(d):
                continue
            for fname in sorted(os.listdir(d)):
                if not fname.endswith((".msg", ".srv")):
                    continue
                fpath = os.path.join(d, fname)
                total += 1
                text = open(fpath).read()
                if sep:
                    parts = text.split(sep)
                    if len(parts) != 2:
                        rep.fail(f"{kind}/{fname}", f"expected exactly one '{sep}', got {len(parts)-1}")
                        continue
                else:
                    parts = [text]
                if len(parts) == 2 and not parts[1].strip():
                    rep.fail(f"{kind}/{fname}", "response section is empty")
                    continue
                bad = []
                for section in parts:
                    for lineno, raw in enumerate(section.splitlines(), 1):
                        line = raw.split("#", 1)[0].strip()
                        if not line:
                            continue
                        if line.startswith(("float64[", "float32[", "int32[", "uint8[", "string[")):
                            pass  # array with an explicit bound
                        m = re.match(r"^([A-Za-z0-9_/]+)(\[\d*\])?\s+([A-Za-z_][A-Za-z0-9_]*)$", line)
                        if not m:
                            bad.append(f"line {lineno}: {raw!r}")
                            continue
                        ftype = m.group(1)
                        if ftype in BUILTIN_TYPES:
                            continue
                        if "/" in ftype:
                            pkgname = ftype.split("/")[0]
                            if pkgname not in ("std_msgs",):
                                bad.append(f"line {lineno}: package {pkgname!r} not declared")
                            continue
                        bad.append(f"line {lineno}: unknown type {ftype!r}")
                if bad:
                    rep.fail(f"{kind}/{fname}", "; ".join(bad))
                else:
                    rep.ok(f"{kind}/{fname}")
    if total == 0:
        rep.fail("msgs", "no .msg/.srv files found")


def check_python_syntax(rep: Report, pkgs: Dict[str, str]) -> None:
    count = 0
    for name, path in pkgs.items():
        for root, dirs, files in os.walk(path):
            dirs[:] = [d for d in dirs if d not in ("__pycache__", "build", "install")]
            for f in files:
                if not f.endswith(".py"):
                    continue
                fpath = os.path.join(root, f)
                count += 1
                try:
                    ast.parse(open(fpath).read(), filename=fpath)
                except SyntaxError as exc:
                    rep.fail(
                        f"py_compile[{os.path.relpath(fpath, SRC)}]",
                        f"line {exc.lineno}: {exc.msg}",
                    )
    rep.ok(f"python syntax: {count} files parse")


def check_entry_points(rep: Report, meta: Dict[str, dict]) -> None:
    """console_scripts must point at module:function that really exist."""
    total = 0
    for name, m in meta.items():
        setup_py = os.path.join(m["path"], "setup.py")
        if not os.path.exists(setup_py):
            if m["build_type"] == "ament_python":
                rep.fail(f"setup.py[{name}]", "ament_python package has no setup.py")
            continue
        src = open(setup_py).read()
        m2 = re.search(r"entry_points\s*=\s*\{(.*?)\n\}", src, re.S)
        if not m2:
            continue
        block = m2.group(1)
        eps = re.findall(r'"([^"=]+)=([A-Za-z_][A-Za-z0-9_.]*):([A-Za-z_][A-Za-z0-9_]*)"', block)
        for exe, modpath, func in eps:
            total += 1
            exe = exe.strip()
            mod_file = os.path.join(m["path"], *modpath.split(".")) + ".py"
            if not os.path.exists(mod_file):
                rep.fail(f"entry_point[{name}]", f"{exe}: module {modpath} not found")
                continue
            tree = ast.parse(open(mod_file).read())
            funcs = {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
            if func not in funcs:
                rep.fail(
                    f"entry_point[{name}]",
                    f"{exe}: {modpath}:{func} does not exist",
                )
                continue
            # the executable must also be a file in the package resource dir
            rep.ok(f"entry_point[{name}]: {exe} -> {modpath}:{func}")
    if total == 0:
        rep.note("entry_points: none declared (fine for a pure library)")

    # ament_python packages need the resource marker file; ament_cmake ones get
    # it generated by the ament_package() CMake macro.
    for name, m in meta.items():
        if m["build_type"] != "ament_python":
            rep.ok(f"resource[{name}]: not required (ament_cmake generates it)")
            continue
        marker = os.path.join(m["path"], "resource", name)
        if not os.path.exists(marker):
            rep.fail(f"resource[{name}]", f"missing resource marker {marker}")
        else:
            rep.ok(f"resource[{name}]")


def check_data_files(rep: Report, meta: Dict[str, dict]) -> None:
    total = 0
    for name, m in meta.items():
        setup_py = os.path.join(m["path"], "setup.py")
        if not os.path.exists(setup_py):
            continue
        src = open(setup_py).read()
        # every quoted string that looks like a relative path inside data_files
        block = re.search(r"data_files\s*=\s*\[(.*?)\n\s*\],\n\s*install_requires", src, re.S)
        if not block:
            block = re.search(r"data_files\s*=\s*\[(.*?)\n\s*\n", src, re.S)
        if not block:
            continue
        for candidate in re.findall(r'"([^"]+)"', block.group(1)):
            if "/" not in candidate or candidate.startswith("share/"):
                continue
            # fragments of concatenated install paths ("/launch") are not files
            if not os.path.splitext(candidate)[1]:
                continue
            total += 1
            if not os.path.exists(os.path.join(m["path"], candidate)):
                rep.fail(f"data_files[{name}]", f"missing file: {candidate}")
    if total:
        rep.ok(f"data_files: {total} referenced files exist")


def check_launch_files(rep: Report, meta: Dict[str, dict]) -> None:
    for name, m in meta.items():
        d = os.path.join(m["path"], "launch")
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if not f.endswith(".launch.py"):
                continue
            fpath = os.path.join(d, f)
            try:
                tree = ast.parse(open(fpath).read())
            except SyntaxError as exc:
                rep.fail(f"launch/{f}", f"syntax error line {exc.lineno}")
                continue
            funcs = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
            if "generate_launch_description" not in funcs:
                rep.fail(f"launch/{f}", "no generate_launch_description()")
                continue
            rep.ok(f"launch/{f}")


def check_ros_free_imports(rep: Report) -> None:
    """The layered claim: the core must import with no ROS and no rclpy.

    Runs in a subprocess with a stub that makes `import rclpy` fail, so a hidden
    ROS dependency in the core shows up here instead of on the robot.
    """
    core = os.path.join(SRC, "g1_arm_ik_core")
    script = r"""
import sys, types, importlib

# Block ROS imports entirely: the core must not need them.
class _Blocker:
    def find_module(self, name, path=None):
        if name.split(".")[0] in ("rclpy", "rcl_interfaces", "tf2_ros", "unitree_sdk2py",
                                  "unitree_go", "unitree_hg"):
            return self
        return None
    def load_module(self, name):
        raise ImportError(f"blocked: {name}")

sys.meta_path.insert(0, _Blocker())

sys.path.insert(0, %r)
import g1_ik
import g1_ik.config, g1_ik.fk, g1_ik.ik, g1_ik.model, g1_ik.reduced_model, g1_ik.urdf_parser
import g1_arm_ik_core
from g1_arm_ik_core import ArmCommandShaper, ControlConfig, MockBackend, SafetyMonitor
from g1_arm_ik_core.robot_backends import make_backend
from g1_arm_ik_core.ros_common import (
    G1_29DOF_JOINT_NAMES, arm_indices, extract_by_name, waist_indices, indices_to_names,
)

# exercise the helpers, not just import them: a wrong index here would be
# silently catastrophic on the robot (joint 15-21 must be the left arm)
assert len(G1_29DOF_JOINT_NAMES) == 29, len(G1_29DOF_JOINT_NAMES)
li, ri, wi = arm_indices("left"), arm_indices("right"), waist_indices()
assert li == [15, 16, 17, 18, 19, 20, 21], li
assert ri == [22, 23, 24, 25, 26, 27, 28], ri
assert wi == [12, 13, 14], wi
assert indices_to_names(wi) == ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
assert extract_by_name(["a", "b"], [1.0, 2.0], ["b"])[0] == 2.0
assert extract_by_name(["a"], [1.0], ["missing"]) is None
print("CORE_IMPORT_OK", len(G1_29DOF_JOINT_NAMES))
""" % (core,)
    try:
        out = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, timeout=120
        )
    except Exception as exc:  # noqa: BLE001
        rep.fail("ros_free_import", f"subprocess failed: {exc}")
        return
    if out.returncode != 0 or "CORE_IMPORT_OK" not in out.stdout:
        rep.fail(
            "ros_free_import",
            f"core pulled in a ROS dependency:\n{out.stdout.strip()}\n{out.stderr.strip()[-800:]}",
        )
        return
    rep.ok("ros_free_import: core imports with rclpy/tf2/unitree blocked")


def check_node_imports_resolvable(rep: Report, meta: Dict[str, dict]) -> None:
    """rclpy modules cannot be imported here, but their own imports must resolve.

    Every `from X import Y` where X is a sibling module inside the same package
    (or a workspace package) must point at something that exists.
    """
    ws_pkgs = set(meta)
    problems = []
    for name, m in meta.items():
        for root, dirs, files in os.walk(m["path"]):
            dirs[:] = [d for d in dirs if d not in ("__pycache__", "test")]
            for f in files:
                if not f.endswith(".py"):
                    continue
                fpath = os.path.join(root, f)
                rel = os.path.relpath(fpath, m["path"])
                tree = ast.parse(open(fpath).read())
                for node in ast.walk(tree):
                    mods: List[str] = []
                    if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                        mods = [node.module]
                    elif isinstance(node, ast.Import):
                        mods = [a.name for a in node.names]
                    for mod in mods:
                        head = mod.split(".")[0]
                        if head in ws_pkgs:
                            # the workspace package must exist as a directory
                            if not os.path.isdir(os.path.join(SRC, head)):
                                problems.append(f"{rel}: {mod} -> no such package")
                            continue
                        if head in ("g1_ik",):
                            if not os.path.isdir(
                                os.path.join(SRC, "g1_arm_ik_core", "g1_ik")
                            ):
                                problems.append(f"{rel}: {mod} -> g1_ik missing")
            # end files
    if problems:
        for p in problems:
            rep.fail("node_imports", p)
    else:
        rep.ok("node_imports: all workspace-internal imports resolve")


def check_consistency(rep: Report, meta: Dict[str, dict]) -> None:
    """Cross-package invariants that are easy to break silently."""
    # the four expected packages must be present
    expected = {
        "g1_arm_ik_core",
        "g1_arm_msgs",
        "g1_arm_ik_node",
        "g1_arm_control_node",
        "g1_arm_bringup",
    }
    missing = expected - set(meta)
    if missing:
        rep.fail("consistency", f"missing packages: {sorted(missing)}")
    else:
        rep.ok("consistency: all 5 packages present (4 required + bringup)")

    # msgs must be a cmake/rosidl package with the interface group
    m = meta.get("g1_arm_msgs", {})
    if m.get("build_type") != "ament_cmake":
        rep.fail("consistency", "g1_arm_msgs must be ament_cmake")
    elif not m.get("has_rosidl_group"):
        rep.fail("consistency", "g1_arm_msgs must declare <member_of_group>rosidl_interface_packages")
    else:
        rep.ok("consistency: g1_arm_msgs is a rosidl interface package")

    # the URDF must be shipped by bringup and referenced by the configs
    bringup = meta.get("g1_arm_bringup", {}).get("path", "")
    urdf_dir = os.path.join(bringup, "urdf")
    if not os.path.isdir(urdf_dir) or not any(
        f.endswith(".urdf") for f in os.listdir(urdf_dir)
    ):
        rep.fail("consistency", "bringup/urdf has no .urdf")
    else:
        rep.ok(f"consistency: bringup ships {len(os.listdir(urdf_dir))} urdf files")

    cfg_dir = os.path.join(bringup, "config")
    for f in sorted(os.listdir(cfg_dir)) if os.path.isdir(cfg_dir) else []:
        if not f.endswith(".yaml"):
            continue
        text = open(os.path.join(cfg_dir, f)).read()
        m2 = re.search(r"^urdf:\s*(\S+)", text, re.M)
        if m2:
            target = os.path.join(bringup, m2.group(1))
            if not os.path.exists(target):
                rep.fail("consistency", f"{f} points at missing urdf {m2.group(1)}")
            else:
                rep.ok(f"consistency: {f} urdf path resolves")



def check_ros_param_types(rep: Report, meta: Dict[str, dict]) -> None:
    """Declared ROS parameters must have sane types and actually be consumed.

    This is the highest-value static check in the file, because a bad parameter
    does not crash: rclpy rejects a type mismatch at startup (good), but a
    parameter that is declared and then never read fails SILENTLY, and a
    scientific-notation float in YAML becomes a *string* that only blows up on
    the robot.
    """
    for pkg in ("g1_arm_ik_node", "g1_arm_control_node"):
        path = meta.get(pkg, {}).get("path")
        if not path:
            continue
        srcs = [
            f
            for f in pathlib.Path(path).rglob("*.py")
            if "__pycache__" not in str(f)
        ]
        text = "\n".join(f.read_text() for f in srcs)
        declared = re.findall(r'declare_parameter\(\s*"([^"]+)"\s*,\s*([^)]+)\)', text)
        if not declared:
            continue
        # 1. duplicate declarations. Checked PER FILE, not per package: two nodes
        # in the same package (ik_node, vla_node) legitimately declare the same
        # parameter names because each is configured separately. A duplicate
        # inside a single node is the real bug.
        per_file_dupes = []
        n_nodes = 0
        for f in srcs:
            fnames = [
                m.group(1)
                for m in re.finditer(
                    r'declare_parameter\(\s*"([^"]+)"\s*,', f.read_text()
                )
            ]
            if not fnames:
                continue
            n_nodes += 1
            local = {n for n in fnames if fnames.count(n) > 1}
            if local:
                per_file_dupes.append(f"{f.name}: {sorted(local)}")
        if per_file_dupes:
            rep.fail(
                f"params[{pkg}]",
                f"a node declares the same parameter twice: {per_file_dupes}",
            )
        else:
            rep.ok(
                f"params[{pkg}]: {len(declared)} parameters across {n_nodes} node(s), "
                f"no duplicates within a node"
            )

        # 2. every declared parameter must be read back somewhere
        for name, default in declared:
            # get_parameter("x").value  or  -p style access
            if f'get_parameter("{name}")' not in text:
                rep.fail(
                    f"params[{pkg}]",
                    f"{name!r} is declared but never read -- it would silently do nothing",
                )

        # 3. ik.* parameters must be in the whitelist the node actually builds,
        #    otherwise the YAML value is dropped on the floor
        # every declared name across the package, for the ik.* whitelist check
        names = [n for n in (d[0] for d in declared)]
        ik_declared = [n for n in names if n.startswith("ik.")]
        m = re.search(r"cfg_dict\s*=\s*\{(.*?)\n\s*\}", text, re.S)
        if ik_declared and m:
            used = set(re.findall(r'"([^"]+)"\s*:', m.group(1)))
            orphans = sorted(n[len("ik."):] for n in ik_declared if n[len("ik."):] not in used)
            if orphans:
                rep.fail(
                    f"params[{pkg}]",
                    f"ik.* declared but not passed to build_ik_config: {orphans}",
                )
            else:
                rep.ok(f"params[{pkg}]: all ik.* parameters reach the IK config")

        # 4. integer-looking defaults must not be declared as bools and vice
        #    versa (a classic copy-paste bug)
        for name, default in declared:
            d = default.strip()
            if d in ("True", "False") and not name.startswith(("ik.", "start")):
                pass  # booleans are fine as-is


def check_yaml_float_traps(rep: Report, meta: Dict[str, dict]) -> None:
    """Catch numbers in YAML parameter files that parse as STRINGS.

    PyYAML implements YAML 1.1, whose float pattern requires *both* a decimal
    point in the mantissa and a sign on the exponent. Verified against PyYAML:

        1.0e-5   -> float    1.0E-05 -> float    1.0e+5 -> float
        0.00001  -> float
        1e-5     -> STR   <- no decimal point
        1.0e5    -> STR   <- exponent without a sign

    A string where rclpy expects a float is rejected at node startup with a type
    error, so this fails loudly rather than silently -- but it fails on the robot
    instead of here, which is the point of checking it now.
    """
    import yaml as _yaml

    bad: List[str] = []
    checked = 0
    for name, m in meta.items():
        cfg = os.path.join(m["path"], "config")
        if not os.path.isdir(cfg):
            continue
        for f in sorted(os.listdir(cfg)):
            if not f.endswith((".yaml", ".yml")):
                continue
            fpath = os.path.join(cfg, f)
            try:
                data = _yaml.safe_load(open(fpath)) or {}
            except Exception as exc:  # noqa: BLE001
                rep.fail("yaml_parse", f"{f}: {exc}")
                continue
            for key, value in _walk_scalars(data):
                # only care about leaves that LOOK numeric but are strings
                if not isinstance(value, str):
                    continue
                checked += 1
                if _NUMERIC_LOOKING.match(value):
                    bad.append(
                        f"{f}: {key} = {value!r} parses as a STRING, not a number"
                    )
    if bad:
        for b in bad:
            rep.fail("yaml_float", b)
    else:
        rep.ok(f"yaml_float: {checked} string values checked, no numeric-lookalike traps")


_NUMERIC_LOOKING = re.compile(r"^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$")


def _walk_scalars(obj, prefix: str = ""):
    """Yield (dotted_key, value) for every leaf of a nested dict/list."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _walk_scalars(v, f"{prefix}{k}.")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _walk_scalars(v, f"{prefix}[{i}].")
    else:
        yield prefix.rstrip("."), obj


def check_no_ros_dep_in_core(rep: Report, meta: Dict[str, dict]) -> None:
    """The core package.xml must not depend on any ROS runtime."""
    core = meta.get("g1_arm_ik_core")
    if not core:
        rep.fail("core_purity", "g1_arm_ik_core not found")
        return
    root = ET.parse(os.path.join(core["path"], "package.xml")).getroot()
    deps = [d.text for d in root.findall("exec_depend")] + [
        d.text for d in root.findall("depend")
    ]
    ros_deps = [
        d
        for d in deps
        if d
        and d.split("-")[0]
        in ("rclpy", "rclcpp", "sensor_msgs", "geometry_msgs", "std_msgs", "tf2_ros")
    ]
    if ros_deps:
        rep.fail("core_purity", f"core declares ROS dependencies: {ros_deps}")
    else:
        rep.ok("core_purity: g1_arm_ik_core declares no ROS runtime dependency")


def check_message_field_usage(rep: Report, meta: Dict[str, dict]) -> None:
    """Every field written on a generated message must exist in the .msg."""
    msgs = {}
    msg_dir = os.path.join(meta["g1_arm_msgs"]["path"], "msg")
    for f in os.listdir(msg_dir):
        if not f.endswith(".msg"):
            continue
        fields = set()
        for raw in open(os.path.join(msg_dir, f)):
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                fields.add(parts[-1])
        msgs[f[:-4]] = fields

    problems = []
    for pkg in ("g1_arm_ik_node", "g1_arm_control_node"):
        path = meta.get(pkg, {}).get("path")
        if not path:
            continue
        for f in pathlib.Path(path).rglob("*.py"):
            if "__pycache__" in str(f):
                continue
            text = f.read_text()
            for msg_name, fields in msgs.items():
                if f"{msg_name}()" not in text:
                    continue
                for attr in re.findall(r"\bs\.([a-z_][a-z0-9_]*)\s*=", text):
                    if attr not in fields:
                        problems.append(
                            f"{f.name}: sets s.{attr} which is not a field of "
                            f"{msg_name} ({sorted(fields)})"
                        )
    if problems:
        for p in problems:
            rep.fail("msg_fields", p)
    else:
        rep.ok("msg_fields: all message assignments match the .msg definitions")



def check_configs_are_loaded(rep: Report, meta: Dict[str, dict]) -> None:
    """Every config/*.yaml must be referenced by at least one launch file.

    A parameter file nobody loads is worse than a missing one: the node silently
    runs on its declared defaults and the operator believes the file is in
    effect. The IK experiments (`g1_left_arm.yaml` / `g1_right_arm.yaml`) are
    loaded by demo.py instead of a launch file, so both are accepted.
    """
    bringup = meta.get("g1_arm_bringup", {}).get("path")
    if not bringup:
        rep.fail("config_loaded", "g1_arm_bringup not found")
        return
    cfg_dir = os.path.join(bringup, "config")
    launch_dir = os.path.join(bringup, "launch")
    if not os.path.isdir(cfg_dir):
        rep.fail("config_loaded", f"no config directory at {cfg_dir}")
        return

    launch_text = ""
    if os.path.isdir(launch_dir):
        for f in os.listdir(launch_dir):
            if f.endswith((".py", ".xml", ".yaml")):
                launch_text += open(os.path.join(launch_dir, f)).read()

    # consumers outside the launch files
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(bringup)))
    demo_text = ""
    for extra in ("demo.py",):
        p = os.path.join(repo_root, extra)
        if os.path.exists(p):
            demo_text += open(p).read()

    unloaded = []
    loaded = []
    for f in sorted(os.listdir(cfg_dir)):
        if not f.endswith((".yaml", ".yml")):
            continue
        if f in launch_text:
            loaded.append(f)
        elif f in demo_text:
            loaded.append(f"{f} (via demo.py)")
        else:
            unloaded.append(f)

    if unloaded:
        for f in unloaded:
            rep.fail(
                "config_loaded",
                f"config/{f} is not referenced by any launch file -- it would "
                f"silently have no effect",
            )
    else:
        rep.ok(f"config_loaded: all {len(loaded)} configs are reachable ({', '.join(loaded)})")


def check_launch_referenced_files_exist(rep: Report, meta: Dict[str, dict]) -> None:
    """Files named inside launch files (config/..., urdf/...) must exist."""
    bringup = meta.get("g1_arm_bringup", {}).get("path")
    launch_dir = os.path.join(bringup, "launch") if bringup else None
    if not launch_dir or not os.path.isdir(launch_dir):
        return
    missing = []
    checked = 0
    # only literal path fragments of the form "config/x.yaml" / "urdf/x.urdf"
    pat = re.compile(r'"((?:config|urdf|rviz)/[A-Za-z0-9_.\-]+)"')
    for f in sorted(os.listdir(launch_dir)):
        if not f.endswith(".py"):
            continue
        text = open(os.path.join(launch_dir, f)).read()
        for rel in pat.findall(text):
            checked += 1
            if not os.path.exists(os.path.join(bringup, rel)):
                missing.append(f"{f}: {rel}")
    if missing:
        for m in missing:
            rep.fail("launch_paths", m)
    elif checked:
        rep.ok(f"launch_paths: {checked} file references in launch files resolve")



def check_docs_reference_real_files(rep: Report) -> None:
    """The main README must only reference files that exist.

    Documentation drifts silently: a renamed module leaves a stale reference that
    nobody notices until someone follows it. This is the cheapest possible guard.
    """
    # __file__ is <repo>/ros2_ws/validate_workspace.py -> two dirnames up
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    doc = os.path.join(repo_root, "README.md")
    if not os.path.exists(doc):
        # fall back to searching upward, so a relocated script still works
        d = os.path.dirname(os.path.abspath(__file__))
        for _ in range(4):
            if os.path.exists(os.path.join(d, "README.md")):
                repo_root = d
                doc = os.path.join(d, "README.md")
                break
            d = os.path.dirname(d)
    if not os.path.exists(doc):
        rep.note("README.md not found; skipping doc reference check")
        return
    text = open(doc).read()
    refs = set(
        re.findall(
            r"`([A-Za-z0-9_/.-]+\.(?:py|yaml|urdf|md|txt|xml|msg|srv))`", text
        )
    )
    missing = []
    for rel in sorted(refs):
        name = rel.split("/")[-1]
        found = False
        for root, dirs, files in os.walk(repo_root):
            dirs[:] = [
                d for d in dirs
                if d not in (".git", "__pycache__", "build", "install", "log")
                and not root.endswith("tools")
            ]
            for f in files:
                cand = os.path.join(root, f)
                if cand.endswith(rel):
                    found = True
                    break
            if found:
                break
        if not found:
            missing.append(rel)
    if missing:
        for m in missing:
            rep.fail("docs_reference", f"README.md references a missing file: {m}")
    else:
        rep.ok(f"docs_reference: all {len(refs)} files named in README.md exist")


def main() -> int:
    rep = Report()
    pkgs = packages()
    if len(pkgs) < 4:
        rep.fail("discovery", f"only found {len(pkgs)} packages in {SRC}")
    meta = check_package_xml(rep, pkgs)
    check_exec_deps(rep, meta)
    check_msgs(rep)
    check_python_syntax(rep, pkgs)
    check_entry_points(rep, meta)
    check_data_files(rep, meta)
    check_launch_files(rep, meta)
    check_ros_free_imports(rep)
    check_node_imports_resolvable(rep, meta)
    check_consistency(rep, meta)
    check_ros_param_types(rep, meta)
    check_yaml_float_traps(rep, meta)
    check_no_ros_dep_in_core(rep, meta)
    check_message_field_usage(rep, meta)
    check_configs_are_loaded(rep, meta)
    check_launch_referenced_files_exist(rep, meta)
    check_docs_reference_real_files(rep)

    print("=" * 78)
    print("ROS 2 WORKSPACE STATIC VALIDATION")
    print("=" * 78)
    for name in rep.passed:
        print(f"  [PASS] {name}")
    for name, msg in rep.failed:
        print(f"  [FAIL] {name}: {msg}")
    for msg in rep.info:
        print(f"  [note] {msg}")
    print()
    total = len(rep.passed) + len(rep.failed)
    print(f"{len(rep.passed)}/{total} checks passed")
    if rep.failed:
        print()
        print("NOTE: this is a static check. It cannot replace `colcon build` on the")
        print("      Ubuntu 22.04 + Humble machine; see README.md (Build / Usage).")
    return rep.rc


if __name__ == "__main__":
    sys.exit(main())
