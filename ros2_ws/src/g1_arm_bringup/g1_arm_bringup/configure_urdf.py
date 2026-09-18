"""Prepare the G1 URDF for ROS: rewrite mesh paths, optionally fetch the meshes.

The official Unitree URDF references meshes *relatively*:

    <mesh filename="meshes/left_elbow_link.STL"/>

That works for MuJoCo (which resolves relative to the URDF's directory) but not
for ROS tooling (robot_state_publisher, any mesh loader), which need
`package://` URIs. This script rewrites the paths in place and can fetch the STL
files from the official repository.

This is OPTIONAL: the IK, the control loop and the TF broadcaster all work from
the URDF's kinematic tree alone and do not read meshes. Only run it if some other
tool of yours needs to load the visuals.

Usage
-----
    # rewrite paths for the URDFs shipped in this package
    ros2 run g1_arm_bringup configure_urdf

    # rewrite and also download the meshes (~110 MB) into share/.../meshes
    ros2 run g1_arm_bringup configure_urdf --fetch-meshes

    # a custom file
    ros2 run g1_arm_bringup configure_urdf --urdf /path/to/g1.urdf --package my_pkg

Note: nothing in this workspace needs the meshes.
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.request
import xml.etree.ElementTree as ET
from typing import List, Tuple

DEFAULT_PACKAGE = "g1_arm_bringup"
MESH_BASE_URL = (
    "https://raw.githubusercontent.com/unitreerobotics/unitree_ros/master/"
    "robots/g1_description/meshes/"
)


def package_share(package: str) -> str:
    try:
        from ament_index_python.packages import get_package_share_directory

        return get_package_share_directory(package)
    except Exception:  # noqa: BLE001 - runnable without a sourced workspace
        # fall back to the source tree layout: <pkg>/../.. is the package dir
        here = os.path.dirname(os.path.abspath(__file__))
        return os.path.dirname(here)


def collect_meshes(urdf_path: str) -> List[Tuple[ET.Element, str]]:
    """Return (element, filename) for every <mesh> in the file."""
    tree = ET.parse(urdf_path)
    out: List[Tuple[ET.Element, str]] = []
    for el in tree.iter("mesh"):
        fn = el.get("filename")
        if fn:
            out.append((el, fn))
    return out


def rewrite(urdf_path: str, package: str, backup: bool = True) -> int:
    tree = ET.parse(urdf_path)
    n = 0
    for el in tree.iter("mesh"):
        fn = el.get("filename")
        if not fn or fn.startswith("package://") or fn.startswith("file://"):
            continue
        # "meshes/x.STL" -> "package://g1_arm_bringup/meshes/x.STL"
        rel = fn.lstrip("./")
        el.set("filename", f"package://{package}/{rel}")
        n += 1
    if n == 0:
        print(f"  {os.path.basename(urdf_path)}: nothing to rewrite")
        return 0
    if backup:
        bak = urdf_path + ".orig"
        if not os.path.exists(bak):
            with open(urdf_path, "rb") as src, open(bak, "wb") as dst:
                dst.write(src.read())
            print(f"  backup written: {os.path.basename(bak)}")
    tree.write(urdf_path)
    print(f"  {os.path.basename(urdf_path)}: rewrote {n} mesh paths")
    return n


def fetch_meshes(urdf_path: str, dest_dir: str) -> int:
    os.makedirs(dest_dir, exist_ok=True)
    meshes = collect_meshes(urdf_path)
    names = sorted({os.path.basename(fn) for _, fn in meshes})
    print(f"  {len(names)} meshes referenced -> {dest_dir}")
    ok = 0
    for i, name in enumerate(names, 1):
        target = os.path.join(dest_dir, name)
        if os.path.exists(target) and os.path.getsize(target) > 0:
            ok += 1
            continue
        url = MESH_BASE_URL + name
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                data = resp.read()
            with open(target, "wb") as fh:
                fh.write(data)
            ok += 1
            print(f"    [{i}/{len(names)}] {name} ({len(data) / 1024:.0f} KB)")
        except Exception as exc:  # noqa: BLE001
            print(f"    [{i}/{len(names)}] FAILED {name}: {exc}")
    print(f"  fetched {ok}/{len(names)}")
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--urdf", action="append", default=None,
                    help="URDF to process (repeatable). Default: the two shipped G1 URDFs.")
    ap.add_argument("--package", default=DEFAULT_PACKAGE,
                    help=f"package:// name to use (default {DEFAULT_PACKAGE})")
    ap.add_argument("--fetch-meshes", action="store_true",
                    help="download the STL meshes (~110 MB total)")
    ap.add_argument("--no-backup", action="store_true",
                    help="do not write a .orig backup before rewriting")
    args = ap.parse_args(argv)

    share = package_share(args.package)
    if args.urdf:
        urdfs = args.urdf
    else:
        urdf_dir = os.path.join(share, "urdf")
        if not os.path.isdir(urdf_dir):
            print(f"error: {urdf_dir} not found. Is g1_arm_bringup installed?", file=sys.stderr)
            return 2
        urdfs = [
            os.path.join(urdf_dir, f)
            for f in sorted(os.listdir(urdf_dir))
            if f.endswith(".urdf")
        ]

    print(f"package share: {share}")
    total = 0
    for u in urdfs:
        if not os.path.exists(u):
            print(f"  missing: {u}", file=sys.stderr)
            return 2
        total += rewrite(u, args.package, backup=not args.no_backup)

    if args.fetch_meshes:
        dest = os.path.join(share, "meshes")
        # use the first URDF to enumerate; all shipped variants share the set
        fetch_meshes(urdfs[0], dest)

    print()
    if total:
        print(f"done: {total} mesh references now use package://{args.package}/")
    else:
        print("done: nothing to change")
    return 0


if __name__ == "__main__":
    sys.exit(main())
