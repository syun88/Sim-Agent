"""Prepare the official TurtleBot3 Lime description as one URDF for the registry.

Why a script: the registry attaches a URDF *and its meshes in place*
(`backend/app/robots/registry.py`, `asset_signature`), so the prepared file has to
sit next to `turtlebot3_lime_description` for `mesh_path` to resolve its
`package://` names. That is how the LeKiwi import works too
(`~/Documents/LeKiwi-src/URDF/LeKiwi.urdf`), and keeping the same shape here means
no robot is vendored into this repository.

Two steps:

1. expand the description's xacro — the full robot: base, wheels and the
   OpenManipulator arm. Needs a ROS install, so it runs in the sim image:

       docker run --rm -v ~/Documents/turtlebot3_lime:/lime --entrypoint bash \\
         course_lime-image_yuu:latest -lc 'source /opt/ros/humble/setup.bash
           source /root/turtlebot3_ws/install/setup.bash
           xacro /lime/turtlebot3_lime_description/urdf/turtlebot3_lime.urdf.xacro \\
             > /lime/lime_full.urdf'

2. drop the camera's visual. It references `realsense2_description`, which is not
   part of the Lime repository (it ships in the ROS/sim image), and no scene needs
   the robot's own camera — the viewer's camera is separate (see
   `simulation/genesis_backend/environment.py`). The link and joint stay.

Writes XML without an encoding declaration: ROS's `urdf_parser_py` hands the text
to lxml, which refuses a string that carries one. The registry reads bytes and does
not care, but probes do, and one file that satisfies both is cheaper than remembering.

Usage:  uv run python scripts/prepare_lime.py [--raw PATH] [--out PATH]
"""
import argparse
import pathlib
import xml.etree.ElementTree as ET

DEFAULT_RAW = pathlib.Path.home() / "Documents" / "turtlebot3_lime" / "lime_full.urdf"
DEFAULT_OUT = pathlib.Path.home() / "Documents" / "turtlebot3_lime" / "lime.urdf"
CAMERA_PACKAGE = "realsense2_description"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw", type=pathlib.Path, default=DEFAULT_RAW,
                        help="xacro output of the full robot (default: %(default)s)")
    parser.add_argument("--out", type=pathlib.Path, default=DEFAULT_OUT,
                        help="prepared URDF next to the description (default: %(default)s)")
    args = parser.parse_args()

    if not args.raw.is_file():
        raise SystemExit(f"raw URDF not found: {args.raw}\nRun the xacro step in the sim image first (see this file's docstring).")
    if not (args.raw.parent / "turtlebot3_lime_description").is_dir():
        raise SystemExit(f"{args.raw.parent / 'turtlebot3_lime_description'} is missing: the prepared URDF must sit next to the description for mesh_path to resolve package:// names")

    root = ET.parse(args.raw).getroot()
    dropped = 0
    for link in root.findall("link"):
        for tag in ("visual", "collision"):
            for element in list(link.findall(tag)):
                meshes = element.findall(".//mesh")
                if any(CAMERA_PACKAGE in (mesh.get("filename") or "") for mesh in meshes):
                    link.remove(element)
                    dropped += 1
    left = [element.get("filename") for element in root.findall(".//mesh")
            if CAMERA_PACKAGE in (element.get("filename") or "")]

    ET.ElementTree(root).write(args.out, encoding="utf-8", xml_declaration=False)
    print(f"{args.out}: {len(root.findall('link'))} links, {len(root.findall('joint'))} joints, "
          f"{len(root.findall('.//mesh'))} meshes (dropped {dropped} {CAMERA_PACKAGE} element(s), {len(left)} left)")
    if left:
        raise SystemExit(f"camera references remain: {left}")


if __name__ == "__main__":
    main()
