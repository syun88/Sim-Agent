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

2. drop the camera's visual, and write the actuators.json that names the wheel
   motors (the arm has no motor named in that description, so it keeps its URDF
   limits). It references `realsense2_description`, which is not
   part of the Lime repository (it ships in the ROS/sim image), and no scene needs
   the robot's own camera — the viewer's camera is separate (see
   `simulation/genesis_backend/environment.py`). The link and joint stay.

Writes XML without an encoding declaration: ROS's `urdf_parser_py` hands the text
to lxml, which refuses a string that carries one. The registry reads bytes and does
not care, but probes do, and one file that satisfies both is cheaper than remembering.

Usage:  uv run python scripts/prepare_lime.py [--raw PATH] [--out PATH]
"""
import argparse
import json
import sys
import pathlib
import xml.etree.ElementTree as ET

DEFAULT_RAW = pathlib.Path.home() / "Documents" / "turtlebot3_lime" / "lime_full.urdf"
DEFAULT_OUT = pathlib.Path.home() / "Documents" / "turtlebot3_lime" / "lime.urdf"
CAMERA_PACKAGE = "realsense2_description"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw", type=pathlib.Path, default=DEFAULT_RAW,
                        help="xacro output of the full robot (default: %(default)s)")
    parser.add_argument("--overlap-threshold-mm", type=float, default=.1,
                        help="declare an adjacent overlap at or above this depth")
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

    # The wheels' motor is named in the description's own comment ("<!--XM430-W210-->"
    # above the wheel limits in bigwheel_orin.urdf.xacro), so record that identity here.
    # The preset's numbers live in backend/app/robots/actuators.py beside the other
    # motors, and a joint with no entry keeps its URDF limit, which is what the arm does.
    # The arm's motor is not named in the package, but it is written in the velocity field: the
    # arm joints declare velocity="4.8" and the wheels 8.063, which are 46 and 77 rev/min - the
    # no-load speeds of the XM430-W350 and the XM430-W210 the description names in a comment for
    # the wheels. Arm mass 0.683 kg and reach 0.353 m match OpenManipulator-X (0.70 kg, 380 mm),
    # whose official actuator is the XM430-W350. Its placeholder effort="1" cannot even overcome
    # the URDF's own 1.0 Nm of joint friction (each joint moved 2-46 percent of a 0.2 rad command
    # before this change), while the arm needs at most 0.55 Nm to hold any posture.
    arm = {'joint%d' % i: {'preset': 'xm430_w350'} for i in range(1, 7)}
    wheels = {"wheel_left_joint": {"preset": "xm430_w210"},
              "wheel_right_joint": {"preset": "xm430_w210"}, **arm}
    (args.out.parent / "actuators.json").write_text(
        json.dumps({"joints": wheels}, indent=2) + "\n")

    # A manufacturer URDF's meshes are visual-quality parts: adjacent links overlap at every
    # joint by design (measured 2026-10-24 on OpenManipulator-X, the arm family this robot
    # carries: 21.4-43.3 mm). The URDF cannot say which overlaps the runtime must ignore, so
    # the package declares it. Without the declaration every IK candidate is rejected as a
    # collision and the arm never moves; reuse the measurement the OpenManipulator-X package
    # already performs instead of writing a second copy of it.
    # Measure before writing: the measurement imports the package, and a declaration left
    # from an earlier run is validated at import time, so it would reject itself.
    stale = args.out.parent / "collision_config.json"
    stale.unlink(missing_ok=True)
    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    from prepare_open_manipulator_x import _collision_declaration
    settings = _collision_declaration(args.out, args.overlap_threshold_mm)
    if settings:
        # The runtime request is all-or-nothing: the engine turns adjacent collision off
        # wholesale, so a partial set would drop a real constraint that the private queries
        # still honour. Declare every adjacent pair that carries collision geometry, which is
        # what a URDF-based planner already does with parent/child pairs, and leave the
        # measured overlaps in the file as the record of why the declaration exists.
        from backend.app.robots.urdf import URDFModel
        model = URDFModel(str(args.out))
        adjacent = {tuple(sorted((joint['parent'], joint['child']))) for joint in model.joints}
        settings["runtime_excluded_adjacent_pairs"] = [list(pair) for pair in sorted(
            {pair for pair in adjacent
             if all(len(model.links[link].findall('collision')) for link in pair)})]
        (args.out.parent / "collision_config.json").write_text(
            json.dumps(settings, indent=2) + "\n")

    ET.ElementTree(root).write(args.out, encoding="utf-8", xml_declaration=False)
    print(f"{args.out}: {len(root.findall('link'))} links, {len(root.findall('joint'))} joints, "
          f"{len(root.findall('.//mesh'))} meshes (dropped {dropped} {CAMERA_PACKAGE} element(s), {len(left)} left)")
    if left:
        raise SystemExit(f"camera references remain: {left}")


if __name__ == "__main__":
    main()
