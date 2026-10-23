"""Stage the OpenMANIPULATOR-X arm for the Genesis import path.

OpenMANIPULATOR-X is the arm Lime (TurtleBot3 manipulation) carries. This stages
its description as a fixed-base single arm and runs the same registry import the
launcher uses, so ``outputs/robots/<robot-id>/`` is produced exactly as a folder
import from the Web UI would produce it.

Upstream is pinned to one commit and every fetched file's sha256 is recorded in
``provenance.json``; a later run re-checks those hashes, so a moved upstream is a
failure rather than a silent model change.

The import itself is static: URDF parse, FK/Jacobian seeds, jaw inference, private
IK and desk-spawn sampling. Reaching, contact, grasp and hold are NOT verified here
- they need Genesis (run ``scripts/verify_auto_gripper.py`` on the host that has it).

The import result measured 2026-10-24 on a host without Genesis:

    robot_id      open_manipulator_x-784078c2a5   (source sha256 784078c2a5bb16ac)
    root link     world            (fixed base, single arm)
    chain         left = joint1..joint4 + gripper_left_joint, IK validation success
    capabilities  motion, joint_control, ik, physical_grasp, place, stack
    gripper       sliding_jaws on gripper_left_joint: aperture 20.0..82.0 mm,
                  cube 25.5 mm, holding_q -0.0078 rad, table clearance 4.3 mm
    spawned props none

Before the jaw fix this gripper was refused outright: the sweep reads a 20.0 mm closed
gap, the finger-length cap allowed only a 22.7 mm cube, and the squeeze rule needs
``0.8 * cube <= 20 mm`` (a 25 mm cube) - so the arm imported with no grasp at all. The
sweep now retries once with the cube floor the measured closure requires, which changes
nothing where a first pass already finds a candidate: Scorpion's analysis is byte-identical
across the change.

Still open, and unchanged by the jaw fix: the desk layout spawns no props for this arm -
``No desk position reachable at object and approach heights`` for both cubes, from 33
candidates each. The gripper is inferred, but nothing is placed in front of it yet.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import urllib.request

REPOSITORY = "ROBOTIS-GIT/open_manipulator"
COMMIT = "ee9a720bcc45acda9c0df8fe689b1182844307db"

# Adopted 2026-10-24 from the commit above; a re-run with a different upstream
# revision must be recorded as a new adoption, not absorbed silently.
URDF_PATH = "open_manipulator_description/urdf/open_manipulator_x/open_manipulator_x.urdf"
MESH_DIR = "open_manipulator_description/meshes/open_manipulator_x"
MESHES = ("gripper_left_palm.stl", "gripper_right_palm.stl", "link1.stl", "link2.stl",
          "link3.stl", "link4.stl", "link5.stl", "plate.stl")


def _fetch(path: str) -> bytes:
    url = f"https://raw.githubusercontent.com/{REPOSITORY}/{COMMIT}/{path}"
    with urllib.request.urlopen(url, timeout=60) as response:
        return response.read()


def _stage(destination: Path, refresh: bool) -> dict:
    """Download the pinned description and record (or verify) its hashes."""
    manifest_path = destination / "provenance.json"
    files = {URDF_PATH: destination / "genesis" / "open_manipulator_x.urdf"}
    for name in MESHES:
        files[f"{MESH_DIR}/{name}"] = destination / "meshes" / "open_manipulator_x" / name
    recorded = json.loads(manifest_path.read_text())["files"] if manifest_path.is_file() else None
    manifest = {"repository": REPOSITORY, "commit": COMMIT,
                "license": "Apache-2.0", "adopted": "2026-10-24", "files": {}}
    for source, target in files.items():
        target.parent.mkdir(parents=True, exist_ok=True)
        if refresh or not target.is_file():
            target.write_bytes(_fetch(source))
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        if recorded and not refresh and recorded[source]["sha256"] != digest:
            raise SystemExit(f"Upstream content changed for {source}: {digest} != {recorded[source]['sha256']}")
        manifest["files"][source] = {"path": str(target.relative_to(destination)),
                                     "sha256": digest, "bytes": target.stat().st_size}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path,
                        default=Path("~/.cache/sim-agent/open_manipulator_x").expanduser())
    parser.add_argument("--refresh", action="store_true", help="re-download and re-record the hashes")
    parser.add_argument("--check", action="store_true", help="verify the recorded hashes only")
    args = parser.parse_args()
    destination = args.dest.expanduser()
    if not (destination / "provenance.json").is_file() and args.check:
        raise SystemExit("Nothing staged yet; run without --check first")
    manifest = _stage(destination, refresh=args.refresh)
    print(f"staged {len(manifest['files'])} files under {destination}")
    if args.check:
        return 0
    from backend.app.robots.registry import import_robot
    profile = import_robot(str(destination))
    directory = Path(profile["directory"])
    analysis = json.loads((directory / "gripper_analysis.json").read_text())
    ik = json.loads((directory / "ik_validation.json").read_text())
    spawn = json.loads((directory / "spawn_validation.json").read_text())
    report = {"robot_id": profile["id"], "source": profile["source"], "source_sha256": profile["source_sha256"],
              "root_link": profile["root_link"], "actuated_joints": [j["name"] for j in profile["joints"]],
              "chains": {side: chain["joints"] for side, chain in profile["chains"].items()},
              "capabilities": profile["capabilities"], "cube_size_m": profile.get("cube_size"),
              "grippers": analysis["grippers"], "gripper_failures": analysis["failures"],
              "ik_validation": ik, "spawned_objects": spawn.get("objects"), "directory": str(directory),
              "physical_grasp_verified": False}
    (destination / "import_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in
                      ("robot_id", "root_link", "chains", "capabilities", "cube_size_m")}, indent=2))
    for side, gripper in analysis["grippers"].items():
        print(f"gripper {side}: joint={gripper['joint']} cube_size={gripper['cube_size']:.4f} "
              f"reference_link={gripper['reference_link']}")
    if analysis["failures"]:
        print("jaw inference failures:", json.dumps(analysis["failures"], indent=2))
    print(f"report: {destination / 'import_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
