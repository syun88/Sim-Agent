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

The import result measured 2026-10-24 (this host, Genesis installed for the last rows):

    robot_id      open_manipulator_x-784078c2a5   (source sha256 784078c2a5bb16ac)
    root link     world            (fixed base, single arm)
    chain         left = joint1..joint4 + gripper_left_joint, IK validation success
    capabilities  motion, joint_control, ik, physical_grasp, place, stack
    gripper       sliding_jaws on gripper_left_joint: aperture 20.0..82.0 mm,
                  cube 25.5 mm, holding_q -0.0078 rad, table clearance 4.3 mm
    setup posture derived by private IK (joint2 -0.724, joint3 0.333, joint4 0.380, jaw open)
    spawned props none

Two blockers were found and fixed on the way: the jaw inference refused a gripper that closes
wider than its finger length allows (now the sweep retries with the closure-derived cube
floor), and the private collision model failed to compile because this URDF's root link is
called 'world' (now robot bodies get a safe name and queries map back).

Still open, with Genesis installed here: the desk props are refused. The source collision
meshes interpenetrate at every adjacent joint - link2/link3 43.1 mm, link3/link4 43.3 mm,
link4/link5 43.2 mm, link1/link2 21.4 mm, each finger against link5 9.1 mm at the setup
posture - so every candidate is rejected as "deeply intersects the robot", and 0 of 8 sampled
directions are clear at every radius from 0.08 to 0.34 m. Preparing collision geometry (the
Scorpion pipeline's inset/refine tools) or declaring the overlapping adjacent pairs is the
next step; this script stages and reports only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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
    """Download the pinned description into ``source/`` and record (or verify) its hashes.

    The hashes pin the *upstream* files. The import-facing copies under ``genesis/`` and
    ``meshes/`` are published from them and may then be skinned, exactly like a prepared
    robot package keeps its original beside the prepared one.
    """
    manifest_path = destination / "provenance.json"
    files = {URDF_PATH: destination / "source" / "open_manipulator_x.urdf"}
    for name in MESHES:
        files[f"{MESH_DIR}/{name}"] = destination / "source" / "meshes" / "open_manipulator_x" / name
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
    _publish(destination, manifest)
    return manifest


def _publish(destination: Path, manifest: dict) -> None:
    """Copy the pinned originals to the paths the importer reads."""
    import shutil
    (destination / "genesis").mkdir(parents=True, exist_ok=True)
    shutil.copy2(destination / "source" / "open_manipulator_x.urdf",
                 destination / "genesis" / "open_manipulator_x.urdf")
    for entry in manifest["files"].values():
        name = Path(entry["path"]).name
        if name.endswith(".stl"):
            target = destination / "meshes" / "open_manipulator_x" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(destination / entry["path"], target)


def _derive_home(destination: Path) -> dict | None:
    """Ask the robot's own IK for a posture clear of the desk and record it.

    This URDF's zero posture lies over the desk, so every desk prop was rejected with
    "spawn cube intersects the robot in its setup posture" (measured 2026-10-24).
    ``import_robot`` reads ``initial_state.json`` from the URDF's folder, which is the
    documented way to give an imported robot a usable setup posture.
    """
    import numpy as np
    from backend.app.robots.registry import import_robot
    from backend.app.robots.urdf import solve_profile_ik
    profile = import_robot(str(destination))
    side = next(iter(profile['chains']))
    target = np.asarray(profile['root_transform'])[:3, 3] + np.array([.20, 0., .22])
    solution = solve_profile_ik(profile, side, target)
    if not solution.get('success'):
        print('no desk-clear setup posture found:', solution.get('detail'), flush=True)
        return None
    state = {name: 0. for name in profile['home']}
    state.update(dict(zip(profile['chains'][side]['joints'], np.asarray(solution['q_rad']).tolist())))
    for gripper in (profile.get('grippers') or {}).values():
        state[gripper['joint']] = float(gripper['open_q'])
    print('setup posture from private IK:', {k: round(v, 4) for k, v in state.items()}, flush=True)
    return state



def _collision_declaration(destination: Path, threshold_mm: float) -> dict | None:
    """Declare the adjacent link pairs whose source meshes overlap in the setup posture.

    These meshes are visual-quality parts: they overlap at every joint by design, so the
    runtime must be told which adjacent pairs not to collide (the URDF cannot say it).
    Only measured overlaps at or above the threshold are declared, and a non-adjacent pair
    is refused by the importer rather than silently accepted.
    """
    from backend.app.robots.registry import import_robot
    from backend.app.robots.collision_planner import PrivateCollisionQueries
    profile = import_robot(str(destination))
    scene = json.loads((Path(profile['directory']) / 'scene.json').read_text())
    queries = PrivateCollisionQueries(profile, scene['table_top'])
    snapshot = dict(profile['home'])
    for gripper in (profile.get('grippers') or {}).values():
        snapshot.update({gripper['joint']: gripper['open_q'],
                         **{name: gripper['open_q'] * k for name, k in gripper['followers'].items()}})
    from backend.app.robots.urdf import URDFModel
    # Adjacency is body adjacency: fixed and mimic children count too, or the declaration
    # would miss the pairs the engine also cannot filter.
    model = URDFModel(profile['urdf'])
    adjacent = {tuple(sorted((joint['parent'], joint['child']))) for joint in model.joints}
    measured = {}
    for contact in queries.contacts(snapshot, tolerance=-.0001):
        names = tuple(sorted(queries.link_names.get(name, name) for name in contact['links']))
        depth_mm = float(contact['penetration_m']) * 1000.
        if names in adjacent and depth_mm >= threshold_mm:
            measured[names] = round(depth_mm, 3)
    if not measured:
        print(f'no adjacent-pair overlap at or above {threshold_mm:g} mm; nothing declared', flush=True)
        return None
    pairs = [list(pair) for pair in sorted(measured)]
    print(f'declaring {len(pairs)} overlapping adjacent pairs: '
          + ', '.join(f'{a}|{b} {measured[(a, b)]:.1f}mm' for a, b in sorted(measured)), flush=True)
    return {'runtime_excluded_adjacent_pairs': pairs,
            'excluded_adjacent_pairs': pairs,
            'measured_setup_posture_overlap_mm': {f'{a}|{b}': measured[(a, b)] for a, b in sorted(measured)},
            'measured_threshold_mm': threshold_mm}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path,
                        default=Path("~/.cache/sim-agent/open_manipulator_x").expanduser())
    parser.add_argument("--refresh", action="store_true", help="re-download and re-record the hashes")
    parser.add_argument("--check", action="store_true", help="verify the recorded hashes only")
    parser.add_argument("--overlap-threshold-mm", type=float, default=5.,
                        help="declare an adjacent pair whose setup-posture meshes overlap by at least this")
    args = parser.parse_args()
    destination = args.dest.expanduser()
    if not (destination / "provenance.json").is_file() and args.check:
        raise SystemExit("Nothing staged yet; run without --check first")
    manifest = _stage(destination, refresh=args.refresh)
    print(f"staged {len(manifest['files'])} files under {destination}")
    if args.check:
        return 0
    from backend.app.robots.registry import import_robot
    initial = destination / "genesis" / "initial_state.json"
    if not initial.is_file():
        state = _derive_home(destination)
        if state:
            initial.write_text(json.dumps(state, indent=2) + "\n")
    declaration = destination / "genesis" / "collision_config.json"
    if not declaration.is_file():
        settings = _collision_declaration(destination, args.overlap_threshold_mm)
        if settings:
            declaration.write_text(json.dumps(settings, indent=2) + "\n")
    profile = import_robot(str(destination))
    directory = Path(profile["directory"])
    analysis = json.loads((directory / "gripper_analysis.json").read_text())
    ik = json.loads((directory / "ik_validation.json").read_text())
    spawn = json.loads((directory / "spawn_validation.json").read_text())
    report = {"robot_id": profile["id"], "source": profile["source"], "source_sha256": profile["source_sha256"],
              "root_link": profile["root_link"], "actuated_joints": [j["name"] for j in profile["joints"]],
              "chains": {side: chain["joints"] for side, chain in profile["chains"].items()},
              "capabilities": profile["capabilities"], "cube_size_m": profile.get("cube_size"),
              "collision_config": profile.get("collision_config"),
              "adjacent_collision": json.loads((directory / "scene.json").read_text()).get("adjacent_collision"),
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
