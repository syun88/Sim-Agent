"""Verify the physical pick & place on Genesis World from many initial object positions.

Every run is a real execution of the same skills the app uses: open gripper, IK the fingertip
midpoint onto the object centre, close the fingers, verify the pinch contacts and the
verification lift, transport, release, retreat - and the independent goal verification. The
initial object position is *scene setup* (the same status as the documented `randomize` reset);
nothing writes the object during an execution, and every claim below is backed by a measurement
the script prints.

Usage:
    uv run python scripts/verify_pick_place_genesis.py [--positions left|right|all] [--json PATH]

Writes `outputs/pick_place_positions.json` and prints one line per run plus a summary.
Exit code 0 only if every run passed the independent goal verification.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.control.motion import OpenArmV2Controller  # noqa: E402
from backend.app.skills.robot_skills import SkillRunner  # noqa: E402
from backend.app.verification.goal_verification import verify_place_goal  # noqa: E402
from backend.app.simulation import create_backend  # noqa: E402

REST_HEIGHT = 1.032

#: One task per arm: object, target zone, and the initial positions to verify. The positions
#: spread across the arm's workspace on the table (x toward the front wall, y lateral), each at
#: least 7 cm from the target zone so every run is a real transport.
TASKS = {
    "left": {
        "arm": "left",
        "object": "red_cube",
        "zone": "blue_zone",
        "positions": [
            (0.44, 0.15),   # the documented demo start
            (0.395, 0.12),  # pulled back toward the robot
            (0.49, 0.10),   # forward and toward the centre line
            (0.36, 0.18),   # close to the robot base
            (0.44, 0.23),   # far out laterally
            (0.52, 0.06),   # forward, near the centre line
        ],
    },
    "right": {
        "arm": "right",
        "object": "blue_cube",
        "zone": "green_zone",
        "positions": [
            (0.50, -0.22),  # the documented demo start
            (0.44, -0.15),  # pulled back toward the centre line
            (0.55, -0.28),  # forward and far out laterally
        ],
    },
}


def run_task(env, skills, task, xy) -> dict:
    """One full pick & place from one initial position; returns the measured evidence."""
    position = np.array([xy[0], xy[1], REST_HEIGHT])
    # Scene setup: place the object, settle, then measure where it actually rests.
    env.reset(object_positions={task["object"]: position})
    env.step(200)
    start = env.object_pose(task["object"])[0].copy()

    plan = [
        {"skill": "approach_object", "arm": task["arm"], "object": task["object"]},
        {"skill": "grasp_object", "arm": task["arm"], "object": task["object"]},
        {"skill": "lift_object", "arm": task["arm"], "object": task["object"]},
        {"skill": "move_to_target", "arm": task["arm"], "target": task["zone"]},
        {"skill": "release_object", "arm": task["arm"], "object": task["object"]},
        {"skill": "retreat", "arm": task["arm"]},
    ]
    started = time.perf_counter()
    results = skills.execute_plan(plan)
    elapsed = time.perf_counter() - started
    verification = verify_place_goal(env, task["object"], task["zone"])

    approach = next((r for r in results if r["skill"] == "approach_object"), {})
    grasp = next((r for r in results if r["skill"] == "grasp_object"), {})
    failed = next((r for r in results if not r.get("success")), None)
    pinch_fingers = grasp.get("pinch_contacts", [])
    return {
        "arm": task["arm"],
        "object": task["object"],
        "zone": task["zone"],
        "initial_position": [round(v, 4) for v in xy],
        "resting_start": [round(v, 4) for v in start.tolist()],
        "skills_success": all(r.get("success") for r in results),
        "failed_skill": {"skill": failed["skill"], "error": failed.get("error"), "detail": failed.get("detail")} if failed else None,
        "approach_displacement_mm": round(approach.get("object_displacement", float("nan")) * 1000, 2),
        "fingertip_alignment_error_mm": round(approach.get("fingertip_midpoint_error", float("nan")) * 1000, 2),
        "pinch_fingers": pinch_fingers,
        "pinch_both_fingers": any("inner" in f for f in pinch_fingers) and any("outer" in f for f in pinch_fingers),
        "verification_lift_follow_mm": round(grasp.get("object_rose", 0.0) * 1000, 1),
        "goal_success": verification["success"],
        "horizontal_error_mm": round(verification["horizontal_position_error"] * 1000, 1),
        "vertical_error_mm": round(verification["vertical_position_error"] * 1000, 1),
        "released": verification["released"],
        "stable": verification["stable"],
        "engine": verification["engine"],
        "grasp_mode": verification["grasp_mode"],
        "duration_s": round(elapsed, 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positions", choices=["left", "right", "all"], default="all")
    parser.add_argument("--json", type=Path, default=ROOT / "outputs" / "pick_place_positions.json")
    args = parser.parse_args()

    print("Building the Genesis World scene (this compiles kernels and takes a while)...")
    env = create_backend()
    skills = SkillRunner(env, OpenArmV2Controller(env))

    tasks = TASKS if args.positions == "all" else {args.positions: TASKS[args.positions]}
    records = []
    for name, task in tasks.items():
        for xy in task["positions"]:
            print(f"\n=== {task['arm']} arm: {task['object']} from {xy} -> {task['zone']} ===", flush=True)
            record = run_task(env, skills, task, xy)
            records.append(record)
            print(
                f"  skills={record['skills_success']} goal={record['goal_success']} "
                f"horizontal_error={record['horizontal_error_mm']}mm "
                f"approach_displacement={record['approach_displacement_mm']}mm "
                f"alignment_error={record['fingertip_alignment_error_mm']}mm "
                f"pinch_fingers={len(record['pinch_fingers'])} both={record['pinch_both_fingers']} "
                f"lift_follow={record['verification_lift_follow_mm']}mm "
                f"released={record['released']} stable={record['stable']} "
                f"[{record['duration_s']}s]"
            )
            if record["failed_skill"]:
                print(f"  FAILED SKILL: {record['failed_skill']}")

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(records, indent=2))

    passed = sum(1 for r in records if r["skills_success"] and r["goal_success"])
    print(f"\n{passed}/{len(records)} runs passed the full physical pick & place and the goal verification")
    print(f"details written to {args.json}")
    return 0 if passed == len(records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
