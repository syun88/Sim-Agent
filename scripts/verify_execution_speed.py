"""Measure native/headless physical task duration, with optional before/after comparison.

Kernel compilation, scene construction and planning are excluded. Each case still
checks physical finger contact, lift-follow, release and stable placement. Run the
same command on the old and new code; do not compare rendered video durations.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.control.motion import OpenArmV2Controller
from backend.app.simulation import create_backend
from backend.app.skills.robot_skills import SkillRunner
from scripts.verify_pick_place_genesis import TASKS, run_task


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", action="store_true")
    parser.add_argument("--all-positions", action="store_true")
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--json", type=Path, default=ROOT / "outputs/execution_speed.json")
    args = parser.parse_args()
    baseline = json.loads(args.compare.read_text()) if args.compare else None
    os.environ["GENESIS_SHOW_VIEWER"] = "1" if args.native else "0"
    env = create_backend()
    result = {"native": args.native, "compute_backend": env.compute_backend,
              "dt": env.timestep, "requested_realtime_factor": env.simulation_speed,
              "timing_scope": "skills only; excludes scene build, compilation, planning and reset",
              "records": []}
    if baseline and any(baseline[key] != result[key] for key in ("native", "compute_backend", "dt")):
        env.close()
        raise ValueError("Compare the same viewer mode, compute backend and physics timestep")
    try:
        env.step(100)
        runner = SkillRunner(env, OpenArmV2Controller(env))
        for task in TASKS.values():
            positions = task["positions"] if args.all_positions else task["positions"][:1]
            for xy in positions:
                record = run_task(env, runner, task, xy)
                record["physical_success"] = all(record[k] for k in (
                    "skills_success", "goal_success", "pinch_both_fingers", "released", "stable"))
                if baseline:
                    old = next((r for r in baseline["records"] if r["arm"] == record["arm"]
                                and r["initial_position"] == record["initial_position"]), None)
                    if old is not None:
                        before = old.get("wall_duration_s", old["duration_s"])
                        record.update(before_wall_s=before, speedup=before / record["wall_duration_s"])
                result["records"].append(record)
                result["success"] = all(r["physical_success"] for r in result["records"])
                args.json.parent.mkdir(parents=True, exist_ok=True)
                args.json.write_text(json.dumps(result, ensure_ascii=False, indent=2))
                print(json.dumps(record, ensure_ascii=False), flush=True)
    finally:
        env.close()
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
