"""Run natural-language tasks through the production planner/executor and real Genesis.

Each case resets only its initial scene. No reset occurs inside an execution or between
the prompts in a follow-up case. Exit 0 requires all physical goals and requested
handover events, rather than merely a completed plan or a provider's success claim.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.agent.world_state import build_world_state
from backend.app.execution.task_executor import TaskExecutor
from backend.app.planning.agent_planner import AgentPlanner
from backend.app.simulation import create_backend

EXAMPLE = "右手がBlue Cubeを取って左手に渡した後に置いて、その左手にあるRED CUBEを掴んで右手がそれを取って右手のところに置く"
CASES = {
    "forward": (["右手で青いキューブを取って左手に渡して左側に置く"], 1),
    "reverse": (["左手で赤いキューブを取って右手に渡して右側に置く"], 1),
    "example": ([EXAMPLE], 2),
    "paraphrase": (["青いやつを右で拾って、空いてるほうの手にパスして、そっちの端に置いといて。そのあと赤い方を左でつまんで右に受け取ってもらって右側に置いといて"], 2),
    "english": (["Pick the blue cube with the right hand, then pass it to the left hand, then place it on the left side. Pick the red cube with the left hand, then pass it to the right hand, then place it on the right side."], 2),
    "followup": (["右手で青いキューブを掴んで", "それを左手に渡して左側に置く"], 1),
    "two_places": (["赤を左側に置いて青を右側に置く"], 0),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=[*CASES, "all"], default="example")
    parser.add_argument("--provider", choices=["fallback", "local"], default="fallback")
    parser.add_argument("--randomize", action="store_true")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    os.environ["OPENARM_LOCAL_PLANNER"] = "1" if args.provider == "local" else "0"
    env = create_backend()
    executor = TaskExecutor(env)
    planner = AgentPlanner()
    planner.api_key = planner.base_url = planner.model = ""
    records = []
    for name in CASES if args.case == "all" else [args.case]:
        env.reset(randomize=args.randomize)
        executor.reset_context()
        env.step(200)
        instructions, expected_handovers = CASES[name]
        results = []
        for instruction in instructions:
            print(f"\n{name}: {instruction}", flush=True)
            try:
                plan = planner.plan(instruction, executor.world_state())
                print(f"provider={plan.provider} planning={plan.planning_latency:.2f}s actions={len(plan.program['actions'])}", flush=True)
                def emit(kind, **payload):
                    if kind == "skill_finished":
                        r = payload["result"]
                        print(f"  {r['skill']} {r['arm']}: {r['success']} {r.get('detail', '')}", flush=True)
                result = executor.run(plan, emit=emit)
            except Exception as exc:
                result = {"success": False, "error": f"{type(exc).__name__}: {exc}"}
                print(result["error"], flush=True)
            results.append(result)
            if not result["success"]:
                break
        handovers = [s for result in results for s in result.get("skills", []) if s["skill"] == "handover_object" and s["success"]]
        passed = all(r["success"] for r in results) and len(results) == len(instructions) and len(handovers) == expected_handovers
        if args.provider == "local":
            passed = passed and all(r.get("plan", {}).get("provider") == "local_ollama" for r in results if len(r.get("plan", {}).get("program", {}).get("actions", [])) > 1)
        records.append({"case": name, "success": passed, "randomize": args.randomize,
                        "expected_handovers": expected_handovers, "observed_handovers": len(handovers), "executions": results})
        output = args.json or ROOT / "outputs" / f"language_tasks_{args.provider}.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(records, ensure_ascii=False, indent=2))
        print(f"{name}: {'PASS' if passed else 'FAIL'}; physical handovers={len(handovers)}/{expected_handovers}; evidence={output}", flush=True)
    return 0 if all(r["success"] for r in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
