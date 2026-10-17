#!/usr/bin/env python3
"""Verify the OpenArm v2 pick & place on Genesis World from multiple initial positions.

Every run uses the real skill sequence (approach → grasp → lift → move → release → retreat)
and the real physics. Object positions are never set directly during execution; the only
teleport is the documented scene reset/setup operation. Results are measured by the
independent verification step, not asserted.
"""

from __future__ import annotations

import sys

import numpy as np

from backend.app.agent.world_state import build_world_state
from backend.app.control.motion import OpenArmV2Controller
from backend.app.planning.agent_planner import AgentPlanner, validate_plan
from backend.app.simulation.genesis_backend.environment import GenesisOpenArmV2Environment
from backend.app.skills.robot_skills import SkillRunner
from backend.app.verification.goal_verification import verify_place_goal


SEEDS = [20261017, 20261018, 20261019, 20261020, 20261021]


def run_one(env, controller, skills, planner, seed: int) -> dict:
    env.reset(randomize=True)
    # Force the scene's randomisation seed so this run starts from a known different pose.
    rng = np.random.default_rng(seed)
    positions = {
        "red_cube": np.array([rng.uniform(0.37, 0.42), rng.uniform(0.11, 0.18), 1.032]),
        "blue_cube": np.array([0.50, -0.22, 1.032]),
    }
    entity = env._entity
    for name, pos in positions.items():
        qpos = np.array([*pos, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        entity.set_qpos(qpos, env._object_qs[name] + np.arange(7), zero_velocity=True)
    env.refresh()

    state = build_world_state(env)
    plan = planner.plan("Move the red cube into the blue zone.", state, [])
    valid, errors = validate_plan(plan, state)
    if not valid:
        return {"seed": seed, "success": False, "errors": errors, "plan_steps": len(plan.steps)}

    results = skills.execute_plan(plan.model_dump()["steps"])
    failed_skills = [r for r in results if not r.get("success")]
    verification = verify_place_goal(env, "red_cube", "blue_zone")
    return {
        "seed": seed,
        "success": verification["success"] and not failed_skills,
        "failed_skills": [r.get("skill") for r in failed_skills],
        "verification": {
            "horizontal_error_mm": round(verification["horizontal_position_error"] * 1000, 1),
            "vertical_error_mm": round(verification["vertical_position_error"] * 1000, 1),
            "stable": verification["stable"],
            "released": verification["released"],
        },
        "initial_position": [round(v, 4) for v in positions["red_cube"]],
        "plan_steps": len(plan.steps),
    }


def main() -> int:
    print("Starting Genesis World multi-position pick & place verification...")
    env = GenesisOpenArmV2Environment()
    controller = OpenArmV2Controller(env)
    skills = SkillRunner(env, controller)
    planner = AgentPlanner()

    results = []
    for seed in SEEDS:
        print(f"\n=== Seed {seed} ===")
        result = run_one(env, controller, skills, planner, seed)
        print(f"  initial: {result['initial_position']}")
        if result.get("verification"):
            print(f"  verification: {result['verification']}")
        if result.get("failed_skills"):
            print(f"  failed skills: {result['failed_skills']}")
        print(f"  success: {result['success']}")
        results.append(result)

    successes = sum(1 for r in results if r["success"])
    print(f"\n{'=' * 50}")
    print(f"Results: {successes}/{len(results)} succeeded")
    for r in results:
        status = "PASS" if r["success"] else "FAIL"
        v = r.get("verification", {})
        print(f"  seed={r['seed']}  {status}  h_err={v.get('horizontal_error_mm', '?')} mm")

    return 0 if successes == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
