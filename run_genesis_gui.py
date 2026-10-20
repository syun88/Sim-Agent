#!/usr/bin/env python3
"""Standalone Genesis native viewer for OpenArm v2.

Genesis World runs on the main thread (required by its native viewer on macOS). The FastAPI
process is only a command/result broker; it never creates a second simulation.
"""

from __future__ import annotations

import time

import httpx

from backend.app.agent.world_state import build_world_state
from backend.app.control.motion import OpenArmV2Controller
from backend.app.planning.agent_planner import AgentPlanner, validate_plan
from backend.app.simulation.genesis_backend.environment import GenesisOpenArmV2Environment
from backend.app.skills.robot_skills import SkillRunner
from backend.app.verification.goal_verification import verify_place_goal

BACKEND_URL = "http://127.0.0.1:8000"


def main() -> None:
    print("Starting Genesis native viewer (OpenArm v2)...")
    env = GenesisOpenArmV2Environment(show_viewer=True)
    controller = OpenArmV2Controller(env)
    skills = SkillRunner(env, controller)
    planner = AgentPlanner()

    print("Ready. Waiting for instructions from the web UI.")
    print("Open http://127.0.0.1:8000/simple and type a command.")
    print("Press Ctrl+C to exit.")

    while True:
        time.sleep(0.5)
        try:
            response = httpx.get(f"{BACKEND_URL}/api/genesis/next", timeout=1.0)
            response.raise_for_status()
            command = response.json()
        except Exception:
            print(".", end="", flush=True)
            continue
        if not command:
            continue
        if command.get("action") == "reset":
            try:
                env.reset(randomize=bool(command.get("randomize", False)))
                _report(command, True, verification={"success": True, "goal_type": "scene_reset"})
            except Exception as exc:
                _report(command, False, detail=f"{type(exc).__name__}: {exc}")
            continue

        instruction = command.get("instruction") or ""
        if len(instruction) < 3:
            continue
        print(f"\nReceived: {instruction}", flush=True)
        try:
            state = build_world_state(env)
            plan = planner.plan(instruction, state, [])
            valid, errors = validate_plan(plan, state)
            if not valid:
                print(f"Plan invalid: {errors}")
                _report(command, False, plan=plan.model_dump(), skills=[], detail=f"INVALID_PLAN: {errors}")
                continue

            print(f"Plan ({plan.provider}):", flush=True)
            for i, step in enumerate(plan.steps, 1):
                description = step.skill
                if step.object:
                    description += f" {step.object}"
                elif step.target:
                    description += f" {step.target}"
                elif step.direction:
                    description += f" {step.direction} {step.distance_cm}cm"
                print(f"  {i}. {description}", flush=True)

            results = skills.execute_plan(plan.model_dump()["steps"])
            all_skills_success = all(result.get("success") for result in results)
            goal_type = plan.goal.get("type", "place_in_zone")
            if goal_type == "place_in_zone":
                object_id = plan.goal.get("object")
                target_id = plan.goal.get("target")
                if not object_id or not target_id:
                    verification = {"success": False, "goal_type": goal_type, "failure_category": "INVALID_GOAL"}
                    success = False
                else:
                    verification = verify_place_goal(env, object_id, target_id)
                    success = verification["success"] and all_skills_success
            else:
                verification = {
                    "success": bool(all_skills_success),
                    "goal_type": goal_type,
                    "method": "completed_skill_execution",
                    "engine": env.name,
                    "grasp_mode": env.grasp_mode,
                    "failure_category": None if all_skills_success else "SKILL_FAILED",
                }
                success = all_skills_success
            if success:
                print("Task completed successfully.", flush=True)
            else:
                print("Task failed.", flush=True)
            _report(command, success, plan=plan.model_dump(), skills=results, verification=verification)
        except Exception as exc:
            print(f"Execution failed: {type(exc).__name__}: {exc}", flush=True)
            _report(command, False, detail=f"{type(exc).__name__}: {exc}")


def _report(command: dict, success: bool, **payload) -> None:
    body = {
        "task_id": command.get("task_id"),
        "instruction": command.get("instruction"),
        "success": bool(success),
        **payload,
    }
    try:
        httpx.post(f"{BACKEND_URL}/api/genesis/result", json=body, timeout=5.0)
    except Exception:
        pass


if __name__ == "__main__":
    main()
