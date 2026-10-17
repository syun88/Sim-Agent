#!/usr/bin/env python3
"""Standalone Genesis native viewer for OpenArm v2.

Runs Genesis World with show_viewer=True on the main thread (required on macOS).
Polls the backend API for instructions submitted from the web UI.
Press Ctrl+C to exit.
"""

from __future__ import annotations

import time

import httpx

from backend.app.agent.world_state import build_world_state
from backend.app.control.motion import OpenArmV2Controller
from backend.app.planning.agent_planner import AgentPlanner, validate_plan
from backend.app.simulation.genesis_backend.environment import GenesisOpenArmV2Environment
from backend.app.skills.robot_skills import SkillRunner

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
        instruction = _poll_next_command()
        if instruction is None:
            continue
        if len(instruction) < 3:
            continue

        print(f"\nReceived: {instruction}")

        state = build_world_state(env)
        plan = planner.plan(instruction, state, [])
        valid, errors = validate_plan(plan, state)
        if not valid:
            print(f"Plan invalid: {errors}")
            _report_result(instruction, False)
            continue

        print(f"Plan ({plan.provider}):")
        for i, step in enumerate(plan.steps, 1):
            print(f"  {i}. {step.skill} {step.object or step.target or ''}")

        results = skills.execute_plan(plan.model_dump()["steps"])
        success = all(r.get("success") for r in results)
        if success:
            print("Task completed successfully.")
        else:
            print("Task failed.")
        _report_result(instruction, success)

    print("Closing Genesis viewer...")


def _poll_next_command() -> str | None:
    try:
        response = httpx.get(f"{BACKEND_URL}/api/genesis/next", timeout=1.0)
        response.raise_for_status()
        return response.json().get("instruction")
    except Exception:
        time.sleep(2.0)
        return None

    time.sleep(0.5)


def _report_result(instruction: str, success: bool) -> None:
    try:
        httpx.post(
            f"{BACKEND_URL}/api/genesis/result",
            json={"instruction": instruction, "success": success},
            timeout=2.0,
        )
    except Exception:
        pass


if __name__ == "__main__":
    main()
