"""Independent physical verification of a place goal.

The check reads the engine's own state through `SimulationBackend` and never consults the
plan or the agent's claim, so a failed execution cannot be reported as a success. The
tolerances are the ones the verified MuJoCo demonstration was measured against.

`released` is physical: the object must be untouched by any gripper finger. With the friction
pinch there is no engine-side grasp state that could be left on, so contact is the evidence.
"""

from __future__ import annotations

import numpy as np

from backend.app.simulation.base import SimulationBackend


def verify_place_goal(
    env: SimulationBackend, object_id: str, target_id: str, tolerance: float = 0.04
) -> dict:
    before = env.object_pose(object_id)[0].copy()
    env.step(100)
    position, _ = env.object_pose(object_id)
    target = env.zone_position(target_id)
    error = float(np.linalg.norm(position[:2] - target[:2]))
    vertical_error = float(abs(position[2] - target[2]))
    speed = env.object_linear_speed(object_id)
    settling_displacement = float(np.linalg.norm(position - before))
    stable = bool(speed < 0.02 and settling_displacement < .002)
    released = not env.object_contacts(object_id)
    success = error <= tolerance and vertical_error <= 0.05 and stable and released
    return {
        "success": bool(success),
        "object_id": object_id,
        "target_id": target_id,
        "position": position.tolist(),
        "target_position": target.tolist(),
        "horizontal_position_error": error,
        "vertical_position_error": vertical_error,
        "object_linear_speed": speed,
        "tolerance": tolerance,
        "stable": stable,
        "settling_displacement": settling_displacement,
        "released": bool(released),
        "engine": env.name,
        "grasp_mode": env.grasp_mode,
        "failure_category": None if success else "GOAL_NOT_SATISFIED",
    }


def verify_task_goal(env: SimulationBackend, plan, results: list[dict], initial_state: dict | None = None) -> dict:
    """Read every final predicate, including the objects placed earlier in a sequence."""
    complete = len(results) == len(plan.steps) and all(r.get("success") for r in results)
    checks = plan.goal.get("checks", [])
    if not checks and plan.goal.get("type") in {"place_in_zone", "already_at_target"}:
        checks = [{"type": "placement", "object": plan.goal["object"], "target": plan.goal["target"]}]
    measured = []
    # Allow release to settle under the same physics; no pose correction or scene reset.
    if any(c["type"] in {"placement", "released"} for c in checks):
        env.step(200)
    for check in checks:
        obj = check["object"]
        if check["type"] == "placement":
            evidence = verify_place_goal(env, obj, check["target"])
        elif check["type"] == "holding":
            contacts = env.object_contacts(obj, check["arm"])
            both = any("inner" in c for c in contacts) and any("outer" in c for c in contacts)
            position = env.object_pose(obj)[0]
            evidence = {"success": bool(both and position[2] > 1.045), "contacts": contacts, "position": position.tolist()}
        elif check["type"] == "released":
            contacts = env.object_contacts(obj)
            evidence = {"success": not contacts, "contacts": contacts}
        elif check["type"] == "approach":
            from backend.app.simulation.openarm_v2.definition import FINGERTIP_MIDPOINT
            position, rotation = env.ee_pose(check["arm"])
            error = float(np.linalg.norm(env.object_pose(obj)[0] - (position + rotation @ FINGERTIP_MIDPOINT[check["arm"]])))
            evidence = {"success": error < .012, "fingertip_midpoint_error": error}
        else:
            evidence = {"success": False, "failure_category": "UNSUPPORTED_GOAL"}
        measured.append({**check, **evidence})
    if initial_state:
        touched = {s.object for s in plan.steps if s.object}
        for item in initial_state["objects"]:
            if item["id"] not in touched:
                displacement = float(np.linalg.norm(env.object_pose(item["id"])[0] - item["position"]))
                measured.append({"type": "undisturbed", "object": item["id"], "success": displacement < .006, "displacement": displacement})
    success = complete and all(c["success"] for c in measured)
    failure = next((r.get("error") for r in results if not r.get("success")), None)
    # Preserve the familiar measured place fields for existing consumers.
    simple = measured[0] if len(measured) == 1 and measured[0]["type"] == "placement" else {}
    return {**simple, "success": bool(success), "goal_type": plan.goal["type"],
            "method": "genesis_postconditions", "checks": measured, "all_skills_completed": complete,
            "engine": env.name, "grasp_mode": env.grasp_mode,
            "failure_category": None if success else (failure or "GOAL_NOT_SATISFIED")}
