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
    position, _ = env.object_pose(object_id)
    target = env.zone_position(target_id)
    error = float(np.linalg.norm(position[:2] - target[:2]))
    vertical_error = float(abs(position[2] - target[2]))
    speed = env.object_linear_speed(object_id)
    stable = bool(speed < 0.08)
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
        "released": bool(released),
        "engine": env.name,
        "grasp_mode": env.grasp_mode,
        "failure_category": None if success else "GOAL_NOT_SATISFIED",
    }
