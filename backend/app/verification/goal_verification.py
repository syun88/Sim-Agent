from __future__ import annotations

import numpy as np

import mujoco

from backend.app.simulation.openarm_v2.environment import OpenArmV2Environment


def verify_place_goal(env: OpenArmV2Environment, object_id: str, target_id: str, tolerance: float = 0.04) -> dict:
    position, _ = env.object_pose(object_id)
    target = env.zone_position(target_id)
    error = float(np.linalg.norm(position[:2] - target[:2]))
    vertical_error = float(abs(position[2] - target[2]))
    stable = bool(np.linalg.norm(env.data.qvel[env._object_joints[object_id]:env._object_joints[object_id] + 6]) < 0.08)
    weld_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_EQUALITY, f"grasp_left_{object_id}")
    released = weld_id >= 0 and not env.data.eq_active[weld_id]
    success = error <= tolerance and vertical_error <= 0.05 and stable and released
    return {
        "success": bool(success),
        "object_id": object_id,
        "target_id": target_id,
        "position": position.tolist(),
        "target_position": target.tolist(),
        "horizontal_position_error": error,
        "vertical_position_error": vertical_error,
        "tolerance": tolerance,
        "stable": stable,
        "released": bool(released),
        "failure_category": None if success else "GOAL_NOT_SATISFIED",
    }
