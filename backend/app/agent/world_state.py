from __future__ import annotations

import numpy as np

from backend.app.simulation.base import SimulationBackend
from backend.app.planning.task_program import TABLE_TARGETS


def physical_holds(env: SimulationBackend) -> dict[str, str | None]:
    """Contact-derived occupancy, never a planner's remembered ownership."""
    holds = {}
    objects = env.frame().objects
    for side in ("left", "right"):
        pinched = []
        for obj in objects:
            contacts = env.object_contacts(obj, side)
            if any("inner" in c for c in contacts) and any("outer" in c for c in contacts):
                pinched.append(obj)
        holds[side] = pinched[0] if len(pinched) == 1 else None
    return holds


def build_world_state(env: SimulationBackend) -> dict:
    frame = env.frame()
    holds = physical_holds(env)
    objects = []
    for name, position in frame.objects.items():
        color = name.split("_")[0]
        objects.append({
            "id": name,
            "type": {"wine_bottle": "bottle", "wine_glass": "glass"}.get(name, "cube"),
            "color": color,
            "position": position,
            "graspable": name != "wine_glass",
            "held_by": [side for side, obj in holds.items() if obj == name],
        })
    targets = []
    for name, position in frame.zones.items():
        targets.append({"id": name, "color": name.split("_")[0], "position": position})
    targets.extend({"id": name, "position": list(position), "type": "table_region"} for name, position in TABLE_TARGETS.items())
    relations = []
    for obj in objects:
        obj_pos = np.asarray(obj["position"])
        for target in targets:
            target_pos = np.asarray(target["position"])
            horizontal = float(np.linalg.norm(obj_pos[:2] - target_pos[:2]))
            if horizontal < 0.06 and abs(obj_pos[2] - target_pos[2]) < 0.06:
                relations.append({"subject": obj["id"], "relation": "inside", "object": target["id"], "distance": horizontal})
            elif horizontal < 0.20:
                relations.append({"subject": obj["id"], "relation": "near", "object": target["id"], "distance": horizontal})
        for other in objects:
            if other["id"] == obj["id"]:
                continue
            other_pos = np.asarray(other["position"])
            if abs(obj_pos[2] - other_pos[2]) < 0.08 and abs(obj_pos[0] - other_pos[0]) < 0.18:
                relations.append({"subject": obj["id"], "relation": "right_of" if obj_pos[1] < other_pos[1] else "left_of", "object": other["id"], "distance": float(abs(obj_pos[1] - other_pos[1]))})
    return {
        "robot": {
            "type": "openarm_v2",
            "active_arm": next((side for side, obj in holds.items() if obj), None),
            "left_arm": {"ee": frame.ee["left"], "gripper": frame.grippers["left"], "held_object": holds["left"]},
            "right_arm": {"ee": frame.ee["right"], "gripper": frame.grippers["right"], "held_object": holds["right"]},
        },
        "objects": objects,
        "targets": targets,
        "spatial_relations": relations,
        "sim_time": frame.sim_time,
        "scene_profile": frame.scene_profile,
        "fluid": frame.fluid,
        "compute_backend": getattr(env, "compute_backend", "cpu"),
        "requested_realtime_factor": getattr(env, "simulation_speed", 1.),
        "requested_motion_speed": getattr(env, "motion_speed", 1.),
    }
