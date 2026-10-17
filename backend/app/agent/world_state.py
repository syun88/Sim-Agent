from __future__ import annotations

import numpy as np

from backend.app.simulation.base import SimulationBackend


def build_world_state(env: SimulationBackend) -> dict:
    frame = env.frame()
    objects = []
    for name, position in frame.objects.items():
        color = name.split("_")[0]
        objects.append({
            "id": name,
            "type": "cube",
            "color": color,
            "position": position,
            "graspable": True,
        })
    targets = []
    for name, position in frame.zones.items():
        targets.append({"id": name, "color": name.split("_")[0], "position": position})
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
            "active_arm": "left",
            "left_arm": {"ee": frame.ee["left"], "gripper": frame.grippers["left"]},
            "right_arm": {"ee": frame.ee["right"], "gripper": frame.grippers["right"]},
        },
        "objects": objects,
        "targets": targets,
        "spatial_relations": relations,
        "sim_time": frame.sim_time,
    }
