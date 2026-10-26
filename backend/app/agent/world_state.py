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
            if env.pinch_established(obj, side):
                pinched.append(obj)
        holds[side] = pinched[0] if len(pinched) == 1 else None
    return holds



def _builtin_skills() -> list[str]:
    """The official cell's list, unchanged: it is trusted as a whole and reports no grippers here."""
    return ["observe_scene", "park_empty_arm", "approach_object", "grasp_object", "lift_object",
            "move_to_target", "move_to_object", "release_object", "retreat", "return_home",
            "open_gripper", "close_gripper", "move_relative", "handover_object",
            "throw_object", "pour_liquid"]


def _imported_skills(grippers, arms, pinch_verified, has_fluid) -> list[str]:
    """What an imported robot offers, derived from what it has rather than from its loader.

    Measured 2026-10-25 on UR5e + 2F-85 and OpenManipulator-X: both can be pinched with physically
    verified (grasp_calibrated), yet the imported path offered neither throwing nor pouring. Throwing
    needs a verified hold - the skill itself refuses one it cannot prove - so it follows the pinch, and
    pouring additionally needs a scene with fluid in it. Handover remains a two-arm skill.
    """
    skills = ["observe_scene", "park_empty_arm", "move_relative", "return_home", "move_joint"]
    if not grippers:
        return skills
    skills += ["approach_object", "grasp_object", "lift_object", "move_to_target", "move_to_object",
               "release_object", "retreat", "open_gripper", "close_gripper"]
    if pinch_verified:
        skills.append("throw_object")
        if has_fluid:
            skills.append("pour_liquid")
    if len(arms) > 1:
        skills.append("handover_object")
    return skills

def build_world_state(env: SimulationBackend) -> dict:
    frame = env.frame()
    # A mobile base reports its own pose; desk scenes have none, and what the planner is told is conditioned
    # on that.
    try:
        base_pose = [float(v) for v in env.base_pose()]
    except (AttributeError, TypeError, NotImplementedError):
        base_pose = None
    holds = physical_holds(env)
    contacts = env.contact_names()
    available = getattr(env,'available_arms',['left','right'])
    objects = []
    for name, position in frame.objects.items():
        color = name.split("_")[0]
        _, attitude = env.object_pose(name)
        admission=(getattr(env,'spawn_validation',None) or {}).get('objects',{}).get(name,{})
        admitted_arm=(admission.get('arm') if admission.get('position') is not None
            and np.linalg.norm(np.asarray(position)-admission['position'])<.005 else None)
        objects.append({
            "id": name,
            "type": {"wine_bottle": "bottle", "wine_glass": "glass"}.get(name, "cube"),
            "color": color,
            "position": position,
            "rotation": attitude.tolist(),
            "linear_speed_m_s": env.object_linear_speed(name),
            "touching": sorted({other for pair in contacts if name in pair.split('|')
                                for other in pair.split('|') if other != name}),
            "graspable": name != "wine_glass" and getattr(env, 'grasp_is_physical_contact', True),
            "held_by": [side for side, obj in holds.items() if obj == name],
            **({'admitted_grasp_arm':admitted_arm} if admitted_arm else {}),
            **({"bounds": env.object_bounds(name).tolist()} if name.endswith("_cube") else {}),
        })
    targets = []
    for name, position in frame.zones.items():
        targets.append({"id": name, "color": name.split("_")[0], "position": position})
    imported = getattr(env, "robot_id", "openarm_v2") != "openarm_v2"
    profile = getattr(env, "profile", None) or {}
    grippers = profile.get("grippers") or {}
    arms = list(getattr(env, "available_arms", ["left", "right"]))
    # A robot's skills follow what it can do, not which loader built it. Measured 2026-10-25: the
    # imported path withheld throw_object and pour_liquid from every imported robot, so a robot with a
    # physically verified pinch could not throw even though every environment API the skill needs is
    # inherited from the same base class - and the refusal blamed a calibration that was in fact valid
    # (the package's gripper_runtime.json verified, hashes matching). The built-in cell keeps exactly
    # the list it had: it reports no grippers through this profile and is trusted as a whole.
    pinch_verified = bool(grippers) and bool(profile.get("grasp_calibrated"))
    has_fluid = bool(getattr(frame, "fluid", None))
    if base_pose is None:
        # Table regions belong to desk scenes; a floor robot's world has floor objects and no table targets.
        targets.extend({"id": name,
                        "position": env.zone_position(name).tolist() if imported else list(position),
                        "type": "table_region"} for name, position in TABLE_TARGETS.items())
    relations = []
    from backend.app.agent.scene_relations import support_relations
    supports = support_relations(objects,contacts)
    relations.extend(supports)
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
    if hasattr(env,'joint_snapshot'):
        joints = env.joint_snapshot()
    else:
        from backend.app.simulation.openarm_v2.definition import ARM_JOINTS
        joints = {name:float(value) for side in available
                  for name,value in zip(ARM_JOINTS[side],env.arm_qpos(side))}
    skills = _builtin_skills() if not imported else _imported_skills(grippers, arms, pinch_verified, has_fluid)
    # Driving the base is a capability of the scene, not of the gripper.
    if base_pose is not None and "navigate_to" not in skills:
        skills = [*skills, "navigate_to"]
    return {
        "robot": {
            "type": getattr(env, "robot_id", "openarm_v2"),
            "name": getattr(env, "robot_name", "OpenArm v2"),
            "available_arms": getattr(env, "available_arms", ["left", "right"]),
            "end_effectors": {side:{'position':frame.ee[side],'rotation':env.ee_pose(side)[1].tolist()}
                              for side in available},
            "joint_positions": joints,
            **({"chains": env.profile["chains"],
                "model_summary": env.model_summary(),
                "joints": [{key: joint[key] for key in ("name", "type", "lower", "upper", "effort", "velocity")}
                           for joint in env.profile["joints"]], "grasp_calibrated": env.profile.get("grasp_calibrated", False), "grippers": {side: {k: g[k] for k in ("joint", "type", "status", "cube_size")} for side, g in env.profile.get("grippers", {}).items()}} if imported else {}),
            "active_arm": next((side for side, obj in holds.items() if obj), None),
            "left_arm": {"ee": frame.ee["left"], "gripper": frame.grippers["left"], "held_object": holds["left"]},
            "right_arm": {"ee": frame.ee["right"], "gripper": frame.grippers["right"], "held_object": holds["right"]},
        },
        "base_pose": base_pose,
        "objects": objects,
        "targets": targets,
        "spatial_relations": relations,
        "support_relations": supports,
        "workspace": {"table_height_m":env.table_height,"position_unit":"m","rotation":"world_from_object"},
        "capabilities": {"navigation": base_pose is not None,
                         "relative_placement": ["on", "left_of", "right_of", "front_of", "behind", "near"],
                         "skills": _builtin_skills() if not imported else _imported_skills(grippers, arms, pinch_verified, has_fluid),
                         "geometry_source": "Genesis collision bounds", "grasp": env.grasp_mode},
        "grasp_mode": env.grasp_mode,
        "spawn_validation": getattr(env, "spawn_validation", None),
        "sim_time": frame.sim_time,
        "scene_profile": frame.scene_profile,
        "fluid": frame.fluid,
        "compute_backend": getattr(env, "compute_backend", "cpu"),
        "requested_realtime_factor": getattr(env, "simulation_speed", 1.),
        "requested_motion_speed": getattr(env, "motion_speed", 1.),
    }
