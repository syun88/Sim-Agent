"""Measure candidate spawn poses on the official MJCF using MuJoCo's own collision/kinematics."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.simulation.openarm_v2.definition import (  # noqa: E402
    ARM_JOINTS,
    EE_SITES,
    GRIPPER_JOINTS,
    OBJECTS,
    ZONES,
)
from backend.app.simulation.openarm_v2.scene import (  # noqa: E402
    INITIAL_OBJECT_POSES,
    resolved_scene_xml,
)


def load_model():
    xml = resolved_scene_xml()
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    return model, data


def set_arm_qpos(model, data, side, values):
    values = np.asarray(values, dtype=float)
    for name, value in zip(ARM_JOINTS[side], values):
        jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        adr = model.jnt_qposadr[jnt_id]
        data.qpos[adr] = value


def get_site_pose(model, data, site_name):
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    return data.site_xpos[site_id].copy(), data.site_xmat[site_id].copy()


def get_joint_qposadr(model, joint_name):
    jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    return model.jnt_qposadr[jnt_id]


def get_body_geom_ids(model, body_name):
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    # All geoms belonging to this body and its descendants
    result = []
    for geom_id in range(model.ngeom):
        if model.geom_bodyid[geom_id] == body_id:
            result.append(geom_id)
    return result


def classify_contact(model, g1, g2, depth, table_geom_ids, cube_geom_ids):
    """Classify a contact pair and return (category, depth, body1, body2)."""
    body1 = model.geom_bodyid[g1]
    body2 = model.geom_bodyid[g2]
    name1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body1)
    name2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body2)
    is_arm1 = "openarm" in (name1 or "")
    is_arm2 = "openarm" in (name2 or "")
    is_table1 = g1 in table_geom_ids
    is_table2 = g2 in table_geom_ids
    all_cube_geoms = set().union(*cube_geom_ids.values())
    is_cube1 = g1 in all_cube_geoms
    is_cube2 = g2 in all_cube_geoms
    pair = f"{name1}|{name2}"
    if is_arm1 and is_arm2:
        return "arm-arm", depth, pair
    if (is_arm1 or is_arm2) and (is_table1 or is_table2):
        return "arm-table", depth, pair
    if (is_arm1 or is_arm2) and (is_cube1 or is_cube2):
        return "arm-cube", depth, pair
    return "other", depth, pair


def measure_pose(model, data, candidate, table_geom_ids, cube_geom_ids):
    data.qpos[:] = model.key_qpos[0].copy()
    for side in ("left", "right"):
        set_arm_qpos(model, data, side, candidate[side])
    mujoco.mj_kinematics(model, data)
    mujoco.mj_collision(model, data)

    result = {}
    for side in ("left", "right"):
        pos, _ = get_site_pose(model, data, EE_SITES[side])
        gripper_qpos_adr = get_joint_qposadr(model, GRIPPER_JOINTS[side])
        result[f"ee_{side}"] = pos.tolist()
        result[f"gripper_{side}_height_above_table"] = float(pos[2] - 1.0)

    distances = {}
    for cube_name in OBJECTS:
        cube_pos = np.array(INITIAL_OBJECT_POSES[cube_name])
        for side in ("left", "right"):
            ee_pos = np.array(result[f"ee_{side}"])
            dist = float(np.linalg.norm(ee_pos - cube_pos))
            distances[f"{side}_ee_to_{cube_name}"] = dist
    result["distances"] = distances

    contacts = {"arm-arm": [], "arm-cube": [], "arm-table": [], "other": []}
    worst = {}
    for i in range(data.ncon):
        con = data.contact[i]
        category, depth, pair = classify_contact(
            model, int(con.geom1), int(con.geom2), float(con.dist), table_geom_ids, cube_geom_ids
        )
        contacts[category].append({"pair": pair, "depth": float(depth)})
        if depth < worst.get(category, 0.0):
            worst[category] = float(depth)
    result["contacts"] = contacts
    result["worst_penetration"] = worst
    result["collision_free"] = all(depth >= -1e-6 for depth in worst.values())
    return result


def find_table_and_cube_geoms(model):
    table_geom_ids = set()
    cube_geom_ids = {}
    for geom_id in range(model.ngeom):
        body_id = model.geom_bodyid[geom_id]
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        if body_name and "table" in body_name.lower():
            table_geom_ids.add(geom_id)
        if body_name and any(cube in body_name for cube in OBJECTS):
            cube_geom_ids.setdefault(body_name, set()).add(geom_id)
    return table_geom_ids, cube_geom_ids


def main():
    model, data = load_model()
    table_geom_ids, cube_geom_ids = find_table_and_cube_geoms(model)

    # Get the official keyframe for each arm
    keyframe_arm = {}
    for side in ("left", "right"):
        keyframe_arm[side] = np.array([
            model.key_qpos[0][model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]]
            for name in ARM_JOINTS[side]
        ])

    print("Official keyframe per arm (rad):", keyframe_arm["left"])
    print("Official keyframe per arm (deg):", np.degrees(keyframe_arm["left"]))

    candidates = []

    # 1. Official keyframe
    candidates.append({
        "name": "official_keyframe",
        "left": keyframe_arm["left"].copy(),
        "right": keyframe_arm["right"].copy(),
    })

    # 2. All seven joints at 90 deg
    all_90 = np.full(7, np.pi / 2)
    candidates.append({
        "name": "all_7_joints_90deg",
        "left": all_90.copy(),
        "right": all_90.copy(),
    })

    # 3. joints 2 and 4 at 90 deg, others keyframe
    j2j4 = keyframe_arm["left"].copy()
    j2j4[1] = np.pi / 2
    j2j4[3] = np.pi / 2
    candidates.append({
        "name": "joint2_and_joint4_90deg",
        "left": j2j4.copy(),
        "right": j2j4.copy(),
    })

    # 4. joint 2 at 45 deg with joint 4 at 90 deg
    j2_45_j4_90 = keyframe_arm["left"].copy()
    j2_45_j4_90[1] = np.pi / 4
    j2_45_j4_90[3] = np.pi / 2
    candidates.append({
        "name": "joint2_45deg_joint4_90deg",
        "left": j2_45_j4_90.copy(),
        "right": j2_45_j4_90.copy(),
    })

    # 5. joint 4 at 90 deg with joint 2 raised (joint2 = -45 deg)
    j4_90_j2_neg45 = keyframe_arm["left"].copy()
    j4_90_j2_neg45[1] = -np.pi / 4
    j4_90_j2_neg45[3] = np.pi / 2
    candidates.append({
        "name": "joint2_minus45deg_joint4_90deg",
        "left": j4_90_j2_neg45.copy(),
        "right": j4_90_j2_neg45.copy(),
    })

    # 6. Only joint 4 at 90 deg
    only_j4 = keyframe_arm["left"].copy()
    only_j4[3] = np.pi / 2
    candidates.append({
        "name": "only_joint4_90deg",
        "left": only_j4.copy(),
        "right": only_j4.copy(),
    })

    # 7. joint 2 at 90 deg (shoulder forward)
    only_j2 = keyframe_arm["left"].copy()
    only_j2[1] = np.pi / 2
    candidates.append({
        "name": "only_joint2_90deg",
        "left": only_j2.copy(),
        "right": only_j2.copy(),
    })

    results = []
    for cand in candidates:
        measured = measure_pose(model, data, cand, table_geom_ids, cube_geom_ids)
        results.append({
            "name": cand["name"],
            "left_deg": np.degrees(cand["left"]).round(2).tolist(),
            "right_deg": np.degrees(cand["right"]).round(2).tolist(),
            **measured,
        })

    with open(ROOT / "outputs" / "spawn_pose_measurements.json", "w") as f:
        json.dump(results, f, indent=2)

    # Print a compact table
    print("\nCandidate spawn poses:")
    for r in results:
        print(f"\n{r['name']}")
        print(f"  joints (deg): left={r['left_deg']} right={r['right_deg']}")
        print(f"  left ee pos:  {tuple(round(x, 3) for x in r['ee_left'])}")
        print(f"  right ee pos: {tuple(round(x, 3) for x in r['ee_right'])}")
        print(f"  left ee height above table:  {r['gripper_left_height_above_table']:.3f} m")
        print(f"  right ee height above table: {r['gripper_right_height_above_table']:.3f} m")
        print(f"  collision_free: {r['collision_free']}")
        print(f"  worst penetration: {r['worst_penetration']}")


if __name__ == "__main__":
    main()
