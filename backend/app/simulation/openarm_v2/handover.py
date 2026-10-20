"""Coupled handover geometry from the attached official model (kinematics only).

The two pads share opposite portions of the cube's side faces. Solve both arms and the
cube orientation together, respecting joint limits and checking the model's collision
meshes. These are candidate poses, not a second physics engine: all motion, contact,
load transfer and success evidence subsequently come from Genesis.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from backend.app.simulation.openarm_v2.definition import ARM_JOINTS, FINGERTIP_MIDPOINT, GRASP_ROTATION, GRIPPER_JOINTS
from backend.app.simulation.openarm_v2.scene import resolved_scene_xml


@dataclass
class HandoverGeometry:
    position: np.ndarray
    donor_rotation: np.ndarray
    donor_seed: np.ndarray
    receiver_seed: np.ndarray
    receiver_relative_rotation: np.ndarray
    receiver_bias: float
    residual: float


def plan_handover_geometry(env, giver: str, obj: str, receiver_bias: float = .015) -> HandoverGeometry:
    import mujoco
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation

    receiver = "right" if giver == "left" else "left"
    ee, rotation = env.ee_pose(giver)
    cube, cube_rotation = env.object_pose(obj)
    grasp_offset = rotation.T @ (cube - ee)
    grasp_rotation = rotation.T @ cube_rotation
    model = mujoco.MjModel.from_xml_string(resolved_scene_xml())
    data = mujoco.MjData(model)
    data.qpos[:] = model.key_qpos[0]
    joints = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for side in ("left", "right") for name in ARM_JOINTS[side]]
    indices = np.array([model.jnt_qposadr[j] for j in joints])
    limits = model.jnt_range[joints].copy()
    limits[:, 0] += .04
    limits[:, 1] -= .04
    home = np.clip(model.key_qpos[0, indices], limits[:, 0] + 1e-6, limits[:, 1] - 1e-6)
    sites = {s: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{s}_ee_control_point") for s in (giver, receiver)}
    frame = env.frame()
    for side in (giver, receiver):
        value = frame.grippers[giver] if side == giver else (.7854 if side == "left" else -.7854)
        for name in (GRIPPER_JOINTS[side], f"openarm_{side}_finger_joint2"):
            joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            data.qpos[model.jnt_qposadr[joint]] = value
    relative = np.diag([-1., -1., 1.]) @ GRASP_ROTATION
    desired_cube = np.array([.44, 0., 1.15])

    def poses(q):
        data.qpos[indices] = q
        mujoco.mj_kinematics(model, data)
        donor_r = data.site_xmat[sites[giver]].reshape(3, 3)
        obj_r = donor_r @ grasp_rotation
        obj_p = data.site_xpos[sites[giver]] + donor_r @ grasp_offset
        receiving_r = obj_r @ relative
        receiving_p = obj_p + receiving_r @ np.array([0., 0., receiver_bias]) - receiving_r @ FINGERTIP_MIDPOINT[receiver]
        return donor_r, obj_r, obj_p, receiving_r, receiving_p

    def residual(q):
        _, _, obj_p, receiving_r, receiving_p = poses(q)
        return np.r_[data.site_xpos[sites[receiver]] - receiving_p,
                     .2 * Rotation.from_matrix(receiving_r @ data.site_xmat[sites[receiver]].reshape(3, 3).T).as_rotvec(),
                     .8 * (obj_p - desired_cube), .0005 * (q - home)]

    rng = np.random.default_rng(17)
    best_error = float("inf")
    collision_rejections = 0
    for sample in range(16):
        seed = home if sample == 0 else np.clip(home + rng.normal(0, 1, 14), limits[:, 0] + 1e-6, limits[:, 1] - 1e-6)
        solved = least_squares(residual, seed, bounds=(limits[:, 0], limits[:, 1]), max_nfev=300,
                               ftol=1e-9, xtol=1e-9, gtol=1e-9)
        error = float(np.linalg.norm(residual(solved.x)[:9]))
        best_error = min(best_error, error)
        if error > .003:
            continue
        donor_r, _, obj_p, _, _ = poses(solved.x)
        donor_r, obj_p = donor_r.copy(), obj_p.copy()
        mujoco.mj_forward(model, data)
        blocked = False
        for contact in data.contact:
            names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(g)) or "" for g in contact.geom]
            arm_pair = ("left" in names[0] and "right" in names[1]) or ("right" in names[0] and "left" in names[1])
            table_pair = "cell_table" in " ".join(names) and any("finger" in n or "link" in n for n in names)
            if (arm_pair or table_pair) and contact.dist < -.001:
                blocked = True
                break
        if blocked:
            collision_rejections += 1
            continue
        slices = {"left": slice(0, 7), "right": slice(7, 14)}
        return HandoverGeometry(obj_p, donor_r, solved.x[slices[giver]], solved.x[slices[receiver]], relative, receiver_bias, error)
    raise ValueError(f"No collision-free coupled handover pose fits the measured grasp (kinematic residual {best_error:.4f}, collision rejections {collision_rejections})")
