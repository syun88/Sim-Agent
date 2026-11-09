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
    meeting: dict | None = None


DEFAULT_MEETING_SPAN_M = .09
DEFAULT_MEETING_STEP_M = .03
#: The point the cell used to hard-code. It stays in the candidate set so that if the measurement says it is
#: the best available, it is chosen *because it was measured*, not because nobody looked anywhere else.
LEGACY_MEETING_BOX = ((0.40, 0.52), 0.0, (1.10, 1.18))


def measured_meeting_point(env, controller, giver: str, receiver: str, obj: str, receiver_bias: float,
                           *, span: float = DEFAULT_MEETING_SPAN_M,
                           step: float = DEFAULT_MEETING_STEP_M) -> dict:
    """Where the two hands should meet, chosen because both arms can hold it - not a fixed table point.

    It was `x = clip(cube_x, .40, .52)`, **`y = 0.0`**, `z = clip(cube_z, 1.10, 1.18)`: a box someone once
    measured to work. The score here is the worse of the two arms' IK margins at that meeting point, so a
    point nearer to whichever arm is struggling wins, and the offsets are searched rather than assumed.
    """
    cube, _ = env.object_pose(obj)
    solve = getattr(controller, 'solve_ik', None)
    if not callable(solve):
        return {'point': [float(value) for value in cube], 'measured': False,
                'reason': 'NO_IK_SOLVER', 'evaluations': []}

    def margin(arm, target, rotation):
        """How close this arm gets to the pose, as a number; None when it cannot be measured."""
        try:
            solution, error, _ = solve(arm, target, rotation)
        except Exception:
            return None
        if solution is None or error is None:
            return None
        return -float(error)

    cube, cube_rotation = env.object_pose(obj)
    ee, rotation = env.ee_pose(giver)
    grasp_offset = rotation.T @ (np.asarray(cube, float) - ee)
    grasp_rotation = rotation.T @ cube_rotation
    opposite = np.diag([-1., -1., 1.])
    relative = opposite @ GRASP_ROTATION
    if obj.endswith('_cube'):
        # Same equivalent-face rule as the coupled solve: opposed approach directions, not the same side.
        relative = min((relative, opposite @ relative),
                       key=lambda candidate: float((grasp_rotation @ candidate)[2, 2]))
    (low_x, high_x), legacy_y, (low_z, high_z) = LEGACY_MEETING_BOX
    candidates = [np.array([float(np.clip(cube[0], low_x, high_x)), float(legacy_y),
                            float(np.clip(cube[2], low_z, high_z))])]
    for offset_x in (-.02, 0., .02):
        for offset_y in np.arange(-span, span + 1e-9, step):
            for offset_z in (-.04, 0., .04):
                candidates.append(np.array([float(cube[0]) + offset_x, float(cube[1]) + float(offset_y),
                                            float(cube[2]) + offset_z]))
    evaluations = []
    best = None
    for point in candidates:
        offset_y = float(point[1] - cube[1])
        donor_target = point - rotation @ grasp_offset
        receiving_rotation = rotation @ grasp_rotation @ relative
        receiver_target = (point + receiving_rotation @ np.array([0., 0., receiver_bias])
                           - receiving_rotation @ FINGERTIP_MIDPOINT[receiver])
        donor_margin = margin(giver, donor_target, rotation)
        receiver_margin = margin(receiver, receiver_target, receiving_rotation)
        score = None
        if donor_margin is not None and receiver_margin is not None:
            # The worse of the two: a meeting point only one arm can hold is not a meeting point.
            score = min(donor_margin, receiver_margin)
        evaluations.append({'point': point.tolist(), 'offset_y_m': offset_y, 'score': score,
                            'donor_margin_m': donor_margin, 'receiver_margin_m': receiver_margin})
        if score is not None and (best is None or score > best['score']):
            best = {'point': point.tolist(), 'score': score, 'offset_y_m': float(offset_y)}
    if best is None:
        # Nothing measurable: keep the point the grasp is already at, and say the choice was not measured.
        return {'point': [float(cube[0]), float(cube[1]), float(cube[2])], 'measured': False,
                'evaluations': evaluations, 'reason': 'UNMEASURED'}
    return {'point': best['point'], 'measured': True, 'score': best['score'],
            'offset_y_m': best['offset_y_m'],
            'legacy_point_chosen': best['point'] == [float(value) for value in candidates[0]],
            'evaluations': evaluations}


def plan_handover_geometry(env, giver: str, obj: str, receiver_bias: float = .015, *,
                           controller=None, meeting: dict | None = None) -> HandoverGeometry:
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
    opposite = np.diag([-1., -1., 1.])
    relative = opposite @ GRASP_ROTATION
    if obj.endswith('_cube'):
        # A previously placed cube may have turned 180 degrees. The same box
        # faces are then geometrically equivalent, but a fixed object-frame
        # receiver attitude would send both hands in from the SAME side.
        # Choose the equivalent face frame with opposed approach directions;
        # this only changes the candidate grasp, never the object's pose.
        relative = min((relative,opposite@relative),
                       key=lambda candidate:float((grasp_rotation@candidate)[2,2]))
    # Where the hands meet is decided by measuring both arms, not by a fixed point in the cell.
    if meeting is None:
        if controller is not None:
            meeting = measured_meeting_point(env, controller, giver, receiver, obj, receiver_bias)
        else:
            meeting = {'point': [float(cube[0]), float(cube[1]), float(cube[2])], 'measured': False,
                       'reason': 'NO_CONTROLLER'}
    desired_cube = np.array(meeting['point'], float)
    if meeting.get('measured') is not True:
        # Fall back to the previously validated working volume rather than an unmeasured point.
        desired_cube = np.array([float(np.clip(cube[0], 0.40, 0.52)), 0.0,
                                 float(np.clip(cube[2], 1.10, 1.18))])

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
                     .4 * (obj_p - desired_cube), .0005 * (q - home)]

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
        geometry = HandoverGeometry(obj_p, donor_r, solved.x[slices[giver]], solved.x[slices[receiver]],
                                    relative, receiver_bias, error)
        geometry.meeting = meeting
        return geometry
    raise ValueError(f"No collision-free coupled handover pose fits the measured grasp (kinematic residual {best_error:.4f}, collision rejections {collision_rejections})")
