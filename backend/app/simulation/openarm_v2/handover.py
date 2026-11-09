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


#: Where the hands may meet, as offsets from the held object. The first entry is the point this cell used to
#: hard-code (x clipped into .40-.52, y = 0, z clipped into 1.10-1.18); it stays as a candidate so that if it
#: wins, it wins because it was measured against the alternatives.
MEETING_CANDIDATES = ((0., 0., 0.), ('legacy',), (0., .03, 0.), (0., -.03, 0.), (0., 0., .02))
#: Seeds spent per candidate. Four keeps the total near what one point used to cost (four candidates x four).
SEEDS_PER_CANDIDATE = 4


def _meeting_points(cube) -> list[tuple[str, np.ndarray]]:
    points = []
    for entry in MEETING_CANDIDATES:
        if entry == ('legacy',):
            point = np.array([float(np.clip(cube[0], 0.40, 0.52)), 0.0, float(np.clip(cube[2], 1.10, 1.18))])
            points.append(('legacy_clip', point))
        else:
            points.append((f'offset_{entry[0]:+.2f}_{entry[1]:+.2f}_{entry[2]:+.2f}',
                           np.asarray(cube, float) + np.asarray(entry, float)))
    return points


def select_winner(evaluations):
    """The candidate with the smallest coupled residual. A candidate that never solved cannot win."""
    solvable = [item for item in evaluations if item.get('solved') is not None]
    if not solvable:
        return None
    return min(solvable, key=lambda item: item['error'])


def plan_handover_geometry(env, giver: str, obj: str, receiver_bias: float = .015, *,
                           meeting: dict | None = None) -> HandoverGeometry:
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
    def attempt(name, desired_cube, seeds):
        """Try one meeting point: solve the coupled pose, screen it, keep the best residual seen."""
        def poses(q):
            data.qpos[indices] = q
            mujoco.mj_kinematics(model, data)
            donor_r = data.site_xmat[sites[giver]].reshape(3, 3)
            obj_r = donor_r @ grasp_rotation
            obj_p = data.site_xpos[sites[giver]] + donor_r @ grasp_offset
            receiving_r = obj_r @ relative
            receiving_p = (obj_p + receiving_r @ np.array([0., 0., receiver_bias])
                           - receiving_r @ FINGERTIP_MIDPOINT[receiver])
            return donor_r, obj_r, obj_p, receiving_r, receiving_p

        def residual(q):
            _, _, obj_p, receiving_r, receiving_p = poses(q)
            return np.r_[data.site_xpos[sites[receiver]] - receiving_p,
                         .2 * Rotation.from_matrix(
                             receiving_r @ data.site_xmat[sites[receiver]].reshape(3, 3).T).as_rotvec(),
                         .4 * (obj_p - desired_cube), .0005 * (q - home)]

        best_error = float("inf")
        collisions = 0
        for seed in seeds:
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
                collisions += 1
                continue
            return {'name': name, 'point': desired_cube.tolist(), 'error': error,
                    'collision_rejections': collisions, 'solved': solved, 'obj_p': obj_p,
                    'donor_r': donor_r}
        return {'name': name, 'point': desired_cube.tolist(),
                'error': None if best_error == float("inf") else best_error,
                'collision_rejections': collisions, 'solved': None}

    # Which meeting point to use is decided by trying them and keeping the one with the best coupled
    # residual: the same measurement the geometry itself is built from. The point this cell used to hard-code
    # is one of the candidates, so it can still win - but only against the alternatives, not by default.
    rng = np.random.default_rng(17)
    evaluations = []
    for name, point in (_meeting_points(cube) if meeting is None
                        else [('given', np.array(meeting['point'], float))]):
        if meeting is not None:
            seeds = [home if sample == 0 else np.clip(home + rng.normal(0, 1, 14),
                                                      limits[:, 0] + 1e-6, limits[:, 1] - 1e-6)
                     for sample in range(16)]
        else:
            seeds = [home] + [np.clip(home + rng.normal(0, 1, 14), limits[:, 0] + 1e-6, limits[:, 1] - 1e-6)
                              for _ in range(SEEDS_PER_CANDIDATE - 1)]
        evaluations.append(attempt(name, point, seeds))
    winner = select_winner(evaluations)
    if winner is None:
        worst = min((item['error'] for item in evaluations if item['error'] is not None), default=None)
        raise ValueError('No collision-free coupled handover pose fits the measured grasp '
                         f'(best kinematic residual {worst if worst is not None else float("nan"):.4f}, '
                         f'candidates {[item["name"] for item in evaluations]})')
    slices = {"left": slice(0, 7), "right": slice(7, 14)}
    geometry = HandoverGeometry(winner['obj_p'], winner['donor_r'],
                                winner['solved'].x[slices[giver]], winner['solved'].x[slices[receiver]],
                                relative, receiver_bias, winner['error'])
    geometry.meeting = {'point': winner['point'], 'measured': True, 'chosen': winner['name'],
                        'residual': winner['error'],
                        'evaluations': [{key: value for key, value in item.items() if key != 'solved'}
                                        for item in evaluations]}
    return geometry
