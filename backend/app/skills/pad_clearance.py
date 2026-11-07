"""Measure whether an in-hand support's own grip blocks the face another cube must land on.

Measured 2026-11-07: a cube pinched at its centre had a stacked cube stop 7.15 mm above it and rest on
the holder's fingers (`support_contact=false`). The project then hard-coded a 15 mm regrasp depth, which
is a number, not a reason. This module answers the question from the official model's own geometry, in a
private MuJoCo scene, so the caller can *decide* how far to lower a grip (or refuse) instead of guessing.

The query is conservative on purpose: a geom's bounding sphere is used for the vertical extent, so it
over-reports blocking rather than under-reporting it. Nothing here moves the robot.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: The bodies that actually pinch: they are also the bodies named in the contact pairs, so a clearance
#: measured here is about the same geometry the grasp is verified against.
def pad_bodies(arm: str) -> tuple[str, ...]:
    return (f'openarm_{arm}_ee_inner_finger', f'openarm_{arm}_ee_outer_finger')


@dataclass(frozen=True)
class PadExtent:
    lowest_m: float
    highest_m: float
    geoms: int

    def blocks(self, top_z: float) -> bool:
        """Whether any pad material reaches above `top_z` (a face that must stay clear)."""
        return self.highest_m > top_z


def pad_extent(kin, joints, arm: str) -> PadExtent | None:
    """Vertical extent of this arm's finger geometry at `joints`, in world coordinates.

    `kin` is the read-only official kinematics object (`OfficialArmKinematics`); `joints` are that arm's
    joint positions. Returns None when the model exposes no finger geometry, so a caller cannot mistake
    "not measured" for "clear".
    """
    import mujoco

    kin.evaluate(np.asarray(joints, float))
    model, data = kin._model, kin._data
    wanted = set(pad_bodies(arm))
    lows, highs, count = [], [], 0
    for geom in range(model.ngeom):
        body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom])) or ''
        if body not in wanted:
            continue
        # The geom's own local AABB, placed by its current world pose: a bounding sphere would count the
        # whole finger link and report metres of interference (measured 2026-11-07: 1.04-1.29 m).
        axes = np.asarray(model.geom_aabb[geom], float).reshape(-1)
        if axes.size != 6:
            axes = None
        if axes is None or not np.any(axes[3:] > axes[:3]):
            centre = float(data.geom_xpos[geom][2])
            radius = float(model.geom_rbound[geom])
            lows.append(centre - radius)
            highs.append(centre + radius)
            count += 1
            continue
        low, high = axes[:3], axes[3:]
        corners = np.array([[x, y, z] for x in (low[0], high[0]) for y in (low[1], high[1]) for z in (low[2], high[2])])
        world = corners @ np.asarray(data.geom_xmat[geom]).reshape(3, 3).T + np.asarray(data.geom_xpos[geom])
        lows.append(float(world[:, 2].min()))
        highs.append(float(world[:, 2].max()))
        count += 1
    if not count:
        return None
    return PadExtent(lowest_m=min(lows), highest_m=max(highs), geoms=count)


def _geom_world_aabb(model, data, geom):
    axes = np.asarray(model.geom_aabb[geom], float).reshape(-1)
    if axes.size != 6 or not np.any(axes[3:] > axes[:3]):
        centre = np.asarray(data.geom_xpos[geom], float)
        radius = float(model.geom_rbound[geom])
        return centre - radius, centre + radius
    low, high = axes[:3], axes[3:]
    corners = np.array([[x, y, z] for x in (low[0], high[0])
                        for y in (low[1], high[1]) for z in (low[2], high[2])])
    world = corners @ np.asarray(data.geom_xmat[geom]).reshape(3, 3).T + np.asarray(data.geom_xpos[geom])
    return world.min(axis=0), world.max(axis=0)


def support_face_clearance(env, arm: str, support_center, support_half_xy: float,
                           support_top_z: float, reach_margin: float = 0.06) -> dict:
    """Clear space above the support's top face *within the support's own footprint*.

    Only pad material that stands over the face a cube must land on matters; a finger link that reaches up
    elsewhere does not block it. Positive clearance means the face is clear; negative means pad material
    rises that far above it, so a stacked cube would rest on the fingers. `unknown` carries the reason when
    the model or the arm state is unavailable - never a silent zero.
    """
    try:
        kin = env.arm_kinematics(arm)
        joints = env.arm_qpos(arm)
    except Exception as error:  # an imported robot may not expose this query at all
        return {'clearance_m': None, 'unknown': f'{type(error).__name__}: {error}'}
    import mujoco

    kin.evaluate(np.asarray(joints, float))
    model, data = kin._model, kin._data
    wanted = set(pad_bodies(arm))
    centre = np.asarray(support_center, float)
    over, count = [], 0
    for geom in range(model.ngeom):
        body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom])) or ''
        if body not in wanted:
            continue
        low, high = _geom_world_aabb(model, data, geom)
        # Pad material that a descending cube would meet: its horizontal distance from the support's
        # centre is within the support's own half-width plus a margin. Material far from the cube (a
        # finger link reaching up towards the wrist) is not in the way, and counting it reported 1.24 m
        # of interference (measured 2026-11-07).
        gap_x = max(0., float(low[0]) - (centre[0] + support_half_xy),
                    (centre[0] - support_half_xy) - float(high[0]))
        gap_y = max(0., float(low[1]) - (centre[1] + support_half_xy),
                    (centre[1] - support_half_xy) - float(high[1]))
        if (gap_x ** 2 + gap_y ** 2) ** .5 > reach_margin:
            continue
        over.append(float(high[2]))
        count += 1
    if not count:
        # Nothing stands over the face. That is a measurement ("no pad material there"), not an infinite
        # clearance: returning `inf` aborted 11 of 24 live tasks when the evidence was serialised as JSON
        # (measured 2026-11-08, "Out of range float values are not JSON compliant").
        return {'clearance_m': None, 'pad_over_face_m': None, 'support_top_m': float(support_top_z),
                'pad_geoms_over_face': 0, 'note': 'no pad geometry over the support face'}
    highest = max(over)
    return {'clearance_m': float(support_top_z - highest), 'pad_over_face_m': highest,
            'support_top_m': float(support_top_z), 'pad_geoms_over_face': count}
