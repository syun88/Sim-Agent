"""How far the jaws must open to let go, and how much room that needs.

A gripper that opens to its full aperture at release sweeps its moving jaw the whole way out, so a
placement beside another object needs that much clearance whether or not the object needs it. Measured
2026-10-25 on OpenManipulator-X (82 mm aperture, 25.5 mm cube): the full-open release put the jaw into
the neighbouring cube, and the clearance that made it work was 71.5 mm centre-to-centre - more than the
task needed. Opening only as far as the object plus a margin shrinks the sweep to the release aperture,
which is the number both the release and the placement clearance should use.
"""
from __future__ import annotations

import os

import numpy as np


def maximum_aperture(env, arm: str) -> float:
    """The gripper's full opening, as measured when the robot was prepared (metres)."""
    gripper = (getattr(env, "grippers", {}) or {}).get(arm, {}) or {}
    return float(gripper.get("maximum_aperture") or 0.)


def object_width(env, obj: str) -> float:
    """The object's wider horizontal dimension, which is what has to pass between the jaws."""
    bounds = env.object_bounds(obj)
    return float(max(bounds[1, :2] - bounds[0, :2]))


def release_aperture(env, arm: str, obj: str, margin: float = .010) -> float:
    """Opening that frees the object with ``margin`` on each side, never more than the gripper has."""
    need = object_width(env, obj) + 2. * margin
    aperture = maximum_aperture(env, arm)
    return min(aperture, need) if aperture else need


def release_command(env, arm: str, obj: str, margin: float = .010) -> float:
    """Gripper command fraction that opens just enough to free the object (1 closed, 0 fully open)."""
    aperture = maximum_aperture(env, arm)
    if aperture <= 0.:
        return 0.
    return max(0., min(1., 1. - release_aperture(env, arm, obj, margin) / aperture))


def jaw_reach(env, arm: str, obj: str, direction=None) -> float:
    """How far the jaws reach from the object's centre, measured along ``direction`` (world frame).

    A world-axis AABB of a jaw includes its length in every direction, so taking the largest corner
    distance reserved 115 mm for a 25.5 mm cube. What a placement actually needs is the extent towards
    the *reference*: the jaw sweeps into whatever sits on that side, and nothing else. ``direction`` is
    the unit vector from the object towards the reference; without it the largest lateral extent is
    returned, which is only a diagnostic.
    """
    gripper = (getattr(env, "grippers", {}) or {}).get(arm, {}) or {}
    links = {name for group in (gripper.get("contact_groups") or []) for name in group}
    if not links or not hasattr(env, "link_bounds"):
        return 0.
    centre = np.asarray(env.object_pose(obj)[0], float)
    aim = None if direction is None else np.asarray(direction, float)
    if aim is not None:
        aim = aim / max(float(np.linalg.norm(aim)), 1e-9)
    reach = 0.
    for name in links:
        bounds = env.link_bounds(name)
        if bounds is None:
            continue
        corners = np.array([[x, y, z] for x in bounds[:, 0] for y in bounds[:, 1] for z in bounds[:, 2]])
        offsets = corners - centre
        if aim is None:
            # Diagnostic only: the extent in the horizontal plane.
            extent = np.linalg.norm(offsets[:, :2], axis=1)
        else:
            extent = offsets @ aim
        reach = max(reach, float(np.max(extent)))
    return reach


def reference_gap(env, arm: str, obj: str, margin: float = .020, direction=None) -> float:
    """Gap between the placed object and a reference so the jaws can still let go of it.

    Bounded by the gripper's FULL aperture, not the release aperture. Measured 2026-10-25: with a partial
    release the moving jaw stops early, but the FIXED jaw does not move at all, and at 54 mm
    centre-to-centre its body was still resting on the reference, so the retreat could not separate
    (RETREAT_CONTACT). Only the full-aperture bound (71.5 mm) separated cleanly, so that is what the
    placement reserves; the partial release is kept because it does help elsewhere (a stack that used to
    jam now places).
    """
    half = object_width(env, obj) / 2.
    aperture = maximum_aperture(env, arm) or release_aperture(env, arm, obj)
    measured = jaw_reach(env, arm, obj, direction)
    # Two bounds, and the larger wins. The aperture bound is what separates cleanly on OpenManipulator-X
    # (71.5 mm measured); the direction-aware jaw reach is the same quantity measured from this robot's
    # own geometry, so where a jaw reaches further than half the aperture it is the binding one. Both are
    # printed when OPENARM_GRIPPER_DEBUG is set: a clearance nobody can read is a clearance nobody can
    # check against the next failure.
    if os.getenv('OPENARM_GRIPPER_DEBUG'):
        print(f'[gripper] {arm}/{obj} aperture_half={aperture / 2.:.4f} jaw_reach={measured:.4f} '
              f'object_half={half:.4f} measured_gap={max(aperture / 2., measured) + margin - half:.4f} '
              f'-> using_aperture={aperture / 2. + margin - half:.4f}', flush=True)
    # The aperture bound is still what runs, because it is the value measured to separate cleanly
    # (71.5 mm centre-to-centre on OpenManipulator-X). The measured reach is a diagnostic until a live
    # run shows it separates too: switching a clearance on a number nobody has tested is how a retreat
    # starts moving a placed object. OPENARM_GRIPPER_DEBUG prints both so the next run can decide.
    return max(0., aperture / 2. + margin - half)
