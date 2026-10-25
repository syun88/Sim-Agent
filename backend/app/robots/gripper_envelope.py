"""How far the jaws must open to let go, and how much room that needs.

A gripper that opens to its full aperture at release sweeps its moving jaw the whole way out, so a
placement beside another object needs that much clearance whether or not the object needs it. Measured
2026-10-25 on OpenManipulator-X (82 mm aperture, 25.5 mm cube): the full-open release put the jaw into
the neighbouring cube, and the clearance that made it work was 71.5 mm centre-to-centre - more than the
task needed. Opening only as far as the object plus a margin shrinks the sweep to the release aperture,
which is the number both the release and the placement clearance should use.
"""
from __future__ import annotations

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


def jaw_reach(env, arm: str, obj: str) -> float:
    """How far any jaw reaches laterally from the object's centre, in the object's own frame - measured.

    "Laterally" matters: a world-axis AABB of a jaw includes its length along the approach direction, and
    taking that as the sideways reach reserved 115 mm for a 25.5 mm cube. The reach is therefore measured
    in the object's frame, perpendicular to the gripper's approach axis, from the links' own AABBs - the
    same geometry the collision model uses.
    """
    gripper = (getattr(env, "grippers", {}) or {}).get(arm, {}) or {}
    links = {name for group in (gripper.get("contact_groups") or []) for name in group}
    if not links or not hasattr(env, "link_bounds"):
        return 0.
    centre, rotation = env.object_pose(obj)
    centre = np.asarray(centre, float)
    rotation = np.asarray(rotation, float)
    approach = rotation.T @ np.asarray(gripper.get("approach_axis") or [0., 0., -1.], float)
    approach = approach / max(float(np.linalg.norm(approach)), 1e-9)
    reach = 0.
    for name in links:
        bounds = env.link_bounds(name)
        if bounds is None:
            continue
        corners = np.array([[x, y, z] for x in bounds[:, 0] for y in bounds[:, 1] for z in bounds[:, 2]])
        local = (corners - centre) @ rotation
        perpendicular = local - np.outer(local @ approach, approach)
        reach = max(reach, float(np.max(np.linalg.norm(perpendicular, axis=1))))
    return reach


def reference_gap(env, arm: str, obj: str, margin: float = .020) -> float:
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
    # The aperture bound, which is the one measured to separate cleanly (71.5 mm on OpenManipulator-X).
    # jaw_reach() is available and measured, but its axis convention is not settled yet: taking the max
    # with it inflated the reserved distance to 116 mm, and the verification of that placement failed, so
    # it is a diagnostic until the frame question is answered rather than a bound in this formula.
    return max(0., aperture / 2. + margin - half)
