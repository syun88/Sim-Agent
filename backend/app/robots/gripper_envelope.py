"""How far the jaws must open to let go, and how much room that needs.

A gripper that opens to its full aperture at release sweeps its moving jaw the whole way out, so a
placement beside another object needs that much clearance whether or not the object needs it. Measured
2026-10-25 on OpenManipulator-X (82 mm aperture, 25.5 mm cube): the full-open release put the jaw into
the neighbouring cube, and the clearance that made it work was 71.5 mm centre-to-centre - more than the
task needed. Opening only as far as the object plus a margin shrinks the sweep to the release aperture,
which is the number both the release and the placement clearance should use.
"""
from __future__ import annotations


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
    return max(0., aperture / 2. + margin - half)
