"""Name the mechanism behind a failed skill from the measurements it already recorded.

The same error code means different things. Measured 2026-10-30/11-07 in this cell:
`MOTION_BLOCKED` was the placing arm stopping 11.7 deg short of a commanded attitude because the other
gripper occupied its approach corridor, not an unreachable target; `APPROACH_COLLISION` was a 249 mm push
of a resting cube, not a weak pinch; `CARRY_COLLISION` was a 7.4 mm shove of a cube that the arm was
carrying past. Reading the code alone sends the next attempt to the wrong layer, so the mechanism is
named from `SkillError.evidence` and travels with the result.

Nothing here loosens a check. Tolerances are read only to compare, and an evidence field that is absent
stays absent: a mechanism is never guessed from the error code alone.
"""

from __future__ import annotations

#: Controller tolerances this module compares against when the record does not carry its own.
#: They mirror `control/motion.py`'s defaults; a record that reports the tolerances it was judged with
#: (``position_tolerance_m`` / ``rotation_tolerance_m``) always wins over these.
POSITION_TOLERANCE_M = 0.012
ROTATION_TOLERANCE_RAD = 0.15
#: A rotation residual this large is not servo lag alone: the commanded attitude may be unreachable.
LARGE_ROTATION_RAD = 0.1
#: A jaw frame is blamed only while the aperture stayed centred to this depth.
FRAME_CENTERED_M = 0.002

GRASP_ERRORS = frozenset({"GRASP_FAILED", "GRASP_LOST", "GRASP_SLIP", "HOLD_LOST", "HAND_CONTACT"})
PLANNING_ERRORS = frozenset({"IK_FAILURE", "COLLISION_PATH", "INVALID_PLAN", "GOAL_NOT_SATISFIED",
                             "GOAL_UNREACHABLE", "GOAL_SEARCH_BUDGET", "HANDOVER_UNREACHABLE",
                             "HANDOVER_APPROACH_COLLISION", "PARK_COLLISION", "PREPARATION_BLOCKED"})
LANGUAGE_MARKERS = ("UNSUPPORTED", "LANGUAGE", "MEANING", "AMBIGUOUS", "NOT_UNDERSTOOD")


def _number(value):
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _last(evidence, key):
    value = evidence.get(key)
    if isinstance(value, list):
        return value[-1] if value else None
    return value if isinstance(value, dict) else None


def _displacement(evidence):
    """The object's measured movement, under either key the skills use."""
    for key in ("displacement_m", "object_displacement_m"):
        value = _number(evidence.get(key))
        if value is not None:
            return value
    return None


def attribute(code, evidence=None):
    """Return ``(mechanism, reason)`` for a failed skill.

    ``mechanism`` is one of GEOMETRY, APPROACH_CONTACT, CARRY_CONTACT, GRASP_CONTACT, GRASP_FRAME,
    TRACKING, PLANNING, LANGUAGE, or UNKNOWN when the recorded evidence cannot decide it - an honest
    unknown, never a guess from the error name.
    """
    evidence = evidence if isinstance(evidence, dict) else {}
    code = str(code or "").upper()
    detail = str(evidence.get("detail") or evidence.get("error") or "")

    pairs = evidence.get("contact_pairs")
    displacement = _displacement(evidence)
    tolerance = _number(evidence.get("tolerance_m"))
    if pairs:
        # The engine reported the arm touching the object before the pose was reached: that is a
        # contact, whatever the displacement happens to be at the moment the motion was stopped.
        moved = "" if displacement is None else f" after {displacement * 1000:.1f} mm of object movement"
        return "APPROACH_CONTACT", f"the measured arm touched the object ({', '.join(pairs)}){moved}"
    if displacement is not None and displacement > 0. and (tolerance is None or displacement > tolerance):
        against = "" if tolerance is None else f" (tolerance {tolerance * 1000:.1f} mm)"
        carried = evidence.get("carrying") or evidence.get("carried_object")
        if carried or evidence.get("phase") == "carry":
            return "CARRY_CONTACT", (f"carrying {carried or 'the held object'} displaced another object "
                                     f"by {displacement * 1000:.1f} mm{against}")
        return "APPROACH_CONTACT", f"the approach displaced the object by {displacement * 1000:.1f} mm{against}"

    collisions = evidence.get("collisions")
    if collisions:
        return "GEOMETRY", f"{len(collisions)} colliding pair(s) recorded"
    peak = _number(evidence.get("maximum_penetration_m"))
    if peak is not None and peak > 0.:
        return "GEOMETRY", f"penetration {peak * 1000:.3f} mm recorded"

    closure = _last(evidence, "close_trace")
    pinch = evidence.get("pinch") if isinstance(evidence.get("pinch"), dict) else None
    if closure is not None or pinch is not None:
        contacts = list((closure or {}).get("object_contacts") or [])
        if not contacts and pinch:
            contacts = [c for group in pinch.get("groups") or [] for c in (group.get("contacts") or [])]
        centered = _number((closure or {}).get("aperture_center_error_m"))
        if contacts:
            return "GRASP_CONTACT", f"{len(contacts)} finger contact(s) but no loaded opposed pinch"
        if code in GRASP_ERRORS or closure is not None:
            if centered is not None and centered > FRAME_CENTERED_M:
                return "GRASP_FRAME", (f"jaws closed on nothing and the aperture drifted "
                                       f"{centered * 1000:.3f} mm off centre")
            return "GRASP_FRAME", "jaws closed on nothing: no finger touched the object"

    position = _number(evidence.get("position_error"))
    rotation = _number(evidence.get("rotation_error"))
    if position is not None or rotation is not None:
        position_tolerance = _number(evidence.get("position_tolerance_m")) or POSITION_TOLERANCE_M
        rotation_tolerance = _number(evidence.get("rotation_tolerance_m")) or ROTATION_TOLERANCE_RAD
        short = ((rotation is not None and rotation > rotation_tolerance)
                 or (position is not None and position > position_tolerance))
        if not short:
            # motion.py raises a blocked motion only when its own tolerances were exceeded. Residuals
            # inside the tolerances a record reports back therefore do not explain the block, and calling
            # it servo lag would be a guess.
            return "UNKNOWN", (f"the motion was blocked but its recorded residuals (position {position}, "
                               f"rotation {rotation}) are inside the tolerances it was judged with "
                               f"({position_tolerance} m / {rotation_tolerance} rad); another measurement decides it")
        if short:
            reason = (f"arm stopped short (position {position}, rotation {rotation}) against tolerances "
                      f"{position_tolerance} m / {rotation_tolerance} rad")
            if rotation is not None and rotation > LARGE_ROTATION_RAD:
                # A residual this large is not servo lag alone: the commanded attitude may be
                # unreachable, or its approach corridor occupied. One sample cannot decide which.
                reason += ("; a residual this large may mean an unreachable or occupied attitude - "
                           "verify reachability and the opposite arm's pose before blaming the servo")
            return "TRACKING", reason

    reached = _number(evidence.get("peak_object_speed_m_s"))
    needed = _number(evidence.get("needed_release_speed_m_s"))
    if reached is not None and needed is not None and reached < needed:
        # The same family as a short move: the arm did not do what it was commanded to do. Named from the
        # two numbers rather than from the error name, so a throw that reports them is attributed even if
        # the category is read out of context (measured 2026-10-28: 0.459 m/s against the 0.767 required).
        return "TRACKING", (f"the arm reached {reached} m/s against the {needed} m/s the launch needed; "
                            f"no release was commanded")

    if code in PLANNING_ERRORS:
        return "PLANNING", f"{code} with no contact and no measured motion"
    if any(marker in code or marker in detail.upper() for marker in LANGUAGE_MARKERS):
        return "LANGUAGE", "instruction could not be turned into a supported task"
    return "UNKNOWN", "the recorded evidence does not decide the mechanism"
