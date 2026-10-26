"""What to compute, what to ask, and in which form.

Measured 2026-10-27 on the local decision model (d1-omni-600M): it answered "how many cubes are visible"
correctly at 0.97 confidence, and did not respond at all to swapping the cubes' sides. That is not a defect to
work around - left/right, colour, contact and ownership are *derivable* from the engine's own state, and a
forward pass spent deriving them is a forward pass that cannot be spent on a judgement.

So this module keeps the split explicit:

* ``fact_packet`` derives everything the scene already answers - who holds what, each object's position, its
  relations to the others, its contacts - and hands the planner a compact, pre-computed description.
* ``QUESTIONS`` lists only the judgements that the facts cannot settle: whether an instruction's outcome has
  been achieved, which admitted recovery fits a failure, whether a reference is ambiguous. Geometry is
  deliberately absent; a guard test fails if a spatial question is ever added back.

The packet is also the right input *shape* for a decision model: short, named, already resolved, with no
coordinates to reason about.
"""
from __future__ import annotations

#: Questions the facts cannot settle, with the answer type that suits a decision model. Each entry says why a
#: forward pass is justified rather than a computation.
QUESTIONS = {
    "outcome_achieved": {
        "type": "noul",
        "instructions": "Has the instruction's requested outcome been achieved, according to these facts?",
        "why": "Interpreting the instruction against observed end state is a language judgement, not geometry.",
    },
    "next_step_blocked": {
        "type": "choice",
        "instructions": "Which single reason best explains why the next step cannot proceed?",
        "criteria": {
            "hand_occupied": "the hand that must act is holding something else",
            "target_occupied": "the destination already holds an object",
            "out_of_reach": "the target is outside the robot's reach",
            "not_observed": "the referenced object is not in the scene",
        },
        "why": "Choosing among named causes from a described situation is a typed decision.",
    },
    "reference_ambiguous": {
        "type": "noul",
        "instructions": "Do the facts leave it genuinely unclear which object the instruction means?",
        "why": "Ambiguity is about what the words admit, not about where things are.",
    },
}

#: Substrings that mean a question is really geometry in disguise. Adding one of these fails the guard.
_GEOMETRY_MARKERS = ("left of", "right of", "left of the blue", "above the", "below the", "distance",
                     "centimet", "closer", "further", "how many cubes are visible")


def relations(objects: list[dict], pair_distance: float = .12) -> list[dict]:
    """Derive the relations the scene already answers, in the robot's own frame (world +Y is left)."""
    derived = []
    for first in objects:
        a = first.get("position") or []
        if len(a) < 3:
            continue
        for second in objects:
            if second is first:
                continue
            b = second.get("position") or []
            if len(b) < 3:
                continue
            horizontal = float(b[0] - a[0])
            lateral = float(b[1] - a[1])
            vertical = float(b[2] - a[2])
            relation = []
            if abs(lateral) >= .02:
                relation.append("right_of" if lateral < 0 else "left_of")
            if abs(horizontal) >= .02:
                relation.append("front_of" if horizontal > 0 else "behind")
            if vertical >= .02:
                relation.append("above")
            elif vertical <= -.02:
                relation.append("below")
            if (horizontal ** 2 + lateral ** 2) ** .5 <= pair_distance:
                relation.append("near")
            if relation:
                derived.append({"object": first.get("id"), "relation": sorted(relation),
                                "reference": second.get("id"),
                                "delta_m": [round(horizontal, 4), round(lateral, 4), round(vertical, 4)]})
    return derived


def fact_packet(state: dict, *, pair_distance: float = .12) -> dict:
    """The compact, pre-computed description a decision model is given instead of a world state."""
    objects = [{k: obj.get(k) for k in ("id", "type", "color", "graspable", "held_by")}
               for obj in (state.get("objects") or [])]
    for obj, source in zip(objects, state.get("objects") or []):
        position = source.get("position") or []
        obj["position"] = [round(float(v), 3) for v in position[:3]] if position else None
        obj["touching"] = sorted(source.get("touching") or [])
    robot = state.get("robot") or {}
    packet = {
        "hands": {side: (robot.get(f"{side}_arm") or {}).get("held_object") for side in ("left", "right")},
        "objects": objects,
        "relations": relations(objects, pair_distance),
        "capabilities": sorted((state.get("capabilities") or {}).get("skills") or []),
        "navigation": bool((state.get("capabilities") or {}).get("navigation")),
        "derived_by": "engine state; no model call was spent on these",
    }
    return {key: value for key, value in packet.items() if value not in (None, [], {})}


def geometry_in_questions(questions: dict | None = None) -> list[str]:
    """Names of questions that are really geometry, which the facts already answer."""
    offenders = []
    for name, spec in (questions or QUESTIONS).items():
        text = " ".join(str(spec.get(key, "")) for key in ("instructions", "why"))
        text += " " + " ".join(str(value) for value in (spec.get("criteria") or {}).values())
        lowered = text.lower()
        if any(marker in lowered for marker in _GEOMETRY_MARKERS):
            offenders.append(name)
    return offenders
