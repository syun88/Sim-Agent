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
        # Twelve labelled outcomes with d1-3B. Bare coordinates: 7/12, every miss spatial. With the derived
        # relations supplied as sentences ("blue_cube is above red_cube"), 9/12 - still missing "stacked",
        # "right of", and one held-versus-placed case. Better, not enough: records only.
        "trusted": False, "measured": "d1-3B 9/12 given derived statements (7/12 without), omni-600M 3/6",
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
        # d1-3B answered all three with 0.96-0.98 confidence and all correct; the 600M got 2/3. Three cases is
        # a small sample, so this is trusted for *labelling a situation*, never as the only reason to act.
        "trusted": True, "measured": "d1-3B 3/3 at 0.96-0.98, omni-600M 2/3 (2026-10-27)",
    },
    "reference_ambiguous": {
        "type": "noul",
        "instructions": "Do the facts leave it genuinely unclear which object the instruction means?",
        "why": "Ambiguity is about what the words admit, not about where things are.",
        # 600M was wrong in both directions; d1-3B got both at 0.92.
        "trusted": True, "measured": "d1-3B 2/2 at 0.92, omni-600M 0/2 (2026-10-27)",
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


#: Words that turn a derived relation into a sentence. The decision model is asked to match language against
#: these, never to reason about coordinates - measured 2026-10-27, it answers spatial questions unreliably
#: (7/12, with every miss spatial) while answering situation questions 3/3 and ambiguity 2/2.
_RELATION_PHRASES = {
    "left_of": "to the left of", "right_of": "to the right of", "front_of": "in front of",
    "behind": "behind", "above": "above", "below": "below", "near": "next to",
}


def summarise_relations(derived: list[dict], limit: int = 12) -> list[str]:
    """Turn derived relations into short statements a language model can match against an instruction."""
    lines = []
    for row in derived[:limit]:
        phrases = [_RELATION_PHRASES.get(name, name) for name in row["relation"]]
        if not phrases:
            continue
        joined = phrases[0] if len(phrases) == 1 else ", ".join(phrases[:-1]) + " and " + phrases[-1]
        lines.append(f"{row['object']} is {joined} {row['reference']}")
    return lines


def fact_packet(state: dict, *, pair_distance: float = .12) -> dict:
    """The compact, pre-computed description a decision model is given instead of a world state."""
    objects = [{k: obj.get(k) for k in ("id", "type", "color", "graspable", "held_by")}
               for obj in (state.get("objects") or [])]
    for obj, source in zip(objects, state.get("objects") or []):
        position = source.get("position") or []
        obj["position"] = [round(float(v), 3) for v in position[:3]] if position else None
        obj["touching"] = sorted(source.get("touching") or [])
    robot = state.get("robot") or {}
    derived = relations(objects, pair_distance)
    packet = {
        "hands": {side: (robot.get(f"{side}_arm") or {}).get("held_object") for side in ("left", "right")},
        "objects": objects,
        "relations": derived,
        "statements": summarise_relations(derived),
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
