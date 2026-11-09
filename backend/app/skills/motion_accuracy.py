"""How far this body lands from where it was told to go, measured rather than assumed.

Parking the arm at the model's home posture is only safe if the arm actually arrives there. Measured
2026-11-09: a placed cube sat 30.8 mm from the home link, which the old fixed 30 mm clearance check called
"clear", and the arm still brushed it - the executed posture was about 35 mm from the planned one, so the
margin was smaller than the positioning error it had to absorb.

So the clearance a home target must have is derived from the error this body has actually shown, plus a
margin, instead of from a constant that happened to be true once. Recordings come from arrival checks; until
one exists the value falls back to the measured 2026-11-09 figure for this cell, which is deliberately
conservative rather than optimistic.
"""

from __future__ import annotations

#: The positioning discrepancy measured on this cell on 2026-11-09 (executed posture vs planned home).
DEFAULT_OBSERVED_ERROR_M = 0.035
#: A home target is only used when it is clear by this multiple of the observed error, plus a floor.
GAIN = 1.5
FLOOR_M = 0.01

_samples: dict[str, list[float]] = {}


#: Moves made while touching something (a grasp, a placed object) legitimately stop short; they describe
#: contact, not the free-space landing error a home target has to absorb.
FREE_SPACE_SOURCES = ('park_arrival', 'move_to_site')


def record(arm: str, position_error_m: float, *, joint_error_rad: float | None = None,
           source: str = 'move') -> None:
    """Note how far an arm landed from its commanded position."""
    if position_error_m is None:
        return
    _samples.setdefault(arm, []).append({'position_error_m': float(position_error_m),
                                        'joint_error_rad': None if joint_error_rad is None else float(joint_error_rad),
                                        'source': source})


def samples(arm: str | None = None, *, sources: tuple | None = None) -> list[dict]:
    entries = [{'arm': name, **entry} for name, values in _samples.items() for entry in values
               if not sources or entry['source'] in sources]
    return [entry for entry in entries if arm is None or entry['arm'] == arm]


def observed_error(arm: str, *, sources: tuple = FREE_SPACE_SOURCES) -> float:
    """The error a home target has to absorb: the worst recorded free-space landing, floored by the default.

    Records may only raise the bar, never lower it. Measured 2026-11-09: ordinary free-space moves land
    within ~12 mm, but the *near-object* posture that caused the collision was ~35 mm off - so a per-move
    maximum that could fall below the measured baseline would call the 30.8 mm margin clear again.
    """
    recorded = [entry['position_error_m'] for entry in samples(arm, sources=sources)]
    return max([DEFAULT_OBSERVED_ERROR_M] + recorded)


#: What a phase is expected to land within when nothing has been recorded for it yet. A carry lands worse
#: than a free-space move: it ends against a support, at a pose the arm reached while holding something.
PHASE_DEFAULTS = {'carry': 0.012, 'move_to_site': 0.010, 'park_arrival': DEFAULT_OBSERVED_ERROR_M}


def expected_error(arm: str, phase: str, *, default: float | None = None) -> float:
    """How far this arm lands from its command in this phase - recorded if possible, else that phase's floor."""
    recorded = [entry['position_error_m'] for entry in samples(arm, sources=(phase,))]
    floor = PHASE_DEFAULTS.get(phase, default if default is not None else 0.012)
    return max([floor] + recorded)


def required_clearance(arm: str, *, gain: float = GAIN, floor: float = FLOOR_M,
                       ceiling: float = 0.12, sources: tuple = FREE_SPACE_SOURCES) -> float:
    """How much room a target pose must have before it is worth commanding, for this body."""
    return float(min(ceiling, max(floor, observed_error(arm, sources=sources) * gain + floor)))


def reset() -> None:
    _samples.clear()
