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


def record(arm: str, position_error_m: float, *, source: str = 'move') -> None:
    """Note how far an arm landed from its commanded position."""
    if position_error_m is None:
        return
    _samples.setdefault(arm, []).append({'m': float(position_error_m), 'source': source}['m'])


def samples(arm: str | None = None) -> list[float]:
    if arm is None:
        return [value for values in _samples.values() for value in values]
    return list(_samples.get(arm, []))


def observed_error(arm: str) -> float:
    """The worst landing error seen for this arm, or the measured default when nothing is recorded."""
    recorded = samples(arm)
    return max(recorded) if recorded else DEFAULT_OBSERVED_ERROR_M


def required_clearance(arm: str, *, gain: float = GAIN, floor: float = FLOOR_M,
                       ceiling: float = 0.12) -> float:
    """How much room a target pose must have before it is worth commanding, for this body."""
    return float(min(ceiling, max(floor, observed_error(arm) * gain + floor)))


def reset() -> None:
    _samples.clear()
