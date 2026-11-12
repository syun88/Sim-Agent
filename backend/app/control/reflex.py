"""The layer that watches a motion while it happens.

The planner decides where to go and the verifier checks where the arm ended up, but nothing watched the arm
*during* the leg. Two measured consequences of that gap (2026-11-10):

* a home route planned with 30.8 mm of clearance was executed ~35 mm off plan and ended 4.7 mm from the cube,
  which surfaced as `PARK_COLLISION` - a post-hoc name for drift nothing had observed;
* `TRACKING` failures could not be predicted from any pre-execution feature (0/34), because the quantity that
  matters only exists while the arm is moving.

A dragonfly does not re-plan its posture against a gust; a local loop holds it. This is that loop, kept
generic so every motion inherits it rather than each skill inventing its own tolerance:

    threshold = expected_error(arm, phase) * 1.5 + 0.01     (the rule validated for the home route)

`expected_error` is the measured execution error for that phase (median over recordings, floored at a measured
default), so a phase that has been seen to drift 35 mm is not policed as if it drifted 10 mm. A phase with no
data keeps its floor and is never treated as zero.

The reflex only *stops* and reports; it never re-plans, never retries and never declares success. Stopping is
the one action that is always safe, and the caller already knows how to turn a failed leg into a decision.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

from backend.app.skills import motion_accuracy

#: Fraction of the leg ignored before watching: every leg starts with an acceleration transient while the
#: servos take up the new reference, and that transient is not drift.
START_FRACTION = 0.15
#: How many consecutive observations must agree before the reflex acts. One sample over threshold is a
#: transient; the point is to catch drift that persists.
CONSECUTIVE_SAMPLES = 3
#: Ceiling: past this the arm is not tracking the plan at all, and a larger threshold would only delay the stop.
MAX_THRESHOLD_M = 0.12
#: How finely the planned path is sampled for the distance question. The samples are forward kinematics, so the
#: cost is small; the spacing is what bounds the measurement, and a fixed count made that bound path-dependent.
PATH_SPACING_M = 0.005
MIN_PATH_SAMPLES = 24
MAX_PATH_SAMPLES = 256
#: How often the comparison is made. A forward-kinematics call is cheap, and 20 Hz is well above the rate at
#: which a 7-joint arm can drift into a neighbouring object.
CHECK_PERIOD_S = 0.05


def enabled() -> bool:
    """Whether the reflex runs. On by default: it can only stop a leg that had already left the plan."""
    return os.getenv('OPENARM_DEVIATION_REFLEX', '1').strip().lower() not in {'0', 'off', 'false', ''}


def threshold_m(arm: str, phase: str) -> float:
    """The deviation that counts as having left the plan for this arm and phase.

    `OPENARM_DEVIATION_REFLEX_MM` forces a value. It exists so the guard can be observed failing: a reflex that
    has never been seen to fire is indistinguishable from one that is not wired in at all.
    """
    forced = os.getenv('OPENARM_DEVIATION_REFLEX_MM', '').strip()
    if forced:
        try:
            return float(forced) / 1000.
        except ValueError:
            pass
    observed = motion_accuracy.expected_error(arm, phase)
    return min(MAX_THRESHOLD_M, max(0.01, observed * 1.5 + 0.01))


@dataclass
class DeviationReflex:
    """Watches one leg of one arm and reports the first persistent departure from its planned path."""

    arm: str
    phase: str
    threshold_m: float = field(default=0.0)
    #: How many consecutive samples must agree. Defaults to CONSECUTIVE_SAMPLES, but a leg that only *produces*
    #: two samples can never satisfy three - measured 2026-11-10: park legs yielded 1-2 samples, so a 110 mm
    #: departure against a 62 mm threshold could not act, and the reflex was blind exactly where drift was worst.
    required: int = 0
    expected_samples: int | None = None
    #: Forward-kinematics positions along the planned path, filled by the motion that owns the leg.
    path_points: list = field(default_factory=list)
    #: The spacing of those samples. The nearest-sample distance cannot resolve finer than half of this, so it
    #: is added to the threshold instead of being mistaken for drift.
    granularity_m: float = 0.0
    samples: int = 0
    worst_m: float = 0.0
    #: Where on the leg the worst sample was, and the longest run above threshold. A leg that drifts far but
    #: never holds it for `CONSECUTIVE_SAMPLES` is either spiky or too short to judge, and those two are
    #: different defects - measured 2026-11-10, a park leg reached 110 mm against a 62 mm threshold and did not
    #: act, which is indistinguishable from "the reflex is not wired in" without these two numbers.
    worst_fraction: float = 0.0
    max_run: int = 0
    over_threshold: int = 0

    def __post_init__(self) -> None:
        if not self.threshold_m:
            self.threshold_m = threshold_m(self.arm, self.phase)
        if not self.required:
            budget = CONSECUTIVE_SAMPLES
            if self.expected_samples is not None:
                budget = min(budget, max(1, int(self.expected_samples)))
            self.required = budget

    def observe(self, path_points, actual_joints, fraction: float,
                kinematics) -> dict | None:
        """Distance from the actual pose to the planned *path*; return evidence when it persists.

        Measured against the path, not against the time-indexed setpoint. The first version compared the arm with
        where the plan said it should be at that instant, so a fast large move that was merely behind schedule
        read as 387 mm of drift and stopped a legitimate handover (measured 2026-11-10). What matters for
        collision is being *on* the path; being late on it is the servo's business.
        """
        if fraction < START_FRACTION:
            return None
        try:
            actual_point = np.asarray(kinematics.evaluate(np.asarray(actual_joints, float))[0], float)
            deviation = min(float(np.linalg.norm(point - actual_point)) for point in path_points)
        except Exception:  # a measurement that cannot be made must not stop the arm
            return None
        self.samples += 1
        if deviation > self.worst_m:
            self.worst_m, self.worst_fraction = deviation, float(fraction)
        if deviation > self.threshold_m + 0.5 * self.granularity_m:
            self.over_threshold += 1
            self.max_run = max(self.max_run, self.over_threshold)
        else:
            self.over_threshold = 0
        if self.over_threshold < self.required:
            return None
        return {'reason': 'MOTION_DEVIATION', 'arm': self.arm, 'phase': self.phase,
                'deviation_m': deviation, 'threshold_m': self.threshold_m, 'worst_m': self.worst_m,
                'worst_fraction': self.worst_fraction, 'fraction': float(fraction), 'samples': self.samples,
                'consecutive': self.over_threshold}

    def evidence(self) -> dict:
        return {'arm': self.arm, 'phase': self.phase, 'threshold_m': self.threshold_m,
                'samples': self.samples, 'worst_m': self.worst_m, 'worst_fraction': self.worst_fraction,
                'max_run': self.max_run, 'required': self.required, 'acted': False}
