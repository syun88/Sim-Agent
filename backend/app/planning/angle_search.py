"""Search a continuous parameter by measuring, instead of picking from a list.

Every angle in this project was a fixed candidate list: quarter turns for the placement attitude, a
fourteen-value yaw table for the approach, five tilts for the leaning search, three staging points. The
agent could *choose* among them and record why, but it could not *derive* an angle - and when none of the
listed values worked, the answer was a refusal rather than a nearby angle that would have.

This module does the smallest thing that changes that: a coarse scan followed by a refinement around the
best measured value. The objective is a measurement callback, so the search is only as good as what the
body can report (IK residual, clearance, path screen); a parameter it cannot measure is `None` and is
skipped, never scored as zero. The full evaluation trace is returned, because "which angle worked and how
good was it" is evidence a reader can check.
"""

from __future__ import annotations

#: How many measurements one refinement pass may spend, on top of the coarse scan.
DEFAULT_REFINEMENT_STEPS = 5


def _sample(value: float, score: float | None) -> dict:
    """Plain data: these traces are returned from the API, so they must serialise as they are."""
    return {'value': float(value), 'score': None if score is None else float(score)}


def refine(objective, start: float, span: float, *, step: float,
           refinement_steps: int = DEFAULT_REFINEMENT_STEPS, minimum_step: float = 0.25):
    """Maximise `objective(value)` around `start` by scanning, then halving the span.

    `objective` returns a score to maximise, or None when that value cannot be measured. Ties keep the
    earliest value, so a search that finds nothing better than the nominal angle returns the nominal one.
    """
    if step <= 0 or span <= 0:
        raise ValueError('span and step must be positive')
    evaluations: list[dict] = []
    best: dict | None = None

    def measure(value: float) -> None:
        nonlocal best
        score = objective(float(value))
        sample = _sample(value, score)
        evaluations.append(sample)
        if sample['score'] is not None and (best is None or sample['score'] > best['score']):
            best = sample

    steps = max(1, int(round(span / step)))
    for index in range(-steps, steps + 1):
        measure(start + index * step)
    local_span = step
    for _ in range(refinement_steps):
        if best is None:
            break
        local_span /= 2.
        if local_span < minimum_step:
            break
        for offset in (-local_span, local_span):
            measure(best['value'] + offset)
    return {'best_value': None if best is None else best['value'],
            'best_score': None if best is None else best['score'],
            'started_at': float(start), 'evaluations': evaluations,
            'measured': sum(1 for sample in evaluations if sample['score'] is not None)}


def best_of(objective, values):
    """The best-scoring of an explicit candidate list, keeping the first on a tie."""
    best_value, best_score = None, None
    for value in values:
        score = objective(float(value))
        if score is None:
            continue
        if best_score is None or score > best_score:
            best_value, best_score = float(value), float(score)
    return best_value, best_score
