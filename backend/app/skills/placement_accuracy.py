"""How precisely this placement has to land, measured from the geometry it lands on.

A placement is feasible when the arm can put the object inside the region the task actually needs:

    probe residual + how far this body lands when carrying  <=  how far off-centre it may land

The third term was the missing one. It is not a constant: a zone is tens of millimetres wide, while a cube
stacked on a cube of the same size needs its centre over the support. Both are read from the scene - the
zone's site radius from the model file, the support face from the body bounds - and when the region cannot be
measured the requirement is `None`, which means "do not gate on it" rather than "allow it".
"""

from __future__ import annotations

import numpy as np


def region_half_extent(env, target: str) -> tuple[float, float]:
    """The half-size of the region a placement may land in, in x and y."""
    bounds = np.asarray(env.object_bounds(target), float)
    return (float((bounds[1, 0] - bounds[0, 0]) / 2.), float((bounds[1, 1] - bounds[0, 1]) / 2.))


def required_accuracy(env, obj: str, target: str, relation: str) -> dict:
    """How far off the requested point the object may land, with the geometry that decided it."""
    try:
        region_x, region_y = region_half_extent(env, target)
        held = np.asarray(env.object_bounds(obj), float)
    except Exception as error:
        return {'required_m': None, 'basis': None, 'unknown': f'{type(error).__name__}: {error}'}
    held_x, held_y = (float((held[1, 0] - held[0, 0]) / 2.), float((held[1, 1] - held[0, 1]) / 2.))
    zones = {}
    try:
        zones = dict(getattr(env.frame(), 'zones', {}) or {})
    except Exception:
        zones = {}
    is_zone = target in zones
    if relation == 'on' and not is_zone:
        # Stacking: the object stays on a support of this half-size when its centre lands within it. The
        # object's own size does not widen that - an overhanging cube still falls.
        requirement = min(region_x, region_y)
        basis = {'support_half_m': [region_x, region_y],
                 'reason': 'the object must keep its centre over the support'}
    else:
        # Inside a region (a zone): the object has to be *inside* it, so its own half-size is subtracted. A
        # zone is an area, not a support face, so "on" describes where it lands, not what it lands on.
        requirement = max(0., min(region_x - held_x, region_y - held_y))
        basis = {'region_half_m': [region_x, region_y], 'object_half_m': [held_x, held_y],
                 'reason': 'the object must lie inside the region'}
    if requirement <= 0:
        return {'required_m': 0., 'basis': basis,
                'unknown': 'the region is no wider than the object'}
    return {'required_m': float(requirement), 'basis': basis}


def feasibility(env, obj: str, target: str, relation: str, *, probe_residual_m: float | None,
                expected_error_m: float) -> dict:
    """Whether a point with this probe residual can be reached closely enough to count as placed."""
    requirement = required_accuracy(env, obj, target, relation)
    needed = None
    if probe_residual_m is not None:
        needed = float(probe_residual_m) + float(expected_error_m)
    feasible = None
    if requirement['required_m'] is not None and needed is not None:
        feasible = bool(needed <= requirement['required_m'])
    return {'required_m': requirement['required_m'], 'basis': requirement.get('basis'),
            'probe_residual_m': probe_residual_m, 'expected_error_m': expected_error_m,
            'estimated_landing_error_m': needed, 'feasible': feasible,
            **({'unknown': requirement['unknown']} if requirement.get('unknown') else {})}
