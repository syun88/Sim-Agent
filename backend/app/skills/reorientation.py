"""The attitudes a held object passes through while it is turned for placement.

A rotation from A to B has two arcs and the arm only has one of them when a wrist joint is near its limit. The
short arc used to be the only one tried, so a reorientation failed with "Cannot reorient" even when the object
could be turned the other way round (measured 2026-11-10). The complementary arc is the same endpoint reached
past 180 degrees, which needs the relative rotation taken *beyond* pi - a slerp cannot express it, because its
interpolation times must stay in [0, 1].
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

#: How many intermediate attitudes an arc is broken into. Eight was what the placement reorientation used and
#: what the existing layouts were measured with.
STEPS = 8


def arc_samples(start, target, mode: str = 'short') -> list[np.ndarray]:
    """The attitudes to move through, ending at `target`.

    mode='short'  the geodesic arc; mode='long' the same endpoint reached past pi; mode='direct' no arc at all.
    """
    start_rotation = Rotation.from_matrix(np.asarray(start, float))
    target_rotation = Rotation.from_matrix(np.asarray(target, float))
    if mode == 'direct':
        return [np.asarray(target, float)]
    fractions = np.linspace(0., 1., STEPS + 1)[1:]
    relative = (start_rotation.inv() * target_rotation).as_rotvec()
    angle = float(np.linalg.norm(relative))
    if angle < 1e-9:
        return [np.asarray(target, float)]
    if mode == 'long':
        # The complementary arc: the same rotation taken the other way (angle - 2*pi).
        relative = relative / angle * (angle - 2. * np.pi)
    return [(start_rotation * Rotation.from_rotvec(relative * fraction)).as_matrix()
            for fraction in fractions]
