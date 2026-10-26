"""Reverse-compute a mobile base placement from a graspable arm posture.

The navigate compiler stops a mobile base at a fixed standoff from the target
object (0.32 m, on the side the base already faces it from, ending facing the
object). For an imported arm that placement can leave only self-colliding IK
postures: on LeKiwi every IK seed at the fixed standoff folds the forearm
into the base, and small standoff changes (0.26 m) fail the same way. The
placement is the free variable, the grasp posture is the constraint, so this
module searches placements instead: it enumerates base placements around the
target object - distance, placement bearing, and a yaw offset from facing the
object - in order of closeness to the default behaviour, and keeps the first
one for which the caller's planning evaluator finds a collision-free grasp
posture (IK seed plus jaw-closure plan, checked on the private collision
model).

The yaw offset is part of the search because the measured blocker was a
shoulder clip grazing the arm's own base plate (1-2 mm at the facing
placements): rotating the base moves the shoulder-pan sector the grasp needs,
which a distance or bearing change alone does not. Placement bearings are
still searched, since with the pan sector moved they change which wheel the
forearm reaches over.

Nothing here moves a robot or writes a pose. The evaluator only *plans*
against hypothetical root transforms; the caller drives the base to the
selected placement through the normal navigate skill, so all motion stays
physical (wheel actuators, measured arrival).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

#: The navigate compiler's fixed standoff; candidate preference is measured
#: from this behaviour.
DEFAULT_STANDOFF_M = 0.32

#: Search grids. Distances span the arm's useful reach around the default
#: standoff; bearing and yaw offsets are stepped coarsely because the
#: evaluator (IK + closure planning per candidate) is the expensive part.
DISTANCES_M = (0.20, 0.24, 0.28, 0.32, 0.36, 0.40)
BEARING_OFFSETS_DEG = (0., -30., 30., -60., 60., -90., 90., 180.)
YAW_OFFSETS_DEG = (0., -20., 20., -40., 40., -60., 60.)

#: One grid step in each dimension costs the same preference penalty, so the
#: search expands roughly spherically from the default placement instead of
#: exhausting one dimension before touching the next.
_DISTANCE_STEP_M = 0.04
_BEARING_STEP_DEG = 30.
_YAW_STEP_DEG = 20.


@dataclass(frozen=True)
class BasePlacement:
    """One candidate pose for the mobile base, in world coordinates."""

    target_xy: tuple[float, float]
    target_yaw: float
    distance_m: float
    bearing_offset_deg: float
    yaw_offset_deg: float

    def as_dict(self) -> dict:
        return {"target_xy": [float(self.target_xy[0]), float(self.target_xy[1])],
                "target_yaw": float(self.target_yaw),
                "distance_m": float(self.distance_m),
                "bearing_offset_deg": float(self.bearing_offset_deg),
                "yaw_offset_deg": float(self.yaw_offset_deg)}


def _preference(placement_values) -> tuple:
    distance_m, bearing_deg, yaw_deg = placement_values
    score = (((distance_m - DEFAULT_STANDOFF_M) / _DISTANCE_STEP_M) ** 2
             + (bearing_deg / _BEARING_STEP_DEG) ** 2
             + (yaw_deg / _YAW_STEP_DEG) ** 2)
    # Deterministic order inside one score: smaller single-dimension
    # deviations first, then the raw values, so the order never depends on
    # grid declaration order.
    return (score, abs(distance_m - DEFAULT_STANDOFF_M) / _DISTANCE_STEP_M,
            abs(bearing_deg) / _BEARING_STEP_DEG, abs(yaw_deg) / _YAW_STEP_DEG,
            distance_m, bearing_deg, yaw_deg)


def candidate_base_placements(target_xy, reference_xy,
                              distances_m=DISTANCES_M,
                              bearing_offsets_deg=BEARING_OFFSETS_DEG,
                              yaw_offsets_deg=YAW_OFFSETS_DEG) -> list[BasePlacement]:
    """Candidate base placements around ``target_xy``, most default-like first.

    ``reference_xy`` is the base's current position: the default bearing is
    the direction from the target toward the reference (the side the base is
    already on, matching the navigate compiler), and each candidate's yaw
    faces the target plus its yaw offset.
    """
    target = np.asarray(target_xy, dtype=float)
    reference = np.asarray(reference_xy, dtype=float)
    offset = reference - target
    if np.linalg.norm(offset) < 1e-9:
        offset = np.array([1., 0.])
    default_bearing = math.atan2(offset[1], offset[0])
    values = [(distance, bearing, yaw)
              for distance in distances_m
              for bearing in bearing_offsets_deg
              for yaw in yaw_offsets_deg]
    values.sort(key=_preference)
    placements = []
    for distance, bearing_deg, yaw_deg in values:
        bearing = default_bearing + math.radians(bearing_deg)
        point = target + distance * np.array([math.cos(bearing), math.sin(bearing)])
        facing = math.atan2(target[1] - point[1], target[0] - point[0])
        placements.append(BasePlacement(
            target_xy=(float(point[0]), float(point[1])),
            target_yaw=float(facing + math.radians(yaw_deg)),
            distance_m=float(distance),
            bearing_offset_deg=float(bearing_deg),
            yaw_offset_deg=float(yaw_deg)))
    return placements


def search_base_placement(target_xy, reference_xy, evaluate, **grid):
    """First candidate (in preference order) the evaluator accepts.

    ``evaluate(placement)`` returns a non-None planning result when the
    placement admits the caller's grasp posture, else None. Evaluation stops
    at the first acceptance. Returns ``(placement, result)`` or None when no
    candidate is accepted.
    """
    for placement in candidate_base_placements(target_xy, reference_xy, **grid):
        result = evaluate(placement)
        if result is not None:
            return placement, result
    return None
