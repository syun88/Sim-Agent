"""Which way to point the hand at an object: the angles an approach may choose from.

A pinch is not a single attitude. Rotating the nominal jaw frame about the vertical changes which side
the hand comes from; leaning it in the vertical plane changes the angle the fingers come down at, and
with it how high the hand has to be while they are beside the object. Both keep the invariant that
makes it the same grasp: the fingertip midpoint still lands on the object centre, so the pads still
close on the object's faces.

The nominal OpenArm attitude is not top-down: its approach axis is about 80 degrees from vertical, so
the hand already comes from the side. A positive lean turns it further down, a negative one brings it
back towards horizontal - which is why the search reports the resulting angle rather than the rotation
it applied.

This module only *generates* the angles, in a fixed order, and says what each one is - the numbers are
not tuned here. Whether a candidate works is decided by the engine: the route screen refuses one with
the links that met, the contact watch stops another, and only a candidate that reaches the pinch pose
without touching the object is accepted. Keeping the geometry separate from that verdict is what lets
the choice be measured rather than assumed.

Measured context (2026-10-27, OpenArm, seed 20261023): rotating the frame about its own approach axis
was refused for every angle from 15 to 90 degrees, and the refusal named a finger inside the table by
0.7 to 1.4 mm - the hand stays at the same height while it spins. Tilting moves the hand off that
height, which is why it is the axis worth searching.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from backend.app.simulation.openarm_v2.definition import GRASP_ROTATION

#: Lean applied to the nominal approach axis, in degrees, tried in this order: the nominal attitude
#: first, then progressively further down. The small negative value reaches a purely horizontal
#: approach (`tilt_from_vertical_deg` reports the resulting angle, which starts near 80).
TILTS_DEGREES = (0., -10., 20., 35., 50.)
#: Which side to come from, in degrees about the vertical, tried in this order for each tilt.
AZIMUTHS_DEGREES = (0., 90., -90., 180.)
#: Horizontal axis the lean rotates about, before the azimuth spins it.
_TILT_AXIS = np.array([0., 1., 0.])


def pinch_frames(tilts=TILTS_DEGREES, azimuths=AZIMUTHS_DEGREES):
    """Candidate jaw frames, nominal first, each as (tilt_degrees, azimuth_degrees, rotation).

    The tilt is applied to the nominal frame about a horizontal axis so the approach leaves vertical,
    then the azimuth rotates the result about the vertical so the hand can come from any side. The
    composition is deliberately fixed and reported: a search whose candidates are not named cannot say
    afterwards which angle worked.
    """
    nominal = np.asarray(GRASP_ROTATION, float)
    for tilt in tilts:
        lean = Rotation.from_rotvec(np.deg2rad(tilt) * _TILT_AXIS).as_matrix()
        for azimuth in azimuths:
            spin = Rotation.from_rotvec(np.deg2rad(azimuth) * np.array([0., 0., 1.])).as_matrix()
            yield float(tilt), float(azimuth), spin @ lean @ nominal


def approach_axis(frame):
    """The direction the fingers point in, in world coordinates (the tool's own z, downwards)."""
    return np.asarray(frame, float) @ np.array([0., 0., -1.])


def tilt_from_vertical_deg(frame):
    """How far the approach axis is from straight down, in degrees: 0 is top-down."""
    axis = approach_axis(frame)
    vertical = np.array([0., 0., -1.])
    cosine = abs(float(axis @ vertical)) / max(float(np.linalg.norm(axis)), 1e-12)
    return float(np.degrees(np.arccos(np.clip(cosine, -1., 1.))))


def describe(frame) -> dict:
    """The candidate's own numbers, for the record."""
    return {'tilt_from_vertical_deg': round(tilt_from_vertical_deg(frame), 1),
            'approach_axis': [round(float(v), 3) for v in approach_axis(frame)]}
