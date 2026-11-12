"""Which measured safeguards are switched on, so they can be compared instead of assumed.

Every safeguard in this project was added because a measurement said it should help. Measured 2026-11-09: with
all of them on, success over a fixed battery swung between 0/9 and 4/6 - which means "it helped" cannot be
concluded while they are stacked. These flags exist so the same layouts and the same instructions can be run
with a safeguard off, and the difference is the evidence.

Default is on: a flag is for measurement, not for shipping a weaker robot.
"""

from __future__ import annotations

import os

#: name -> what turning it off restores.
FLAGS = {
    'approach_margin': 'approach screen is the nominal 20 mm only, without the measured approach error',
    'destination_search': 'place at the requested point only, with no search inside the region',
    'arm_preflight': 'do not measure the requested hand before moving',
    'home_margin': 'home clearance is the old fixed 30 mm, not the measured landing error',
    'destination_hand': 'do not refuse a hand that cannot reach the destination it was told to use',
    'requested_hand_preflight': ('do not refuse before moving when the named hand measurably cannot reach '
                                 'the object in this layout while the other hand can'),
}


#: Flags that a measurement showed to be harmful as a default. Measured 2026-11-09 on identical layouts and
#: instructions (seeds 11,12 x 4 tasks): with the arm preflight refusing, 2/8 succeeded and 4 tasks were
#: refused that the robot could actually do; with it off, 3/8 succeeded and nothing was refused. The
#: measurement is still recorded (the preflight rows are computed either way) - it just does not decide.
DEFAULT_OFF = {'arm_preflight'}

#: `requested_hand_preflight` ships on, by measurement rather than by preference. It is the cheap form of the
#: refusal that `arm_preflight` got wrong: that one answered from the arm's momentary posture and refused work
#: the robot could do. Measured 2026-11-10 on both seeds with the same six instructions - seed 21 (the layout
#: that works): 6/6 with the refusal on, 6/6 with it off, and nothing refused. Seed 22 (two tasks name a hand
#: the layout cannot serve): 2/6 either way, and the refusal turned two 60-90 s failures into an immediate
#: failure carrying the measured reason ("the left hand cannot reach red_cube here; the right one can").
#: Same success, no discarded work, two wasted attempts gone.


def enabled(name: str) -> bool:
    if name not in FLAGS:
        raise KeyError(f'unknown measurement flag {name!r}; known: {sorted(FLAGS)}')
    default = '0' if name in DEFAULT_OFF else '1'
    return os.environ.get(f'OPENARM_MEASURE_{name.upper()}', default) not in ('0', 'false', 'False', '')


def describe() -> dict[str, bool]:
    return {name: enabled(name) for name in FLAGS}
