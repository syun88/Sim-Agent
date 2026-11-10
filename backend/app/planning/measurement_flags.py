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
}


#: Flags that a measurement showed to be harmful as a default. Measured 2026-11-09 on identical layouts and
#: instructions (seeds 11,12 x 4 tasks): with the arm preflight refusing, 2/8 succeeded and 4 tasks were
#: refused that the robot could actually do; with it off, 3/8 succeeded and nothing was refused. The
#: measurement is still recorded (the preflight rows are computed either way) - it just does not decide.
DEFAULT_OFF = {'arm_preflight'}


def enabled(name: str) -> bool:
    if name not in FLAGS:
        raise KeyError(f'unknown measurement flag {name!r}; known: {sorted(FLAGS)}')
    default = '0' if name in DEFAULT_OFF else '1'
    return os.environ.get(f'OPENARM_MEASURE_{name.upper()}', default) not in ('0', 'false', 'False', '')


def describe() -> dict[str, bool]:
    return {name: enabled(name) for name in FLAGS}
