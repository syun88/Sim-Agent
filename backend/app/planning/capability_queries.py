"""Read-only questions the planner may ask about its own body before committing to a motion.

The project's failures kept being answered by a human looking at numbers and then hard-coding a value:
"stage at x=0.45, y=-0.04", "lower the pinch 15 mm", "try yaw 270". Each of those was a measurement turned
into a constant. This module is the same measurements as *callable questions*, so the planner can ask them
per scene instead of inheriting a constant that happened to be true once.

Every query is non-destructive: private forward kinematics, IK solves and distance checks. None of them
moves the robot, and none of them returns a value it did not measure - an unavailable query reports
`unknown` with the reason.
"""

from __future__ import annotations

import numpy as np

from backend.app.simulation.openarm_v2.definition import FINGERTIP_MIDPOINT, GRASP_ROTATION
from backend.app.skills.pad_clearance import support_face_clearance

#: Yaw candidates for a hand that must reach a pose: the calibrated attitude first, then quarter turns.
#: A cube is four-fold symmetric, so these are the same grasp with a different approach side.
ATTITUDE_YAWS_DEG = (0.0, 90.0, 180.0, 270.0)


def catalog() -> list[dict]:
    """What the planner may ask, in the shape a prompt or a model needs."""
    return [
        {'name': 'reachable_pose',
         'question': 'can this arm put its pinch point here with this hand attitude?',
         'inputs': ['arm', 'pinch_point', 'attitude_yaw_deg'],
         'returns': ['ok', 'position_error_m', 'rotation_error_rad']},
        {'name': 'pad_blocks_support_face',
         'question': "does the holder's own finger geometry stand above the held support's top face?",
         'inputs': ['holder_arm', 'support'],
         'returns': ['clearance_m', 'pad_over_face_m']},
    ]


def attitude(yaw_deg: float) -> np.ndarray:
    from scipy.spatial.transform import Rotation
    return Rotation.from_euler('z', yaw_deg, degrees=True).as_matrix() @ GRASP_ROTATION


def pinch_point(arm: str, object_center, yaw_deg: float = 0.0) -> np.ndarray:
    """Where the control point must be so the fingertip pads close on `object_center`."""
    return np.asarray(object_center, float) - attitude(yaw_deg) @ FINGERTIP_MIDPOINT[arm]


def reachable(env, controller, arm: str, pinch_target, yaw_deg: float = 0.0) -> dict:
    """Whether this arm can hold its pinch at `pinch_target` with that attitude, and how far off it is."""
    solve = getattr(controller, 'solve_ik', None)
    if not callable(solve):
        return {'ok': None, 'unknown': 'no IK solver on this controller'}
    target = np.asarray(pinch_target, float)
    rotation = attitude(yaw_deg)
    try:
        solution, position_error, _ = solve(arm, target, rotation)
    except Exception as error:
        return {'ok': None, 'unknown': f'{type(error).__name__}: {error}'}
    if solution is None:
        return {'ok': False, 'position_error_m': None, 'yaw_deg': yaw_deg}
    return {'ok': bool(position_error is not None and position_error <= .012),
            'position_error_m': None if position_error is None else float(position_error),
            'yaw_deg': yaw_deg}


def pad_blocks_support_face(env, holder_arm: str, support: str) -> dict:
    """Whether the holder's fingers occupy the space above the support's top face."""
    try:
        bounds = env.object_bounds(support)
        centre = env.object_pose(support)[0]
        half = float(max(bounds[1][0] - bounds[0][0], bounds[1][1] - bounds[0][1])) / 2.
        measured = support_face_clearance(env, holder_arm, centre, half, float(bounds[1][2]))
    except Exception as error:
        return {'blocks': None, 'unknown': f'{type(error).__name__}: {error}'}
    clearance = measured.get('clearance_m')
    if clearance is None:
        return {'blocks': None, 'unknown': measured.get('unknown', 'unmeasured'), 'measured': measured}
    return {'blocks': clearance < 0., 'clearance_m': float(clearance), 'measured': measured}
