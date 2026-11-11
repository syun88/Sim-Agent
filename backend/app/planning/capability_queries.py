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


def reachable(env, controller, arm: str, pinch_target, yaw_deg: float = 0.0, *,
              posture: str = 'current') -> dict:
    """Whether this arm can hold its pinch at `pinch_target` with that attitude, and how far off it is.

    Answered by the one solver (`planning.body_solver`), so this and every other caller cannot disagree about
    the same pose.
    """
    from backend.app.planning.body_solver import pose_feasible as _solve
    answer = _solve(env, controller, arm, pinch_target, attitude(yaw_deg), posture=posture)
    if answer.get('feasible') is None:
        return {'ok': None, 'unknown': answer.get('unknown', 'unmeasured'), 'yaw_deg': yaw_deg}
    return {'ok': bool(answer['feasible']), 'position_error_m': answer.get('residual_m'),
            'yaw_deg': yaw_deg, 'posture': answer.get('posture'),
            'joint_limit_margin_rad': answer.get('joint_limit_margin_rad')}


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


#: Postures a *layout* question is asked from. Fixed set: the same layout must get the same answer whenever
#: it is asked, which is what makes the answer a property of the layout rather than of the moment.
def canonical_postures(env, arm: str) -> list[np.ndarray]:
    """Where to stand when asking "can this arm grasp that object here at all".

    A fixed, deterministic set: home, plus two configurations taken from the joint ranges. One posture is not
    enough (measured 2026-11-09: home answers False for a layout the same arm can approach from a folded
    posture) and the current posture is not allowed (the answer would depend on the moment).
    """
    postures = []
    try:
        postures.append(np.asarray(env.home_arm_qpos(arm), float))
    except Exception:
        pass
    try:
        limits = np.asarray(env.arm_limits(arm), float)
        low, high = limits[:, 0], limits[:, 1]
        for fraction in (0.25, 0.75):
            # Alternate the direction per joint so the two derived postures are not mirror images of each other
            # and the set covers both halves of each joint's range.
            weights = np.array([fraction if index % 2 == 0 else 1. - fraction
                                for index in range(low.size)])
            postures.append(low + (high - low) * weights)
    except Exception:
        pass
    unique = []
    for posture in postures:
        if posture.size and np.all(np.isfinite(posture)):
            if not any(np.allclose(posture, seen) for seen in unique):
                unique.append(posture)
    return unique


#: Answers already computed, keyed by layout. A layout question is expensive (a few approach plans) and its
#: answer cannot change while the layout is the same, so it is remembered. Posture is *not* in the key: that
#: is the point of asking it this way.
_FEASIBLE_CACHE: dict[tuple, dict] = {}


def clear_feasibility_cache() -> None:
    _FEASIBLE_CACHE.clear()


def approach_feasible(env, controller, arm: str, obj: str, *, max_postures: int | None = None,
                      max_frames: int | None = None) -> dict:
    """Whether this arm has a *screened route* to the object from any canonical posture.

    Two questions live here, and conflating them was a root cause: "is there a route from where the arm is
    standing right now" is momentary (the motion answers it), while "can this arm grasp that object in this
    layout" is a property of the layout. Measured 2026-11-09 on one fixed layout: asking from the current
    posture answered False from home, True from a folded posture and True from a spread one - the same
    question flipping with the arm's whereabouts, which is why refusals and hand switches appeared
    inconsistent. The answer below is a property of the layout because the postures it is asked from are a
    fixed set.
    """
    try:
        from backend.app.skills.approach import cube_approach_frames, plan_cube_approach
        from backend.app.skills.robot_skills import APPROACH_OVERFLY, APPROACH_STANDOFF, STAGING_ALTITUDE

        try:
            key = (arm, obj, tuple(np.round(np.asarray(env.object_pose(obj)[0], float), 3)))
        except Exception:
            key = None
        if key is not None and key in _FEASIBLE_CACHE:
            return {**_FEASIBLE_CACHE[key], 'cached': True}
        postures = canonical_postures(env, arm)
        if max_postures is not None:
            # A bounded question for callers that must stay cheap (a reset admission runs on every scene). Measured
            # 2026-11-10: the full set costs ~89 s per reset across two objects and two arms, which is right for an
            # offline corpus and wrong interactively.
            postures = postures[:max(1, max_postures)]
        if not postures:
            return {'feasible': None, 'unknown': 'no canonical posture for this body'}
        centre = np.asarray(env.object_pose(obj)[0], float)
        frames = cube_approach_frames(arm, centre, APPROACH_STANDOFF)
        if max_frames is not None:
            frames = frames[:max(1, max_frames)]
        attempts = []
        for index, posture in enumerate(postures):
            route, planning = plan_cube_approach(env, controller, arm, obj, frames[0]['pinch'],
                                                frames[0]['standoff'], APPROACH_OVERFLY, STAGING_ALTITUDE,
                                                frames=frames, start_override=posture)
            attempts.append({'posture': index,
                             'feasible': route is not None,
                             'reason': planning.get('error'),
                             'grasp_yaw_deg': None if route is None else planning.get('grasp_yaw_deg')})
            if route is not None:
                answer = {'feasible': True, 'posture': index, 'grasp_yaw_deg': planning.get('grasp_yaw_deg'),
                          'attempts': attempts, 'asked_from': 'canonical_postures'}
                if key is not None:
                    if len(_FEASIBLE_CACHE) > 128:
                        _FEASIBLE_CACHE.clear()
                    _FEASIBLE_CACHE[key] = answer
                return answer
        answer = {'feasible': False, 'reason': attempts[-1].get('reason'), 'attempts': attempts,
                  'asked_from': 'canonical_postures'}
        if key is not None:
            if len(_FEASIBLE_CACHE) > 128:
                _FEASIBLE_CACHE.clear()
            _FEASIBLE_CACHE[key] = answer
        return answer
    except AttributeError as error:
        # A body that cannot answer the planner's own questions is not thereby unable to approach: report the
        # question as unknown rather than turning a missing method into a refusal.
        return {'feasible': None, 'unknown': f'{type(error).__name__}: {error}'}
    except Exception as error:
        return {'feasible': None, 'unknown': f'{type(error).__name__}: {error}'}


def zone_reachability(env, controller, *, arms=('left', 'right')) -> dict:
    """Whether each zone is within an arm's reach at all - the question a layout has to answer before it is used.

    Measured 2026-11-09: of 8 tasks on two fixed layouts, 4 failed and 2 of those were `DESTINATION_UNREACHABLE`
    - the zone could not be reached by the arm holding the cube from any posture. Nothing in the reset checked
    that, so the layout was handed to a task it could not satisfy.
    """
    found = {}
    try:
        zones = list((getattr(env.frame(), 'zones', {}) or {}))
    except Exception:
        return found
    for zone in zones:
        try:
            centre = np.asarray(env.zone_position(zone), float)
        except Exception:
            continue
        for arm in arms:
            postures = canonical_postures(env, arm)
            best, asked = None, 0
            for posture in postures:
                for yaw in ATTITUDE_YAWS_DEG:
                    target = pinch_point(arm, centre, yaw)
                    try:
                        solution, error, _ = controller.solve_ik(arm, target, attitude(yaw), q_init=posture)
                    except Exception:
                        continue
                    if error is None:
                        continue
                    asked += 1
                    if solution is not None and (best is None or error < best):
                        best = float(error)
            found.setdefault(zone, {})[arm] = {
                'reachable': None if not asked else bool(best is not None and best <= .012),
                'best_residual_m': best, 'asked': asked}
    return found


def layout_grasp_admission(env, controller, arms=('left', 'right'), *, yaws=ATTITUDE_YAWS_DEG,
                           deep: bool | str = False) -> dict:
    """Whether every object in the current layout can be pinched by *some* arm from a canonical posture.

    Measured 2026-11-10 on a 18-task round: of 7,312 rejected approach routes, 6,971 were `IK_FAILURE` - the arm
    could not even solve the pregrasp for the object. 341 were clearance. So the dominant reason a task failed was
    a layout the body could not work in, which is answerable before the task starts.
    """
    answer = {'objects': {}, 'admitted': True}
    try:
        names = list((getattr(env.frame(), 'objects', {}) or {}))
    except Exception:
        return {'objects': {}, 'admitted': None, 'unknown': 'no scene'}
    for name in names:
        try:
            centre = np.asarray(env.object_pose(name)[0], float)
        except Exception:
            continue
        per_arm = {}
        for arm in arms:
            ok = None
            if deep:
                # The route question, not the pinch question. Measured 2026-11-10: pinch reachability admitted every
                # layout the planner then failed on (0 resamples), because a screened 4-waypoint approach needs more
                # than an IK solution at the pinch point.
                budget = {} if deep == 'deep' else {'max_postures': 1, 'max_frames': 4}
                probe = approach_feasible(env, controller, arm, name, **budget)
                per_arm[arm] = probe.get('feasible')
                continue
            for yaw in yaws:
                probe = reachable(env, controller, arm, pinch_point(arm, centre, yaw), yaw, posture='canonical')
                if probe.get('ok') is True:
                    ok = True
                    break
                if probe.get('ok') is None:
                    ok = None
            per_arm[arm] = ok
        reachable_arms = [arm for arm, ok in per_arm.items() if ok is True]
        answer['objects'][name] = {'arms': per_arm, 'reachable_arms': reachable_arms}
        if per_arm and not reachable_arms and all(ok is not None for ok in per_arm.values()):
            answer['admitted'] = False
    return answer
