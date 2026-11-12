"""One owner for "can this arm put its hand here, and with which joints".

The same question had four answers (`capability_queries.reachable`, `approach_feasible`, the placement search
and `move_to_site`'s own re-solve), and they disagreed: 30.8 mm planned vs -4.7 mm executed, 20 mm screened vs
6.7-9.1 mm of displacement, 2 mm probed vs 27 mm executed. Each fix removed one disagreement and the class
remained, because nothing owned the answer.

This module owns it. Every caller asks the same function, with an explicit *posture policy*:

* `canonical` - the fixed posture set. The answer is a property of the layout, so it cannot change because the
  arm happens to be standing somewhere else (measured: one layout answered False/True/True from three stances).
* `current`   - from where the arm is now. Momentary, and only the motion should ask it that way.

Answers are cached by (arm, pose, posture) so asking twice costs nothing and returns the identical solution -
which is what makes "the solver is deterministic" a testable claim instead of an intention.
"""

from __future__ import annotations

import numpy as np

#: Same acceptance the capability queries used, so "reachable" keeps meaning one thing.
REACH_TOLERANCE_M = 0.012
#: A pose whose joints sit this close to a limit cannot be held by the servo, even though IK solves it.
LIMIT_MARGIN_RAD = 0.05

_cache: dict[tuple, dict] = {}


def clear_cache() -> None:
    _cache.clear()


def canonical_postures(env, arm: str) -> list[np.ndarray]:
    """The fixed posture set a layout question is asked from (home, plus two from the joint ranges)."""
    postures = []
    try:
        postures.append(np.asarray(env.home_arm_qpos(arm), float))
    except Exception:
        pass
    try:
        limits = np.asarray(env.arm_limits(arm), float)
        low, high = limits[:, 0], limits[:, 1]
        for fraction in (0.25, 0.75):
            weights = np.array([fraction if index % 2 == 0 else 1. - fraction for index in range(low.size)])
            postures.append(low + (high - low) * weights)
    except Exception:
        pass
    unique = []
    for posture in postures:
        if posture.size and np.all(np.isfinite(posture)) and not any(np.allclose(posture, seen) for seen in unique):
            unique.append(posture)
    return unique


def _limits_margin(env, arm, configuration):
    try:
        limits = np.asarray(env.arm_limits(arm), float)
    except Exception:
        return None
    q = np.asarray(configuration, float)
    if q.shape != (limits.shape[0],):
        return None
    return float(np.min(np.minimum(q - limits[:, 0], limits[:, 1] - q)))


def solve_pose(env, controller, arm: str, point, rotation, *, posture: str = 'canonical',
               seed=None) -> dict:
    """Can this arm put its end effector at `point` with `rotation`, and with which joints?

    `posture='canonical'` answers from the fixed set (layout property); `'current'` from the arm's present
    configuration. A `seed` overrides both, for a caller that already knows where it wants the solve to start.
    """
    solve = getattr(controller, 'solve_ik', None)
    if not callable(solve):
        return {'ok': None, 'unknown': 'no IK solver on this controller'}
    target = np.asarray(point, float)
    attitude = np.asarray(rotation, float)
    # The key includes the body it was asked about. Without it, an answer about one controller would be
    # returned for another - a real hazard, and the reason the cache is also cleared between tests.
    key = (id(controller), arm, tuple(np.round(target, 4)), tuple(np.round(attitude, 4).ravel()),
           posture, None if seed is None else tuple(np.round(np.asarray(seed, float), 4)))
    if key in _cache:
        return {**_cache[key], 'cached': True}
    if seed is not None:
        starts = [(None, np.asarray(seed, float))]
    elif posture == 'current':
        starts = [(None, None)]
    else:
        postures = canonical_postures(env, arm)
        # A body that cannot describe a canonical posture still has a solver: fall back to asking from where it
        # is, and say so, rather than reporting the question as unanswerable.
        starts = ([(index, value) for index, value in enumerate(postures)] if postures else [(None, None)])
    best = None
    asked = 0
    for index, start in starts:
        try:
            solution, residual, _ = solve(arm, target, attitude, **({'q_init': start} if start is not None else {}))
        except Exception as error:
            return {'ok': None, 'unknown': f'{type(error).__name__}: {error}'}
        if solution is None:
            # "No solution" is an answer: this arm cannot put its hand there. It must not be read as unknown.
            asked += 1
            candidate = {'ok': False, 'residual_m': None, 'posture': index,
                         'joint_limit_margin_rad': None, 'solution': None, 'reason': 'NO_SOLUTION'}
            if best is None:
                best = candidate
            continue
        if residual is None:
            continue
        asked += 1
        margin = _limits_margin(env, arm, solution)
        candidate = {'ok': bool(residual <= REACH_TOLERANCE_M and (margin is None or margin >= LIMIT_MARGIN_RAD)),
                     'residual_m': float(residual), 'posture': index,
                     'joint_limit_margin_rad': margin,
                     'solution': np.asarray(solution, float).tolist()}
        if candidate['ok']:
            best = candidate
            break
        if best is None or candidate['residual_m'] < best['residual_m']:
            best = candidate
    if best is None:
        answer = {'ok': None, 'unknown': 'no measurable solve'}
    else:
        answer = {**best, 'asked': asked, 'posture_policy': posture}
    if len(_cache) > 512:
        _cache.clear()
    _cache[key] = answer
    return answer


def pose_feasible(env, controller, arm: str, point, rotation, **kwargs) -> dict:
    """The boolean question, answered by the same solver, with the numbers that decided it."""
    answer = solve_pose(env, controller, arm, point, rotation, **kwargs)
    return {'feasible': answer.get('ok'), 'residual_m': answer.get('residual_m'),
            'posture': answer.get('posture'), 'joint_limit_margin_rad': answer.get('joint_limit_margin_rad'),
            'solution': answer.get('solution'),
            **({'unknown': answer['unknown']} if answer.get('unknown') else {})}
