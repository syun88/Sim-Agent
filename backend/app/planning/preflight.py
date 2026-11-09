"""Answer the cheap geometrical questions about a plan *before* it runs.

The mechanism predictor could not recognise the contact mechanism from plan shape alone (measured 2026-11-08:
`APPROACH_CONTACT` 0/7 recalled). The missing information was geometrical - is the placement pose reachable,
does the holder's own grip cover the face - and all of it can be asked in advance, without motion. This
module runs those queries once per plan and stores the answers next to the plan, so a later predictor (or a
reader) has them before anything moves.

Answers that cannot be measured stay `None`. A plan with no relative placement has nothing to ask, and an
environment without the queries is reported as unavailable rather than as "no problem".
"""

from __future__ import annotations

import math

import numpy as np

from backend.app.planning.capability_queries import (ATTITUDE_YAWS_DEG, pad_blocks_support_face,
                                                    pinch_point, reachable)


def _finite(value):
    """A measurement that a JSON reader can consume: non-finite values are reported as absent.

    A single `inf` from a geometry query aborted 11 of 24 live tasks at serialisation time (measured
    2026-11-08). Sanitising here means one bad measurement can never cost a run.
    """
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)):
        return float(value) if math.isfinite(float(value)) else None
    return value


def _placement_questions(actions):
    return [action for action in actions
            if action.get('op') == 'place' and action.get('relation') and action.get('target')]


#: Operations that need one arm to reach one object.
_GRASP_OPS = ('pick', 'approach')


def arm_feasibility(env, controller, actions, *, arms=('left', 'right')) -> list[dict]:
    """Whether the arm the plan names can actually reach the object, measured, with the alternative.

    Measured 2026-11-09: an instruction naming the right hand for a cube at y=+0.264 was attempted twelve
    times and failed every time in the approach search - the layout simply put the cube beyond that arm. The
    honest answer is available before any motion: ask each arm, and say which one can.
    """
    rows = []
    for action in actions:
        if action.get('op') not in _GRASP_OPS or not action.get('object'):
            continue
        arm = action.get('arm')
        if arm not in arms:
            continue
        row = {'op': action.get('op'), 'object': action.get('object'), 'arm': arm, 'alternatives': {}}
        for candidate in (arm, *(name for name in arms if name != arm)):
            try:
                centre = np.asarray(env.object_pose(action['object'])[0], float)
                answers = [reachable(env, controller, candidate, pinch_point(candidate, centre, yaw), yaw)
                           for yaw in ATTITUDE_YAWS_DEG]
                decisive = [item for item in answers if item.get('ok') is not None]
                best = min((item.get('position_error_m') for item in answers
                            if item.get('position_error_m') is not None), default=None)
                unknown = next((item.get('unknown') for item in answers if item.get('unknown')), None)
                row['alternatives'][candidate] = {
                    'reachable_yaws': sum(1 for item in decisive if item['ok']),
                    'asked_yaws': len(decisive),
                    'best_position_error_m': _finite(best),
                    **({'unknown': unknown} if unknown else {})}
            except Exception as error:
                row['alternatives'][candidate] = {'error': f'{type(error).__name__}: {error}'}
        requested = row['alternatives'].get(arm) or {}
        others = [(name, answer) for name, answer in row['alternatives'].items() if name != arm]
        row['requested_reachable'] = (None if not requested.get('asked_yaws')
                                      else bool(requested.get('reachable_yaws')))
        reachable_others = [name for name, answer in others
                            if answer.get('asked_yaws') and answer.get('reachable_yaws')]
        row['reachable_arms'] = reachable_others
        row['verdict'] = ('REQUESTED_ARM_UNREACHABLE' if row['requested_reachable'] is False
                          and reachable_others else
                          'REQUESTED_ARM_REACHABLE' if row['requested_reachable'] else 'UNMEASURED')
        rows.append(row)
    return rows


def run_preflight(env, controller, plan) -> dict:
    """Read-only queries about every relative placement in `plan`. Never moves the robot."""
    # Accept either an AgentPlan dump (`program.actions`) or a TaskProgram dump (`actions`).
    program = plan.get('program') if isinstance(plan.get('program'), dict) else plan
    actions = (program or {}).get('actions') or []
    questions = _placement_questions(actions)
    if env is None or controller is None:
        if not questions:
            return {'available': True, 'questions': [], 'arms': [],
                    'note': 'no relative placement to query'}
        return {'available': False, 'reason': 'no environment or controller to query',
                'questions': len(questions), 'arms': []}
    arms = arm_feasibility(env, controller, actions)
    if not questions:
        return {'available': True, 'questions': [], 'arms': arms,
                'note': 'no relative placement to query'}

    answers = []
    for action in questions:
        arm, target = action.get('arm'), action.get('target')
        row = {'arm': arm, 'object': action.get('object'), 'target': target}
        try:
            held = bool((env.object_pose(target) is not None)
                        and any((env.frame().objects or {})))
        except Exception:
            held = None
        row['target_known'] = held
        try:
            blocked = pad_blocks_support_face(env, 'right' if arm == 'left' else 'left', target)
            row['pad_blocks_face'] = blocked.get('blocks')
            row['pad_clearance_m'] = blocked.get('clearance_m')
        except Exception as error:
            row['pad_blocks_face'] = None
            row['pad_error'] = f'{type(error).__name__}: {error}'
        try:
            centre = np.asarray(env.object_pose(target)[0], float)
            heights = [reachable(env, controller, arm, pinch_point(arm, centre, yaw), yaw)
                       for yaw in ATTITUDE_YAWS_DEG]
            decisive = [item for item in heights if item.get('ok') is not None]
            row['reachable_yaws'] = sum(1 for item in decisive if item['ok'])
            row['reachable_asked'] = len(decisive)
        except Exception as error:
            row['reachable_yaws'] = None
            row['reach_error'] = f'{type(error).__name__}: {error}'
        answers.append({key: _finite(value) for key, value in row.items()})
    return {'available': True, 'questions': answers, 'arms': arms}
