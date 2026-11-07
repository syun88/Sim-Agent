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


def run_preflight(env, controller, plan) -> dict:
    """Read-only queries about every relative placement in `plan`. Never moves the robot."""
    # Accept either an AgentPlan dump (`program.actions`) or a TaskProgram dump (`actions`).
    program = plan.get('program') if isinstance(plan.get('program'), dict) else plan
    actions = (program or {}).get('actions') or []
    questions = _placement_questions(actions)
    if not questions:
        return {'available': True, 'questions': [], 'note': 'no relative placement to query'}
    if env is None or controller is None:
        return {'available': False, 'reason': 'no environment or controller to query',
                'questions': len(questions)}

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
    return {'available': True, 'questions': answers}
