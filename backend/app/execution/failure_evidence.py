"""Task-local, read-only failure observations. Labels never invent a physical measurement.

These receipts describe what an executor returned; they are not independent proof that
its sensor or planner was correct. Replanning may cite their IDs, never forge them.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, default=str)


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def relevant_state(state, *, obj=None, arm=None):
    """Preconditions visible to this layer; absent data stays absent, not zero.

    Other object poses are relevant to collision screening. Contact/link geometry and
    joint constraints are not exposed here; this is not a proof of an identical IK query.
    """
    robot = state.get('robot') or {}
    objects = state.get('objects') or []
    return {
        'object': obj, 'arm': arm,
        'objects': {o['id']: {k: deepcopy(o[k]) for k in ('position', 'rotation', 'bounds', 'held_by', 'touching') if k in o}
                    for o in objects if 'id' in o},
        'hands': {side: robot.get(f'{side}_arm', {}).get('held_object') for side in ('left', 'right')},
        'end_effector': deepcopy((robot.get('end_effectors') or {}).get(arm)) if arm else None,
        'base_pose': deepcopy(state.get('base_pose')),
    }


def state_fingerprint(state, *, obj=None, arm=None):
    return hashlib.sha256(_canonical(relevant_state(state, obj=obj, arm=arm)).encode()).hexdigest()


def _classification(code, raw, phase):
    if code == 'COLLISION_PATH' or code in {'MOTION_BLOCKED', 'PARK_COLLISION'} and raw.get('collision'):
        return 'PATH_COLLISION'
    if code in {'GRASP_FAILED', 'GRASP_LOST', 'GRASP_SLIP', 'HOLD_LOST', 'HAND_CONTACT'}:
        return 'HOLD'
    if code in {'TRACKING_ERROR', 'TRACKING_FAILED'}:
        return 'TRACKING'
    if code in {'ROBOT_CAPABILITY', 'UNSUPPORTED_GOAL'} or (phase == 'preflight' and code == 'INVALID_PLAN'
                                                       and raw.get('kind') == 'capability_validation'):
        return 'UNSUPPORTED'
    if code in {'IK_FAILURE', 'HANDOVER_UNREACHABLE'} and _finite(raw.get('unreachable_by_m')):
        return 'REACHABILITY'
    return 'UNKNOWN'


def failure_observation(contract, epoch, attempt, result, before, after):
    """Normalize the last failed skill, or a structured preflight/verification rejection."""
    # Goal search retains older failed macros even after later recovery. Only a terminal
    # failed skill is the failure of this result; otherwise inspect terminal verification.
    skills = result.get('skills') or []
    failed = skills[-1] if skills and not skills[-1].get('success') else None
    terminal_category = (result.get('verification') or {}).get('failure_category')
    if (result.get('verification') or {}).get('execution_mode') == 'observed_goal_search' and failed \
            and terminal_category not in {None, failed.get('error')}:
        failed = None
    phase = result.get('failure_phase') or ('skill' if failed else 'preflight' if result.get('preflight') else 'verification')
    raw = deepcopy((result.get('failure_evidence') or {}) if result.get('failure_phase') else
                   (failed.get('evidence') or {}) if failed else
                   (result.get('preflight') or {}) if phase == 'preflight' else (result.get('verification') or {}))
    code = (failed or {}).get('error') or (result.get('verification') or {}).get('failure_category') or 'UNKNOWN'
    obj, arm = (failed or {}).get('object'), (failed or {}).get('arm')
    skill_before = (failed or {}).get('before_world_state') or before
    skill_after = (failed or {}).get('world_state') or after
    action = (failed or {}).get('action_index')
    step = (failed or {}).get('step_index')
    provenance = {'task_id': contract.task_id, 'scene_epoch': epoch, 'attempt': attempt,
                  'action_index': action, 'step_index': step, 'phase': phase}
    # No wall clock is substituted for an absent simulation timestamp.
    time_observed = (failed or {}).get('world_state', {}).get('sim_time')
    if time_observed is None:
        time_observed = after.get('sim_time')
    numbers = {k: v for k, v in raw.items() if _finite(v)}
    return {'id': f"{contract.task_id}/epoch/{epoch}/attempt/{attempt}/"
                  f"{phase}/{action if action is not None else 'unknown'}/{step if step is not None else 'unknown'}",
            **provenance, 'skill': (failed or {}).get('skill'), 'object': obj, 'arm': arm,
            'observed_sim_time': time_observed, 'code': code,
            'classification': _classification(code, raw, phase), 'measurement': numbers,
            'raw': raw, 'detail': (failed or {}).get('detail') or result.get('detail'),
            'before_fingerprint': state_fingerprint(skill_before, obj=obj, arm=arm),
            'after_fingerprint': state_fingerprint(skill_after, obj=obj, arm=arm),
            'known_inputs': relevant_state(skill_before, obj=obj, arm=arm),
            'unknown_inputs': ['collision_geometry', 'ik_seed', 'joint_limits']}
