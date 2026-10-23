"""Observe after each semantic action and search only the remaining final goals.

The attempts ledger retains unsuccessful physical actions. Recovery never resets
the scene, changes object poses, or asks an LLM to assert a successful outcome.
"""
from __future__ import annotations

import time
import math

from backend.app.models.schemas import AgentPlan, PlanStep
from backend.app.planning.goal_search import GoalDomain, action_key, compile_goal_program
from backend.app.planning.task_program import GroundingError, TaskProgram, compile_program


def _can_recover(executor, result, action):
    if not result.get('skills'):
        return False
    failed = next((step for step in result['skills'] if not step['success']), {})
    if failed.get('error') not in {'IK_FAILURE', 'COLLISION_PATH', 'MOTION_BLOCKED', 'GRASP_FAILED'}:
        return False
    # A partly completed exchange or a loaded transport must not be retried as a
    # table pick. The existing controller owns local, contact-sensitive recovery.
    if action.op != 'pick':
        return False
    env, obj = executor.env, action.object
    return (not any(env.object_contacts(obj, side) for side in ('left', 'right'))
            and abs(env.object_bounds(obj)[0, 2]-env.table_height) < .006
            and env.object_linear_speed(obj) < .02)


def run_goal_program(executor, plan, *, emit=None, should_stop=None, force_failure=False):
    started = time.perf_counter()
    source = TaskProgram.model_validate(plan.goal_program)
    initial = executor.world_state()
    blocked, attempts, skills, trace = set(), [], [], []
    prepared = {}
    def signature(state, arm):
        ee = state.get('robot', {}).get('end_effectors', {}).get(arm)
        return (*ee['position'], *(v for row in ee['rotation'] for v in row)) if ee else None
    def observed_state():
        state = executor.world_state()
        ready = []
        for arm, previous in prepared.items():
            current_pose = signature(state, arm)
            if current_pose == previous or (current_pose is not None and previous is not None
                    and math.dist(current_pose[:3], previous[:3]) < .003
                    and math.dist(current_pose[3:], previous[3:]) < .05):
                ready.append(arm)
        state['robot']['handover_ready'] = ready
        return state
    repairs, retries = 0, 0
    verification = {'success': False, 'failure_category': 'GOAL_NOT_SATISFIED'}
    detail, stopped, success = None, False, False
    try:
        for iteration in range(25):
            if should_stop and should_stop():
                stopped = True
                break
            observed = observed_state()
            current = compile_goal_program(source, observed, initial_state=initial, blocked=blocked)
            # Retain the language provenance when the suffix is synthesised again.
            for field in ('provider', 'model', 'provider_error'):
                setattr(current, field, getattr(plan, field))
            domain = GoalDomain(source, observed, initial)
            before = domain.project(observed)
            pending = current.program['actions']
            trace.append({'iteration': iteration, 'pending_actions': pending,
                          'planning_evidence': current.planning_evidence})
            if not pending:
                # This is an actual independent read/settle/verification step,
                # including preserved positions and every requested final goal.
                check = current.model_copy(deep=True)
                check.goal_program = None
                check.goal['allowed_object_motion'] = sorted(domain.movable)
                check.steps = [PlanStep(skill='observe_scene', arm=current.arm)]
                final = executor.run(check, emit=emit, should_stop=should_stop)
                verification = final['verification']
                skills.extend(final['skills'])
                success = final['success']
                stopped = final['status'] == 'STOPPED'
                if not success:
                    detail = 'Observed final goals did not pass independent Genesis verification'
                break
            if iteration == 24:
                detail = 'GOAL_SEARCH_BUDGET: 24 semantic actions executed without reaching every final goal'
                break
            from backend.app.planning.task_program import TaskAction
            action = TaskAction.model_validate(pending[0])
            if emit:
                emit('goal_replanned', status='REPLANNING', plan=current.model_dump(),
                     detail=f'Observed scene; {len(pending)} remaining actions')
            macro = compile_program(TaskProgram(actions=[action]), observed)
            # Frame constraints belong to the whole requested task. A different
            # requested cube may move while it remains physically held; truly
            # protected objects still have frozen-position checks at every step.
            macro.goal['allowed_object_motion'] = sorted(domain.movable)
            # Preserve grasp look-ahead from the complete suffix. Recompiling an
            # isolated pick would lose the pad offset needed for a later handover.
            macro.steps = [step.model_copy(deep=True) for step in current.steps if step.action_index == 0]
            result = executor.run(macro, emit=emit, should_stop=should_stop,
                                  force_failure=force_failure and iteration == 0)
            attempts.append({'action': action.model_dump(), 'result': result})
            retries += result.get('retry_count', 0)
            skills.extend(result['skills'])
            prepared.pop(action.arm, None)
            if action.receiver:
                prepared.pop(action.receiver, None)
            if result['success'] and action.op == 'home':
                prepared[action.arm] = signature(executor.world_state(), action.arm)
            domain.assert_preserved(executor.world_state())
            if result['status'] == 'STOPPED':
                stopped = True
                break
            after = domain.project(observed_state())
            if not result['success']:
                if repairs >= 2 or not _can_recover(executor, result, action):
                    verification = result['verification']
                    detail = 'Physical action failed; partial holds and non-table failures require explicit recovery'
                    break
                blocked.add((after, action_key(action)))
                repairs += 1
                trace[-1]['feedback'] = {'failed_action': action.model_dump(),
                    'failure_category': result['verification'].get('failure_category'),
                    'blocked_in_observed_state': True}
            elif before == after:
                # An unfulfilled effect cannot loop forever just because a
                # controller returned success. Search a different transition.
                blocked.add((after, action_key(action)))
                trace[-1]['feedback'] = {'failure_category': 'EFFECT_NOT_OBSERVED'}
    except GroundingError as error:
        detail = str(error)
        verification = {'success': False, 'failure_category': str(error).split(':', 1)[0]}
    verification = {**verification, 'success': bool(success and not stopped),
        'execution_mode': 'observed_goal_search', 'replanning_trace': trace,
        'execution_attempts': attempts, 'recovery_count': repairs,
        'all_attempts_succeeded': all(a['result']['success'] for a in attempts)}
    return {'success': bool(success and not stopped),
        'status': 'STOPPED' if stopped else 'SUCCEEDED' if success else 'FAILED',
        'detail': detail, 'plan': plan.model_dump(), 'skills': skills,
        'verification': verification, 'initial_world_state': initial,
        'final_world_state': executor.world_state(), 'retry_count': retries+repairs,
        'total_duration': time.perf_counter()-started}
