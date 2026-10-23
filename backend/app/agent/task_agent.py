"""Observe -> bind intent -> prepare -> execute -> diagnose -> recover, in one world.

The language model may select only admitted recovery tools. Object identities and
the final goal stay frozen across retries. Reset is a recorded scene restart, never
evidence that a physical attempt succeeded. All physics calls stay on the caller.
"""
from __future__ import annotations

from copy import deepcopy
import time

from backend.app.agent.scene_reasoning import SceneReferenceError, bind_scene_references
from backend.app.planning.agent_planner import PlannerProviderError
from backend.app.planning.preparation import compile_prepared_program
from backend.app.planning.spatial_language import preservation_spans
from backend.app.planning.task_program import GroundingError, TaskProgram


RECOVERABLE = {'IK_FAILURE','GRASP_FAILED','COLLISION_PATH','MOTION_BLOCKED','HOLD_LOST',
               'GOAL_NOT_SATISFIED','INVALID_PLAN','HANDOVER_FAILED','PARK_COLLISION','GRASP_LOST','GRASP_SLIP',
               'HANDOVER_UNREACHABLE','HANDOVER_APPROACH_COLLISION','HANDOVER_NO_RECEIVER_PINCH',
               'HANDOVER_DONOR_CONTACT','HANDOVER_LOST','HAND_CONTACT'}
RESTARTABLE = {'pick','handover','place','approach','home'}


def completed_prefix(plan, result):
    """A successful approach alone never marks a pick/handover as completed."""
    count = 0
    for index, _ in enumerate(plan.program.get('actions', [])):
        expected = [s for s in plan.steps if s.action_index==index]
        actual = [s for s in result.get('skills',[]) if s.get('action_index')==index]
        if not expected or len(actual)!=len(expected) or not all(s['success'] for s in actual):
            break
        count += 1
    return count


class TaskAgent:
    def __init__(self, executor, planner, *, plan_call=None, recovery_call=None, max_attempts=3, max_resets=1):
        self.executor, self.planner = executor, planner
        self.plan_call = plan_call or planner.plan
        self.recovery_call = recovery_call or planner.assess_failure
        self.max_attempts, self.max_resets = max_attempts, max_resets

    def run(self, instruction, *, emit=None, should_stop=None, force_failure=False):
        started = time.perf_counter()
        initial = self.executor.world_state()
        epoch_initial = initial
        decisions, attempts, resets = [], [], 0
        def announce(decision):
            decisions.append(decision)
            if emit:
                emit('agent_decision',status='REPLANNING',agent_trace=deepcopy(decisions), detail=decision['explanation'])
        def finish(result):
            result = {**result,'agent_trace':decisions,'agent_attempts':attempts,
                'initial_world_state':initial,'final_world_state':self.executor.world_state(),
                'total_duration':time.perf_counter()-started}
            result['verification'] = {**result.get('verification',{}),'agent_attempts':len(attempts),
                'scene_resets':resets,'all_agent_attempts_succeeded':all(a['result']['success'] for a in attempts)}
            result['retry_count'] = sum(a['result'].get('retry_count',0) for a in attempts)+max(0,len(attempts)-1)
            return result
        def blocked(detail, category, diagnosis=None):
            return finish({'success':False,'status':'FAILED','detail':detail,'skills':[],
                'diagnosis':diagnosis or {},'verification':{'success':False,'failure_category':category}})
        if should_stop and should_stop():
            return finish({'success':False,'status':'STOPPED','skills':[],'verification':{'success':False}})
        try:
            binding = bind_scene_references(instruction, initial)
            for decision in binding.decisions:
                announce(decision)
            plan = self.plan_call(binding.instruction, initial, decisions)
        except SceneReferenceError as error:
            announce({'kind':'missing_reference','explanation':str(error),'observed':error.diagnosis})
            return blocked(str(error),'REFERENCE_NOT_PRESENT',error.diagnosis)
        except PlannerProviderError as error:
            holds = {s:initial.get('robot',{}).get(s+'_arm',{}).get('held_object') for s in ('left','right')}
            diagnosis = {'category':error.category,'observed_hands':holds,
                'required':'指示の対象・順序が既存の操作で表現でき、対象が観測され、必要な手が空いていること。',
                'capabilities':initial.get('capabilities',{})}
            announce({'kind':'planning_blocked','explanation':str(error),'observed':diagnosis})
            return blocked(f'{error.category}: {error}',error.category,diagnosis)
        original = plan.model_copy(deep=True)
        for item in (plan.planning_evidence or {}).get('preparations',[]):
            announce({'kind':'prepare_precondition',**item})
        for decision in binding.decisions:
            plan.assumptions.append(decision['explanation'])
        # Replay is restricted to idempotent rigid-cube tasks. No automatic
        # double pour, second throw, joint increment or relative arm displacement.
        source = TaskProgram.model_validate(original.goal_program or original.program)
        if source.goals and (original.planning_evidence or {}).get('requested_program'):
            source=TaskProgram.model_validate(original.planning_evidence['requested_program'])
        if source.actions:
            # Freeze resolved object IDs but separate the user's requested
            # actions from occupancy preparation and incidental arm choices.
            evidence = original.planning_evidence or {}
            inserted = {i for p in evidence.get('preparations',[]) for i in (p['action_index'],p['action_index']+1)}
            source.actions = [a for i,a in enumerate(source.actions) if i not in inserted]
            requested = evidence.get('requested_program',{}).get('actions',[])
            if len(requested)==len(source.actions):
                for action, raw in zip(source.actions,requested):
                    if raw.get('arm') is None and action.op in {'pick','place','approach'}:
                        action.arm = None
        restartable = bool(source.goals or source.actions) and (bool(source.goals) or all(a.op in RESTARTABLE for a in source.actions))
        restartable = restartable and all((not a.object or a.object.endswith('_cube')) for a in source.actions)
        reset_allowed = restartable and not source.keep_in_place and not preservation_spans(instruction, initial)
        reset_allowed = reset_allowed and getattr(self.executor.env,'scene_profile','cubes')=='cubes'
        for iteration in range(self.max_attempts):
            if should_stop and should_stop():
                return finish({'success':False,'status':'STOPPED','plan':plan.model_dump(),'skills':[], 'verification':{'success':False}})
            if emit:
                emit('plan_generated',status='VALIDATING',plan=plan.model_dump(),agent_trace=deepcopy(decisions))
            result = self.executor.run(plan,emit=emit,should_stop=should_stop,force_failure=force_failure and iteration==0)
            if result['success'] and restartable:
                from backend.app.verification.goal_verification import verify_task_goal
                # Measure all final predicates against this scene epoch, not
                # just the resumed suffix. Attempt history remains separate.
                check = original.model_copy(deep=True)
                check.steps = []
                check.goal['allowed_object_motion'] = sorted({s.object for s in original.steps if s.object})
                if resets:
                    check.goal = deepcopy(plan.goal)
                    check.goal['allowed_object_motion'] = sorted({s.object for s in original.steps if s.object})
                final = verify_task_goal(self.executor.env,check,[],initial_state=epoch_initial)
                result['verification']['agent_final_postconditions'] = final
                if not final['success']:
                    result.update(success=False,status='FAILED')
                    result['verification'].update(success=False,failure_category=final['failure_category'])
            attempts.append({'attempt':iteration+1,'plan':plan.model_dump(),'result':deepcopy(result)})
            if result['success'] or result.get('status')=='STOPPED':
                return finish(result)
            category = result.get('verification',{}).get('failure_category')
            failed = next((s for s in reversed(result.get('skills',[])) if not s['success']),{})
            category = failed.get('error') or category
            observed = self.executor.world_state()
            announce({'kind':'execution_failed','explanation':f'{category}。現在の保持・接触と残りの作業を再確認します。',
                'failure_category':category,'observed_hands':{s:observed.get('robot',{}).get(s+'_arm',{}).get('held_object') for s in ('left','right')}})
            if iteration+1==self.max_attempts or category not in RECOVERABLE or not restartable:
                result['detail'] = result.get('detail') or f'{category}: 自動復旧できる操作の範囲または試行上限に達しました。'
                return finish(result)
            legal, continuation = [], None
            prefix = 0 if plan.goal_program else completed_prefix(plan,result)
            if plan.goal_program:
                # The goal executor already replans locally. Avoid restarting its
                # exhausted failure transition under a fresh, forgotten ban list.
                prefix = 0
            elif prefix>0 and prefix<len(plan.program['actions']):
                try:
                    remaining = TaskProgram(actions=plan.program['actions'][prefix:])
                    continuation = compile_prepared_program(remaining,observed)
                    # Retain every original final predicate, including objects
                    # placed before this suffix. The scene has not been reset.
                    continuation.goal = deepcopy(original.goal)
                    continuation.goal['allowed_object_motion'] = sorted({s.object for s in original.steps if s.object})
                    legal.append('resume')
                except GroundingError:
                    pass
            if reset_allowed and resets<self.max_resets:
                legal.append('reset_and_retry')
            if not legal:
                result['detail'] = f'{category}: 現在の制約を保った復旧手順がありません。'+('位置を保持する指示があるためシーンを初期化しません。' if not reset_allowed else '')
                return finish(result)
            assessment = self.recovery_call(instruction,observed,
                {'category':category,'detail':failed.get('detail') or result.get('detail'),
                 'completed_action_prefix':prefix,'attempt':iteration+1,'resets':resets},legal)
            if assessment.get('decision') not in legal:
                assessment = {'decision':legal[0],'reason':'観測状態で実行可能な復旧手順を選びます。','provider':'deterministic_fallback'}
            if should_stop and should_stop():
                result.update(success=False,status='STOPPED')
                return finish(result)
            announce({'kind':assessment['decision'],'explanation':assessment.get('reason','復旧します。'),
                'assessment':assessment,'completed_action_prefix':prefix})
            if assessment['decision']=='resume':
                plan = continuation
            else:
                if emit:
                    emit('agent_reset',status='PREPARING_SCENE',detail='失敗した試行を記録し、シーンを初期化して同じ対象・目標で再試行します。')
                self.executor.env.reset()
                self.executor.reset_context()
                self.executor.env.step(200)
                resets += 1
                epoch_initial = self.executor.world_state()
                try:
                    plan = compile_prepared_program(source,epoch_initial)
                except GroundingError as error:
                    return blocked(str(error),'RECOVERY_PRECONDITION_FAILED')
            for field in ('provider','model','provider_error'):
                setattr(plan,field,getattr(original,field))
        return blocked('AGENT_ATTEMPT_BUDGET','AGENT_ATTEMPT_BUDGET')
