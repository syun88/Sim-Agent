"""Observe -> bind intent -> prepare -> execute -> diagnose -> recover, in one world.

The language model may select only admitted recovery tools. Object identities and
the final goal stay frozen across retries. Reset is a recorded scene restart, never
evidence that a physical attempt succeeded. All physics calls stay on the caller.
"""
from __future__ import annotations

from copy import deepcopy
import time

from backend.app.agent.scene_reasoning import SceneReferenceError, bind_scene_references
from backend.app.agent.recovery_policy import reset_evidence
from backend.app.planning.agent_planner import PlannerProviderError
from backend.app.planning.preparation import compile_prepared_program
from backend.app.planning.spatial_language import preservation_spans
from backend.app.planning.task_program import GroundingError, TaskProgram


RECOVERABLE = {'IK_FAILURE','GRASP_FAILED','COLLISION_PATH','MOTION_BLOCKED','HOLD_LOST',
               'GOAL_NOT_SATISFIED','INVALID_PLAN','HANDOVER_FAILED','PARK_COLLISION','GRASP_LOST','GRASP_SLIP',
               'HANDOVER_UNREACHABLE','HANDOVER_APPROACH_COLLISION','HANDOVER_NO_RECEIVER_PINCH',
               'HANDOVER_DONOR_CONTACT','HANDOVER_LOST','HAND_CONTACT',
               'THROW_FAILED'}
#: Operations a task may retry from the observed scene. All table-safe and re-derivable from world state. The
#: model writes approach/grasp/lift/throw as separate actions, so anything missing here makes whole plans
#: unrestartable - measured 2026-10-27: a plan containing a throw never offered a recovery because `lift` was
#: absent from this set.
RESTARTABLE = {'pick', 'handover', 'place', 'approach', 'home', 'navigate', 'throw', 'pour', 'lift',
               'retreat', 'move'}


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
    def __init__(self, executor, planner, *, plan_call=None, recovery_call=None, max_attempts=4, max_resets=1):
        self.executor, self.planner = executor, planner
        self.plan_call = plan_call or planner.plan
        self.recovery_call = recovery_call or planner.assess_failure
        self.max_attempts, self.max_resets = max_attempts, max_resets

    def run(self, instruction, *, emit=None, should_stop=None, force_failure=False):
        started = time.perf_counter()
        initial = self.executor.world_state()
        epoch_initial = initial
        decisions, attempts, recoveries, resets = [], [], [], 0
        staging_used=False
        replan_used=False
        alternate_used=False
        def announce(decision):
            decisions.append(decision)
            if emit:
                emit('agent_decision',status='REPLANNING',agent_trace=deepcopy(decisions), detail=decision['explanation'])
        def finish(result):
            result = {**result,'agent_trace':decisions,'agent_attempts':attempts,'agent_recoveries':recoveries,
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
        announce({'kind':'scene_observed','explanation':'現在のシーンを観測し、この配置・支持関係・保持状態から新しい作業を計画します。',
            'sim_time':initial.get('sim_time'),'objects':[{k:o.get(k) for k in ('id','position','held_by','touching')} for o in initial['objects']],
            'support_relations':initial.get('support_relations',[])})
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
        plan.planning_evidence={**(plan.planning_evidence or {}),'task_start_observation':{
            'sim_time':initial.get('sim_time'),'objects':deepcopy(initial['objects']),
            'support_relations':deepcopy(initial.get('support_relations',[]))}}
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
            if result['success']:
                # Homing is part of task completion, not a reset. A requested
                # physical hold is retained; empty arms return to model home.
                home_required=bool(source.goals) or (source.actions[-1].op in {'pick','handover','place','throw','pour'})
                if home_required:
                    home=self.executor.home_empty_arms(emit=emit,should_stop=should_stop)
                    result['post_task_homing']=home
                    result['verification']['task_goal_before_homing']=True
                    if not home['success']:
                        result.update(success=False,status=home.get('status','FAILED'),
                            detail='作業の配置は維持していますが、空の腕のホーム復帰を完了できませんでした。')
                        result['verification'].update(success=False,failure_category='POST_TASK_HOME_FAILED')
                    elif restartable:
                        final=verify_task_goal(self.executor.env,check,[],initial_state=epoch_initial)
                        result['verification']['after_homing_postconditions']=final
                        if not final['success']:
                            result.update(success=False,status='FAILED',detail='ホーム復帰後に作業の最終状態を確認できませんでした。')
                            result['verification'].update(success=False,failure_category='POST_TASK_SCENE_CHANGED')
                return finish(result)
            if result.get('status')=='STOPPED':
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
            legal, continuation, alternate = [], None, None
            prefix = 0 if plan.goal_program else completed_prefix(plan,result)
            if plan.goal_program:
                # The goal executor already replans locally. Avoid restarting its
                # exhausted failure transition under a fresh, forgotten ban list.
                prefix = 0
            elif prefix<len(plan.program['actions']):
                try:
                    remaining = TaskProgram(actions=plan.program['actions'][prefix:])
                    continuation = compile_prepared_program(remaining,observed)
                    # Retain every original final predicate, including objects
                    # placed before this suffix. The scene has not been reset.
                    continuation.goal = deepcopy(original.goal)
                    continuation.goal['allowed_object_motion'] = sorted({s.object for s in original.steps if s.object})
                    # First try useful progress. With no completed action,
                    # clear/reposition free arms before another approach.
                    if prefix>0 and not any(r.get('decision')=='resume' for r in recoveries):
                        legal.append('resume')
                except GroundingError:
                    pass
            if not staging_used:
                legal.append('recover_in_place')
            if (not alternate_used and len(source.actions)==1 and source.actions[0].op=='pick'
                    and source.actions[0].arm is None and not any(observed['robot'].get(a+'_arm',{}).get('held_object') for a in ('left','right'))):
                previous_arm=plan.program['actions'][0]['arm']
                for arm in observed['robot'].get('available_arms',['left','right']):
                    if arm==previous_arm:continue
                    try:
                        request=source.model_copy(deep=True)
                        request.actions[0].arm=arm
                        alternate=compile_prepared_program(request,observed)
                        legal.append('alternate_arm')
                        break
                    except GroundingError:
                        pass
            certificate=reset_evidence(observed,source)
            # Measured failures can be answered with a different plan; only when they carry numbers, so the
            # agent can still run out of options.
            # Replanning is offered when the failure carries something a different plan could use: a design, or
            # measurements plus the levers that produced them (a throw that reports which lever would help).
            # Conditional, not unconditional: the agent must still be able to run out of options.
            evidence = failed.get('evidence') or {}
            carries_measurements = ('design' in str(failed.get('detail') or '')
                                    or bool(evidence.get('levers')) or bool(evidence.get('unreachable_by_m')))
            if not replan_used and carries_measurements:
                legal.append('replan')
            if not legal and certificate and reset_allowed and resets<self.max_resets:
                legal.append('reset_and_retry')
            if not legal:
                result['detail'] = f'{category}: 同じシーン内で利用できる復旧案を試しました。配置を維持して停止します。'
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
                'assessment':assessment,'completed_action_prefix':prefix,'reset_evidence':certificate})
            recovery={'decision':assessment['decision'],'before':observed}
            recoveries.append(recovery)
            if assessment['decision']=='resume':
                plan = continuation
            elif assessment['decision']=='alternate_arm':
                alternate_used=True
                plan=alternate
                # No arm was requested: the same cube may be held by either
                # admitted arm. Preserve explicit ownership everywhere else.
                original.goal=deepcopy(plan.goal)
            elif assessment['decision']=='replan':
                replan_used=True
                try:
                    from backend.app.planning.agent_planner import compact_scene
                    # A replanned task starts from where the robot actually is: give it the hands and the
                    # compact scene, not only the failure, so a plan cannot assume an object is already held.
                    replanned=self.plan_call(instruction,observed,
                        [{'recent_failure':failed.get('detail'),'measured':failed.get('evidence'),
                          'hands':{side:(observed.get('robot',{}).get(f'{side}_arm') or {}).get('held_object')
                                   for side in ('left','right')},
                          'scene':compact_scene(observed)}])
                except Exception:
                    replanned=None
                if replanned is None or not getattr(replanned,'steps',None):
                    result['detail']=f'{category}: 失敗の数値を踏まえた再計画を作れませんでした。'
                    return finish(result)
                plan=replanned
                original=replanned
                recovery['replanned_steps']=[step.skill for step in plan.steps]
            elif assessment['decision']=='recover_in_place':
                staging_used=True
                staging=self.executor.home_empty_arms(emit=emit,should_stop=should_stop,clear_contacts=True)
                recovery['staging']=staging
                recovery['after']=self.executor.world_state()
                if staging.get('status')=='STOPPED':
                    result.update(success=False,status='STOPPED')
                    return finish(result)
                try:
                    observed=self.executor.world_state()
                    if source.goals:
                        # The inner goal executor already tracks failed effects;
                        # repositioning the arms is a real change before retrying.
                        plan=compile_prepared_program(source,observed)
                    else:
                        # ``prefix`` counts completed actions; when it equals the list length there is nothing
                        # left to re-compile, and TaskProgram refuses an empty action list (measured
                        # 2026-10-27: the recovery surfaced as a raw ValidationError instead of a decision).
                        remaining_actions = plan.program['actions'][prefix:]
                        if not remaining_actions:
                            remaining_actions = plan.program['actions']
                        remaining=TaskProgram(actions=remaining_actions)
                        plan=compile_prepared_program(remaining,observed)
                        plan.goal=deepcopy(original.goal)
                    plan.goal['allowed_object_motion']=sorted({s.object for s in original.steps if s.object})
                except GroundingError as error:
                    result['detail']='現状を維持した復旧後も計画できませんでした: '+str(error)
                    return finish(result)
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
