"""Observe -> bind intent -> prepare -> execute -> diagnose -> recover, in one world.

The language model may select only admitted recovery tools. Object identities and
the final goal stay frozen across retries. Reset is a recorded scene restart, never
evidence that a physical attempt succeeded. All physics calls stay on the caller.
"""
from __future__ import annotations

from copy import deepcopy
import time

from backend.app.agent.scene_reasoning import SceneReferenceError
from backend.app.planning.grounding import ground
from backend.app.agent.recovery_policy import admit_reset, reset_evidence
from backend.app.verification.failure_attribution import cross_check, withholds
from backend.app.planning.agent_planner import PlannerProviderError, grounded_recovery_reason
from backend.app.planning.preparation import compile_prepared_program
from backend.app.planning.spatial_language import preservation_spans
from backend.app.planning.task_program import GroundingError, TaskProgram
from backend.app.planning.task_contract import TaskContract, requested_program
from backend.app.execution.contract_evidence import ContractEvidence, completed_prefix
from backend.app.execution.failure_evidence import failure_observation


RECOVERABLE = {'IK_FAILURE','GRASP_FAILED','COLLISION_PATH','MOTION_BLOCKED','HOLD_LOST',
               'GOAL_NOT_SATISFIED','INVALID_PLAN','HANDOVER_FAILED','PARK_COLLISION','GRASP_LOST','GRASP_SLIP',
               'HANDOVER_UNREACHABLE','HANDOVER_APPROACH_COLLISION','HANDOVER_NO_RECEIVER_PINCH',
               'HANDOVER_DONOR_CONTACT','HANDOVER_LOST','HAND_CONTACT',
               'APPROACH_COLLISION','CARRY_COLLISION',
               'THROW_SPEED_NOT_REACHED',
               'THROW_FAILED'}
#: Operations a task may retry from the observed scene. All table-safe and re-derivable from world state. The
#: model writes approach/grasp/lift/throw as separate actions, so anything missing here makes whole plans
#: unrestartable - measured 2026-10-27: a plan containing a throw never offered a recovery because `lift` was
#: absent from this set.
# Relative displacement cannot be restarted from an unknown partial endpoint: repeating the
# same distance would change the original requested amount. It needs scoped effect evidence first.
RESTARTABLE = {'pick', 'handover', 'place', 'approach', 'home', 'navigate', 'throw', 'pour', 'lift',
               'retreat'}


#: Mechanisms that are produced by the arm's path meeting an object or the robot itself.
CONTACT_MECHANISMS = frozenset({'APPROACH_CONTACT', 'CARRY_CONTACT', 'GEOMETRY'})


def _repeats_measured_contact(mechanism, receipt):
    """Whether resuming would drive the same arm through the same unchanged obstacle.

    A contact mechanism plus an unchanged relevant-state fingerprint means nothing the contact depended on
    moved, so the same suffix repeats it. A changed fingerprint (the object shifted, a hand let go) is new
    evidence and may be resumed. This only withholds a candidate; it never turns a failure into success.
    """
    if mechanism not in CONTACT_MECHANISMS:
        return False
    before, after = receipt.get('before_fingerprint'), receipt.get('after_fingerprint')
    if before is None or after is None:
        return False
    return before == after


def _switch_plan_arm(plan, old_arm: str, new_arm: str) -> None:
    """Rewrite an inferred hand in both the program and the compiled steps.

    Only used when the instruction did not name a hand: then the hand is a resolution of the request, and
    resolving it to the arm that measurably has an approach route is what the request actually asked for.
    """
    program = getattr(plan, 'program', None)
    for action in (getattr(program, 'actions', None) or []):
        if getattr(action, 'arm', None) == old_arm:
            action.arm = new_arm
        if getattr(action, 'receiver', None) == old_arm:
            action.receiver = new_arm
    for step in (getattr(plan, 'steps', None) or []):
        if getattr(step, 'arm', None) == old_arm:
            step.arm = new_arm
        if getattr(step, 'receiver', None) == old_arm:
            step.receiver = new_arm


class TaskAgent:
    def __init__(self, executor, planner, *, plan_call=None, recovery_call=None, max_attempts=4, max_resets=1,
                 mechanism_reviewer=None):
        self.executor, self.planner = executor, planner
        self.plan_call = plan_call or planner.plan
        self.recovery_call = recovery_call or planner.assess_failure
        self.max_attempts, self.max_resets = max_attempts, max_resets
        #: Optional second opinion on a failure's mechanism. It can only withhold the irreversible
        #: recovery (see `verification.failure_attribution.cross_check`); None disables it, which is the
        #: default because it costs a model call per failure.
        self.mechanism_reviewer = mechanism_reviewer

    def run(self, instruction, *, emit=None, should_stop=None, force_failure=False):
        started = time.perf_counter()
        initial = self.executor.world_state()
        epoch_initial = initial
        decisions, attempts, recoveries, resets = [], [], [], 0
        #: Reviewer verdicts and reset admissions, kept across attempts: a decision taken on an earlier
        #: attempt must not disappear when a later attempt produces the final result.
        mechanism_reviews, reset_decisions = [], []
        evidence_ledger = []
        staging_used=False
        replan_used=False
        alternate_used=False
        contract = None
        def announce(decision):
            decisions.append(decision)
            if emit:
                emit('agent_decision',status='REPLANNING',agent_trace=deepcopy(decisions), detail=decision['explanation'])
        def finish(result):
            result = {**result,'agent_trace':decisions,'agent_attempts':attempts,'agent_recoveries':recoveries,
                'initial_world_state':initial,'final_world_state':self.executor.world_state(),
                'evidence_ledger':deepcopy(evidence_ledger),
                'mechanism_reviews':deepcopy(mechanism_reviews),
                'reset_decisions':deepcopy(reset_decisions),
                'total_duration':time.perf_counter()-started}
            result['verification'] = {**result.get('verification',{}),'agent_attempts':len(attempts),
                'scene_resets':resets,'all_agent_attempts_succeeded':all(a['result']['success'] for a in attempts)}
            if contract is not None:
                result['task_contract'] = contract.describe()
                result['verification']['contract_id'] = contract.contract_id
            result['retry_count'] = sum(a['result'].get('retry_count',0) for a in attempts)+max(0,len(attempts)-1)
            return result
        def blocked(detail, category, diagnosis=None):
            return finish({'success':False,'status':'FAILED','detail':detail,'skills':[],
                'diagnosis':diagnosis or {},'verification':{'success':False,'failure_category':category,
                    **({'contract_status':'UNKNOWN'} if category=='CONTRACT_UNVERIFIABLE' else {})}})
        if should_stop and should_stop():
            return finish({'success':False,'status':'STOPPED','skills':[],'verification':{'success':False}})
        announce({'kind':'scene_observed','explanation':'現在のシーンを観測し、この配置・支持関係・保持状態から新しい作業を計画します。',
            'sim_time':initial.get('sim_time'),'objects':[{k:o.get(k) for k in ('id','position','held_by','touching')} for o in initial['objects']],
            'support_relations':initial.get('support_relations',[])})
        try:
            # One grounding pass, recorded. The planner receives the resolved text and its own copy of the
            # decisions, so the same sentence is not rewritten twice with different offsets.
            binding = ground(instruction, initial)
            for decision in binding['decisions']:
                announce({'kind':decision['kind'],'explanation':decision['explanation'],
                          'object':decision['resolved'].get('object'),'arm':decision['resolved'].get('arm'),
                          'basis':decision['basis'],'step':decision['step'],
                          'surface':decision['surface'],'replacement':decision['replacement'],
                          'candidates':decision['candidates'],'alternatives':decision['alternatives']})
            plan = self.plan_call(binding['instruction'], {**initial, 'instruction': instruction}, decisions)
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
        # Cheap geometrical questions about the plan, answered before anything moves. Stored with the plan
        # so a later predictor has contact/path information that does not depend on the outcome.
        preflight = None
        try:
            from backend.app.planning.preflight import run_preflight
            preflight = run_preflight(getattr(self.executor, 'env', None),
                                      getattr(self.executor, 'controller', None), plan.program)
            plan.planning_evidence = {**(plan.planning_evidence or {}), 'preflight': preflight}
        except Exception as error:
            preflight = {'available': False, 'reason': f'{type(error).__name__}: {error}'}
            plan.planning_evidence = {**(plan.planning_evidence or {}), 'preflight': preflight}
        # An arm that cannot reach the object is measured before any motion. Measured 2026-11-09: a right-hand
        # instruction for a cube at y=+0.264 failed twelve attempts in the approach search; both arms answer
        # this question in milliseconds.
        #
        # What happens next depends on who chose the hand. If the *instruction* named it, the request cannot be
        # met and the task stops with the alternative named. If the hand was only inferred, refusing would be
        # wrong - measured 2026-11-09: "青いキューブを赤いキューブの上に置いて" was refused for the inferred
        # left hand while the right hand could reach it, so the inferred choice is changed and recorded.
        from backend.app.planning.language_fallback import arm_mentions
        from backend.app.planning.measurement_flags import enabled as _measured
        # The *destination* decides the hand when the instruction does not name one. Measured 2026-11-09: each
        # zone in these layouts is reachable by exactly one arm (blue by the left, green by the right), so an
        # inferred hand that cannot reach the target makes the task impossible for a reason that is known
        # before any motion - and already measured in preflight['zones'].
        explicit_arms = set(arm_mentions(instruction))
        zone_switch = []
        for step in (getattr(plan, 'steps', None) or []):
            target, arm = getattr(step, 'target', None), getattr(step, 'arm', None)
            zones = (preflight or {}).get('zones') or {}
            if not target or arm not in ('left', 'right') or arm in explicit_arms:
                continue
            if not isinstance(zones.get(target), dict):
                continue
            if (zones[target].get(arm) or {}).get('reachable') is False:
                alternatives = [name for name in ('left', 'right') if name != arm
                                and (zones[target].get(name) or {}).get('reachable') is True]
                if not alternatives:
                    continue
                for other_step in (getattr(plan, 'steps', None) or []):
                    if getattr(other_step, 'arm', None) == arm:
                        other_step.arm = alternatives[0]
                for action in (getattr(getattr(plan, 'program', None), 'actions', None) or []):
                    if getattr(action, 'arm', None) == arm:
                        action.arm = alternatives[0]
                zone_switch.append({'target': target, 'from': arm, 'to': alternatives[0],
                                    'basis': 'the other arm is the one that can reach that destination',
                                    'residual_m': (zones[target].get(alternatives[0]) or {}).get('best_residual_m')})
        if zone_switch:
            plan.planning_evidence = {**(plan.planning_evidence or {}),
                                      'arm_choice': (plan.planning_evidence or {}).get('arm_choice', []) + zone_switch}
            announce({'kind': 'arm_choice',
                      'explanation': '指示に手の指定が無いため、目的地に届く手を選びました。',
                      'arm_choice': zone_switch})
        unreachable = ([row for row in (preflight or {}).get('arms', [])
                        if row.get('verdict') == 'REQUESTED_ARM_UNREACHABLE']
                       if _measured('arm_preflight') else [])
        if unreachable:
            from backend.app.planning.language_fallback import arm_mentions
            explicit = set(arm_mentions(instruction))
            switchable = [row for row in unreachable
                          if row['arm'] not in explicit and row.get('reachable_arms')]
            if switchable:
                changes = []
                for row in switchable:
                    alternative = row['reachable_arms'][0]
                    _switch_plan_arm(plan, row['arm'], alternative)
                    changes.append({'object': row['object'], 'from': row['arm'], 'to': alternative,
                                    'basis': 'the hand was not requested; the other one has a screened approach',
                                    'screen': (row.get('screen') or {}).get(alternative)})
                plan.planning_evidence = {**(plan.planning_evidence or {}),
                                          'arm_choice': changes}
                announce({'kind': 'arm_choice', 'explanation':
                          '指示に手の指定が無いため、接近経路が確認できた手を選びました。',
                          'arm_choice': changes})
                preflight = run_preflight(getattr(self.executor, 'env', None),
                                          getattr(self.executor, 'controller', None), plan.program)
                plan.planning_evidence = {**(plan.planning_evidence or {}), 'preflight': preflight}
                unreachable = [row for row in (preflight or {}).get('arms', [])
                               if row.get('verdict') == 'REQUESTED_ARM_UNREACHABLE']
        if unreachable:
            row = unreachable[0]
            alternative = ', '.join(row.get('reachable_arms') or [])
            # Keep the whole measurement with the plan before refusing: measured 2026-11-09, refusals raised
            # here left no preflight rows behind, so a refused round could not be diagnosed afterwards.
            plan.planning_evidence = {**(plan.planning_evidence or {}),
                                      'preflight': preflight,
                                      'arm_refusal': {'object': row.get('object'), 'arm': row.get('arm'),
                                                      'reachable_arms': row.get('reachable_arms'),
                                                      'alternatives': row.get('alternatives'),
                                                      'screen': row.get('screen')}}
            raise GroundingError(
                f"ARM_CANNOT_REACH: the {row['arm']} hand cannot reach {row['object']} here "
                f"(reachable: {alternative or 'neither hand'})")
        for decision in binding['decisions']:
            if decision['explanation']:
                plan.assumptions.append(decision['explanation'])
        # Replay is restricted to idempotent rigid-cube tasks. No automatic
        # double pour, second throw, joint increment or relative arm displacement.
        source = requested_program(original)
        contract = TaskContract.freeze(instruction, original, source)
        receipts = ContractEvidence(contract)
        announce({'kind':'task_contract_bound','explanation':'元の目的・対象・順序を固定し、以後の計画と実行証拠を別に管理します。',
                  'task_contract':contract.describe()})
        restartable = bool(source.goals or source.actions) and (bool(source.goals) or all(a.op in RESTARTABLE for a in source.actions))
        restartable = restartable and all((not a.object or a.object.endswith('_cube')) for a in source.actions)
        reset_allowed = restartable and not source.keep_in_place and not preservation_spans(instruction, initial)
        reset_allowed = reset_allowed and getattr(self.executor.env,'scene_profile','cubes')=='cubes'
        for iteration in range(self.max_attempts):
            if should_stop and should_stop():
                return finish({'success':False,'status':'STOPPED','plan':plan.model_dump(),'skills':[], 'verification':{'success':False}})
            try:
                contract.admit(plan, receipts)
                plan = contract.execution_plan(plan, receipts)
            except GroundingError as error:
                category = 'CONTRACT_UNVERIFIABLE' if str(error).startswith('CONTRACT_UNVERIFIABLE') else 'CONTRACT_VIOLATION'
                return blocked(str(error), category)
            plan.planning_evidence = {**(plan.planning_evidence or {}), 'task_contract':contract.describe()}
            if emit:
                emit('plan_generated',status='VALIDATING',plan=plan.model_dump(),agent_trace=deepcopy(decisions))
            before_attempt = self.executor.world_state()
            result = self.executor.run(plan,emit=emit,should_stop=should_stop,force_failure=force_failure and iteration==0)
            receipts.record(plan, result, iteration + 1)
            if result['success']:
                # State predicates are observed now; events use task/epoch-scoped receipts.
                final = receipts.verify(self.executor.env,plan,epoch_initial)
                result['verification']['agent_final_postconditions'] = final
                if not final['success']:
                    result.update(success=False,status='FAILED')
                    result['verification'].update(success=False,failure_category=final['failure_category'])
            failure_receipt = None
            if not result['success']:
                failure_receipt = failure_observation(contract, receipts.epoch, iteration + 1,
                    result, before_attempt, self.executor.world_state())
                evidence_ledger.append(failure_receipt)
            attempts.append({'attempt':iteration+1,'plan':plan.model_dump(),'result':deepcopy(result),
                             'failure_evidence_id':failure_receipt['id'] if failure_receipt else None})
            if result['success']:
                # Homing is part of task completion, not a reset. A requested
                # physical hold is retained; empty arms return to model home.
                home_required=bool(source.goals) or (source.actions[-1].op in {'pick','handover','place','throw','pour'})
                if home_required:
                    home_before = self.executor.world_state()
                    home=self.executor.home_empty_arms(emit=emit,should_stop=should_stop)
                    result['post_task_homing']=home
                    result['verification']['task_goal_before_homing']=True
                    if not home['success']:
                        result.update(success=False,status=home.get('status','FAILED'),
                            detail='作業の配置は維持していますが、空の腕のホーム復帰を完了できませんでした。')
                        result['verification'].update(success=False,failure_category='POST_TASK_HOME_FAILED')
                    else:
                        final=receipts.verify(self.executor.env,plan,epoch_initial)
                        result['verification']['after_homing_postconditions']=final
                        if not final['success']:
                            result.update(success=False,status='FAILED',detail='ホーム復帰後に作業の最終状態を確認できませんでした。')
                            result['verification'].update(success=False,failure_category='POST_TASK_SCENE_CHANGED')
                    if not result['success']:
                        phase = 'homing' if not home['success'] else 'post_homing_verification'
                        raw = home if phase == 'homing' else final
                        terminal = failure_observation(contract, receipts.epoch, iteration + 1,
                            {**result, 'failure_phase':phase, 'failure_evidence':raw},
                            home_before, self.executor.world_state())
                        evidence_ledger.append(terminal)
                        attempts[-1]['failure_evidence_id'] = terminal['id']
                        attempts[-1]['result'] = deepcopy(result)
                        announce({'kind':'execution_failed','explanation':f"{terminal['code']}。完了後の検査が失敗しました。",
                                  'failure_category':terminal['code'],'evidence_id':terminal['id'],
                                  'classification':terminal['classification']})
                return finish(result)
            if result.get('status')=='STOPPED':
                return finish(result)
            category = result.get('verification',{}).get('failure_category')
            failed = next((s for s in reversed(result.get('skills',[])) if not s['success']),{})
            category = failed.get('error') or category
            observed = self.executor.world_state()
            # The code names the layer that raised the failure; the mechanism names what the recorded
            # measurements show. Say both, so a reader does not chase 'IK_FAILURE' when the numbers say
            # the arm was blocked short of a commanded attitude (see verification/failure_attribution).
            mechanism = failure_receipt.get('mechanism')
            mechanism_reason = failure_receipt.get('mechanism_reason')
            explanation = f'{category}。'
            if mechanism and mechanism != 'UNKNOWN':
                explanation += f'計測からの機序は{mechanism}（{mechanism_reason}）。'
            explanation += '現在の保持・接触と残りの作業を再確認します。'
            announce({'kind':'execution_failed','explanation':explanation,
                'failure_category':category,'evidence_id':failure_receipt['id'],
                'classification':failure_receipt['classification'],
                'mechanism':mechanism,'mechanism_reason':mechanism_reason,
                'observed_hands':{s:observed.get('robot',{}).get(s+'_arm',{}).get('held_object') for s in ('left','right')}})
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
                    remaining = TaskProgram(actions=source.actions[receipts.completed:])
                    continuation = compile_prepared_program(remaining,observed)
                    # Retain every original final predicate, including objects
                    # placed before this suffix. The scene has not been reset.
                    continuation.goal = contract.verification_plan(continuation).goal
                    # First try useful progress. With no completed action,
                    # clear/reposition free arms before another approach.
                    # Resuming assumes the world still matches the plan. After a throw the object has left the hand
                    # and is moving or landed elsewhere, so the remaining steps would act on a stale picture:
                    # measured 2026-10-27, the resumed approach found the cube in the air at z=0.93 and reported a
                    # collision, and on another run approached where the cube no longer was.
                    object_left_the_hand = str(failed.get('skill') or '') in {'throw_object', 'release_object'}
                    repeats_the_same_contact = _repeats_measured_contact(mechanism, failure_receipt)
                    if (prefix > 0 and not object_left_the_hand and not repeats_the_same_contact
                            and not any(r.get('decision') == 'resume' for r in recoveries)):
                        legal.append('resume')
                    elif repeats_the_same_contact:
                        # The measurements show a contact the scene has not changed around; replaying the
                        # same suffix would drive the same arm through the same obstacle.
                        announce({'kind':'resume_withheld',
                                  'explanation':('同じ接触が再発する条件が変わっていないため、残り手順の再開を見送ります'
                                                 f'（機序: {mechanism}）。'),
                                  'mechanism':mechanism,'mechanism_reason':failure_receipt.get('mechanism_reason'),
                                  'evidence_id':failure_receipt['id']})
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
            # A second opinion, asked once per failure and only able to withhold: if it disagrees with the
            # rule table's mechanism, the scene is not restarted - an uncertain cause is not a licence to
            # throw the layout away. Every other recovery stays available.
            review = cross_check(failure_receipt.get('mechanism') or 'UNKNOWN',
                                 failure_receipt.get('raw'), self.mechanism_reviewer)
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
            offered, why_not = admit_reset(
                certificate if not receipts.receipts else None, review,
                legal_is_empty=not legal, reset_allowed=reset_allowed, resets=resets,
                max_resets=self.max_resets)
            if certificate is not None:
                # Why the irreversible option was or was not offered, recorded rather than discarded:
                # "no reset" and "reset refused for a contested mechanism" are different states.
                reset_decisions.append({'attempt': iteration + 1, 'offered': offered, 'reason': why_not})
            if review.get('verdict') != 'unavailable':
                mechanism_reviews.append({'attempt': iteration + 1, **review})
            if offered:
                legal.append('reset_and_retry')
            if not legal:
                if withholds(review, action='reset_and_retry') and certificate:
                    result['detail'] = (f'{category}: 機序の判定が一致しないため配置を維持して停止します'
                                        f'（rule={review["mechanism"]}, '
                                        f'reviewer={review["reviewer_mechanism"]}）。')
                else:
                    result['detail'] = f'{category}: 同じシーン内で利用できる復旧案を試しました。配置を維持して停止します。'
                return finish(result)
            assessment = self.recovery_call(instruction,observed,
                {'category':category,'detail':failed.get('detail') or result.get('detail'),
                 'completed_action_prefix':prefix,'attempt':iteration+1,'resets':resets,
                 'evidence_id':failure_receipt['id'],'failure_observation':deepcopy(failure_receipt)},legal)
            if assessment.get('decision') not in legal:
                default_reason = grounded_recovery_reason(legal[0], {'category': category, 'detail': failed.get('detail')})
                assessment = {'decision': legal[0], 'reason': default_reason, 'provider': 'deterministic_fallback'}
            else:
                assessment['reason'] = grounded_recovery_reason(assessment['decision'], {'category': category, 'detail': failed.get('detail')}, assessment.get('reason'))
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
                # The contract records unspecified ownership as a choice, not a mutable goal.
            elif assessment['decision']=='replan':
                replan_used=True
                try:
                    from backend.app.planning.agent_planner import compact_scene
                    # A replanned task starts from where the robot actually is: give it the hands and the
                    # compact scene, not only the failure, so a plan cannot assume an object is already held.
                    replanned=self.plan_call(instruction,observed,
                        [{'recent_failure':failed.get('detail'),'measured':failed.get('evidence'),
                          'evidence_id':failure_receipt['id'], 'failure_observation':deepcopy(failure_receipt),
                          'hands':{side:(observed.get('robot',{}).get(f'{side}_arm') or {}).get('held_object')
                                   for side in ('left','right')},
                          'scene':compact_scene(observed)}])
                except Exception:
                    replanned=None
                if replanned is None or not getattr(replanned,'steps',None):
                    result['detail']=f'{category}: 失敗の数値を踏まえた再計画を作れませんでした。'
                    return finish(result)
                plan=replanned
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
                        # Recover from user-action progress, never from compiler-inserted parking actions.
                        # Idempotent state repair may need the full request if no suffix remains; the
                        # contract gate still rejects replay of any already-executed event.
                        remaining_actions = source.actions[receipts.completed:]
                        if not remaining_actions:
                            remaining_actions = source.actions
                        remaining=TaskProgram(actions=remaining_actions)
                        plan=compile_prepared_program(remaining,observed)
                        plan.goal=contract.verification_plan(plan).goal
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
                receipts.reset_epoch()
                epoch_initial = self.executor.world_state()
                try:
                    plan = compile_prepared_program(source,epoch_initial)
                except GroundingError as error:
                    return blocked(str(error),'RECOVERY_PRECONDITION_FAILED')
            for field in ('provider','model','provider_error'):
                setattr(plan,field,getattr(original,field))
        return blocked('AGENT_ATTEMPT_BUDGET','AGENT_ATTEMPT_BUDGET')
