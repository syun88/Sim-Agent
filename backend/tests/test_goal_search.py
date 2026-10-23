from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError

from backend.app.planning.agent_planner import AgentPlanner, validate_plan
from backend.app.planning.goal_search import GoalDomain, action_key, compile_goal_program, goal_holds
from backend.app.planning.language_fallback import parse_fallback
from backend.app.planning.task_program import ActionProposal, GroundingError, InstructionMeaning, TaskAction, TaskGoal, TaskProgram, compile_program


def set_pose(world, name, position):
    obj = next(item for item in world['objects'] if item['id'] == name)
    obj.update(position=list(position), bounds=[[v-.023 for v in position], [v+.023 for v in position]])


@pytest.fixture
def world():
    result = {'objects': [{'id': 'red_cube', 'type': 'cube'}, {'id': 'blue_cube', 'type': 'cube'}],
        'targets': [{'id': 'center', 'position': [.50, 0., 1.008]},
                    {'id': 'left_side', 'position': [.46, .30, 1.008]},
                    {'id': 'right_side', 'position': [.46, -.30, 1.008]}],
        'workspace': {'table_height_m': 1.008}, 'support_relations': [],
        'robot': {'available_arms': ['left', 'right'], 'left_arm': {'held_object': None}, 'right_arm': {'held_object': None}}}
    for name, position in [('red_cube', [.44, .18, 1.031]), ('blue_cube', [.50, -.22, 1.031])]:
        set_pose(result, name, position)
    for obj in result['objects']:
        obj.update(rotation=[[1,0,0],[0,1,0],[0,0,1]], touching=[], held_by=[], linear_speed_m_s=0.)
    return result


def apply(world, action):
    """Independent small reference interpreter, not the search transition code."""
    arm, name = action['arm'], action['object']
    hand = world['robot'][f'{arm}_arm']
    if action['op'] == 'home':
        assert hand['held_object'] is None
        ready = world['robot'].setdefault('handover_ready', [])
        if arm not in ready:
            ready.append(arm)
        return world
    world['robot']['handover_ready'] = [a for a in world['robot'].get('handover_ready', []) if a != arm]
    obj = next(x for x in world['objects'] if x['id'] == name)
    if action['op'] == 'pick':
        assert hand['held_object'] in (None, name)
        assert not any(world['robot'][f'{other}_arm']['held_object'] == name for other in ('left', 'right') if other != arm)
        hand['held_object'] = name
        obj['held_by'] = [arm]
        set_pose(world, name, [*obj['position'][:2], 1.15])
        world['support_relations'] = [r for r in world['support_relations'] if r['subject'] != name]
    elif action['op'] == 'handover':
        receiver = world['robot'][f"{action['receiver']}_arm"]
        assert hand['held_object'] == name and receiver['held_object'] is None
        hand['held_object'], receiver['held_object'] = None, name
        obj['held_by'] = [action['receiver']]
    else:
        assert action['op'] == 'place' and hand['held_object'] == name
        assert action['relation'] is None  # Relative physics is separately tested in Genesis.
        target = next(x['position'] for x in world['targets'] if x['id'] == action['target'])
        set_pose(world, name, [*target[:2], target[2]+.023])
        hand['held_object'] = None
        obj['held_by'] = []
    return world


def swapped_goals():
    return TaskProgram(goals=[TaskGoal(type='holding', object='blue_cube', arm='left'),
                              TaskGoal(type='holding', object='red_cube', arm='right')])


def occupied_world(world):
    apply(world, TaskAction(op='pick', object='red_cube', arm='left').model_dump())
    apply(world, TaskAction(op='pick', object='blue_cube', arm='right').model_dump())
    return world


def test_goal_search_frees_hands_where_direct_handover_is_invalid(world):
    occupied_world(world)
    with pytest.raises(GroundingError, match='cannot receive'):
        compile_program(TaskProgram(actions=[TaskAction(op='handover', object='blue_cube', arm='right', receiver='left')]), world)
    program = swapped_goals()
    plan = compile_program(program, world)
    valid, errors = validate_plan(plan, world)
    assert valid, errors
    assert any(a['op'] == 'place' for a in plan.program['actions'])
    observed = deepcopy(world)
    for action in plan.program['actions']:
        apply(observed, action)
    assert all(goal_holds(goal, observed) for goal in program.goals)
    assert plan.planning_evidence['physical_execution_verified'] is False


def test_reobserving_each_action_shrinks_the_remaining_plan(world):
    occupied_world(world)
    initial = deepcopy(world)
    program = swapped_goals()
    lengths = []
    for _ in range(12):
        plan = compile_goal_program(program, world, initial_state=initial)
        lengths.append(len(plan.program['actions']))
        if not plan.program['actions']:
            break
        apply(world, plan.program['actions'][0])
    assert lengths[-1] == 0 and lengths[0] > lengths[-1]
    assert all(goal_holds(goal, world) for goal in program.goals)


def test_goal_already_true_observes_without_replaying_a_pick(world):
    apply(world, TaskAction(op='pick', object='blue_cube', arm='right').model_dump())
    plan = compile_program(TaskProgram(goals=[TaskGoal(type='holding', object='blue_cube', arm='right')]), world)
    assert [step.skill for step in plan.steps] == ['observe_scene']
    assert plan.program['actions'] == []


def test_table_pinch_still_requires_an_actual_lift(world):
    world['robot']['right_arm']['held_object'] = 'blue_cube'
    plan = compile_program(TaskProgram(goals=[TaskGoal(type='holding', object='blue_cube', arm='right')]), world)
    assert [step.skill for step in plan.steps] == ['lift_object']


def test_rejected_pick_transition_selects_another_hand_without_reset(world):
    source = TaskProgram(goals=[TaskGoal(type='holding', object='red_cube', arm='left')])
    domain = GoalDomain(source, world)
    failed = TaskAction(op='pick', object='red_cube', arm='left')
    plan = compile_goal_program(source, world, blocked={(domain.project(world), action_key(failed))})
    assert next(a for a in plan.program['actions'] if a['op'] == 'pick')['arm'] == 'right'
    for action in plan.program['actions']:
        apply(world, action)
    assert goal_holds(source.goals[0], world)


def test_unmentioned_objects_are_not_moved_to_make_a_hand_free(world):
    occupied_world(world)
    source = TaskProgram(goals=[TaskGoal(type='holding', object='blue_cube', arm='left')])
    with pytest.raises(GroundingError, match='GOAL_UNREACHABLE'):
        compile_program(source, world)


def test_preserved_object_is_compared_with_the_original_snapshot(world):
    source = TaskProgram(goals=[TaskGoal(type='holding', object='blue_cube', arm='right')])
    initial = deepcopy(world)
    set_pose(world, 'red_cube', [.7, .2, 1.031])
    with pytest.raises(GroundingError, match='PRESERVATION_VIOLATION'):
        compile_goal_program(source, world, initial_state=initial)


def test_keep_in_place_cannot_be_ignored_to_satisfy_a_goal(world):
    source = TaskProgram(goals=[TaskGoal(type='holding', object='blue_cube', arm='right')], keep_in_place=['blue_cube'])
    with pytest.raises(GroundingError, match='GOAL_UNREACHABLE'):
        compile_program(source, world)


def test_support_must_be_cleared_before_the_lower_cube_is_picked(world):
    world['support_relations'] = [{'subject': 'blue_cube', 'object': 'red_cube', 'relation': 'on'}]
    source = TaskProgram(goals=[TaskGoal(type='holding', object='blue_cube', arm='right'),
                               TaskGoal(type='placement', object='red_cube', target='center')])
    plan = compile_program(source, world)
    assert next(a for a in plan.program['actions'] if a['object'])['object'] == 'blue_cube'


@pytest.mark.parametrize('goals', [
    [{'type':'holding','object':'red_cube','arm':'left'}, {'type':'holding','object':'blue_cube','arm':'left'}],
    [{'type':'holding','object':'red_cube','arm':'left'}, {'type':'placement','object':'red_cube','target':'center'}],
    [{'type':'holding','object':'unknown_cube','arm':'left'}],
])
def test_contradictory_or_unknown_goals_fail_before_motion(world, goals):
    with pytest.raises(GroundingError):
        compile_program(TaskProgram(goals=goals), world)


def test_goal_search_is_bounded(world):
    occupied_world(world)
    with pytest.raises(GroundingError, match='after 1 states'):
        compile_goal_program(swapped_goals(), world, max_states=1)


def test_schema_cannot_mix_ordered_actions_and_unordered_goals():
    with pytest.raises(ValidationError):
        TaskProgram(actions=[TaskAction(op='home')], goals=[TaskGoal(type='holding', object='red_cube', arm='left')])


def test_model_envelope_has_one_array_with_an_explicit_interpretation():
    proposal = ActionProposal(mode='actions', items=[{'op':'pick','object':'blue_cube','arm':'left'}])
    program = proposal.task_program()
    assert program.actions and not program.goals
    with pytest.raises(ValidationError):
        ActionProposal(mode='actions', items=[{'type':'holding','object':'blue_cube','arm':'left'}])
    with pytest.raises(ValidationError):
        ActionProposal.model_validate({'mode':'actions','items':[], 'goals':[{'type':'holding'}]})
    with pytest.raises(ValueError):
        ActionProposal(mode='error', items=proposal.items, error='No').task_program()


def test_meaning_stage_cannot_silently_erase_a_procedural_request():
    assert InstructionMeaning(interpretation='ordered_actions').goals == []
    with pytest.raises(ValidationError):
        InstructionMeaning(interpretation='ordered_actions', goals=swapped_goals().goals)
    with pytest.raises(ValidationError):
        InstructionMeaning(interpretation='final_state')


def test_literal_target_holders_override_model_copying_current_ownership(world, monkeypatch):
    occupied_world(world)
    monkeypatch.setenv('OPENARM_LOCAL_PLANNER', '1')
    for key in ('AI_AND_API_KEY', 'AI_AND_BASE_URL', 'AI_AND_MODEL'):
        monkeypatch.delenv(key, raising=False)
    planner = AgentPlanner()
    wrong = TaskProgram(goals=[TaskGoal(type='holding',object='red_cube',arm='left'),
                              TaskGoal(type='holding',object='blue_cube',arm='right')])
    monkeypatch.setattr(planner, '_ai_plan', lambda *args: {'payload':wrong.model_dump(),'raw':'wrong source roles'})
    plan = planner.plan('Blue should finish in the left gripper and red in the right.', world)
    assert {(g['object'],g['arm']) for g in plan.goal_program['goals']} == {('blue_cube','left'),('red_cube','right')}
    assert len(plan.planning_evidence['semantic_binding_repairs']) == 2
    assert plan.program['actions']  # Must not return a no-op based on current ownership.


def test_final_holder_constraints_do_not_turn_ordered_intermediate_holders_into_goals(world):
    from backend.app.planning.goal_language import holding_conditions
    assert not holding_conditions('Pick blue with the right hand, then pass blue to the left hand.', world)


@pytest.mark.parametrize('text', ['最終的に左手に赤、右手に青を持たせて', '最後は左手に赤、右手に青を持っていて'])
def test_final_state_fallback_preserves_japanese_case_particles(world, text):
    program = parse_fallback(text, world)
    assert {(g.object,g.arm) for g in program.goals} == {('red_cube','left'),('blue_cube','right')}


@pytest.mark.parametrize('text', ['完成形は青が赤の上にある状態。赤の位置は変えないで。',
                                'Blue should be on top of red. Keep red in place.'])
def test_literal_relative_goal_and_preserved_reference_are_separated(world, text):
    from backend.app.planning.goal_language import relative_conditions
    from backend.app.planning.spatial_language import preservation_spans
    assert relative_conditions(text,world) == [{'type':'relative_placement','object':'blue_cube','target':'red_cube','relation':'on'}]
    assert {name for _,_,name in preservation_spans(text,world)} == {'red_cube'}


def test_group_request_cannot_be_silently_replaced_by_one_pick(world, monkeypatch):
    from backend.app.planning.agent_planner import PlannerProviderError
    planner = AgentPlanner()
    monkeypatch.setattr(planner, '_ai_plan', lambda *args: pytest.fail('Unsupported grouping must be checked before model planning'))
    with pytest.raises(PlannerProviderError) as error:
        planner.plan('積んだ二つを同時に一つの手で掴んで渡して', world)
    assert error.value.category == 'UNSUPPORTED_TASK'


def test_using_both_hands_is_not_mistaken_for_grasping_two_objects_at_once():
    from backend.app.planning.goal_language import unsupported_group_request
    assert not unsupported_group_request('Use both hands together: pick blue with the right, then pass it to the left.')
    assert not unsupported_group_request('両方の手を一緒に使って青を取って左に渡して')
    assert unsupported_group_request('Pick both cubes together with one hand')


def test_full_exchange_request_uses_measured_objects_not_fixed_colour_assignment(world):
    occupied_world(world)
    program = parse_fallback('両手のキューブを入れ替えて', world)
    assert {(g.object, g.arm) for g in program.goals} == {('blue_cube','left'), ('red_cube','right')}
    with pytest.raises(GroundingError):
        parse_fallback('両手のキューブを入れ替えて、そのあと未知の作業をして', world)


def test_free_language_model_can_request_goals_instead_of_guessing_buffer_steps(world, monkeypatch):
    occupied_world(world)
    monkeypatch.setenv('OPENARM_LOCAL_PLANNER', '1')
    for key in ('AI_AND_API_KEY', 'AI_AND_BASE_URL', 'AI_AND_MODEL'):
        monkeypatch.delenv(key, raising=False)
    planner = AgentPlanner()
    monkeypatch.setattr(planner, '_ai_plan', lambda *args: {'payload': swapped_goals().model_dump(), 'raw': 'goal program'})
    plan = planner.plan('I want each cube to finish in the opposite hand; work out the intermediate steps.', world)
    assert plan.provider == 'local_ollama' and plan.goal_program
    assert any(a['op'] == 'place' for a in plan.program['actions'])


class ObservedExecutor:
    """Control-flow fixture. Physical acceptance is covered by the Genesis script."""
    def __init__(self, world, fail_picks=0):
        self.world, self.fail_picks = deepcopy(world), fail_picks
        self.calls = []
        self.env = SimpleNamespace(table_height=1.008, object_contacts=lambda *args: [],
            object_bounds=lambda name: np.asarray(next(x['bounds'] for x in self.world['objects'] if x['id'] == name)),
            object_linear_speed=lambda name: 0.)

    def world_state(self):
        return deepcopy(self.world)

    def run(self, plan, **kwargs):
        self.calls.append(plan)
        actions = plan.program['actions']
        failed = bool(actions and actions[0]['op'] == 'pick' and self.fail_picks)
        if failed:
            self.fail_picks -= 1
        elif actions:
            apply(self.world, actions[0])
        verified = not failed
        if not actions:
            for check in plan.goal['checks']:
                if check['type'] == 'preserved':
                    pos = next(x['position'] for x in self.world['objects'] if x['id'] == check['object'])
                    verified = verified and np.linalg.norm(np.asarray(pos)-check['position']) < .006
                else:
                    verified = verified and goal_holds(TaskGoal.model_validate(check), self.world)
        return {'success': verified, 'status': 'SUCCEEDED' if verified else 'FAILED',
                'retry_count': 0, 'verification': {'success': verified, 'failure_category': 'GRASP_FAILED' if failed else None},
                'skills': [{'skill': plan.steps[0].skill, 'success': not failed, 'error': 'GRASP_FAILED' if failed else None}]}


def test_goal_executor_uses_new_observations_and_independent_final_check(world):
    from backend.app.execution.goal_executor import run_goal_program
    occupied_world(world)
    executor = ObservedExecutor(world)
    result = run_goal_program(executor, compile_program(swapped_goals(), world))
    assert result['success']
    assert len(result['verification']['replanning_trace']) > 1
    assert executor.calls[-1].steps[0].skill == 'observe_scene'
    assert all(goal_holds(g, executor.world) for g in swapped_goals().goals)


def test_recovered_physical_attempt_remains_in_the_result_ledger(world):
    from backend.app.execution.goal_executor import run_goal_program
    program = TaskProgram(goals=[TaskGoal(type='holding', object='red_cube', arm='left')])
    executor = ObservedExecutor(world, fail_picks=1)
    result = run_goal_program(executor, compile_program(program, world))
    assert result['success'] and result['retry_count'] == 1
    assert result['verification']['all_attempts_succeeded'] is False
    assert result['verification']['execution_attempts'][0]['result']['success'] is False
    assert any(not step['success'] for step in result['skills'])
    assert goal_holds(program.goals[0], executor.world)


def test_unrecoverable_goal_attempt_cannot_become_success(world):
    from backend.app.execution.goal_executor import run_goal_program
    program = TaskProgram(goals=[TaskGoal(type='holding', object='red_cube', arm='left')])
    executor = ObservedExecutor(world, fail_picks=10)
    result = run_goal_program(executor, compile_program(program, world))
    assert not result['success']
    assert result['verification']['failure_category'] == 'GOAL_UNREACHABLE'
    assert sum(bool(p.program['actions']) and p.program['actions'][0]['op'] == 'pick' for p in executor.calls) == 2
    assert len(executor.calls) <= 4


def test_stop_before_goal_execution_does_not_move_or_verify_success(world):
    from backend.app.execution.goal_executor import run_goal_program
    executor = ObservedExecutor(world)
    result = run_goal_program(executor, compile_program(swapped_goals(), world), should_stop=lambda: True)
    assert result['status'] == 'STOPPED' and not result['success']
    assert executor.calls == []


def test_goal_macro_preserves_pick_geometry_for_a_later_handover(world):
    from backend.app.execution.goal_executor import run_goal_program
    source = TaskProgram(goals=[TaskGoal(type='relative_placement', object='blue_cube', target='red_cube', relation='on')])
    executor = ObservedExecutor(world)
    run_goal_program(executor, compile_program(source, world), should_stop=lambda: len(executor.calls) >= 2)
    approach = next(step for call in executor.calls for step in call.steps if step.skill == 'approach_object')
    assert approach.grasp_offset_x == -.015


def test_task_frame_permission_does_not_relax_hold_or_preservation_checks(world):
    from backend.app.models.schemas import AgentPlan, PlanStep
    from backend.app.verification.goal_verification import verify_task_goal
    initial = deepcopy(world)
    positions = {'blue_cube': np.array([.5, -.2, 1.2]), 'red_cube': np.array([.44, .18, 1.2])}
    pinch = {'left': True, 'right': True}
    env = SimpleNamespace(name='test-observation', grasp_mode='contact-test', table_height=1.008,
        step=lambda *args: None, object_pose=lambda name:(positions[name],np.eye(3)),
        object_contacts=lambda *args:['pad'], pinch_established=lambda obj, arm:pinch[arm],
        object_bounds=lambda obj:np.stack([positions[obj]-.023,positions[obj]+.023]))
    plan = AgentPlan(goal={'type':'goal_program', 'allowed_object_motion':['blue_cube','red_cube'],
        'checks':[{'type':'holding','object':'blue_cube','arm':'left'}, {'type':'holding','object':'red_cube','arm':'right'}]},
        steps=[PlanStep(skill='observe_scene')])
    results = [{'success':True}]
    assert verify_task_goal(env,plan,results,initial_state=initial)['success']
    pinch['right'] = False
    assert not verify_task_goal(env,plan,results,initial_state=initial)['success']
    pinch['right'] = True
    plan.goal['checks'].append({'type':'preserved','object':'red_cube','position':initial['objects'][0]['position']})
    assert not verify_task_goal(env,plan,results,initial_state=initial)['success']
