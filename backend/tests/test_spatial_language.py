"""The support noun is not a second task; negations never become actions."""
import pytest

from backend.app.planning.agent_planner import AgentPlanner, PlannerProviderError
from backend.app.planning.spatial_language import validate_instruction_relations
from backend.app.planning.task_program import TaskProgram, TaskAction, compile_program, GroundingError

WORLD = {'objects': [{'id': 'red_cube', 'position': [.44, .15, 1.031]},
                     {'id': 'blue_cube', 'position': [.5, -.22, 1.031]}], 'targets': []}
REQUEST = '右手で青いキューブを取って左手に渡して左側の赤いキューブの上に乗せて'


@pytest.fixture(autouse=True)
def no_provider(monkeypatch):
    for key in ('AI_AND_API_KEY', 'AI_AND_BASE_URL', 'AI_AND_MODEL', 'OPENARM_LOCAL_PLANNER', 'OPENARM_FAST_LANGUAGE'):
        monkeypatch.delenv(key, raising=False)


def test_user_request_preserves_three_actions_and_red_as_support():
    p = AgentPlanner().plan(REQUEST, WORLD)
    assert [(a['op'], a['object']) for a in p.program['actions']] == [
        ('pick', 'blue_cube'), ('handover', 'blue_cube'), ('place', 'blue_cube')]
    assert p.program['actions'][-1]['relation'] == 'on'
    assert p.program['actions'][-1]['target'] == 'red_cube'
    assert p.goal['checks'] == [{'type': 'relative_placement', 'object': 'blue_cube', 'target': 'red_cube',
                                'relation': 'on', 'distance_cm': None, 'arm': 'left'}]


@pytest.mark.parametrize('text,target,relation', [
    ('青を赤の上に置いて', 'red_cube', 'on'),
    ('青い箱を赤い箱に重ねて', 'red_cube', 'on'),
    ('赤を青の上に載せて', 'blue_cube', 'on'),
    ('赤の左に青を置いて', 'red_cube', 'left_of'),
    ('青を赤の右隣に置いて', 'red_cube', 'right_of'),
    ('青を赤の手前に置いて', 'red_cube', 'front_of'),
    ('赤の奥に青を置いて', 'red_cube', 'behind'),
    ('Put blue on top of red', 'red_cube', 'on'),
    ('Place red next to blue', 'blue_cube', 'near'),
    ('Place blue to the left of red', 'red_cube', 'left_of'),
])
def test_object_relative_arguments_survive_paraphrases(text, target, relation):
    p = AgentPlanner().plan(text, WORLD)
    assert p.program['actions'][-1]['target'] == target
    assert p.program['actions'][-1]['relation'] == relation
    assert all(action['object'] != target for action in p.program['actions'])


def test_negative_handover_is_not_executed_as_an_affirmative_handover():
    p = AgentPlanner().plan('Pick the red cube with the left hand and keep holding it. Do not pass it.', WORLD)
    assert [action['op'] for action in p.program['actions']] == ['pick']


def test_leave_support_in_place_is_not_an_extra_place_command():
    p = AgentPlanner().plan('Put blue on top of red. Leave red where it is.', WORLD)
    assert len(p.program['actions']) == 1
    assert p.program['actions'][0]['object'] == 'blue_cube'


def test_model_cannot_reuse_the_old_unrelated_two_handover_example(monkeypatch):
    planner = AgentPlanner()
    planner.enabled = planner.local_enabled = True
    wrong = {'actions': [{'op': 'pick', 'object': 'blue_cube', 'arm': 'right'},
                         {'op': 'handover', 'object': 'it', 'arm': 'right', 'receiver': 'left'},
                         {'op': 'place', 'object': 'it', 'arm': 'left', 'target': 'left_side'},
                         {'op': 'pick', 'object': 'red_cube', 'arm': 'left'}]}
    monkeypatch.setattr(planner, '_ai_plan', lambda *args: {'payload': wrong, 'raw': 'old unrelated example'})
    plan = planner.plan(REQUEST, WORLD)
    assert plan.provider_error == 'PROVIDER_INVALID_PROGRAM'
    assert len(plan.program['actions']) == 3
    assert plan.program['actions'][-1]['relation'] == 'on'


def test_directional_next_to_is_a_valid_refinement_of_near():
    p = compile_program(TaskProgram(actions=[TaskAction(op='place', object='red_cube', target='blue_cube', relation='left_of')]), WORLD)
    validate_instruction_relations(p, 'Put the red cube next to the blue one, on its left.', WORLD)


def test_a_held_reference_cannot_be_used_as_a_stationary_support():
    state = {**WORLD, 'robot': {'left_arm': {'held_object': 'red_cube'}}}
    with pytest.raises(GroundingError, match='reference cube'):
        compile_program(TaskProgram(actions=[TaskAction(op='place', object='blue_cube', target='red_cube', relation='on')]), state)


def test_physical_handover_and_stack(env):
    from backend.app.execution.task_executor import TaskExecutor
    executor = TaskExecutor(env)
    env.step(200)
    plan = AgentPlanner().plan(REQUEST, executor.world_state())
    result = executor.run(plan)
    assert result['success'], result
    stack = next(c for c in result['verification']['checks'] if c['type'] == 'relative_placement')
    assert stack['support_contact'] and stack['released'] and stack['stable']
    assert stack['supported_fraction'] > .8
    assert abs(stack['support_gap']) < .004
    assert next(c for c in result['verification']['checks'] if c['type'] == 'undisturbed')['displacement'] < .006
