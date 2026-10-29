"""Instruction relations must survive every provider and reach the success oracle."""
import pytest

from backend.app.planning.agent_planner import AgentPlanner, PlannerProviderError


WORLD = {'objects': [{'id': 'red_cube', 'position': [.44, .15, 1.031]},
                     {'id': 'blue_cube', 'position': [.5, -.22, 1.031]}], 'targets': []}


@pytest.fixture(autouse=True)
def no_provider(monkeypatch):
    for key in ('AI_AND_API_KEY', 'AI_AND_BASE_URL', 'AI_AND_MODEL',
                'OPENARM_LOCAL_PLANNER', 'OPENARM_FAST_LANGUAGE', 'OPENARM_CAPABILITY_MODE'):
        monkeypatch.delenv(key, raising=False)


def test_unresolved_support_never_becomes_a_table_side():
    # Reported 2026-10-29: this request became pick red + place blue at right_side, SUCCEEDED.
    # A held support is currently unsupported; refusing is honest, substituting is not.
    with pytest.raises(PlannerProviderError) as error:
        AgentPlanner().plan('赤いキューブを左手で持ち上げて、その上に右手で青いキューブを重ねて', WORLD)
    assert error.value.category == 'ROBOT_CAPABILITY'
    assert 'held reference cube' in str(error.value)


def test_rejected_fallback_is_not_resurrected_when_the_model_fails(monkeypatch):
    planner = AgentPlanner()
    planner.enabled = planner.local_enabled = True
    planner.capability_mode = True
    monkeypatch.setattr(planner, '_ai_plan', lambda *args: {
        'payload': {'actions': [{'op': 'place', 'object': 'blue_cube', 'target': 'right_side'}]},
        'raw': 'lost the support'})
    with pytest.raises(PlannerProviderError):
        planner.plan('赤いキューブを左手で持ち上げて、その上に右手で青いキューブを重ねて', WORLD)


@pytest.mark.parametrize('support,subject,word,relation', [
    ('赤', '青', '上', 'on'), ('青', '赤', '左', 'left_of'),
    ('赤', '青', '右隣', 'right_of'), ('青', '赤', '手前', 'front_of'),
    ('赤', '青', '奥', 'behind'), ('青', '赤', 'そば', 'near'),
])
def test_relative_pronouns_preserve_arbitrary_support_and_relation(support,subject,word,relation):
    from backend.app.planning.spatial_language import placement_references
    text = f'{support}いキューブを持ち上げて、その{word}に{subject}いキューブを置いて'
    refs = placement_references(text, WORLD)
    expected = 'red_cube' if support == '赤' else 'blue_cube'
    assert [(r['target'], r['relation']) for r in refs] == [(expected, relation)]


def test_model_cannot_put_down_a_required_held_support_to_make_the_task_feasible(monkeypatch):
    planner = AgentPlanner()
    planner.enabled = planner.local_enabled = True
    planner.capability_mode = True
    monkeypatch.setattr(planner, '_ai_plan', lambda *args: {'payload': {'actions': [
        {'op': 'pick', 'object': 'red_cube', 'arm': 'left'},
        {'op': 'place', 'object': 'red_cube', 'arm': 'left', 'target': 'left_side'},
        {'op': 'place', 'object': 'blue_cube', 'arm': 'right', 'target': 'red_cube', 'relation': 'on'}]},
        'raw': 'put the held support down first'})
    with pytest.raises(PlannerProviderError):
        planner.plan('赤いキューブを左手で持ち上げて、その上に右手で青いキューブを重ねて', WORLD)


def test_hand_reference_is_not_an_object_relative_position():
    from backend.app.planning.spatial_language import placement_references
    assert placement_references('青を右手で取って、その左手にある赤を掴んで', WORLD) == []


def test_ambiguous_antecedent_is_not_guessed():
    from backend.app.planning.spatial_language import placement_references
    from backend.app.planning.task_program import GroundingError
    with pytest.raises(GroundingError, match='Unresolved'):
        placement_references('赤と青を持ち上げて、その上に置いて', WORLD)


@pytest.mark.parametrize('text', ['赤と青を重ねて', 'Stack red and blue'])
def test_support_free_stacking_never_means_two_table_placements(text):
    with pytest.raises(PlannerProviderError):
        AgentPlanner().plan(text, WORLD)


def test_intermediate_relations_need_not_remain_true_at_completion():
    from backend.app.planning.spatial_language import validate_instruction_relations
    from backend.app.planning.task_program import compile_program, TaskProgram
    plan = compile_program(TaskProgram.model_validate({'actions': [
        {'op': 'place', 'object': 'blue_cube', 'target': 'red_cube', 'relation': 'left_of'},
        {'op': 'place', 'object': 'blue_cube', 'target': 'red_cube', 'relation': 'right_of'}]}), WORLD)
    validate_instruction_relations(plan, 'Place blue to the left of red, then place blue to the right of red', WORLD)


def test_explicit_level_roles_are_resolved_supports():
    from backend.app.planning.spatial_language import validate_instruction_relations
    from backend.app.planning.task_program import compile_program, TaskProgram
    plan = compile_program(TaskProgram.model_validate({'goals': [
        {'type': 'relative_placement', 'object': 'blue_cube', 'target': 'red_cube', 'relation': 'on'}]}), WORLD)
    validate_instruction_relations(plan, '赤が下、青が上になるように重ねて', WORLD)


@pytest.mark.parametrize('text', [
    'Do not stack red and blue, pick blue',
    '赤と青を重ねないで、青を取って',
    '赤には触らないで、青を重ねずに右側に置いて',
])
def test_prohibiting_support_does_not_require_a_positive_support_goal(text):
    from backend.app.planning.spatial_language import validate_instruction_relations
    from backend.app.planning.task_program import TaskProgram, compile_program
    plan = compile_program(TaskProgram.model_validate({'actions': [
        {'op': 'pick', 'object': 'blue_cube', 'arm': 'right'}]}), WORLD)
    validate_instruction_relations(plan, text, WORLD)


def test_success_oracle_must_contain_the_requested_relation():
    from backend.app.planning.spatial_language import validate_instruction_relations
    from backend.app.planning.task_program import compile_program, TaskProgram, GroundingError
    plan = compile_program(TaskProgram.model_validate({'actions': [
        {'op': 'place', 'object': 'blue_cube', 'target': 'red_cube', 'relation': 'on'}]}), WORLD)
    # The path may still stack correctly; a success oracle checking a table side is not admissible.
    plan.goal['checks'] = [{'type': 'placement', 'object': 'blue_cube', 'target': 'right_side'}]
    with pytest.raises(GroundingError, match='success'):
        validate_instruction_relations(plan, '青を赤の上に置いて', WORLD)
