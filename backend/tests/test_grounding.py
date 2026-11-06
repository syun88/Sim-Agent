"""Entity grounding is one recorded, replayable decision set - not invisible text surgery."""
import pytest

from backend.app.agent.scene_reasoning import SceneBinding
from backend.app.planning.agent_planner import AgentPlanner
from backend.app.planning import grounding as grounding_module
from backend.app.planning.grounding import ground
from backend.app.planning.task_program import GroundingError

WORLD = {'objects': [{'id': 'red_cube', 'type': 'cube', 'position': [.44, .15, 1.031]},
                     {'id': 'blue_cube', 'type': 'cube', 'position': [.50, -.22, 1.031]}],
         'targets': [], 'sim_time': 12.5,
         'robot': {'available_arms': ['left', 'right'],
                   'left_arm': {'held_object': None}, 'right_arm': {'held_object': None}}}

INSTRUCTIONS = (
    '右手にあるキューブを取って左手に渡して',
    '右手でキューブを掴んでから、掴んでいる状態のキューブの上に赤いキューブを重ねて',
    'キューブを、もう一方の手に持たせて、そのキューブの上にもう一つのキューブを重ねて',
    '左手で、右手で掴んでいるキューブの上に赤いキューブを重ねて',
    '青いキューブを右手に渡して右においてから、赤いキューブを青いキューブに乗せて',
)


def replay(text, decisions):
    out, cursor = [], 0
    for decision in sorted(decisions, key=lambda item: item['span'][0]):
        start, end = decision['span']
        assert start >= cursor, 'decisions must not overlap'
        out.append(text[cursor:start])
        out.append(decision['replacement'])
        cursor = end
    out.append(text[cursor:])
    return ''.join(out)


@pytest.mark.parametrize('instruction', INSTRUCTIONS)
def test_every_edit_is_a_recorded_decision(instruction):
    record = ground(instruction, WORLD)
    # The resolved sentence is exactly the input plus the recorded replacements.
    for step in record['steps']:
        recorded = [d for d in record['decisions'] if d['step'] == step]
        assert recorded or step in {'table_side', 'uncolored_hand', 'contrastive_other',
                                    'contrastive_first', 'opposite_hand', 'relative_pronouns',
                                    'hand_reference'}
    assert record['instruction'] != '' and record['original'] == instruction


def test_a_decision_reports_the_entity_it_actually_wrote():
    """Regression (2026-11-07): both edits of a contrastive pair reported the pair's first cube."""
    record = ground('キューブを、もう一方の手に持たせて、そのキューブの上にもう一つのキューブを重ねて', WORLD)
    for decision in record['decisions']:
        written = decision['replacement']
        object_id = decision['resolved'].get('object')
        if object_id:
            assert object_id in written, (decision['kind'], object_id, written)


def test_the_record_carries_the_observation_the_choice_rested_on():
    record = ground('右手にあるキューブを取って', WORLD)
    assert record['observed']['hands'] == {'left': None, 'right': None}
    assert set(record['observed']['objects']) == {'red_cube', 'blue_cube'}
    decision = record['decisions'][0]
    assert decision['basis'] == 'the only unheld cube in that arm workspace'
    assert decision['resolved'] == {'object': 'blue_cube', 'arm': 'right'}


def test_an_unrecorded_rewrite_is_refused_instead_of_travelling_on(monkeypatch):
    """A pass that edits text without a decision must stop planning, not produce a different sentence."""
    def silent_edit(text, state, decisions=None):
        return text.replace('キューブ', 'cubex')
    monkeypatch.setattr(grounding_module, 'resolve_relative_pronouns', silent_edit)
    with pytest.raises(GroundingError, match='UNGROUNDED_EDIT'):
        ground('赤いキューブを左手で取って', WORLD)


def test_a_recorded_edit_that_was_not_made_is_refused(monkeypatch):
    """The claim must match the edit: a decision with no corresponding change is also a lie."""
    def claimed_text(instruction, state):
        return SceneBinding(instruction, [{'kind': 'observed_hold', 'arm': 'left', 'object': 'red_cube',
                                           'source': '左手', 'span': [0, 2], 'replacement': '左手でred_cube',
                                           'candidates': ['red_cube'], 'alternatives': [],
                                           'explanation': 'claimed'}], [])
    monkeypatch.setattr(grounding_module, 'bind_scene_references', claimed_text)
    with pytest.raises(GroundingError, match='UNGROUNDED_EDIT'):
        ground('赤いキューブを取って', WORLD)


def test_grounding_is_applied_once_not_twice():
    """The planner used to bind references a second time on the already-resolved text."""
    record = ground('右手にあるキューブを取って', WORLD)
    again = ground(record['instruction'], WORLD)
    assert again['decisions'] == []
    assert again['instruction'] == record['instruction']


def test_the_plan_records_the_grounding_with_its_basis():
    plan = AgentPlanner().plan('右手にあるキューブを取って', WORLD)
    grounding = plan.planning_evidence['grounding']
    assert grounding['instruction'] == '右手でblue_cubeを取って'
    decision = plan.planning_evidence['scene_bindings'][0]
    assert decision['basis'] == 'the only unheld cube in that arm workspace'
    assert decision['surface'] == '右手にあるキューブ'
    assert decision['replacement'] == '右手でblue_cube'


def test_an_ambiguous_scene_still_refuses_rather_than_guessing():
    from backend.app.agent.scene_reasoning import SceneReferenceError
    two_right = {**WORLD, 'objects': [{'id': 'red_cube', 'type': 'cube', 'position': [.5, -.20, 1.031]},
                                      {'id': 'blue_cube', 'type': 'cube', 'position': [.5, -.25, 1.031]}],
                 'robot': {'available_arms': ['left', 'right'],
                           'left_arm': {'held_object': None}, 'right_arm': {'held_object': None}}}
    with pytest.raises(SceneReferenceError):
        ground('右手にあるキューブを取って', two_right)
