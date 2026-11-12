"""A composite task is a file. What it produces is still only a proposal for the same verifier."""
import json

import pytest

from backend.app.planning import task_templates as templates
from backend.app.planning.task_program import ActionProposal

STATE = {
    'objects': [{'id': 'red_cube'}, {'id': 'blue_cube'}],
    'targets': [{'id': 'blue_zone'}],
    'robot': {'available_arms': ['left', 'right']},
}


def _write(tmp_path, payload):
    (tmp_path / f'{payload["id"]}.json').write_text(json.dumps(payload, ensure_ascii=False))
    return tmp_path


def test_the_shipped_procedures_expand_into_ordered_actions():
    entries = templates.load()
    assert 'handover_then_place' in entries
    expansion = templates.expand(entries['handover_then_place'],
                                 {'object': 'red_cube', 'zone': 'blue_zone', 'giver': 'left', 'receiver': 'right'},
                                 STATE)
    # The expansion is a proposal envelope, so the ordinary compiler and contract still apply.
    program = ActionProposal.model_validate(expansion['envelope']).task_program()
    assert [action.op for action in program.actions] == ['pick', 'handover', 'place']
    assert [action.object for action in program.actions] == ['red_cube'] * 3
    assert expansion['goal'] == {'type': 'place_in_zone', 'object': 'red_cube', 'target': 'blue_zone'}
    assert expansion['template'] == 'handover_then_place'


def test_a_new_composite_task_is_a_new_file(tmp_path):
    _write(tmp_path, {'id': 'sweep_two', 'description': 'place both cubes into one zone',
                      'slots': {'zone': 'zone'}, 'requires': [],
                      'goal': {'type': 'place_in_zone', 'object': 'red_cube', 'target': '{zone}'},
                      'actions': [{'op': 'pick', 'object': 'red_cube'},
                                  {'op': 'place', 'object': 'red_cube', 'target': '{zone}'},
                                  {'op': 'pick', 'object': 'blue_cube'},
                                  {'op': 'place', 'object': 'blue_cube', 'target': '{zone}'}]})
    entries = templates.load(tmp_path)
    expansion = templates.expand(entries['sweep_two'], {'zone': 'blue_zone'}, STATE)
    assert [item['object'] for item in expansion['envelope']['items']] == [
        'red_cube', 'red_cube', 'blue_cube', 'blue_cube']


def test_an_object_the_scene_does_not_contain_refuses_the_template():
    entries = templates.load()
    with pytest.raises(templates.TemplateError, match='SLOT_NOT_OBSERVED'):
        templates.expand(entries['pick_then_place'], {'object': 'green_block', 'zone': 'blue_zone', 'arm': 'left'},
                         STATE)


def test_a_missing_slot_is_named_rather_than_guessed():
    entries = templates.load()
    with pytest.raises(templates.TemplateError, match='SLOT_MISSING'):
        templates.expand(entries['handover_then_place'], {'object': 'red_cube'}, STATE)


def test_a_body_without_the_required_capability_refuses_the_template():
    entries = templates.load()
    with pytest.raises(templates.TemplateError, match='CAPABILITY_UNAVAILABLE'):
        templates.expand(entries['handover_then_place'],
                         {'object': 'red_cube', 'zone': 'blue_zone', 'giver': 'left', 'receiver': 'right'},
                         STATE, {'skills': ['grasp', 'place']})


def test_a_malformed_template_is_reported_not_skipped(tmp_path):
    _write(tmp_path, {'id': 'broken', 'slots': {'object': 'blob'}, 'actions': [{'op': 'pick'}]})
    with pytest.raises(templates.TemplateError, match='unknown slot kinds'):
        templates.load(tmp_path)


def test_an_unknown_op_is_refused_at_load_time(tmp_path):
    _write(tmp_path, {'id': 'broken', 'slots': {'object': 'object'}, 'actions': [{'op': 'teleport'}]})
    with pytest.raises(templates.TemplateError, match='unknown op'):
        templates.load(tmp_path)


def test_the_phrase_list_only_points_at_a_template():
    entries = templates.load()
    assert templates.phrase_hint('赤いキューブを左手で持って右手に渡してから青いゾーンに置いて', entries) \
        == ['handover_then_place']
    assert templates.phrase_hint('赤いキューブを左において', entries) == []


def test_a_template_reaches_the_planner_by_name_and_still_compiles(monkeypatch):
    """The model names a procedure; the expansion is compiled and grounded like any other plan."""
    from backend.app.planning.agent_planner import AgentPlanner

    monkeypatch.setenv('OPENARM_LOCAL_PLANNER', '1')
    monkeypatch.setenv('OPENARM_CAPABILITY_MODE', '0')
    planner = AgentPlanner()
    monkeypatch.setattr(planner, '_complete', lambda *a, **k: json.dumps(
        {'template': 'handover_then_place',
         'slots': {'object': 'red_cube', 'zone': 'blue_zone', 'giver': 'left', 'receiver': 'right'}}))
    response = planner._ai_plan('赤いキューブを左手で持って右手に渡してから青いゾーンに置いて', STATE, [])
    actions = [item['op'] for item in response['payload']['actions']]
    assert actions == ['pick', 'handover', 'place']


def test_an_unknown_template_name_is_refused(monkeypatch):
    from backend.app.planning.agent_planner import AgentPlanner
    from backend.app.planning.agent_planner import PlannerProviderError

    monkeypatch.setenv('OPENARM_LOCAL_PLANNER', '1')
    planner = AgentPlanner()
    monkeypatch.setattr(planner, '_complete', lambda *a, **k: json.dumps({'template': 'invented', 'slots': {}}))
    with pytest.raises(PlannerProviderError, match='Unknown template'):
        planner._ai_plan('何か', STATE, [])


def test_a_template_whose_object_is_not_here_is_refused(monkeypatch):
    from backend.app.planning.agent_planner import AgentPlanner
    from backend.app.planning.agent_planner import PlannerProviderError

    monkeypatch.setenv('OPENARM_LOCAL_PLANNER', '1')
    planner = AgentPlanner()
    monkeypatch.setattr(planner, '_complete', lambda *a, **k: json.dumps(
        {'template': 'pick_then_place',
         'slots': {'object': 'green_block', 'zone': 'blue_zone', 'arm': 'left'}}))
    with pytest.raises(PlannerProviderError, match='SLOT_NOT_OBSERVED'):
        planner._ai_plan('緑のブロックを拾って置いて', STATE, [])
