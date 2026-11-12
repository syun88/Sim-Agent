"""The annotation export must state what the record measured, and nothing more."""
import json
import subprocess
import sys
from pathlib import Path

from scripts.export_lerobot_annotations import main, records_from_execution

REPO = Path(__file__).resolve().parents[2]

EXECUTION = {
    'task_id': 'task-1',
    'instruction': '右手にあるキューブを取って左手に渡して',
    'agent_attempts': [{
        'attempt': 1,
        'plan': {
            'program': {'actions': [{'op': 'pick', 'object': 'blue_cube', 'arm': 'right'},
                                    {'op': 'handover', 'object': 'blue_cube', 'arm': 'right', 'receiver': 'left'},
                                    {'op': 'place', 'object': 'blue_cube', 'arm': 'left', 'target': 'left_side'}]},
            'steps': [{'skill': 'approach_object', 'arm': 'right', 'object': 'blue_cube', 'action_index': 0,
                       'before_world_state': {'sim_time': 10.0, 'objects': [
                           {'id': 'blue_cube', 'position': [.5, -.22, 1.031], 'bounds': [[.47, -.25, 1.005], [.53, -.19, 1.057]],
                            'held_by': [], 'rotation': [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}]},
                       'world_state': {'sim_time': 11.0, 'objects': [
                           {'id': 'blue_cube', 'position': [.5, -.22, 1.031], 'bounds': [[.47, -.25, 1.005], [.53, -.19, 1.057]],
                            'held_by': ['right'], 'rotation': [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}]},
                       'success': True},
                      {'skill': 'grasp_object', 'arm': 'right', 'object': 'blue_cube', 'action_index': 0,
                       'success': True},
                      {'skill': 'handover_object', 'arm': 'right', 'object': 'blue_cube', 'action_index': 1,
                       'success': False, 'error': 'HANDOVER_NO_RECEIVER_PINCH'}],
            'planning_evidence': {'scene_bindings': [
                {'kind': 'acquire_missing_hold', 'basis': 'the only unheld cube in that arm workspace',
                 'surface': '右手にあるキューブ', 'replacement': '右手でblue_cube',
                 'resolved': {'object': 'blue_cube', 'arm': 'right'}}]},
        },
        'result': {'skills': [{'skill': 'approach_object', 'success': True},
                              {'skill': 'handover_object', 'success': False,
                               'error': 'HANDOVER_NO_RECEIVER_PINCH'}],
                   'final_world_state': {'objects': [
                       {'id': 'blue_cube', 'position': [.46, .3, 1.031], 'bounds': [[.43, .27, 1.005], [.49, .33, 1.057]],
                        'held_by': [], 'rotation': [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}]}},
    }],
    'evidence_ledger': [{'attempt': 1, 'code': 'HANDOVER_NO_RECEIVER_PINCH', 'mechanism': 'GRASP_CONTACT',
                         'mechanism_reason': '1 finger contact(s) but no loaded opposed pinch',
                         'classification': 'HOLD'}],
}


def test_one_record_per_subtask_with_measured_world_boxes():
    records = records_from_execution(EXECUTION)
    assert [r['subtask'] for r in records] == ['pick blue_cube', 'hand_over blue_cube to left hand',
                                               'place blue_cube to left_side']
    first = records[0]
    assert first['box_space'] == 'world_aabb'
    assert first['objects_before']['blue_cube']['bounds'] == [[.47, -.25, 1.005], [.53, -.19, 1.057]]
    assert first['objects_after']['blue_cube']['held_by'] == ['right']
    assert first['outcome']['success'] is True


def test_image_boxes_are_reported_as_absent_not_invented():
    """No camera frame is recorded with an execution, so an image box would be fabricated."""
    for record in records_from_execution(EXECUTION):
        assert record['image_boxes'] is None


def test_entity_grounding_lists_the_decisions_about_that_object_only():
    records = records_from_execution(EXECUTION)
    assert records[0]['entity_grounding'][0]['basis'] == 'the only unheld cube in that arm workspace'
    # The decision resolves a mention, so it is listed for every subtask about that same entity ...
    assert records[2]['entity_grounding'] == records[0]['entity_grounding']
    # ... and an action on another object must not inherit it.
    other = json.loads(json.dumps(EXECUTION))
    other['agent_attempts'][0]['plan']['program']['actions'].append(
        {'op': 'pick', 'object': 'red_cube', 'arm': 'left'})
    assert records_from_execution(other)[-1]['entity_grounding'] == []


def test_a_failed_subtask_names_the_skill_and_the_error():
    records = records_from_execution(EXECUTION)
    failed = records[1]['outcome']
    assert failed['success'] is False
    assert failed['failed_skill'] == 'handover_object'
    assert failed['error'] == 'HANDOVER_NO_RECEIVER_PINCH'


def test_a_run_without_attempts_is_skipped_rather_than_emitted_empty():
    assert records_from_execution({'task_id': 'x', 'instruction': 'y', 'agent_attempts': []}) == []


def test_the_cli_writes_jsonl_and_reports_skipped_files(tmp_path):
    good = tmp_path / 'good.json'
    good.write_text(json.dumps(EXECUTION), encoding='utf-8')
    empty = tmp_path / 'empty.json'
    empty.write_text(json.dumps({'task_id': 'x', 'instruction': 'y', 'agent_attempts': []}), encoding='utf-8')
    out = tmp_path / 'annotations.jsonl'
    code = main(['--input', str(tmp_path), '--out', str(out), '--with-mechanisms'])
    assert code == 0
    lines = [json.loads(line) for line in out.read_text(encoding='utf-8').splitlines()]
    assert len(lines) == 3
    assert lines[1]['failure_mechanisms'][0]['mechanism'] == 'GRASP_CONTACT'
