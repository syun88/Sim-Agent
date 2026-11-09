"""The exported annotation is body- and skill-independent: capability, measurements, and who produced it."""
import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / 'scripts' / 'export_lerobot_annotations.py'


def _record(*, actions, skills, robot_type='openarm_v2', homing=None, instruction='pick and place'):
    plan = {'steps': [{'skill': 'approach_object', 'action_index': 0, 'arm': 'left',
                       'object': 'red_cube', 'success': True}],
            'program': {'actions': actions}}
    result = {'skills': skills, 'success': True, 'status': 'SUCCEEDED',
              'final_world_state': {'objects': [{'id': 'red_cube', 'position': [0.4, 0.1, 1.03]}],
                                    'robot': {'type': robot_type}}}
    return {'task_id': 't1', 'instruction': instruction, 'status': 'SUCCEEDED', 'retry_count': 0,
            'post_task_homing': homing,
            'agent_attempts': [{'attempt': 1, 'plan': plan, 'result': result}]}


def _export(tmp_path, record):
    source = tmp_path / 'execution.json'
    source.write_text(json.dumps(record))
    out = tmp_path / 'out.jsonl'
    result = subprocess.run([sys.executable, str(SCRIPT), '--input', str(source), '--out', str(out)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    return [json.loads(line) for line in out.read_text().splitlines() if line.strip()]


def test_a_subtask_carries_the_capability_not_only_the_skill_name(tmp_path):
    record = _record(actions=[{'op': 'handover', 'object': 'red_cube', 'arm': 'left', 'receiver': 'right'}],
                     skills=[{'skill': 'handover_object', 'capability': None, 'action_index': 0,
                              'arm': 'left', 'object': 'red_cube', 'success': True}])
    item = _export(tmp_path, record)[0]
    assert item['schema'].startswith('sim-agent.annotation/')
    assert item['capability'] == 'hand_over'
    assert item['skills'][0]['capability'] == 'hand_over'


def test_an_unknown_operation_is_flagged_rather_than_dropped(tmp_path):
    record = _record(actions=[{'op': 'stack', 'object': 'red_cube'}],
                     skills=[{'skill': 'stack_object', 'action_index': 0, 'success': True}])
    item = _export(tmp_path, record)[0]
    assert item['capability'] is None
    assert item['capability_unmapped'] is True


def test_measured_values_travel_with_the_subtask_and_say_where_they_came_from(tmp_path):
    record = _record(actions=[{'op': 'approach', 'object': 'red_cube'}],
                     skills=[{'skill': 'approach_object', 'action_index': 0, 'success': True,
                              'planning': {'grasp_yaw_deg': -15.0, 'approach_overfly_m': 0.05,
                                           'measured_clearance_m': 0.0264}}])
    item = _export(tmp_path, record)[0]
    values = item['measurements']['values']
    assert values['grasp_yaw_deg'] == -15.0 and values['measured_clearance_m'] == 0.0264
    assert item['measurements']['sources']['measured_clearance_m'] == 'approach_object'


def test_a_missing_measurement_is_absent_not_zero(tmp_path):
    record = _record(actions=[{'op': 'place', 'object': 'red_cube'}],
                     skills=[{'skill': 'move_to_object', 'action_index': 0, 'success': True}])
    item = _export(tmp_path, record)[0]
    assert item['measurements']['values'] == {}


def test_the_body_is_named_and_an_unrecorded_body_is_not_guessed(tmp_path):
    named = _export(tmp_path, _record(actions=[{'op': 'pick', 'object': 'red_cube'}],
                                      skills=[{'skill': 'grasp_object', 'action_index': 0, 'success': True}],
                                      robot_type='openarm_v2'))[0]
    assert named['body'] == {'id': 'openarm_v2', 'source': 'world_state.robot.type'}

    unknown = _record(actions=[{'op': 'pick', 'object': 'red_cube'}],
                      skills=[{'skill': 'grasp_object', 'action_index': 0, 'success': True}])
    unknown['agent_attempts'][0]['result']['final_world_state'].pop('robot')
    item = _export(tmp_path, unknown)[0]
    assert item['body'] == {'id': None, 'source': 'unrecorded'}


def test_the_tidy_up_after_the_subtasks_is_recorded_once_per_episode(tmp_path):
    homing = {'success': True, 'arms': [{'arm': 'left', 'success': True, 'route': 'lift_then_home',
                                        'home_mode': 'scene_clearance_home',
                                        'home_clearance_required_m': 0.0625,
                                        'observed_positioning_error_m': 0.035}]}
    record = _record(actions=[{'op': 'place', 'object': 'red_cube'}],
                     skills=[{'skill': 'move_to_object', 'action_index': 0, 'success': True}], homing=homing)
    item = _export(tmp_path, record)[0]
    arm = item['episode']['post_task_homing']['arms'][0]
    assert arm['home_mode'] == 'scene_clearance_home'
    assert arm['home_clearance_required_m'] == 0.0625
