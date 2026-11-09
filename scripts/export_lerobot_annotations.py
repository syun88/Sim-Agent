#!/usr/bin/env python3
"""Turn saved executions into annotation records a policy-training pipeline can consume.

A LeRobot-style dataset carries, per training frame, an instruction, the subtask the policy was on,
where the objects were, and the actions it took. This repo already produces that evidence on every run
(`initial_world_state`, `agent_attempts[*].plan`, `skills`, `verification`, `evidence_ledger`), so the
records can be derived from what was measured instead of annotated by hand.

What this writes, per subtask:

* `subtask` - the action's own verb/object/target in the planner's vocabulary, plus `entity_grounding`:
  the recorded decisions about that object (`planning_evidence.scene_bindings`). Those decisions resolve
  *mentions* in the instruction, not individual actions, so all decisions about the same entity are listed
  and none is claimed to have produced this subtask.
* `objects` - world-frame axis-aligned bounding boxes and positions at the start and end of the subtask,
  taken from the recorded world states. **These are 3D world boxes, not image boxes**: this repo records no
  camera frames with the execution, so an image-space box would have to be invented. `image_boxes` is
  therefore `null` and `box_space` says `world_aabb`.
* `outcome` - the measured verification checks and, for a failure, the attributed mechanism.

Nothing is inferred: a field the record does not carry stays `null`, and a run whose world states are
missing is skipped with a reason rather than emitted with empty boxes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

#: The schema this exporter writes. Bump it when a field changes meaning, not when one is added.
SCHEMA = 'sim-agent.annotation/2'

#: Planner verb -> the subtask word a policy dataset usually uses.
SUBTASK_VERBS = {'pick': 'pick', 'handover': 'hand_over', 'place': 'place', 'approach': 'approach',
                 'move': 'move', 'throw': 'throw', 'pour': 'pour', 'home': 'home', 'open': 'open',
                 'close': 'close', 'joint': 'move_joint', 'navigate': 'navigate'}

#: Planner verb -> what the robot is being asked to do, independent of this repo's skill names. A dataset
#: that has to be reusable across bodies cannot be keyed on `move_to_object` or `park_empty_arm`.
CAPABILITIES = {'pick': 'grasp', 'handover': 'hand_over', 'place': 'place', 'approach': 'reach',
                'move': 'move', 'throw': 'throw', 'pour': 'pour', 'home': 'home', 'open': 'actuate_gripper',
                'close': 'actuate_gripper', 'joint': 'actuate_joint', 'navigate': 'navigate'}

#: Skill name -> capability, for the steps recorded with their own skill names.
SKILL_CAPABILITIES = {'approach_object': 'reach', 'grasp_object': 'grasp', 'lift_object': 'carry',
                      'move_to_object': 'place', 'move_to_target': 'place', 'release_object': 'release',
                      'handover_object': 'hand_over', 'retreat': 'retreat', 'return_home': 'home',
                      'park_empty_arm': 'home', 'throw_object': 'throw', 'pour_liquid': 'pour',
                      'navigate_to_object': 'navigate'}

#: Measurements the skills record, by the field they appear under. Only measured values travel; a missing
#: one stays absent rather than becoming a zero.
MEASUREMENT_FIELDS = ('grasp_yaw_deg', 'grasp_tilt_deg', 'approach_overfly_m', 'measured_clearance_m',
                      'measured_clearance_whole_path_m', 'noncontact_clearance_m', 'target_body_clearance_m',
                      'staging', 'placer_yaw_deg', 'home_clearance_required_m',
                      'observed_positioning_error_m', 'deviation_m', 'joint_error_rad',
                      'relief_fraction', 'drop_m', 'clearance_height_m')


def _body(record: dict) -> dict:
    """Which body produced this episode. Unrecorded stays unrecorded: sim and real must not be guessed."""
    for key in ('body', 'robot', 'robot_id'):
        value = record.get(key)
        if isinstance(value, str) and value:
            return {'id': value, 'source': key}
        if isinstance(value, dict) and value.get('id'):
            return {'id': value['id'], 'source': key}
    candidates = [record.get('final_world_state'), record.get('initial_world_state')]
    candidates += [((attempt.get('result') or {}).get('final_world_state'))
                   for attempt in (record.get('agent_attempts') or [])]
    candidates += [((attempt.get('result') or {}).get('initial_world_state'))
                   for attempt in (record.get('agent_attempts') or [])]
    for world in candidates:
        robot = (world.get('robot') or {}) if isinstance(world, dict) else {}
        if robot.get('type'):
            return {'id': str(robot['type']), 'source': 'world_state.robot.type'}
    return {'id': None, 'source': 'unrecorded'}


def _episode(record: dict) -> dict:
    """What happened outside the subtasks: the tidy-up that runs after them, and how many attempts ran."""
    homing = record.get('post_task_homing') or {}
    arms = []
    for arm in homing.get('arms') or []:
        arms.append({'arm': arm.get('arm'), 'success': arm.get('success'), 'route': arm.get('route'),
                     'home_mode': arm.get('home_mode'),
                     'home_clearance_required_m': arm.get('home_clearance_required_m'),
                     'observed_positioning_error_m': arm.get('observed_positioning_error_m'),
                     'error': arm.get('error'), 'detail': arm.get('detail')})
    return {'post_task_homing': {'success': homing.get('success'), 'arms': arms} if homing else None,
            'retry_count': record.get('retry_count'), 'status': record.get('status')}


def _measurements(steps_here: list) -> dict:
    """Every measured number the skills recorded for this subtask, with the skill it came from."""
    found, sources = {}, {}
    for step in steps_here:
        for container in (step, step.get('evidence') or {}, step.get('planning') or {}):
            if not isinstance(container, dict):
                continue
            for field in MEASUREMENT_FIELDS:
                if field in container and container[field] is not None:
                    found[field] = container[field]
                    sources[field] = step.get('skill')
    return {'values': found, 'sources': sources}


def _objects(state):
    if not isinstance(state, dict):
        return None
    objects = state.get('objects')
    if not isinstance(objects, list):
        return None
    view = {}
    for obj in objects:
        if not isinstance(obj, dict) or 'id' not in obj:
            continue
        entry = {'position': obj.get('position'), 'rotation': obj.get('rotation'),
                 'bounds': obj.get('bounds'), 'held_by': obj.get('held_by') or []}
        view[obj['id']] = entry
    return view


def _subtask_label(action):
    verb = SUBTASK_VERBS.get(action.get('op'), action.get('op'))
    parts = [verb] if verb else []
    if action.get('object'):
        parts.append(action['object'])
    if action.get('relation') and action.get('target'):
        parts.append(f"{action['relation']} {action['target']}")
    elif action.get('target'):
        parts.append(f"to {action['target']}")
    if action.get('receiver'):
        parts.append(f"to {action['receiver']} hand")
    return ' '.join(str(part) for part in parts)


def _entity_grounding(record, action):
    """The recorded decisions about this action's object.

    Named `entity_grounding`, not `grounding`: a decision resolves a **mention** ("右手にあるキューブ"),
    and the plan does not record which mention produced which action. Claiming that a decision *caused*
    this subtask would be an invention, so every decision about the same entity is listed and the reader
    can see the surface it came from.
    """
    evidence = ((record.get('agent_attempts') or [{}])[0].get('plan') or {}).get('planning_evidence') or {}
    object_id = action.get('object')
    if not object_id:
        return []
    found = []
    for decision in evidence.get('scene_bindings') or []:
        if (decision.get('resolved') or {}).get('object') == object_id:
            found.append({'kind': decision.get('kind'), 'basis': decision.get('basis'),
                          'surface': decision.get('surface'), 'replacement': decision.get('replacement')})
    return found


def records_from_execution(record: dict) -> list[dict]:
    """One annotation record per executed subtask, or an empty list with the reason in `.skipped`."""
    attempts = record.get('agent_attempts') or []
    if not attempts:
        return []
    out = []
    initial = record.get('initial_world_state')
    for attempt in attempts:
        plan = attempt.get('plan') or {}
        program = plan.get('program') or {}
        actions = program.get('actions') or []
        result = attempt.get('result') or {}
        steps = plan.get('steps') or []
        by_index = {}
        for step in steps:
            by_index.setdefault(step.get('action_index'), []).append(step)
        if not actions and not steps:
            continue
        executed_all = [step for step in (result.get('skills') or []) if isinstance(step, dict)]
        for index, action in enumerate(actions):
            steps_here = by_index.get(index, [])
            executed = [step for step in executed_all if step.get('action_index') == index]
            before = next((s.get('before_world_state') for s in steps_here if s.get('before_world_state')), None)
            after = next((s.get('world_state') for s in reversed(steps_here) if s.get('world_state')), None)
            failures = [s for s in steps_here if s.get('success') is False]
            evidence_steps = executed or steps_here
            capability = CAPABILITIES.get(action.get('op'))
            out.append({
                'schema': SCHEMA,
                'body': _body(record),
                'task_id': record.get('task_id'),
                'attempt': attempt.get('attempt'),
                'instruction': record.get('instruction'),
                'subtask_index': index,
                'subtask': _subtask_label(action),
                'action': action,
                'capability': capability,
                **({'capability_unmapped': True} if capability is None else {}),
                'measurements': _measurements(evidence_steps),
                'entity_grounding': _entity_grounding(record, action),
                'skills': [{'skill': s.get('skill'), 'capability': SKILL_CAPABILITIES.get(s.get('skill')),
                            'arm': s.get('arm'), 'object': s.get('object'),
                            'success': s.get('success'), 'error': s.get('error')}
                           for s in evidence_steps],
                'objects_before': _objects(before) or _objects(initial),
                'objects_after': _objects(after) or _objects(result.get('final_world_state')),
                'box_space': 'world_aabb',
                'image_boxes': None,
                'outcome': {'success': all(s.get('success') for s in steps_here) if steps_here else None,
                            'failed_skill': (failures[0].get('skill') if failures else None),
                            'error': (failures[0].get('error') if failures else None)},
                'episode': _episode(record),
                'provenance': {'source': 'sim-agent execution record',
                               'sim_time': (before or {}).get('sim_time'),
                               'scene_epoch': 0 if attempt.get('attempt') == 1 else attempt.get('attempt')},
            })
    return out


def mechanisms_by_subtask(record: dict) -> dict:
    """Attributed mechanisms from the evidence ledger, keyed by the attempt they belong to."""
    found = {}
    for receipt in record.get('evidence_ledger') or []:
        attempt = receipt.get('attempt')
        found.setdefault(attempt, []).append({'code': receipt.get('code'),
                                              'mechanism': receipt.get('mechanism'),
                                              'reason': receipt.get('mechanism_reason'),
                                              'classification': receipt.get('classification')})
    return found


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--input', required=True, help='execution JSON file, or a directory of them')
    parser.add_argument('--out', default='-', help='output JSONL path, or - for stdout')
    parser.add_argument('--with-mechanisms', action='store_true',
                        help='attach attributed failure mechanisms per attempt')
    args = parser.parse_args(argv)

    source = Path(args.input)
    files = sorted(source.glob('*.json')) if source.is_dir() else [source]
    lines, skipped = [], []
    for path in files:
        try:
            record = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError) as error:
            skipped.append({'file': str(path), 'reason': f'unreadable: {type(error).__name__}'})
            continue
        if not isinstance(record, dict):
            skipped.append({'file': str(path), 'reason': 'not an execution record'})
            continue
        records = records_from_execution(record)
        if not records:
            skipped.append({'file': str(path), 'reason': 'no executed subtasks recorded'})
            continue
        mechanisms = mechanisms_by_subtask(record) if args.with_mechanisms else {}
        for item in records:
            if mechanisms:
                item['failure_mechanisms'] = mechanisms.get(item['attempt']) or []
            lines.append(json.dumps(item, ensure_ascii=False))

    body = '\n'.join(lines) + ('\n' if lines else '')
    if args.out == '-':
        sys.stdout.write(body)
    else:
        Path(args.out).write_text(body, encoding='utf-8')
    for entry in skipped:
        print(f"skipped {entry['file']}: {entry['reason']}", file=sys.stderr)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
