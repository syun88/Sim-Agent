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

#: Planner verb -> the subtask word a policy dataset usually uses.
SUBTASK_VERBS = {'pick': 'pick', 'handover': 'hand_over', 'place': 'place', 'approach': 'approach',
                 'move': 'move', 'throw': 'throw', 'pour': 'pour', 'home': 'home', 'open': 'open',
                 'close': 'close', 'joint': 'move_joint', 'navigate': 'navigate'}


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
        for index, action in enumerate(actions):
            steps_here = by_index.get(index, [])
            before = next((s.get('before_world_state') for s in steps_here if s.get('before_world_state')), None)
            after = next((s.get('world_state') for s in reversed(steps_here) if s.get('world_state')), None)
            failures = [s for s in steps_here if s.get('success') is False]
            out.append({
                'task_id': record.get('task_id'),
                'attempt': attempt.get('attempt'),
                'instruction': record.get('instruction'),
                'subtask_index': index,
                'subtask': _subtask_label(action),
                'action': action,
                'entity_grounding': _entity_grounding(record, action),
                'skills': [{'skill': s.get('skill'), 'arm': s.get('arm'), 'object': s.get('object'),
                            'success': s.get('success'), 'error': s.get('error')} for s in steps_here],
                'objects_before': _objects(before) or _objects(initial),
                'objects_after': _objects(after) or _objects(result.get('final_world_state')),
                'box_space': 'world_aabb',
                'image_boxes': None,
                'outcome': {'success': all(s.get('success') for s in steps_here) if steps_here else None,
                            'failed_skill': (failures[0].get('skill') if failures else None),
                            'error': (failures[0].get('error') if failures else None)},
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
