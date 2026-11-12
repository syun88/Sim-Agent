"""Turn execution records into (pre-execution features -> observed mechanism) examples.

The features are what a planner knows **before** it acts: the instruction, the proposed actions, the arms
and the scene. The label is what the measurements later showed - the mechanism from the evidence ledger, or
the same attribution (`verification/failure_attribution`) applied to the failed skill's recorded evidence.
Keeping the two apart is the whole point: a feature that leaks the post-execution evidence would make the
predictor look good and tell a planner nothing it could not already see.
"""

from __future__ import annotations

import json
from pathlib import Path

from backend.app.verification.failure_attribution import attribute

#: Stable feature order. Every feature is available before running the plan.
FEATURES = (
    'action_count', 'pick_count', 'place_count', 'handover_count', 'approach_count',
    'relative_placements', 'uses_left', 'uses_right', 'both_arms', 'support_held_at_start',
    'mentions_maintain_hold', 'instruction_chars', 'object_count', 'model_called',
    # Pre-execution geometry (`planning/preflight.py`). Plan shape alone could not tell the contact
    # mechanism apart (APPROACH_CONTACT 0/7, measured 2026-11-08); these are answered before motion.
    'preflight_missing', 'pad_blocks_face', 'pad_clearance_mm', 'reachable_yaws', 'reachable_asked',
)

#: A successful run is a label too: a predictor that only knows failures cannot say "this will work".
SUCCESS = 'NONE'
ACTIONS_WITH_OBJECT = ('pick', 'place', 'handover', 'approach', 'throw', 'pour')


def _plan_of(record):
    attempts = record.get('agent_attempts') or []
    if not attempts:
        return None
    plan = attempts[0].get('plan') or {}
    return plan


def _held_at_start(record, support):
    robot = (record.get('initial_world_state') or {}).get('robot') or {}
    return any((robot.get(f'{side}_arm') or {}).get('held_object') == support for side in ('left', 'right'))


def _support_of(actions):
    for action in actions:
        if action.get('op') == 'place' and action.get('relation') and action.get('target'):
            return action['target']
    return None


def _preflight_features(plan) -> dict:
    evidence = (plan.get('planning_evidence') or {}).get('preflight') or {}
    questions = evidence.get('questions') or []
    rows = [row for row in questions if isinstance(row, dict)] if isinstance(questions, list) else []
    if not evidence.get('available') or not rows:
        # Older records predate the preflight pass. A missing measurement is a fact about the record, so
        # it gets its own feature instead of being silently imputed as "nothing was blocking".
        return {'preflight_missing': 1, 'pad_blocks_face': 0, 'pad_clearance_mm': 0.,
                'reachable_yaws': 0, 'reachable_asked': 0}
    blocks = [row.get('pad_blocks_face') for row in rows if row.get('pad_blocks_face') is not None]
    clearances = [row.get('pad_clearance_m') for row in rows if isinstance(row.get('pad_clearance_m'), (int, float))]
    reachable_yaws = sum(int(row.get('reachable_yaws') or 0) for row in rows)
    asked = sum(int(row.get('reachable_asked') or 0) for row in rows)
    return {'preflight_missing': 0,
            'pad_blocks_face': int(any(blocks)),
            'pad_clearance_mm': (min(clearances) * 1000.) if clearances else 0.,
            'reachable_yaws': reachable_yaws, 'reachable_asked': asked}


def features_of(record: dict) -> dict | None:
    plan = _plan_of(record)
    if plan is None:
        return None
    actions = (plan.get('program') or {}).get('actions') or []
    if not actions:
        return None
    instruction = record.get('instruction') or ''
    arms = {action.get('arm') for action in actions if action.get('arm')}
    support = _support_of(actions)
    return {
        'action_count': len(actions),
        'pick_count': sum(a.get('op') == 'pick' for a in actions),
        'place_count': sum(a.get('op') == 'place' for a in actions),
        'handover_count': sum(a.get('op') == 'handover' for a in actions),
        'approach_count': sum(a.get('op') == 'approach' for a in actions),
        'relative_placements': sum(1 for a in actions if a.get('relation')),
        'uses_left': int('left' in arms),
        'uses_right': int('right' in arms),
        'both_arms': int({'left', 'right'} <= arms),
        'support_held_at_start': int(bool(support) and _held_at_start(record, support)),
        'mentions_maintain_hold': int(bool(__import__('re').search(
            r'持ったまま|維持したまま|持っている状態|掴んだまま|掴んでいる状態|while\s+holding', instruction))),
        'instruction_chars': len(instruction),
        'object_count': len((record.get('initial_world_state') or {}).get('objects') or []),
        'model_called': int(bool((plan.get('model_called')))),
        **_preflight_features(plan),
    }


def label_of(record: dict) -> str | None:
    """The observed mechanism, or `NONE` when the run succeeded. `None` when neither is recorded."""
    if record.get('success'):
        return SUCCESS
    for receipt in record.get('evidence_ledger') or []:
        if receipt.get('mechanism'):
            return receipt['mechanism']
    for attempt in record.get('agent_attempts') or []:
        for skill in reversed((attempt.get('result') or {}).get('skills') or []):
            if not skill.get('success'):
                mechanism, _ = attribute(skill.get('error'), skill.get('evidence') or {})
                return mechanism
    return None


def example_of(record: dict) -> dict | None:
    features = features_of(record)
    label = label_of(record)
    if features is None or label is None:
        return None
    return {'task_id': record.get('task_id'), 'instruction': record.get('instruction'),
            'features': features, 'label': label}


def load_examples(paths) -> tuple[list[dict], list[dict]]:
    """Read records into examples, reporting every record that could not become one."""
    examples, skipped = [], []
    for path in paths:
        path = Path(path)
        try:
            record = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError) as error:
            skipped.append({'file': str(path), 'reason': f'unreadable: {type(error).__name__}'})
            continue
        example = example_of(record) if isinstance(record, dict) else None
        if example is None:
            skipped.append({'file': str(path),
                            'reason': 'no plan or no measurable label'})
            continue
        examples.append(example)
    return examples, skipped
