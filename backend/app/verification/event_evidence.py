"""Check recorded event measurements without substituting the engine's later state."""
from __future__ import annotations


def missing_event_fields(check, event):
    if not event:
        return ['receipt']
    if check['type'] in {'throw', 'hit'}:
        missing = [] if event.get('trajectory') else ['trajectory']
        if check['type'] == 'hit' and event.get('target_position') is None:
            missing.append('target_position')
        return missing
    if check['type'] == 'pour':
        evidence = event.get('evidence') or {}
        return [name for name in ('initial_liquid', 'fluid_trace') if not (event.get(name) or evidence.get(name))]
    return ['unsupported_event_type']


def verify_recorded_flight(check, event):
    missing = missing_event_fields(check, event)
    if missing:
        return {'success':False, 'contract_status':'UNKNOWN', 'missing_fields':missing,
                'failure_category':'EVENT_EVIDENCE_MISSING'}
    from backend.app.skills.dynamic_skills import evaluate_throw, closest_approach, HIT_TOLERANCE_M
    if check['type'] == 'throw':
        return evaluate_throw(event['trajectory'], check['min_height'])
    # Target coordinates were observed by the executing skill. A later homing, collision or
    # placement must not retroactively move the target of a historical hit check.
    closest = closest_approach(event['trajectory'], event['target_position'])
    return {'success':bool(closest is not None and closest <= HIT_TOLERANCE_M),
            'closest_approach_m':closest, 'tolerance':HIT_TOLERANCE_M,
            'target':check['target'], 'target_position':event['target_position']}
