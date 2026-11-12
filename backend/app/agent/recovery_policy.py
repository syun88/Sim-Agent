"""Scene reset requires an observed loss of tabletop capability, not an IK error."""
import math


def reset_evidence(state, source):
    requested={g.object for g in source.goals} | {a.object for a in source.actions if a.object}
    objects={o['id']:o for o in state['objects']}
    # A held object is useful recoverable progress, including a partial handover.
    if any(state.get('robot',{}).get(arm+'_arm',{}).get('held_object') for arm in ('left','right')):
        return None
    for name in sorted(requested):
        obj=objects.get(name)
        if obj is None:
            return {'category':'OBJECT_MISSING_FROM_PHYSICS','object':name}
        position=obj.get('position',[])
        if len(position)!=3 or not all(math.isfinite(v) for v in position):
            return {'category':'NONFINITE_PHYSICS_STATE','object':name}
        bottom=(obj.get('bounds') or [position])[0][2]
        table=state.get('workspace',{}).get('table_height_m')
        if table is not None and bottom<table-.10 and not obj.get('held_by'):
            return {'category':'BELOW_TABLE_NO_RECOVERY_SKILL','object':name,
                    'bottom_z':bottom,'table_height_m':table,
                    'reason':'対象が机上操作の範囲から落下し、机の下から回収する動作が実装されていません。'}
    # Ordinary reachable-scene grasp/path failures never certify a reset.
    return None


def admit_reset(certificate, review, *, legal_is_empty, reset_allowed, resets, max_resets):
    """Whether the irreversible recovery may be offered as a candidate, and why not.

    A scene reset throws the layout away, so it needs both an observed cause (the certificate) and an
    uncontested mechanism. A second opinion that disagrees with the rule table's mechanism withholds it;
    nothing else can change this decision, and no reviewer answer can add a candidate.
    """
    if certificate is None:
        return False, 'no certificate for the scene state'
    if not legal_is_empty:
        return False, 'other recoveries are available'
    if not reset_allowed:
        return False, 'a reset is not allowed for this task'
    if resets >= max_resets:
        return False, 'the reset budget is spent'
    if review.get('verdict') == 'disagree':
        return False, (f'the mechanism is contested: rule={review.get("mechanism")}, '
                       f'reviewer={review.get("reviewer_mechanism")}')
    return True, 'certified and uncontested'
