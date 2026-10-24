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
