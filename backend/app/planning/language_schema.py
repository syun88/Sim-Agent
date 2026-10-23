"""Provider schemas encode the same conditional contracts as Python validation.

Pydantic's after validators do not appear in its JSON schema. Without these
branches a constrained decoder can still generate holding goals with no hand,
or fill placement fields on a holding goal, repeatedly consuming repair calls.
"""
from __future__ import annotations

from backend.app.planning.task_program import InstructionMeaning, TABLE_TARGETS


def meaning_schema(state: dict) -> dict:
    ids = [o['id'] for o in state['objects']]
    targets = sorted(set(TABLE_TARGETS) | {t['id'] for t in state['targets']})
    arms = state.get('robot', {}).get('available_arms', ['left','right'])
    def branch(properties, required):
        return {'type':'object', 'properties':properties, 'required':required, 'additionalProperties':False}
    goal = {'anyOf':[
        branch({'type':{'const':'holding'}, 'object':{'enum':ids}, 'arm':{'enum':arms}}, ['type','object','arm']),
        branch({'type':{'const':'placement'}, 'object':{'enum':ids}, 'target':{'enum':targets}}, ['type','object','target']),
        branch({'type':{'const':'relative_placement'}, 'object':{'enum':ids}, 'target':{'enum':ids},
                'relation':{'enum':['on','left_of','right_of','front_of','behind','near']},
                'distance_cm':{'type':'number','minimum':1,'maximum':30}}, ['type','object','target','relation'])]}
    schema = InstructionMeaning.model_json_schema()
    schema['$defs']['TaskGoal'] = goal
    # Mode and nonempty payload must agree in the decoder as well as validation.
    # The selected backend may ignore schema constraints. Python validation is
    # authoritative; conditional goal branches chiefly reduce irrelevant slots.
    return schema
