"""Resolve present-tense references against observation, with visible assumptions.

An empty hand is a missing precondition, not evidence of a hidden held object.
The one unheld cube on that side can be acquired as preparation; ambiguity is
reported with candidates instead of silently choosing a colour or old discourse.
"""
from __future__ import annotations

from dataclasses import dataclass
import re

from backend.app.planning.references import named_objects
from backend.app.planning.task_program import observed_holds


class SceneReferenceError(ValueError):
    def __init__(self, explanation, diagnosis):
        super().__init__(explanation)
        self.diagnosis = diagnosis


@dataclass
class SceneBinding:
    instruction: str
    decisions: list[dict]


def bind_scene_references(instruction: str, state: dict) -> SceneBinding:
    held = observed_holds(state)
    objects = {o['id']:o for o in state['objects']}
    decisions = []
    pattern = re.compile(
        r'(?P<jp>左|右)(?:手|腕)(?:にある|の|で持って(?:いる|る)|に持って(?:いる|る))'
        r'(?P<object>(?:(?:赤|青)(?:い|の)?\s*|(?:red|blue)\s+)?(?:キューブ|ブロック|物体|物|cube))'
        r'|(?:the\s+)?(?P<enobject>(?:(?:blue|red)\s+)?cube)\s+in\s+(?:your|the)\s+(?P<en>left|right)\s+(?:hand|gripper)', re.I)
    def bind(match):
        arm = match.group('en') or ('left' if match.group('jp')=='左' else 'right')
        arm = arm.lower()
        side = '左手' if arm=='left' else '右手'
        literal = named_objects(match.group(), state)
        alternatives = []
        if held[arm]:
            chosen, kind = held[arm], 'observed_hold'
            if literal and chosen not in literal:
                chosen = None
            candidates = [held[arm]]
        else:
            candidates = [o['id'] for o in objects.values() if o.get('type','cube')=='cube'
                and o.get('graspable',True) and o['id'] not in held.values()
                and (o['id'] in literal if literal else o['position'][1]*(1 if arm=='left' else -1)>.025)
                and (not o.get('bounds') or abs(o['bounds'][0][2]-state.get('workspace',{}).get('table_height_m',1.008))<.008)]
            chosen, kind = (candidates[0] if len(candidates)==1 else None), 'acquire_missing_hold'
            if chosen is None and len(candidates) > 1:
                ranked = sorted(candidates, key=lambda name: (-float(objects[name]['position'][2]), name))
                if float(objects[ranked[0]]['position'][2]) - float(objects[ranked[1]]['position'][2]) > .02:
                    # Only a stack is resolved here, not a free choice between equals. Of two cubes at
                    # materially different heights only the upper one can be grasped without
                    # disturbing the other, so the scene itself answers which one was meant: take it,
                    # record the choice and the alternatives, and let the task continue. Two cubes at
                    # the same height stay ambiguous - there the scene offers no answer and stopping
                    # is still the honest outcome (see the guard in test_task_agent.py).
                    chosen, kind, alternatives = ranked[0], 'assumed_ambiguity_resolution', ranked[1:]
        if chosen is None:
            diagnosis = {'category':'REFERENCE_NOT_PRESENT','arm':arm,'observed_holder':held[arm],
                         'source':match.group(),'candidates':candidates,
                         'required':'受け渡す物体を一意に指定するか、対象を机の上に準備してください。'}
            raise SceneReferenceError(f'{side}の現在の保持物体は{held[arm] or "ありません"}。対象を一意に決められません。'+diagnosis['required'], diagnosis)
        explanation = (f'{side}が実際に保持している{chosen}を対象にします。' if kind=='observed_hold' else
                       f'{side}は空です。現在の配置で対象候補は{chosen}だけなので、まず取得してから指示を続けます。' if kind=='acquire_missing_hold' else
                       f'{side}は空で、指示に合う候補が{len(candidates)}個あります。重なりでは上の物体だけが'
                       f'把持できるため、いちばん高い{chosen}を対象にします（他: {", ".join(alternatives)}）。')
        decisions.append({'kind':kind,'arm':arm,'object':chosen,'source':match.group(),
                          'candidates':candidates,'alternatives':alternatives,'explanation':explanation})
        return f'{side}で{chosen}' if match.group('jp') else f'{chosen} with the {arm} hand'
    text = pattern.sub(bind, instruction)
    def table_reference(match):
        arm='left' if match.group('side')=='左' else 'right'
        cubes=sorted((o for o in objects.values() if o.get('type','cube')=='cube'),key=lambda o:o['position'][1])
        if not cubes or (len(cubes)>1 and abs(cubes[0]['position'][1]-cubes[-1]['position'][1])<.01):
            raise SceneReferenceError('左右の配置だけでは対象を一意に決められません。色か物体名を指定してください。',
                {'category':'AMBIGUOUS_SPATIAL_REFERENCE','source':match.group(),'candidates':[o['id'] for o in cubes]})
        obj=cubes[-1 if arm=='left' else 0]['id']
        decisions.append({'kind':'observed_spatial_reference','object':obj,'source':match.group(),
            'explanation':f'現在の物体位置から「{match.group()}」を{obj}に対応付けました。'})
        return obj
    text=re.sub(r'(?P<side>左|右)(?:側)?(?:にある|の)(?:方|ほう|やつ|物体|物|キューブ)',table_reference,text)
    return SceneBinding(text, decisions)
