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
        r'(?P<jp>左|右)(?:手|腕)(?:にある|の|で(?:持って|掴んで|つかんで|握って|保持して)(?:いる|る)?|に(?:持って|持たせて|掴んで|つかんで|握って|保持して)(?:いる|る)?)'
        r'(?P<object>(?:(?:赤|青)(?:い|の)?\s*|(?:red|blue)\s+)?(?:キューブ|ブロック|物体|物|状態|もの|やつ|cube))'
        r'|(?:the\s+)?(?P<enobject>(?:(?:blue|red)\s+)?cube)\s+(?:in|held\s+by)\s+(?:your|the)\s+(?P<en>left|right)\s+(?:hand|gripper|arm)', re.I)
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
        suffix_spatial = re.match(r'(?:の)?(?:上|下|左|右|前|後|奥|真上|てっぺん|土台|近く|隣|そば| atop| on| under| next)', instruction[match.end():], re.I)
        prefix_spatial = re.search(r'\b(?:on|atop|under|beside|next\s+to|left\s+of|right\s+of)\s+(?:(?:top|the)\s+of\s+)?$', instruction[:match.start()], re.I)
        has_earlier_actor = bool(re.search(r'[左右](?:手|腕)(?:で|が|は)', instruction[:match.start()]))
        is_modifying_phrase = any(w in match.group().lower() for w in ('で掴んで', 'でつかんで', 'で握って', 'で保持して', 'で持った', 'で掴んだ', 'held by', 'in your', 'in the'))
        if suffix_spatial or prefix_spatial or has_earlier_actor or is_modifying_phrase:
            return f'{chosen}'
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

    # 照応・補集合・相対的手参照の解決
    cubes = sorted([o['id'] for o in objects.values() if o.get('type', 'cube') == 'cube'])

    # 「右手でキューブを」「左手でキューブを」など、腕が指定されている無標キューブの解決
    uncolored_hand_pattern = re.compile(r'(?P<arm>[左右])(?:手|腕)で(?P<cube>キューブ|ブロック|物体|物)(?=[をにへと、,\s])')
    def bind_uncolored_hand(m):
        arm = 'left' if m.group('arm') == '左' else 'right'
        matched_cubes = [o['id'] for o in objects.values() if o.get('type', 'cube') == 'cube'
                         and o['id'] not in held.values()
                         and o['position'][1] * (1 if arm == 'left' else -1) > 0.02]
        if len(matched_cubes) == 1:
            chosen = matched_cubes[0]
            decisions.append({
                'kind': 'arm_reach_cube_binding', 'object': chosen, 'arm': arm,
                'explanation': f'{m.group("arm")}手の作業域にある{chosen}を「{m.group()}」に対応付けました。'
            })
            return f'{m.group("arm")}手で{chosen}'
        return m.group()
    text = uncolored_hand_pattern.sub(bind_uncolored_hand, text)
    other_cube_pattern = re.compile(
        r'(?:もう(?:一つ|片方|1つ|1個)の|残りの|もう一方の)(?:キューブ|ブロック|物体|物)|\b(?:the\s+other|another)\s+cube\b', re.I)
    explicit_cubes = named_objects(text, state)
    if other_cube_pattern.search(text) and len(cubes) >= 2:
        if explicit_cubes:
            first_cube = explicit_cubes[0]
            remaining = [c for c in cubes if c != first_cube]
            if len(remaining) == 1:
                other_cube = remaining[0]
                text = other_cube_pattern.sub(other_cube, text)
                decisions.append({
                    'kind': 'complement_cube_reference', 'object': other_cube,
                    'explanation': f'指示の「もう一つのキューブ」を、未言及の{other_cube}に対応付けました。'
                })
        else:
            # "キューブを、... もう一つのキューブを..." 対比構造
            first_cube, other_cube = cubes[0], cubes[1]
            text = other_cube_pattern.sub(other_cube, text)
            unmarked_pattern = re.compile(r'(?<![赤青redbluetheotheranother\w])キューブ(?=[をにはへと、,\s])')
            text = unmarked_pattern.sub(first_cube, text, count=1)
            decisions.append({
                'kind': 'unmarked_contrastive_resolution',
                'object': first_cube, 'other_object': other_cube,
                'explanation': f'未指定の2つのキューブ参照に対し、順に{first_cube}と{other_cube}を割り当てました。'
            })

    other_hand_pattern = re.compile(
        r'(?:もう(?:一方|片方)の|反対の|逆の)(?:手|腕|アーム)|\b(?:the\s+other|opposite)\s+(?:hand|arm)\b', re.I)
    if other_hand_pattern.search(text):
        acting = None
        if '左手' in text or '左腕' in text or 'left' in text.lower():
            acting = 'left'
        elif '右手' in text or '右腕' in text or 'right' in text.lower():
            acting = 'right'
        elif any(held.values()):
            acting = next(arm for arm, obj in held.items() if obj)
        opposite = 'right' if acting == 'left' else 'left' if acting == 'right' else None
        if opposite is None and len(state.get('robot', {}).get('available_arms', ['left', 'right'])) == 2:
            first_cube_id = named_objects(text, state)[0] if named_objects(text, state) else None
            if first_cube_id:
                pos = objects[first_cube_id].get('position')
                prim_arm = 'left' if (pos and pos[1] >= 0) else 'right'
                opposite = 'right' if prim_arm == 'left' else 'left'
        if opposite:
            side_str = '右手' if opposite == 'right' else '左手'
            text = other_hand_pattern.sub(side_str, text)
            decisions.append({
                'kind': 'opposite_hand_reference', 'arm': opposite,
                'explanation': f'指示の「もう一方の手」を、対向する{side_str}に対応付けました。'
            })

    return SceneBinding(text, decisions)
