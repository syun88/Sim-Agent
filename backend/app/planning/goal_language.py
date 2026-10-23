"""Literal end-state role obligations, independent of model/state anchoring.

This is not an instruction parser: it only checks unambiguous object/hand
bindings and exchanges. Unrecognised wording still needs model interpretation.
"""
from __future__ import annotations

import re
import unicodedata

from backend.app.planning.references import object_mentions, named_objects
from backend.app.planning.spatial_language import placement_references, strip_preservation


def unsupported_group_request(instruction: str) -> bool:
    text = unicodedata.normalize('NFKC', instruction).lower()
    if re.search(r"\b(?:don't|do not|never)\b|しない|さない|ないで", text):
        return False
    quantity = re.search(r'(?:二つ|2つ|二個|2個|両方)(?!の?(?:手|腕|グリッパー))|\b(?:two|both)\b(?!\s+(?:hands?|arms?|grippers?)\b)', text)
    together = re.search(r'同時|一緒|まとめて|一つの手|片手|\b(?:together|simultaneously)\b|\b(?:one|single)\s+(?:hand|gripper)\b', text)
    grasp = re.search(r'掴|つか|持ち上げ|取っ|\b(?:grasp|grab|pick|lift|take)\b', text)
    return bool(quantity and together and grasp)


def relative_conditions(instruction: str, state: dict) -> list[dict]:
    text = strip_preservation(instruction, state)
    # Subject/complement roles also express support without a placement verb:
    # "red is bottom, blue is top". This adds a literal obligation to the
    # language model, rather than inventing a second manipulated support goal.
    levels = {'top':set(), 'bottom':set()}
    for _, end, name in object_mentions(text, state):
        for level, pattern in {
            'top':r'(?:が|は|を)(?:一番)?上(?:[、,。\s]|に|で|$)|\s+(?:is\s+)?(?:on\s+)?(?:the\s+)?top\b',
            'bottom':r'(?:が|は|を)(?:一番)?下(?:[、,。\s]|に|で|$)|\s+(?:is\s+)?(?:on\s+)?(?:the\s+)?bottom\b',
        }.items():
            if re.match(pattern, text[end:], re.I):
                levels[level].add(name)
    if len(levels['top']) == len(levels['bottom']) == 1 and levels['top'] != levels['bottom']:
        return [{'type':'relative_placement','object':next(iter(levels['top'])),
                 'target':next(iter(levels['bottom'])), 'relation':'on'}]
    references = placement_references(text, state)
    masked = list(text)
    for reference in references:
        start, end = reference['span']
        masked[start:end] = ' '*(end-start)
    subjects = set(named_objects(''.join(masked), state))
    if len(subjects) != 1:
        return []
    subject = next(iter(subjects))
    return [{'type':'relative_placement', 'object':subject, 'target':r['target'], 'relation':r['relation']}
            for r in references if r['target'] != subject and not (r['relation']=='near' and any(
                other['target']==r['target'] and other['relation'] in {'left_of','right_of','front_of','behind'} for other in references))]


def holding_conditions(instruction: str, state: dict) -> list[dict]:
    text = unicodedata.normalize('NFKC', instruction).lower()
    # Intermediate holders in an ordered procedure are not final predicates.
    if re.search(r'\bthen\b|\bafter\b|その後|した後|してから|渡して', text):
        return []
    if re.search(r"\b(?:don't|do not|never)\b|しない|さない|ないで", text):
        return []
    result = []
    hand_context = bool(re.search(r'\b(?:hands?|grippers?)\b|[左右両]手|グリッパー|ハンド', text))
    if hand_context:
        for clause in re.split(r'\band\b|[、,;。.!?]', text):
            objects = {name for _, _, name in object_mentions(clause, state)}
            arms = set()
            for match in re.finditer(r'\b(left|right)\b|([左右])(?:の)?(?:手|腕|アーム|グリッパー|ハンド)', clause):
                arms.add(match.group(1) or ('left' if match.group(2) == '左' else 'right'))
            if (len(objects) == len(arms) == 1
                    and not re.search(r'\b(?:side|zone|next|beside)\b|側|隣', clause)):
                result.append({'object': next(iter(objects)), 'arm': next(iter(arms)), 'source': clause.strip()[:240]})
            elif len(objects) == 1 and len(arms) > 1 and re.search(r'それぞれ|一個ずつ|\beach\b', clause):
                # One registered cube cannot be duplicated into two separate
                # holders. Expose the contradiction instead of adding another
                # colour to make a feasible but unrelated request.
                result.extend({'object':next(iter(objects)), 'arm':arm, 'source':clause.strip()[:240]}
                              for arm in sorted(arms))
    exchange = re.search(r'\b(?:swap|exchange)\b|\b(?:opposite|other)\s+hands?\b|(?:両手|左右|中身|手に).*?(?:逆|入れ替)', text)
    if exchange and not result:
        robot = state.get('robot', {})
        held = {arm: robot.get(f'{arm}_arm', {}).get('held_object') for arm in ('left', 'right')}
        if all(held.values()) and held['left'] != held['right']:
            result = [{'object': held['right'], 'arm': 'left', 'source': 'Exchange the observed held objects'},
                      {'object': held['left'], 'arm': 'right', 'source': 'Exchange the observed held objects'}]
    return result
