"""Ground literal spatial arguments without treating a support as a picked object."""
import re
from collections import Counter

from backend.app.planning.references import named_objects, object_mentions
from backend.app.planning.task_program import GroundingError

JAPANESE = {
    "on": r"(?:の)?(?:真上|上|てっぺん)|(?:に|へ)(?:重ね|(?:載|乗)せ|積)|を土台に",
    "left_of": r"(?:の)?(?:左側|左隣|左)",
    "right_of": r"(?:の)?(?:右側|右隣|右)",
    "front_of": r"(?:の)?(?:手前|前側|前)",
    "behind": r"(?:の)?(?:後ろ側|後ろ|奥)",
    "near": r"(?:の)?(?:隣|となり|そば|近く|横)",
}
ENGLISH = {
    "on": r"(?:on\s+top\s+of|on|atop|onto)",
    "left_of": r"(?:to\s+the\s+)?left\s+of",
    "right_of": r"(?:to\s+the\s+)?right\s+of",
    "front_of": r"(?:in\s+front\s+of)",
    "behind": r"behind",
    "near": r"(?:next\s+to|beside|near)",
}


def preservation_spans(text, state):
    spans = []
    for start, end, name in object_mentions(text, state):
        suffix = re.match(r"(?:(?:の)?(?:位置|場所)(?:は|を)?|には|は|を|に)?(?:触ら(?:ないで|ない(?:こと)?|ず(?:に)?)|動かさ(?:ないで|ない(?:こと)?|ず(?:に)?)|変え(?:ないで|ない(?:こと)?|ず(?:に)?)|そのまま)(?:[、,\s]*)", text[end:])
        english = re.search(r"(?:leave\s+|keep\s+|without\s+(?:moving|touching)\s+|(?:do\s+not|don't)\s+(?:move|touch)\s+)(?:the\s+)?$", text[:start], re.I)
        ending = re.match(r"\s*(?:where\s+it\s+is|alone|untouched|in\s+place|unchanged)[.!\s]*", text[end:], re.I)
        if suffix:
            spans.append((start, end + suffix.end(), name))
        elif english and (ending or re.search(r"without|do\s+not|don't", english.group(), re.I)):
            spans.append((english.start(), end + (ending.end() if ending else 0), name))
    return spans


def strip_preservation(text, state):
    masked = list(text)
    for start, end, _ in preservation_spans(text, state):
        masked[start:end] = " " * (end - start)
    return "".join(masked)


def placement_references(text, state):
    references = []
    for start, end, name in object_mentions(text, state):
        possessive = re.match(r"['’]s\s+(left|right|front|back)\b",text[end:],re.I)
        prefix = re.search(r'\b(?:on|to|at)\s+(?:the\s+)?$',text[:start],re.I)
        if possessive and prefix:
            references.append({'target':name,'relation':{'left':'left_of','right':'right_of','front':'front_of','back':'behind'}[possessive.group(1).lower()],
                               'span':(prefix.start(),end+possessive.end())})
            continue
        for relation, pattern in JAPANESE.items():
            match = re.match(pattern, text[end:])
            if match:
                references.append({"target": name, "relation": relation, "span": (start, end + match.end())})
                break
        else:
            for relation, pattern in ENGLISH.items():
                match = re.search(r"(?:" + pattern + r")\s+(?:the\s+|that\s+)?$", text[:start], re.I)
                if match:
                    references.append({"target": name, "relation": relation, "span": (match.start(), end)})
                    break
    # Resolve object-relative pronouns only from a unique antecedent in the preceding clause.
    # Never turn a lost support ('その上') into an unspecified table placement.
    # 'その左手' refers to a hand, not the left of an object.
    pronouns = r'(?:その|それの)(?:' + '|'.join(JAPANESE.values()) + r')(?!手|腕|アーム|グリッパー|ハンド)'
    for match in re.finditer(pronouns, text):
        prefix = re.split(r'[、,;。.!！\n]', text[:match.start()])
        previous = next((part for part in reversed(prefix) if part.strip()), '')
        candidates = set(named_objects(previous, state))
        if len(candidates) != 1:
            raise GroundingError('Unresolved object-relative reference: ' + match.group())
        suffix = re.sub(r'^(?:その|それの)', '', match.group())
        relation = next(key for key, pattern in JAPANESE.items() if re.fullmatch(pattern, suffix))
        references.append({'target':next(iter(candidates)), 'relation':relation, 'span':match.span()})
    return references


def resolve_relative_pronouns(text, state):
    """Bind full-instruction context before a parser splits it into individual actions."""
    for ref in sorted(placement_references(text, state), key=lambda item: item['span'][0], reverse=True):
        start, end = ref['span']
        phrase = text[start:end]
        if phrase.startswith(('その', 'それの')):
            suffix = re.sub(r'^(?:その|それの)', '', phrase)
            text = text[:start] + ref['target'] + 'の' + suffix + text[end:]
    return text


def _prohibited_clause(clause):
    return bool(re.search(r"\b(?:do\s+not|don't|never)\b|渡さない|投げない|置かない|"
                          r"(?:積み)?重ね(?:ない|ず)", clause, re.I))


def validate_instruction_relations(plan, instruction, state, *, required_holders=()):
    from backend.app.planning.goal_language import relative_conditions
    measured_holders = {(c.get('object'), c.get('arm')) for c in plan.goal.get('checks', [])
                        if c.get('type') == 'holding'}
    for holder in required_holders:
        if (holder['object'], holder['arm']) not in measured_holders:
            raise GroundingError('The success checks omit a requested final holder: ' + str(holder))
    references = placement_references(instruction, state)
    roles = relative_conditions(instruction, state)
    # A stacking verb asks for support, not two independent table placements. Explicit level roles
    # ('red is bottom, blue is top') are also a support specification.
    affirmative_support = any(re.search(r'重ね|積み重|\b(?:stack|atop)\b', clause, re.I)
                              and not _prohibited_clause(clause)
                              for clause in re.split(r'[。.!！,、;\n]+', instruction))
    if affirmative_support and not any(ref['relation'] == 'on' for ref in [*references, *roles]):
        raise GroundingError('Unresolved support relation: specify what is above what')
    requested = Counter((ref["target"], ref["relation"]) for ref in references)
    actual = Counter((action["target"], action["relation"]) for action in plan.program["actions"]
                     if action["op"] == "place" and action.get("relation"))
    actual.update((goal['target'], goal['relation']) for goal in
                  (plan.goal_program or {}).get('goals', []) if goal['type'] == 'relative_placement')
    for target, relation in list(requested):
        if relation == "near":
            actual[(target, "near")] += sum(actual[(target, specific)] for specific in ("left_of", "right_of", "front_of", "behind"))
    if requested - actual:
        raise GroundingError("Preserve the requested object-relative placement: " + str(dict(requested)))
    # Ordered actions carry intermediate obligations; only the last manipulation of each object
    # describes its final relation. Do not demand 'left of' AND 'right of' after sequential moves.
    last = {a['object']: a for a in plan.program['actions'] if a.get('object')}
    final_relations = [a for a in last.values() if a['op'] == 'place' and a.get('relation')]
    final_relations += [g for g in (plan.goal_program or {}).get('goals', [])
                        if g['type'] == 'relative_placement']
    def key(item):
        return item.get('object'), item.get('target'), item.get('relation')
    expected = {key(item) for item in final_relations}
    measured = {key(check) for check in plan.goal.get('checks', [])
                if check.get('type') == 'relative_placement'}
    if expected - measured:
        raise GroundingError('The success checks omit a final object-relative placement: ' + str(expected - measured))
    # Explicit top/bottom roles have no reference span and express final-state obligations.
    # Named-reference spans were checked against ordered actions above, not promoted to final goals.
    if not references:
        for role in roles:
            if key(role) not in expected:
                raise GroundingError('Preserve the requested relative subject and support: ' + str(role))
    # A noun in 'on the red cube' denotes the support, not a command to pick red.
    masked = list(instruction)
    for ref in references:
        start, end = ref["span"]
        masked[start:end] = " " * (end - start)
    subjects = set(named_objects("".join(masked), state))
    anchors = {ref["target"] for ref in references} - subjects
    preparations = {p['action_index']:p for p in (plan.planning_evidence or {}).get('preparations', [])
                    if p.get('kind') == 'ground_held_reference'}
    for index, action in enumerate(plan.program["actions"]):
        if action["object"] in anchors and not (index in preparations
                and action['op'] == 'place' and action['object'] == preparations[index]['object']):
            raise GroundingError(f"{action['object']} is only a placement reference; do not add a manipulation of it")
        if action["object"] in {name for _, _, name in preservation_spans(instruction, state)}:
            raise GroundingError(f"The instruction requires leaving {action['object']} in place")
    # Explicitly prohibited actions are constraints, never extra positive tasks.
    from backend.app.planning.language_fallback import VERBS
    prohibited, positive = set(), set()
    for clause in re.split(r"[。.!！,、;]+", instruction):
        verbs = {match.lastgroup for match in VERBS.finditer(clause)}
        if _prohibited_clause(clause):
            prohibited.update(verbs)
        else:
            positive.update(verbs)
    for action in plan.program["actions"]:
        if action["op"] in prohibited - positive:
            raise GroundingError(f"The instruction prohibits {action['op']}")
