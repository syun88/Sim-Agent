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
    return references


def validate_instruction_relations(plan, instruction, state):
    references = placement_references(instruction, state)
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
    # A noun in 'on the red cube' denotes the support, not a command to pick red.
    masked = list(instruction)
    for ref in references:
        start, end = ref["span"]
        masked[start:end] = " " * (end - start)
    subjects = set(named_objects("".join(masked), state))
    anchors = {ref["target"] for ref in references} - subjects
    for action in plan.program["actions"]:
        if action["object"] in anchors:
            raise GroundingError(f"{action['object']} is only a placement reference; do not add a manipulation of it")
        if action["object"] in {name for _, _, name in preservation_spans(instruction, state)}:
            raise GroundingError(f"The instruction requires leaving {action['object']} in place")
    # Explicitly prohibited actions are constraints, never extra positive tasks.
    from backend.app.planning.language_fallback import VERBS
    prohibited, positive = set(), set()
    for clause in re.split(r"[。.!！,、;]+", instruction):
        verbs = {match.lastgroup for match in VERBS.finditer(clause)}
        if re.search(r"\b(?:do\s+not|don't|never)\b|渡さない|投げない|置かない", clause, re.I):
            prohibited.update(verbs)
        else:
            positive.update(verbs)
    for action in plan.program["actions"]:
        if action["op"] in prohibited - positive:
            raise GroundingError(f"The instruction prohibits {action['op']}")
