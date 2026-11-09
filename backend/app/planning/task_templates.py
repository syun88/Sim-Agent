"""Composite tasks as data: a procedure written in JSON, not in code.

The grammar can already express "pick, hand over, place" one action at a time. What it cannot do is carry a
*new* composition - sweep three cubes into a zone, place two objects on one target, hand over then pour -
without someone editing the planner. A template is that composition as a file:

    {"id": "handover_then_place",
     "description": "pick with one hand, pass it to the other, then place it",
     "phrases": ["渡してから", "hand it over then"],
     "slots": {"object": "object", "zone": "zone", "giver": "arm", "receiver": "arm"},
     "requires": ["grasp", "handover", "place"],
     "goal": {"type": "place_in_zone", "object": "{object}", "target": "{zone}"},
     "actions": [
       {"op": "pick", "object": "{object}", "arm": "{giver}"},
       {"op": "handover", "object": "{object}", "arm": "{giver}", "receiver": "{receiver}"},
       {"op": "place", "object": "{object}", "target": "{zone}"}]}

A template is only a *proposal*: the expansion goes through the same grounding, compiler and contract as any
other plan, and nothing here may widen what the body can do. A slot that the scene does not contain, or a
`requires` the body does not report, refuses the template instead of guessing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_TASKS_DIR = Path(__file__).resolve().parents[3] / 'assets' / 'tasks'

#: What a slot may name. Grounding differs per kind, so an unknown kind is refused.
SLOT_KINDS = ('object', 'zone', 'arm', 'relation', 'number', 'text')
ACTIONS = ('pick', 'handover', 'place', 'approach', 'move', 'open', 'close', 'home', 'throw', 'pour',
           'joint', 'navigate')


class TemplateError(ValueError):
    """The template is unusable for this scene or this body. Never silently ignored."""

    def __init__(self, reason: str, detail: str = ''):
        super().__init__(f'{reason}: {detail}' if detail else reason)
        self.reason = reason
        self.detail = detail


@dataclass(frozen=True)
class Template:
    id: str
    description: str
    slots: dict
    requires: tuple
    goal: dict
    actions: tuple
    phrases: tuple = field(default_factory=tuple)
    source: str = ''

    def slot_names(self) -> list[str]:
        return list(self.slots)


def _template(payload: dict, source: str) -> Template:
    missing = [key for key in ('id', 'slots', 'actions') if key not in payload]
    if missing:
        raise TemplateError('INVALID_TEMPLATE', f'{source} lacks {missing}')
    slots = payload['slots']
    unknown = {kind for kind in slots.values() if kind not in SLOT_KINDS}
    if unknown:
        raise TemplateError('INVALID_TEMPLATE', f'{payload["id"]} declares unknown slot kinds {sorted(unknown)}')
    for action in payload['actions']:
        if action.get('op') not in ACTIONS:
            raise TemplateError('INVALID_TEMPLATE', f'{payload["id"]} uses unknown op {action.get("op")!r}')
    return Template(id=payload['id'], description=payload.get('description', ''),
                    slots=dict(slots), requires=tuple(payload.get('requires', ())),
                    goal=dict(payload.get('goal') or {}), actions=tuple(payload['actions']),
                    phrases=tuple(payload.get('phrases', ())), source=source)


def load(directory: str | Path | None = None) -> dict[str, Template]:
    """Every `*.json` in the tasks directory, keyed by id. A bad file is reported, not skipped."""
    root = Path(directory) if directory is not None else DEFAULT_TASKS_DIR
    if not root.is_dir():
        return {}
    templates = {}
    for path in sorted(root.glob('*.json')):
        try:
            payload = json.loads(path.read_text())
            entry = _template(payload, str(path))
        except (OSError, ValueError) as error:
            raise TemplateError('INVALID_TEMPLATE', f'{path}: {error}') from None
        templates[entry.id] = entry
    return templates


def phrase_hint(instruction: str, templates: dict[str, Template] | None = None) -> list[str]:
    """Templates whose declared phrasing appears in the instruction. A hint for the planner, never a decision."""
    templates = load() if templates is None else templates
    lowered = instruction.lower()
    return [entry.id for entry in templates.values()
            if any(phrase.lower() in lowered for phrase in entry.phrases)]


def _ground(kind: str, value, state: dict) -> str:
    objects = {item.get('id') for item in state.get('objects', []) if item.get('id')}
    zones = {item.get('id') for item in state.get('targets', []) if item.get('id')}
    robot = state.get('robot') or {}
    arms = set(robot.get('available_arms') or ['left', 'right'])
    if kind == 'object':
        if value not in objects:
            raise TemplateError('SLOT_NOT_OBSERVED', f'{value!r} is not in this scene')
        return value
    if kind == 'zone':
        if value not in zones and value not in objects:
            raise TemplateError('SLOT_NOT_OBSERVED', f'{value!r} is neither a zone nor an object here')
        return value
    if kind == 'arm':
        if value not in arms:
            raise TemplateError('SLOT_NOT_OBSERVED', f'{value!r} is not an arm of this body')
        return value
    return value


def expand(template: Template, values: dict, state: dict, capabilities: dict | None = None) -> dict:
    """Fill a template for this scene. Returns a proposal envelope; it is not an accepted plan."""
    missing = [name for name in template.slots if name not in values]
    if missing:
        raise TemplateError('SLOT_MISSING', f'{template.id} needs {missing}')
    grounded = {name: _ground(template.slots[name], values[name], state) for name in template.slots}
    skills = set((capabilities or {}).get('skills') or [])
    unavailable = [need for need in template.requires if skills and need not in skills]
    if unavailable:
        raise TemplateError('CAPABILITY_UNAVAILABLE', f'{template.id} needs {unavailable}, body has {sorted(skills)}')

    def fill(item):
        return {key: (grounded[value[1:-1]] if isinstance(value, str) and value.startswith('{')
                      and value.endswith('}') and value[1:-1] in grounded else value)
                for key, value in item.items() if value is not None}

    actions = [fill(dict(action)) for action in template.actions]
    goal = fill(dict(template.goal)) if template.goal else {}
    # `envelope` is shaped like any other proposal, so it is validated and compiled by the same path.
    return {'envelope': {'mode': 'actions', 'items': actions, 'assumptions': [f'template:{template.id}']},
            'template': template.id, 'goal': goal, 'requires': list(template.requires)}


def prompt_line(templates: dict[str, Template] | None = None) -> str:
    """One line for the model: which composite tasks exist and what they need."""
    templates = load() if templates is None else templates
    if not templates:
        return ''
    parts = [f'{entry.id} ({entry.description}; slots: {", ".join(entry.slot_names())})'
             for entry in templates.values()]
    return ('Composite procedures you may request instead of listing their steps: '
            + '; '.join(parts) + '. Reply {"template": "<id>", "slots": {...}} to use one. ')
