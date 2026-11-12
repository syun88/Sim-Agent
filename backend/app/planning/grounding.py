"""One owner for entity grounding: every byte the instruction loses is a recorded decision.

Reference resolution used to be string surgery spread over two modules with no span, no basis and no
record of what it looked at. That is where this project kept failing: a duplicated actor particle
(`右手でblue_cube`) produced "Multiple acting hands in one clause" on 2026-11-07, and the mutation that
caused it was invisible in the plan evidence.

This module runs the declared passes in a fixed order and then **proves** that the resolved instruction
differs from the input only at recorded spans. An unrecorded rewrite raises `UNGROUNDED_EDIT` instead of
travelling on as a quietly different sentence: the failure becomes a refusal at planning time, not a
confusing parse or model error three layers later.

The decisions are the interface a reader (or a later learned model) needs: surface, span, basis, resolved
entity, candidates, and the observation the choice rested on.
"""

from __future__ import annotations

from backend.app.agent.scene_reasoning import SceneReferenceError, bind_scene_references
from backend.app.planning.spatial_language import resolve_relative_pronouns
from backend.app.planning.task_program import GroundingError, observed_holds

#: Why a decision was allowed to choose this entity. A basis is a rule about the scene, not a model
#: opinion, so it can be re-derived from the recorded observation.
BASIS = {
    'observed_hold': 'the arm was measured holding this object',
    'acquire_missing_hold': 'the only unheld cube in that arm workspace',
    'assumed_ambiguity_resolution': 'highest of two stacked cubes; only the top one can be grasped',
    'observed_spatial_reference': 'observed side of the table',
    'arm_reach_cube_binding': 'the only unheld cube in that arm workspace',
    'complement_cube_reference': 'the remaining cube after the one already named',
    'unmarked_contrastive_resolution': 'first/second unmarked cube in a contrastive pair',
    'opposite_hand_reference': 'the arm opposite the one acting or holding',
    'pronoun_reference': 'the unique object named in the preceding clause',
}

#: Fixed order. A pass may only read the text produced by the passes before it.
PASSES = ('scene_hands', 'relative_pronouns')


def _replay(text: str, decisions: list[dict]) -> str:
    """Rebuild the resolved text from the recorded spans alone."""
    out, cursor = [], 0
    for decision in sorted(decisions, key=lambda item: item['span'][0]):
        start, end = decision['span']
        if start < cursor:
            raise GroundingError(f"GROUNDING_OVERLAP: {decision.get('kind')} spans {start}-{end} "
                                 f"overlaps an earlier edit ending at {cursor}")
        out.append(text[cursor:start])
        out.append(decision['replacement'])
        cursor = end
    out.append(text[cursor:])
    return ''.join(out)


def _first_difference(left: str, right: str) -> str:
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            window = slice(max(0, index - 12), index + 12)
            return f"at {index}: ...{left[window]!r} vs ...{right[window]!r}"
    return f"lengths {len(left)} vs {len(right)}"


def ground(instruction: str, state: dict) -> dict:
    """Resolve entity references and return the resolved text with its decisions.

    Raises `SceneReferenceError` when a reference cannot be resolved (a scene fact is missing) and
    `GroundingError` when a pass rewrote text it did not record, or recorded an edit it did not make.
    """
    record = {
        'original': instruction,
        'instruction': instruction,
        'decisions': [],
        'basis': BASIS,
        'passes': list(PASSES),
        # What the decisions were allowed to read. A reader can re-derive every choice from this.
        'observed': {
            'sim_time': state.get('sim_time'),
            'hands': observed_holds(state),
            'objects': {o['id']: {'position': list(o.get('position') or []),
                                  'type': o.get('type', 'cube')}
                        for o in state.get('objects') or [] if 'id' in o},
            'available_arms': list((state.get('robot') or {}).get('available_arms') or ['left', 'right']),
        },
    }

    binding = bind_scene_references(instruction, state)
    scene_decisions = list(binding.decisions)
    # Each rewrite step is proved separately: a pass that rewrites more than once shifts offsets, and
    # checking the whole pass at once would either miss the shift or reject a correct edit.
    steps = list(binding.steps) or [{'label': 'scene_hands', 'before': instruction,
                                     'after': binding.instruction,
                                     'decision_index': [0, len(scene_decisions)]}]
    for step in steps:
        begin, end = step['decision_index']
        recorded = scene_decisions[begin:end]
        expected = _replay(step['before'], recorded)
        if expected != step['after']:
            raise GroundingError(f"UNGROUNDED_EDIT: step {step['label']} changed the instruction outside "
                                 'its recorded decisions ' + _first_difference(expected, step['after']))
        for decision in recorded:
            record['decisions'].append(_normalise(decision, 'scene_hands', step['before'], step['label']))

    pronoun_decisions: list[dict] = []
    resolved = resolve_relative_pronouns(binding.instruction, state, pronoun_decisions)
    expected = _replay(binding.instruction, pronoun_decisions)
    if expected != resolved:
        raise GroundingError('UNGROUNDED_EDIT: the pronoun pass changed the instruction outside its '
                             'recorded decisions ' + _first_difference(expected, resolved))
    for decision in pronoun_decisions:
        record['decisions'].append(_normalise(decision, 'relative_pronouns', binding.instruction,
                                              'relative_pronouns'))

    record['instruction'] = resolved
    record['steps'] = [step['label'] for step in steps] + ['relative_pronouns']
    return record


def _normalise(decision: dict, rule: str, text: str, step: str) -> dict:
    """One shape for every grounding decision, whichever pass produced it."""
    start, end = decision['span']
    resolved = {key: decision[key] for key in ('object', 'arm', 'receiver', 'relation', 'target',
                                               'paired_object')
                if decision.get(key) is not None}
    return {
        'rule': rule,
        'step': step,
        'kind': decision.get('kind'),
        'basis': BASIS.get(decision.get('kind'), 'unclassified'),
        # Offsets into that step's input text; `surface` is the same substring and survives any shift.
        'span': [start, end],
        'surface': decision.get('source') or text[start:end],
        'replacement': decision['replacement'],
        'resolved': resolved,
        'candidates': list(decision.get('candidates') or []),
        'alternatives': list(decision.get('alternatives') or []),
        'explanation': decision.get('explanation') or '',
    }
