"""Immutable task intent, separate from replaceable plans and execution receipts.

This consumes an already grounded interpretation; it does not claim to understand arbitrary language.
Only the execution owner can add receipts. Replanning never supplies its own completion evidence.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import re
import uuid

from backend.app.planning.task_program import GroundingError, TaskProgram, TaskAction

EVENT_SKILLS = {'throw': 'throw_object', 'hit': 'throw_object', 'pour': 'pour_liquid'}
_INTENT_FIELDS = ('op', 'object', 'arm', 'receiver', 'target', 'relation', 'direction',
                  'distance_cm', 'height_cm', 'volume_ml', 'joint', 'position_deg')


def canonical_json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


def source_actions(plan):
    """Grounded user actions, excluding compiler-owned occupancy preparation."""
    inserted = {i for p in (plan.planning_evidence or {}).get('preparations', [])
                if p.get('action_index') is not None for i in (p['action_index'], p['action_index'] + 1)}
    return [(i, a) for i, a in enumerate((plan.program or {}).get('actions', [])) if i not in inserted]


def intent_checks(plan):
    # A compiler-added parking destination is a procedure choice, not a user final goal.
    evidence = plan.planning_evidence or {}
    # A requested intermediate pick is not a final table-placement obligation. Only an
    # explicitly requested final placement/hold of that object can freeze its destination.
    if plan.goal_program:
        explicit = {g['object'] for g in evidence.get('requested_program', {}).get('goals', [])}
    else:
        actions = [a for _,a in source_actions(plan)]
        last = {a.get('object'):(i,a) for i,a in enumerate(actions) if a.get('object')}
        explicit = {obj for obj,(i,a) in last.items()
                    if a['op'] == 'place' or (a['op'] in {'pick','handover'}
                    and not any(later['op']=='place' and later.get('relation')
                                and later.get('target')==obj for later in actions[i+1:]))}
    incidental = {(p['object'],p['target']) for p in evidence.get('preparations', [])
                  if p.get('kind') in {'free_hand', 'free_hand_goal','ground_held_reference'}
                  and p['object'] not in explicit}
    requested = evidence.get('requested_program') or plan.goal_program or plan.program or {}
    explicit_preservation = set(requested.get('keep_in_place', []))
    # Derived protection is re-observed by the goal executor and the contract's untouched-object
    # check. Its new baseline after a legitimate parking operation is not a new user requirement.
    return [c for c in plan.goal.get('checks', [])
            if not (c['type']=='placement' and (c['object'],c.get('target')) in incidental)
            and (c['type'] != 'preserved' or c['object'] in explicit_preservation)]


def requested_program(plan):
    """Retain grounded IDs while removing compiler-owned preparation and unrequested hands."""
    source = TaskProgram.model_validate(plan.goal_program or plan.program)
    evidence = plan.planning_evidence or {}
    if source.goals and evidence.get('requested_program'):
        return TaskProgram.model_validate(evidence['requested_program'])
    if source.actions:
        source.actions = [TaskAction.model_validate(a) for _, a in source_actions(plan)]
        requested = evidence.get('requested_program', {}).get('actions', [])
        if len(requested) == len(source.actions):
            for action, raw in zip(source.actions, requested):
                if raw.get('arm') is None and action.op != 'handover':
                    action.arm = None
    return source


@dataclass(frozen=True)
class TaskContract:
    task_id: str
    contract_id: str
    payload: str

    @classmethod
    def freeze(cls, instruction, plan, source):
        actions = [a.model_dump() for a in source.actions]
        checks = deepcopy(intent_checks(plan))
        goal = deepcopy(plan.goal)
        goal['checks'] = checks
        # Unspecified hands are choices, not constraints inherited from the first plan's default.
        choices = {}
        for index, check in enumerate(checks):
            relevant = [a for a in actions if a.get('object') == check.get('object')]
            last = relevant[-1] if relevant else None
            fields = []
            if last and last['op'] != 'handover' and last.get('arm') is None and 'arm' in check:
                fields.append('arm')
            if last and last['op'] == 'place' and last.get('target') is None and not last.get('relation'):
                fields.append('target')
            choices[str(index)] = fields
        # No continuous hold monitor exists yet. Do not silently downgrade "while holding" to a final hold.
        temporal = [m.group() for m in re.finditer(
            r'持ったまま|持ちながら|保持したまま|\bwhile\s+holding\b', instruction, re.I)]
        positions = {action_index:position for position,(action_index, _) in enumerate(source_actions(plan))}
        event_actions = {str(i):positions[c['action_index']] for i,c in enumerate(checks)
                         if c['type'] in EVENT_SKILLS}
        data = {'instruction': instruction,
                'interpretation': {'source':plan.provider, 'instruction_span':[0,len(instruction)],
                                   'resolution':'grounded_initial_plan_not_independent_language_proof'},
                'goal': goal, 'actions': actions,
                'goals': [g.model_dump() for g in source.goals], 'choices': choices,
                'allowed_object_motion': sorted({s.object for s in plan.steps if s.object} |
                                               {s.target for s in plan.steps if s.skill == 'pour_liquid' and s.target} |
                                               {g.object for g in source.goals}),
                'keep_in_place': list(source.keep_in_place), 'temporal': temporal, 'event_actions':event_actions}
        payload = canonical_json(data)
        return cls(str(uuid.uuid4()), hashlib.sha256(payload.encode()).hexdigest(), payload)

    def obligation_id(self, index):
        return f'{self.contract_id}/check/{index}'

    def describe(self):
        data = json.loads(self.payload)
        obligations = [{'id':self.obligation_id(i), 'kind':'event' if c['type'] in EVENT_SKILLS else 'final_state',
                        'condition':c} for i,c in enumerate(self.checks())]
        obligations.extend({'id':f'{self.contract_id}/order/{i}', 'kind':'ordered_action', 'condition':a}
                           for i,a in enumerate(data['actions']))
        obligations.extend({'id':f'{self.contract_id}/during/{i}', 'kind':'during', 'source_text':span,
                            'monitor':'UNSUPPORTED'} for i,span in enumerate(data['temporal']))
        return {'task_id': self.task_id, 'contract_id': self.contract_id, **data, 'obligations':obligations}

    def checks(self):
        return json.loads(self.payload)['goal'].get('checks', [])

    def _matches(self, index, candidate):
        data = json.loads(self.payload)
        expected = data['goal']['checks'][index]
        ignored = {'action_index', *data['choices'].get(str(index), [])}
        return {k:v for k,v in expected.items() if k not in ignored} == {
            k:v for k,v in candidate.items() if k not in ignored}

    def action_start(self, plan, completed):
        data = json.loads(self.payload)
        expected = data['actions']
        if not expected:
            return 0
        actual = [a for _, a in source_actions(plan)]
        for start in range(min(completed, len(expected)) + 1):
            if len(actual) != len(expected) - start:
                continue
            if all(all(e.get(k) is None or e.get(k) == a.get(k) for k in _INTENT_FIELDS)
                   for e, a in zip(expected[start:], actual)):
                if any(a['op'] in {'throw', 'pour'} for a in expected[start:completed]):
                    raise GroundingError('CONTRACT_VIOLATION: an executed event would be repeated')
                return start
        raise GroundingError('CONTRACT_VIOLATION: requested actions, bindings or order changed')

    def admit(self, plan, ledger):
        data = json.loads(self.payload)
        if data['temporal']:
            raise GroundingError('CONTRACT_UNVERIFIABLE: continuous holding evidence is not implemented')
        self.action_start(plan, ledger.completed)
        remaining = list(enumerate(self.checks()))
        for candidate in intent_checks(plan):
            match = next(((i,c) for i,c in remaining if self._matches(i, candidate)), None)
            if match is None:
                raise GroundingError('CONTRACT_VIOLATION: final conditions changed or were added')
            remaining.remove(match)
        for index, check in remaining:
            if check['type'] not in EVENT_SKILLS or not ledger.has(index):
                raise GroundingError('CONTRACT_VIOLATION: a required final condition was omitted')
        allowed = set(data['allowed_object_motion'])
        if any(s.object and s.object not in allowed for s in plan.steps):
            raise GroundingError('CONTRACT_VIOLATION: plan manipulates an unrequested object')

    def execution_plan(self, plan, ledger):
        """Attach fixed state goals and only pending events, with this plan's action indices."""
        current = self.verification_plan(plan)
        current.goal_program = deepcopy(plan.goal_program)
        data = json.loads(self.payload)
        start = self.action_start(plan, ledger.completed)
        pairs = source_actions(plan)
        pending = []
        for index, check in enumerate(current.goal.get('checks', [])):
            if check['type'] in EVENT_SKILLS:
                if ledger.has(index):
                    continue
                position = data['event_actions'][str(index)] - start
                if position < 0 or position >= len(pairs):
                    raise GroundingError('CONTRACT_VIOLATION: pending event has no action')
                check['action_index'] = pairs[position][0]
            pending.append(check)
        current.goal['checks'] = pending
        return current

    def verification_plan(self, plan):
        data = json.loads(self.payload)
        check = plan.model_copy(deep=True)
        check.goal = data['goal']
        check.goal['allowed_object_motion'] = data['allowed_object_motion']
        for index, original in enumerate(check.goal.get('checks', [])):
            candidate = next((c for c in plan.goal.get('checks', []) if self._matches(index, c)), None)
            if candidate:
                for field in data['choices'].get(str(index), []):
                    original[field] = candidate[field]
        check.goal_program = None
        return check

