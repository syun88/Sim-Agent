"""Task-owned action progress and immutable event receipts for independent verification."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json

from backend.app.models.schemas import PlanStep
from backend.app.planning.task_contract import EVENT_SKILLS, source_actions, canonical_json


def completed_prefix(plan, result):
    count = 0
    for index, _ in enumerate((plan.program or {}).get('actions', [])):
        expected = [s for s in plan.steps if s.action_index == index]
        actual = [s for s in result.get('skills', []) if s.get('action_index') == index]
        if not expected or len(actual) != len(expected) or not all(s['success'] for s in actual):
            break
        count += 1
    return count


@dataclass(frozen=True)
class EventReceipt:
    task_id: str
    obligation: int
    epoch: int
    attempt: int
    result: str
    observed_at: float | None


class ContractEvidence:
    """A task-local owner of receipts; model-provided claims never enter here."""
    def __init__(self, contract):
        self.contract = contract
        self.epoch = 0
        self.completed = 0
        self.receipts = []

    def has(self, obligation):
        return any(r.task_id == self.contract.task_id and r.epoch == self.epoch and r.obligation == obligation
                   for r in self.receipts)

    def reset_epoch(self, new_epoch=None):
        new_epoch = self.epoch + 1 if new_epoch is None else new_epoch
        if not isinstance(new_epoch, int) or new_epoch <= self.epoch:
            raise ValueError('scene epoch must advance to an integer')
        self.epoch = new_epoch
        self.completed = 0

    def record(self, plan, result, attempt):
        start = self.contract.action_start(plan, self.completed)
        prefix = completed_prefix(plan, result)
        self.completed = max(self.completed, start + sum(i < prefix for i, _ in source_actions(plan)))
        unused = [(i,c) for i,c in enumerate(self.contract.checks()) if c['type'] in EVENT_SKILLS and not self.has(i)]
        for check in plan.goal.get('checks', []):
            if check['type'] not in EVENT_SKILLS:
                continue
            match = next(((i,c) for i,c in unused if self.contract._matches(i, check)), None)
            if not match:
                continue
            index, original = match
            event = next((r for r in result.get('skills', []) if r.get('skill') == EVENT_SKILLS[check['type']]
                          and r.get('action_index') == check.get('action_index')
                          and r.get('object') == check['object'] and r.get('success')
                          and any(s.skill == r['skill'] and s.action_index == r.get('action_index')
                                  and s.arm == r.get('arm') for s in plan.steps)), None)
            if event:
                event = deepcopy(event)
                event['action_index'] = original.get('action_index')
                self.receipts.append(EventReceipt(self.contract.task_id, index, self.epoch, attempt, canonical_json(event),
                                                  (event.get('world_state') or {}).get('sim_time')))
                unused.remove(match)

    def current_receipts(self):
        return [r for r in self.receipts if r.task_id == self.contract.task_id and r.epoch == self.epoch]

    def inspect_events(self):
        from backend.app.verification.event_evidence import missing_event_fields, verify_recorded_flight
        observations = []
        for index, check in enumerate(self.contract.checks()):
            if check['type'] not in EVENT_SKILLS:
                continue
            receipt = next((r for r in self.current_receipts() if r.obligation == index), None)
            event = json.loads(receipt.result) if receipt else {}
            missing = missing_event_fields(check, event)
            if not missing and check['type'] in {'throw', 'hit'}:
                measured = verify_recorded_flight(check, event)
                status = 'SATISFIED' if measured['success'] else 'UNSATISFIED'
            else:
                measured = {'reason':'missing_event_measurement' if missing else 'requires_live_fluid_and_bottle_state'}
                status = 'UNKNOWN'
            observations.append({'obligation':index, 'obligation_id':self.contract.obligation_id(index),
                                 'status':status, 'missing_fields':missing, 'measurement':measured})
        return observations

    def verify(self, env, plan, initial_state):
        from backend.app.verification.goal_verification import verify_task_goal
        check = self.contract.verification_plan(plan)
        inspection = self.inspect_events()
        missing = [r['obligation'] for r in inspection if r['missing_fields']]
        unknown = []
        known = []
        for index, condition in enumerate(check.goal.get('checks', [])):
            condition['obligation_id'] = self.contract.obligation_id(index)
            if index in missing:
                unknown.append({**condition, 'success':False, 'contract_status':'UNKNOWN',
                                'failure_category':'EVENT_EVIDENCE_MISSING'})
            else:
                known.append(condition)
        check.goal['checks'] = known
        events = [json.loads(r.result) for r in self.current_receipts() if r.obligation not in missing]
        # These step/result pairs are receipts, not new motor commands. State predicates are
        # read freshly; absent event measurements remain UNKNOWN even if the skills claimed success.
        check.steps = [PlanStep(skill=r['skill'], object=r['object'], action_index=r['action_index']) for r in events]
        result = verify_task_goal(env, check, events, initial_state=initial_state)
        status = 'UNSATISFIED' if not result['success'] else 'UNKNOWN' if missing else 'SATISFIED'
        if missing:
            result = {**result, 'success':False, 'failure_category':'EVENT_EVIDENCE_MISSING',
                      'checks':result.get('checks', []) + unknown}
        return {**result, 'contract_id':self.contract.contract_id, 'scene_epoch':self.epoch,
                'contract_status':status, 'missing_event_obligations':missing,
                'event_receipts':[{'obligation_id':self.contract.obligation_id(r.obligation),
                                   'attempt':r.attempt, 'epoch':r.epoch, 'observed_at':r.observed_at}
                                  for r in self.current_receipts()]}
