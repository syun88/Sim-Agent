#!/usr/bin/env python3
"""CPU-only contract diff and recorded-event inspection; never connects to the live simulator."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app.execution.contract_evidence import ContractEvidence
from backend.app.models.schemas import AgentPlan
from backend.app.planning.preparation import compile_prepared_program
from backend.app.planning.task_contract import TaskContract, requested_program
from backend.app.planning.task_program import GroundingError, TaskProgram


def _plan(value, state):
    if 'steps' in value:
        return AgentPlan.model_validate(value)
    return compile_prepared_program(TaskProgram.model_validate(value), state)


def _diff(before, after, path=''):
    if isinstance(before, dict) and isinstance(after, dict):
        return [d for k in sorted(before.keys() | after.keys())
                for d in _diff(before.get(k), after.get(k), f'{path}/{k}')]
    if isinstance(before, list) and isinstance(after, list) and len(before) == len(after):
        return [d for i,(a,b) in enumerate(zip(before,after)) for d in _diff(a,b,f'{path}/{i}')]
    return [] if before == after else [{'path':path, 'before':before, 'after':after}]


def inspect_case(case):
    state = case['state']
    initial = _plan(case['initial_plan'], state)
    contract = TaskContract.freeze(case['instruction'], initial, requested_program(initial))
    ledger = ContractEvidence(contract)
    ignored = []
    for index, attempt in enumerate(case.get('attempts', []), 1):
        epoch = attempt['epoch']
        if epoch < ledger.epoch:
            ignored.append({'attempt':index, 'reason':'STALE_SCENE_EPOCH', 'epoch':epoch})
            continue
        if ledger.epoch < epoch:
            ledger.reset_epoch(epoch)
        plan = _plan(attempt['plan'], state)
        contract.admit(plan, ledger)
        plan = contract.execution_plan(plan, ledger)
        ledger.record(plan, attempt['result'], index)
    epoch = case.get('scene_epoch', ledger.epoch)
    if epoch < ledger.epoch:
        raise ValueError('scene_epoch cannot move backward')
    if ledger.epoch < epoch:
        ledger.reset_epoch(epoch)
    candidate = _plan(case.get('candidate_plan', case['initial_plan']), state)
    error = None
    try:
        contract.admit(candidate, ledger)
    except GroundingError as rejected:
        error = str(rejected)
    events = ledger.inspect_events()
    state_obligations = [o['id'] for o in contract.describe()['obligations'] if o['kind']=='final_state']
    # Offline receipt input is diagnostic data, not authenticated physical evidence. State predicates
    # and pour's current bottle/fluid predicates cannot be measured by this read-only CLI.
    status = ('UNSATISFIED' if any(e['status']=='UNSATISFIED' for e in events) else
              'UNKNOWN' if state_obligations or not events or any(e['status']=='UNKNOWN' for e in events) else
              'SATISFIED')
    code = 1 if error or status=='UNSATISFIED' else 2 if status=='UNKNOWN' else 0
    return {'mode':'offline_diagnostic_not_live_physical_proof', 'contract':contract.describe(),
            'candidate_admitted':error is None, 'rejection':error,
            'intent_diff':_diff(requested_program(initial).model_dump(),candidate.goal_program or candidate.program),
            'events':events, 'scene_epoch':ledger.epoch, 'ignored_receipts':ignored,
            'unobserved_state_obligations':state_obligations,
            'recorded_contract_status':status, 'exit_code':code}


def demo_case(name):
    state = {'objects':[{'id':'parcel_A','type':'cube','position':[.44,.15,1.031]},
                        {'id':'parcel_B','type':'cube','position':[.44,-.15,1.031]}],
             'targets':[], 'robot':{'left_arm':{'held_object':None}, 'right_arm':{'held_object':None}}}
    program = {'actions':[{'op':'throw','object':'parcel_B','arm':'right','height_cm':3.}]}
    plan = _plan(program,state)
    trace = [{'t':.24+i*.002, 'position':[0.,0.,1.2+i*.002-4.905*(i*.002)**2],
              'velocity':[0.,0.,1.-9.81*i*.002], 'contacts':[]} for i in range(65)]
    skills = [{'skill':s.skill,'object':s.object,'arm':s.arm,'action_index':s.action_index,'success':True,
               **({'trajectory':trace} if s.skill=='throw_object' and name!='missing-evidence' else {})}
              for s in plan.steps]
    candidate = program if name!='changed-object' else {'actions':[
        {'op':'throw','object':'parcel_A','arm':'right','height_cm':3.}]}
    # Do not propose a second throw after a receipt: an empty, observation-only plan carries the
    # remaining state goals, while the contract still owns the event requirement.
    observe = plan.model_copy(deep=True)
    observe.program = {'actions':[], 'goals':[], 'keep_in_place':[], 'assumptions':[], 'error':None}
    observe.steps = []
    observe.goal['checks'] = []
    return {'instruction':'Throw parcel_B upward using the right hand; required free rise is 3 cm.',
            'state':state, 'initial_plan':program,
            'candidate_plan':candidate if name in {'changed-object','stale-epoch'} else observe.model_dump(),
            'attempts':[] if name=='changed-object' else [{'epoch':0,'plan':program,'result':{'skills':skills}}],
            'scene_epoch':1 if name=='stale-epoch' else 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--input', type=Path, help='Local JSON: instruction, state, initial_plan, candidate_plan, attempts, scene_epoch')
    group.add_argument('--demo', choices=['measured-event','changed-object','missing-evidence','stale-epoch'])
    args = parser.parse_args()
    started = datetime.now(timezone.utc).isoformat()
    try:
        case = json.loads(args.input.read_text()) if args.input else demo_case(args.demo)
        report = inspect_case(case)
    except (ValueError, KeyError, TypeError) as error:
        report = {'exit_code':1, 'error':str(error), 'mode':'offline_diagnostic_not_live_physical_proof'}
    report.update(started_at=started, finished_at=datetime.now(timezone.utc).isoformat())
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return report['exit_code']


if __name__=='__main__':
    raise SystemExit(main())
