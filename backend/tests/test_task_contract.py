"""Intent and receipt guards run without a Genesis world, a model, or a GPU."""
from copy import deepcopy
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import numpy as np
import pytest

from backend.app.execution.contract_evidence import ContractEvidence
from backend.app.planning.preparation import compile_prepared_program
from backend.app.planning.task_contract import TaskContract, requested_program
from backend.app.planning.task_program import GroundingError, TaskProgram


@pytest.fixture
def scene():
    return {'objects':[{'id':'parcel_A','type':'cube','position':[.44,.15,1.031]},
                       {'id':'parcel_B','type':'cube','position':[.44,-.15,1.031]},
                       {'id':'support_C','type':'cube','position':[.55,0.,1.031]},
                       {'id':'wine_bottle','type':'bottle','position':[.44,.29,1.09]},
                       {'id':'wine_glass','type':'glass','graspable':False,'position':[.55,.05,1.006]}],
            'targets':[{'id':'left_side','position':[.46,.30,1.008]},
                       {'id':'right_side','position':[.46,-.30,1.008]},
                       {'id':'wine_glass','position':[.55,.05,1.006]}],
            'robot':{'left_arm':{'held_object':None},'right_arm':{'held_object':None}},
            'fluid':{'bottle_ml':50.},'workspace':{'table_height_m':1.008}}


def compiled(scene, actions):
    return compile_prepared_program(TaskProgram(actions=actions),scene)


def bound(scene, actions, instruction='typed fixture: preserve these grounded intentions'):
    plan=compiled(scene,actions)
    contract=TaskContract.freeze(instruction,plan,requested_program(plan))
    return plan,contract,ContractEvidence(contract)


def execution(plan, **event_fields):
    return {'skills':[{'skill':s.skill,'object':s.object,'arm':s.arm,'action_index':s.action_index,
                       'success':True,**(event_fields if s.skill in {'throw_object','pour_liquid'} else {})}
                      for s in plan.steps]}


def inert_engine(scene):
    return SimpleNamespace(name='recorded-state-test-double',grasp_mode='physical-friction-pinch',
        object_pose=lambda obj:(np.array(next(o['position'] for o in scene['objects'] if o['id']==obj)),np.eye(3)))


@pytest.mark.parametrize('original,changed',[
    ({'op':'pick','object':'parcel_A','arm':'left'}, {'op':'pick','object':'parcel_B','arm':'left'}),
    ({'op':'pick','object':'parcel_A','arm':'left'}, {'op':'pick','object':'parcel_A','arm':'right'}),
    ({'op':'place','object':'parcel_A','target':'parcel_B','relation':'on'},
     {'op':'place','object':'parcel_A','target':'support_C','relation':'on'}),
    ({'op':'place','object':'parcel_A','target':'parcel_B','relation':'left_of'},
     {'op':'place','object':'parcel_A','target':'parcel_B','relation':'right_of'}),
    ({'op':'pour','object':'wine_bottle','target':'wine_glass','volume_ml':10.},
     {'op':'pour','object':'wine_bottle','target':'wine_glass','volume_ml':20.}),
])
def test_changed_fixed_bindings_are_rejected_before_execution(scene,original,changed):
    _,contract,ledger=bound(scene,[original])
    with pytest.raises(GroundingError,match='CONTRACT_VIOLATION'):
        contract.admit(compiled(scene,[changed]),ledger)


def test_unspecified_hand_is_not_frozen_to_the_first_plan_default(scene):
    first,contract,ledger=bound(scene,[{'op':'pick','object':'parcel_A'}])
    assert first.arm=='left'
    alternate=compiled(scene,[{'op':'pick','object':'parcel_A','arm':'right'}])
    contract.admit(alternate,ledger)
    assert contract.execution_plan(alternate,ledger).goal['checks']==[
        {'type':'holding','object':'parcel_A','arm':'right'}]


def test_executable_goal_plan_keeps_observed_goal_dispatch(scene,monkeypatch):
    from backend.app.execution.task_executor import TaskExecutor
    scene['objects'][1]['id']='payload_cube'  # Existing GoalDomain requires cube-suffixed IDs.
    source=TaskProgram(goals=[{'type':'holding','object':'payload_cube','arm':'right'}])
    plan=compile_prepared_program(source,scene)
    contract=TaskContract.freeze('hold parcel_B',plan,requested_program(plan))
    ledger=ContractEvidence(contract)
    executor=TaskExecutor.__new__(TaskExecutor)
    monkeypatch.setattr('backend.app.execution.goal_executor.run_goal_program',
                        lambda *a,**k:{'path':'observed_goal_execution'})
    assert executor.run(contract.execution_plan(plan,ledger))=={'path':'observed_goal_execution'}


def test_explicit_preservation_is_not_dropped_with_derived_protection(scene):
    scene['objects'][1]['id']='payload_cube'
    goals=[{'type':'holding','object':'payload_cube','arm':'right'}]
    source=TaskProgram(goals=goals,keep_in_place=['parcel_A'])
    plan=compile_prepared_program(source,scene)
    contract=TaskContract.freeze('hold payload_cube and keep parcel_A in place',plan,requested_program(plan))
    ledger=ContractEvidence(contract)
    contract.admit(plan,ledger)
    assert {'type':'preserved','object':'parcel_A','position':scene['objects'][0]['position']} in contract.checks()
    weakened=compile_prepared_program(TaskProgram(goals=goals),scene)
    with pytest.raises(GroundingError,match='CONTRACT_VIOLATION'):
        contract.admit(weakened,ledger)


def test_design_can_change_without_changing_the_required_event(scene):
    _,contract,ledger=bound(scene,[{'op':'throw','object':'parcel_B','arm':'right','height_cm':3.,'stroke_cm':20.}])
    candidate=compiled(scene,[{'op':'throw','object':'parcel_B','arm':'right','height_cm':3.,'stroke_cm':12.}])
    contract.admit(candidate,ledger)
    assert contract.execution_plan(candidate,ledger).steps[-1].stroke_cm==12.
    weaker=compiled(scene,[{'op':'throw','object':'parcel_B','arm':'right','height_cm':1.,'stroke_cm':12.}])
    with pytest.raises(GroundingError,match='CONTRACT_VIOLATION'):
        contract.admit(weaker,ledger)


def test_order_is_not_reduced_to_an_equal_set_of_final_states(scene):
    actions=[{'op':'place','object':'parcel_A','arm':'left','target':'left_side'},
             {'op':'place','object':'parcel_A','arm':'left','target':'right_side'}]
    _,contract,ledger=bound(scene,actions)
    # Dropping the explicit first placement leaves an identical final-state predicate.
    candidate=compiled(scene,actions[1:])
    assert candidate.goal['checks']==contract.checks()
    with pytest.raises(GroundingError,match='order changed'):
        contract.admit(candidate,ledger)


def test_contract_data_does_not_mutate_with_the_plan_or_display_copy(scene):
    plan,contract,_=bound(scene,[{'op':'throw','object':'parcel_B','arm':'right','height_cm':3.}])
    original=contract.describe()
    display=contract.describe()
    display['goal']['checks'][0]['min_height']=0.
    plan.goal['checks'].clear()
    assert contract.describe()==original
    with pytest.raises(FrozenInstanceError):
        contract.payload='{}'
    assert {o['kind'] for o in original['obligations']}=={'event','ordered_action'}


def test_missing_trajectory_and_old_epoch_cannot_verify_an_event(scene,upward_flight):
    plan,contract,ledger=bound(scene,[{'op':'throw','object':'parcel_B','arm':'right','height_cm':3.}])
    engine=inert_engine(scene)
    ledger.record(plan,execution(plan),1)
    assert not ledger.verify(engine,plan,scene)['success']
    # A model's success boolean alone is not measured flight evidence.
    measured=ContractEvidence(contract)
    measured.record(plan,execution(plan,trajectory=upward_flight),1)
    assert measured.verify(engine,plan,scene)['contract_status']=='SATISFIED'
    measured.reset_epoch()
    stale=measured.verify(engine,plan,scene)
    assert not stale['success'] and stale['contract_status']=='UNKNOWN'
    assert stale['missing_event_obligations']==[0]
    other_plan,_,other=bound(scene,[{'op':'throw','object':'parcel_B','arm':'right','height_cm':3.}])
    other.receipts.extend(measured.receipts)
    assert not other.verify(engine,other_plan,scene)['success'], 'same content is not the same task'


def test_a_success_claim_with_a_measured_drop_is_not_a_successful_throw(scene,upward_flight):
    plan,_,ledger=bound(scene,[{'op':'throw','object':'parcel_B','arm':'right','height_cm':3.}])
    drop=[{**row,'velocity':[0.,0.,-.1]} for row in upward_flight]
    ledger.record(plan,execution(plan,trajectory=drop),1)
    result=ledger.verify(inert_engine(scene),plan,scene)
    assert not result['success'] and result['contract_status']=='UNSATISFIED'
    assert ledger.inspect_events()[0]['measurement']['failure_category']=='NO_UPWARD_FREE_FLIGHT'


def test_event_binding_survives_a_resumed_action_index_change(scene,upward_flight):
    actions=[{'op':'place','object':'parcel_A','arm':'left','target':'left_side'},
             {'op':'throw','object':'parcel_B','arm':'right','height_cm':3.}]
    first,contract,ledger=bound(scene,actions)
    prefix=execution(first)
    prefix['skills']=[s for s in prefix['skills'] if s['action_index']==0]
    ledger.record(first,prefix,1)
    suffix=compiled(scene,actions[1:])
    # TaskAgent keeps all original final conditions when it resumes a suffix.
    suffix.goal=deepcopy(first.goal)
    contract.admit(suffix,ledger)
    current=contract.execution_plan(suffix,ledger)
    event=next(c for c in current.goal['checks'] if c['type']=='throw')
    assert event['action_index']==0
    ledger.record(current,execution(current,trajectory=upward_flight),2)
    assert ledger.has(1)
    receipt=ledger.receipts[0]
    assert receipt.attempt==2 and receipt.obligation==1
    import json
    assert json.loads(receipt.result)['action_index']==1
    assert contract.describe()['obligations'][1]['id']==contract.obligation_id(1)
    with pytest.raises(GroundingError,match='executed event would be repeated'):
        contract.admit(first,ledger)


def test_hit_uses_observed_event_target_not_a_later_target_pose(scene,upward_flight):
    from backend.app.verification.goal_verification import verify_task_goal
    plan=compiled(scene,[{'op':'throw','object':'parcel_B','arm':'right','target':'parcel_A'}])
    result=execution(plan,trajectory=upward_flight,target_position=upward_flight[0]['position'])
    engine=inert_engine(scene)
    engine.object_pose=lambda _:(_ for _ in ()).throw(AssertionError('must not read a historical target from live state'))
    measured=verify_task_goal(engine,plan,result['skills'])
    assert measured['success']
    result['skills'][-1].pop('target_position')
    missing=verify_task_goal(engine,plan,result['skills'])
    assert not missing['success'] and missing['checks'][0]['contract_status']=='UNKNOWN'


@pytest.mark.parametrize('demo,code,status',[
    ('measured-event',0,'SATISFIED'), ('changed-object',1,'UNKNOWN'),
    ('missing-evidence',2,'UNKNOWN'), ('stale-epoch',2,'UNKNOWN')])
def test_review_cli_reports_admission_and_measurement_separately(demo,code,status):
    import json
    from pathlib import Path
    import subprocess
    import sys
    process=subprocess.run([sys.executable,'scripts/check_task_contract.py','--demo',demo],
                           cwd=Path(__file__).resolve().parents[2],capture_output=True,text=True,timeout=20)
    report=json.loads(process.stdout)
    assert process.returncode==code,process.stdout+process.stderr
    assert report['recorded_contract_status']==status
    assert report['mode']=='offline_diagnostic_not_live_physical_proof'
    assert report['started_at'] <= report['finished_at']
    if demo=='changed-object':
        assert not report['candidate_admitted']
        assert any(d['after']=='parcel_A' for d in report['intent_diff'])
    if demo=='measured-event':
        assert report['events'][0]['measurement']['free_flight_rise']>.03


@pytest.mark.parametrize('instruction',['parcel_Aを持ったままparcel_Bを動かす',
                                         'Move parcel_B while holding parcel_A'])
def test_unobservable_during_is_not_accepted_as_final_holding(scene,instruction):
    plan,contract,ledger=bound(scene,[{'op':'pick','object':'parcel_A','arm':'left'}],instruction)
    with pytest.raises(GroundingError,match='CONTRACT_UNVERIFIABLE'):
        contract.admit(plan,ledger)
    assert any(o['kind']=='during' for o in contract.describe()['obligations'])
