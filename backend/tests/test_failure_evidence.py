from copy import deepcopy
from types import SimpleNamespace

from backend.app.execution.failure_evidence import failure_observation, state_fingerprint
from backend.app.planning.task_contract import TaskContract
from backend.app.planning.task_program import TaskProgram
from backend.app.planning.preparation import compile_prepared_program


def scenario():
    state={'objects':[{'id':'parcel_cube','position':[.4,.1,1.03],
                       'bounds':[[.38,.08,1.01],[.42,.12,1.05]]},
                      {'id':'obstacle','position':[.5,.1,1.03]}],
           'robot':{'left_arm':{'held_object':None},'right_arm':{'held_object':None},
                    'end_effectors':{'right':{'position':[.4,.1,1.2]}}},
           'targets':[]}
    plan=compile_prepared_program(TaskProgram(actions=[{'op':'pick','object':'parcel_cube','arm':'right'}]),state)
    contract=TaskContract.freeze('pick the parcel',plan,TaskProgram.model_validate(plan.program))
    return state,plan,contract


def test_fingerprint_ignores_time_and_prose_but_tracks_relevant_poses_and_holds():
    state,_,_=scenario()
    baseline=state_fingerprint(state,obj='parcel_cube',arm='right')
    later=deepcopy(state)
    later['sim_time']=999
    later['discourse']={'note':'a different explanation'}
    assert baseline==state_fingerprint(later,obj='parcel_cube',arm='right')
    for path,value in [('object',[.41,.1,1.03]),('obstacle',[.52,.1,1.03]),('hand','parcel_cube'),('arm',[.42,.1,1.2])]:
        changed=deepcopy(later)
        if path=='object':changed['objects'][0]['position']=value
        if path=='obstacle':changed['objects'][1]['position']=value
        if path=='hand':changed['robot']['right_arm']['held_object']=value
        if path=='arm':changed['robot']['end_effectors']['right']['position']=value
        assert state_fingerprint(changed,obj='parcel_cube',arm='right')!=baseline,path


def test_base_pose_is_a_relevant_precondition_even_when_nested_robot_is_unchanged():
    state,_,_=scenario()
    state['base_pose']=[0.,0.,0.]
    moved=deepcopy(state)
    moved['base_pose']=[1.,0.,0.]
    assert state_fingerprint(state)!=state_fingerprint(moved)


def test_goal_search_historical_failure_is_not_its_terminal_failure():
    state,plan,contract=scenario()
    result={'success':False,'skills':[{'skill':'approach_object','success':False,
        'error':'IK_FAILURE','world_state':dict(state,sim_time=1)},
        {'skill':'observe_scene','success':True,'world_state':dict(state,sim_time=2)}],
        'verification':{'success':False,'failure_category':'GOAL_NOT_SATISFIED',
                        'execution_mode':'observed_goal_search'}}
    receipt=failure_observation(contract,0,1,result,state,dict(state,sim_time=2))
    assert receipt['phase']=='verification' and receipt['code']=='GOAL_NOT_SATISFIED'
    assert receipt['observed_sim_time']==2 and receipt['action_index'] is None
    # The final skill can also fail for a different cause than the terminal goal rejection.
    result['skills'].pop()
    receipt=failure_observation(contract,0,1,result,state,dict(state,sim_time=2))
    assert receipt['phase']=='verification' and receipt['code']=='GOAL_NOT_SATISFIED'


def test_skill_fingerprint_uses_its_own_preconditions_not_the_outer_attempt_start():
    state,plan,contract=scenario()
    moved=deepcopy(state)
    moved['objects'][1]['position']=[.6,.1,1.03]
    skill={'skill':'approach_object','object':'parcel_cube','arm':'right',
           'action_index':0,'step_index':1,'success':False,'error':'COLLISION_PATH',
           'before_world_state':moved,'world_state':moved}
    receipt=failure_observation(contract,0,1,{'skills':[skill],
        'verification':{'failure_category':'COLLISION_PATH'}},state,state)
    assert receipt['before_fingerprint']==state_fingerprint(moved,obj='parcel_cube',arm='right')
    assert receipt['before_fingerprint']!=state_fingerprint(state,obj='parcel_cube',arm='right')


def test_unknown_ik_is_not_asserted_as_reachability_and_no_measurement_is_invented():
    state,plan,contract=scenario()
    result={'success':False,'skills':[{'skill':'approach_object','object':'parcel_cube','arm':'right',
            'action_index':0,'step_index':0,'success':False,'error':'IK_FAILURE'}],
            'verification':{'failure_category':'IK_FAILURE'}}
    unknown=failure_observation(contract,0,1,result,state,state)
    assert unknown['classification']=='UNKNOWN' and unknown['measurement']=={}
    assert unknown['observed_sim_time'] is None
    assert unknown['before_fingerprint']==unknown['after_fingerprint']
    result['skills'][0]['evidence']={'unreachable_by_m':.023, 'levers':{'stroke_cm':'change'}}
    measured=failure_observation(contract,0,1,result,state,state)
    assert measured['classification']=='REACHABILITY'
    assert measured['measurement']=={'unreachable_by_m':.023}
    assert measured['raw']['levers']=={'stroke_cm':'change'}
    assert unknown['id']==measured['id']
    result['skills'][0]['error']='COLLISION_PATH'
    result['skills'][0]['evidence']={'candidate': 'screened'}
    blocked=failure_observation(contract,0,2,result,state,state)
    assert blocked['classification']=='PATH_COLLISION'
    assert 'collision_link' not in blocked['raw']
    assert blocked['id']!=measured['id']


def test_saved_receipt_replay_checks_provenance_without_touching_the_scene(tmp_path):
    import json
    import subprocess
    import sys
    from pathlib import Path
    from scripts.replay_failure_evidence import replay
    state,plan,contract=scenario()
    result={'success':False,'skills':[{'skill':'approach_object','object':'parcel_cube','arm':'right',
        'action_index':0,'step_index':0,'success':False,'error':'IK_FAILURE'}],
        'verification':{'failure_category':'IK_FAILURE'}}
    receipt=failure_observation(contract,0,1,result,state,state)
    saved={'evidence_ledger':[receipt], 'agent_attempts':[{'failure_evidence_id':receipt['id']}],
           'agent_trace':[{'kind':'execution_failed','evidence_id':receipt['id']}]}
    path=tmp_path/'result.json'
    path.write_text(json.dumps(saved))
    command=[sys.executable,str(Path(__file__).resolve().parents[2]/'scripts/replay_failure_evidence.py'),str(path)]
    observed=subprocess.run(command,capture_output=True,text=True,timeout=10)
    assert observed.returncode==0,observed.stderr
    assert json.loads(observed.stdout)['failures'][0]['unchanged_visible_inputs']
    saved['agent_trace'][0]['evidence_id']='invented'
    assert replay(saved)['findings']==['UNKNOWN_TRACE_EVIDENCE_ID']
    saved['agent_trace'][0]['evidence_id']=receipt['id']
    saved['evidence_ledger'][0]['known_inputs']['objects']['parcel_cube']['position'][0]=999
    assert replay(saved)['findings']==['INPUT_FINGERPRINT_MISMATCH']


def test_retry_receipt_uses_retry_precondition_not_the_first_attempt(monkeypatch):
    from backend.app.execution.task_executor import TaskExecutor
    state,plan,_=scenario()
    initial=deepcopy(state)
    retry_state=deepcopy(state)
    retry_state['objects'][1]['position']=[.6,.1,1.03]
    executor=TaskExecutor.__new__(TaskExecutor)
    executor.controller=SimpleNamespace(cancelled=False,should_stop=lambda:False,
                                        wait_gripper=lambda *args:None,stop_requested=lambda:False)
    executor.env=SimpleNamespace(object_contacts=lambda *args:[],contact_names=lambda:[])
    executor.world_state=lambda:retry_state
    class Skills:
        def execute_plan(self,steps):
            return [{'skill':'approach_object','success':False,'error':'IK_FAILURE'}]
        def held_object(self,arm):return None
        def approach_object(self,arm,obj,**kwargs):
            return {'skill':'approach_object','success':False,'error':'IK_FAILURE'}
    executor.skills=Skills()
    executor.discourse={}
    executor._monitor_step=lambda *args:None
    monkeypatch.setattr('backend.app.execution.task_executor.validate_plan',lambda *args:(True,[]))
    states=iter([initial,retry_state,retry_state,retry_state])
    monkeypatch.setattr('backend.app.execution.task_executor.build_world_state',lambda *_:next(states))
    monkeypatch.setattr('backend.app.execution.task_executor.verify_task_goal',
                        lambda *args,**kwargs:{'success':False,'failure_category':'IK_FAILURE'})
    result=executor.run(plan)
    assert result['skills'][0]['before_world_state']['objects'][1]['position']==[.6,.1,1.03]
    assert len(result['skills'][0]['attempts'])==2


def test_real_executor_validation_rejection_is_structured(monkeypatch):
    from backend.app.execution.task_executor import TaskExecutor
    state,plan,_=scenario()
    executor=TaskExecutor.__new__(TaskExecutor)
    executor.controller=SimpleNamespace(cancelled=False,should_stop=None)
    executor.world_state=lambda:state
    monkeypatch.setattr('backend.app.execution.task_executor.validate_plan',
                        lambda *args:(False,['Selected robot does not support grasp_object']))
    result=executor.run(plan)
    assert not result['success'] and result['skills']==[]
    assert result['preflight']=={'kind':'plan_validation','errors':['Selected robot does not support grasp_object']}
