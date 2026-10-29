"""Control-flow tests; real contact/reset acceptance is verify_task_agent.py."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from backend.app.agent.scene_reasoning import SceneReferenceError, bind_scene_references
from backend.app.agent.task_agent import TaskAgent
from backend.app.planning.agent_planner import AgentPlanner
from backend.app.planning.preparation import compile_prepared_program
from backend.app.planning.task_program import GroundingError, TaskAction, TaskProgram, compile_program


@pytest.fixture
def world(monkeypatch):
    for key in ('AI_AND_API_KEY','AI_AND_BASE_URL','AI_AND_MODEL','OPENARM_LOCAL_PLANNER','OPENARM_FAST_LANGUAGE'):
        monkeypatch.delenv(key, raising=False)
    return {'objects':[{'id':'red_cube','type':'cube','position':[.44,.15,1.031]},
                       {'id':'blue_cube','type':'cube','position':[.50,-.22,1.031]}],
        'targets':[], 'robot':{'left_arm':{'held_object':None},'right_arm':{'held_object':None}},
        'workspace':{'table_height_m':1.008}}


def test_empty_hand_reference_proposes_acquisition_without_claiming_a_hold(world):
    result = bind_scene_references('右手にあるキューブを取って左手に渡して',world)
    assert result.instruction == '右手でblue_cubeを取って左手に渡して'
    assert result.decisions[0]['kind']=='acquire_missing_hold'
    assert world['robot']['right_arm']['held_object'] is None


def test_present_hold_outweighs_object_side_and_old_discourse(world):
    world['robot']['right_arm']['held_object']='red_cube'
    world['discourse']={'last_object':'blue_cube'}
    result=bind_scene_references('右手のキューブを左手に渡して',world)
    assert result.decisions[0]['object']=='red_cube'
    assert result.decisions[0]['kind']=='observed_hold'


def test_two_candidates_are_not_replaced_by_a_stale_discourse_default(world):
    world['objects'][0]['position'][1]=-.3
    world['discourse']={'last_object':'red_cube'}
    with pytest.raises(SceneReferenceError) as error:
        bind_scene_references('右手にあるキューブを取って左手に渡して',world)
    assert set(error.value.diagnosis['candidates'])=={'red_cube','blue_cube'}


def test_explicit_empty_hold_alias_cannot_fall_through_to_last_object(world):
    world['discourse']={'last_object':'red_cube'}
    with pytest.raises(GroundingError,match='hand is empty'):
        compile_program(TaskProgram(actions=[TaskAction(op='pick',object='held:right',arm='right')]),world)


def test_prepare_receiver_freezes_pronoun_before_staging_another_object(world):
    world['robot']['left_arm']['held_object']='red_cube'
    world['objects'][0]['held_by']=['left']
    requested=TaskProgram(actions=[TaskAction(op='pick',object='blue_cube',arm='right'),
        TaskAction(op='handover',object='it',arm='right',receiver='left')])
    with pytest.raises(GroundingError,match='cannot receive'):
        compile_program(requested,world)
    plan=compile_prepared_program(requested,world)
    actions=plan.program['actions']
    assert [a['op'] for a in actions]==['pick','place','home','handover']
    assert actions[1]['object']=='red_cube' and actions[-1]['object']=='blue_cube'
    assert len(plan.planning_evidence['preparations'])==1
    assert any(s.skill=='park_empty_arm' for s in plan.steps)


def test_goal_receiver_obstruction_becomes_an_explained_preparation_goal(world):
    for obj in world['objects']:
        x,y,z=obj['position']
        obj['bounds']=[[x-.023,y-.023,z-.023],[x+.023,y+.023,z+.023]]
    world['robot']['left_arm']['held_object']='red_cube'
    world['objects'][0]['position'][2]=1.15
    world['objects'][0]['bounds'][0][2]=1.127
    world['objects'][0]['bounds'][1][2]=1.173
    goal=TaskProgram(goals=[{'type':'holding','object':'blue_cube','arm':'left'}])
    plan=compile_prepared_program(goal,world)
    assert {(g['type'],g['object']) for g in plan.goal_program['goals']}=={('holding','blue_cube'),('placement','red_cube')}
    assert len(plan.planning_evidence['requested_program']['goals'])==1
    assert plan.planning_evidence['preparations'][0]['object']=='red_cube'


class FakeExecutor:
    def __init__(self,world,outcomes):
        self.state=deepcopy(world)
        self.default=deepcopy(world)
        self.outcomes=iter(outcomes)
        self.calls=[]
        self.resets=0
        self.homes=[]
        self.env=SimpleNamespace(reset=self.reset,step=lambda n:None,scene_profile='cubes')
    def reset(self):
        self.resets+=1
        self.state=deepcopy(self.default)
    def reset_context(self):
        self.state.pop('discourse',None)
    def world_state(self):return deepcopy(self.state)
    def home_empty_arms(self,**kwargs):
        self.homes.append(kwargs)
        return {'success':True,'status':'SUCCEEDED','arms':[]}
    def run(self,plan,**kwargs):
        self.calls.append(plan)
        outcome=next(self.outcomes)
        if callable(outcome):return outcome(self,plan)
        return {'success':outcome,'status':'SUCCEEDED' if outcome else 'FAILED','plan':plan.model_dump(),
            'skills':[{'skill':s.skill,'action_index':s.action_index,'success':outcome,
                       'error':None if outcome else 'MOTION_BLOCKED'} for s in plan.steps[:None if outcome else 1]],
            'verification':{'success':outcome,'failure_category':None if outcome else 'MOTION_BLOCKED'},'retry_count':0}


@pytest.fixture
def postconditions(monkeypatch):
    monkeypatch.setattr('backend.app.verification.goal_verification.verify_task_goal',
                        lambda *a,**k:{'success':True,'failure_category':None})


def test_failed_attempt_is_retained_after_in_place_recovery(world,postconditions):
    executor=FakeExecutor(world,[False,True])
    result=TaskAgent(executor,AgentPlanner()).run('青いキューブを掴んで')
    assert result['success'] and executor.resets==0
    assert [a['result']['success'] for a in result['agent_attempts']]==[False,True]
    assert result['verification']['all_agent_attempts_succeeded'] is False
    assert result['verification']['scene_resets']==0
    assert result['agent_recoveries'][0]['decision']=='recover_in_place'


def test_repeated_motion_failure_keeps_scene_and_stops(world):
    executor=FakeExecutor(world,[False,False,False])
    result=TaskAgent(executor,AgentPlanner()).run('青いキューブを掴んで')
    assert not result['success'] and executor.resets==0 and len(executor.calls)==3


def test_alternate_hand_keeps_object_identity_and_scene(world,postconditions):
    executor=FakeExecutor(world,[False,False,True])
    executor.state['objects'][1]['position'][1]=.22
    result=TaskAgent(executor,AgentPlanner()).run('青いキューブを掴んで')
    assert result['success']
    assert executor.calls[0].program['actions'][0]['arm']=='left'
    assert executor.calls[1].program['actions'][0]['arm']=='left'
    assert executor.calls[2].program['actions'][0]['arm']=='right'
    assert {p.program['actions'][0]['object'] for p in executor.calls}=={'blue_cube'}
    assert executor.resets==0


def test_stop_during_diagnosis_never_resets_or_retries(world):
    executor=FakeExecutor(world,[False,True])
    stopped=False
    def assess(*args):
        nonlocal stopped
        stopped=True
        return {'decision':'reset_and_retry','reason':'test'}
    result=TaskAgent(executor,AgentPlanner(),recovery_call=assess).run('青を取って',should_stop=lambda:stopped)
    assert result['status']=='STOPPED' and executor.resets==0 and len(executor.calls)==1


def test_ambiguous_reference_reports_requirements_without_reset(world):
    executor=FakeExecutor(world,[])
    executor.state['objects'][1]['position'][1]=.2
    result=TaskAgent(executor,AgentPlanner()).run('右手のキューブを左手に渡して')
    assert not result['success'] and not executor.calls and executor.resets==0
    assert result['diagnosis']['category']=='REFERENCE_NOT_PRESENT'
    assert result['diagnosis']['required']


def test_preservation_constraint_prevents_automatic_scene_reset(world):
    executor=FakeExecutor(world,[False,False])
    result=TaskAgent(executor,AgentPlanner()).run('赤には触らないで、青を右手で掴んで')
    assert not result['success'] and executor.resets==0 and len(executor.calls)==2


def test_resume_does_not_replay_a_completed_pick(world,postconditions):
    def first(executor,plan):
        executor.state['robot']['right_arm']['held_object']='blue_cube'
        skills=[{'skill':s.skill,'action_index':s.action_index,'success':s.action_index==0,
                 'error':None if s.action_index==0 else 'IK_FAILURE'} for s in plan.steps]
        return {'success':False,'status':'FAILED','skills':skills,'verification':{'success':False,'failure_category':'IK_FAILURE'}}
    executor=FakeExecutor(world,[first,True])
    result=TaskAgent(executor,AgentPlanner(),recovery_call=lambda *a:{'decision':'resume','reason':'continue remaining'}).run('右手で青を取って左手に渡して')
    assert result['success'] and executor.resets==0
    assert [a['op'] for a in executor.calls[1].program['actions']]==['handover']
    assert executor.calls[1].goal['checks'][0]['object']=='blue_cube'


def test_model_cannot_reset_a_usable_scene_even_when_it_requests_reset(world,postconditions):
    choices=[]
    def assess(instruction,state,failure,legal):
        choices.append(legal)
        return {'decision':'reset_and_retry','reason':'invalid early reset'}
    executor=FakeExecutor(world,[False,True])
    result=TaskAgent(executor,AgentPlanner(),recovery_call=assess).run('右手で青を取って')
    assert result['success'] and executor.resets==0
    assert all('reset_and_retry' not in legal for legal in choices)


def test_reset_requires_scene_loss_and_exhausted_in_place_options(world,postconditions):
    def fall(executor,plan):
        executor.state['objects'][1]['position'][2]=.02
        return {'success':False,'status':'FAILED','skills':[{'success':False,'error':'GRASP_LOST','action_index':0}],
                'verification':{'success':False,'failure_category':'GRASP_LOST'}}
    executor=FakeExecutor(world,[fall,False,True])
    result=TaskAgent(executor,AgentPlanner()).run('右手で青を取って')
    assert result['success'] and executor.resets==1
    assert [r['decision'] for r in result['agent_recoveries']]==['recover_in_place','reset_and_retry']
    reset=next(d for d in result['agent_trace'] if d['kind']=='reset_and_retry')
    assert reset['reset_evidence']['category']=='BELOW_TABLE_NO_RECOVERY_SKILL'


def test_home_failure_preserves_completed_task_without_replay_or_reset(world,postconditions):
    executor=FakeExecutor(world,[True])
    executor.home_empty_arms=lambda **kw:{'success':False,'status':'FAILED','failure_category':'PARK_COLLISION'}
    result=TaskAgent(executor,AgentPlanner()).run('青を中央に置いて')
    assert not result['success'] and len(executor.calls)==1 and executor.resets==0
    assert result['verification']['task_goal_before_homing']
    assert result['verification']['failure_category']=='POST_TASK_HOME_FAILED'


def test_next_prompt_plans_from_previous_result_after_homing(world,postconditions):
    observations=[]
    def stack(executor,plan):
        executor.state['objects'][1]['position']=[.44,.15,1.08]
        executor.state['support_relations']=[{'subject':'blue_cube','relation':'on','object':'red_cube'}]
        executor.state['discourse']={'last_object':'blue_cube'}
        return {'success':True,'status':'SUCCEEDED','skills':[], 'verification':{'success':True}}
    executor=FakeExecutor(world,[stack,True])
    planner=AgentPlanner()
    def plan(text,state,memory):
        observations.append(deepcopy(state))
        return planner.plan(text,state,memory)
    agent=TaskAgent(executor,planner,plan_call=plan)
    assert agent.run('青を赤の上に置いて')['success']
    second=agent.run('青を右側に置いて')
    assert second['success'] and executor.resets==0 and len(executor.homes)==2
    assert observations[1]['objects'][1]['position']==[.44,.15,1.08]
    assert observations[1]['support_relations'][0]['object']=='red_cube'
    assert second['agent_trace'][0]['kind']=='scene_observed'
    assert second['initial_world_state']['discourse']['last_object']=='blue_cube'


def test_a_stack_is_resolved_to_its_upper_cube_and_the_choice_is_recorded(world):
    """'Grab the blue cube' with two blue cubes: only the upper one can be grasped, so proceed.

    Stopping would hand back a task the scene already answers. What must not happen is an unrecorded
    guess: two cubes at the same height stay ambiguous (the guard above), and the stack case names the
    object it chose and the candidates it chose over.
    """
    world['objects'] = [
        {'id': 'blue_low', 'type': 'cube', 'position': [.44, -.1, 1.031]},
        {'id': 'blue_high', 'type': 'cube', 'position': [.44, -.1, 1.083]},
    ]
    result = bind_scene_references('右手にある青いキューブを取って', world)
    decision = result.decisions[0]
    assert decision['kind'] == 'assumed_ambiguity_resolution'
    assert decision['object'] == 'blue_high'
    assert decision['alternatives'] == ['blue_low']
    assert set(decision['candidates']) == {'blue_low', 'blue_high'}
    assert 'blue_high' in result.instruction


def test_two_cubes_at_the_same_height_still_stop_rather_than_guess(world):
    world['objects'] = [
        {'id': 'blue_a', 'type': 'cube', 'position': [.44, -.1, 1.031]},
        {'id': 'blue_b', 'type': 'cube', 'position': [.44, -.1, 1.031]},
    ]
    with pytest.raises(SceneReferenceError):
        bind_scene_references('右手にある青いキューブを取って', world)


def test_a_launch_that_never_reached_the_speed_retries_in_place(world,postconditions):
    """The jaws never opened, so the cube is still pinched: the same arm can retry it with another stroke.

    Measured 2026-10-28 on the cell: 12 cm reached 1.245 m/s and cleared 3 cm while 20 cm reached 0.472 m/s
    and could not, and the failure carries `stroke_cm` as the lever. What must not happen is a scene reset -
    nothing about the desk changed.
    """
    executor=FakeExecutor(world,[False,True],error='THROW_SPEED_NOT_REACHED')
    result=TaskAgent(executor,AgentPlanner()).run('青いキューブを投げて')
    assert result['success'] and executor.resets==0
    assert result['agent_recoveries'][0]['decision']=='recover_in_place'
