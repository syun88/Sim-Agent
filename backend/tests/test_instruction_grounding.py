"""Regressions for wrong accepted intentions, independent of physical controllers."""
from copy import deepcopy

import pytest

from backend.app.planning.agent_planner import AgentPlanner, PlannerProviderError
from backend.app.planning.goal_language import relative_conditions


@pytest.fixture
def world(monkeypatch):
    for key in ('AI_AND_API_KEY','AI_AND_BASE_URL','AI_AND_MODEL','OPENARM_LOCAL_PLANNER','OPENARM_FAST_LANGUAGE'):
        monkeypatch.delenv(key, raising=False)
    return {'objects':[{'id':'red_cube','position':[.44,.15,1.031]},
                       {'id':'blue_cube','position':[.50,-.22,1.031]}],
            'targets':[{'id':'center','position':[.50,0,1.008]}]}


@pytest.mark.parametrize('text,object_id,arm', [
    ('右側にある青いキューブを左手で取って','blue_cube','left'),
    ('左側の赤いキューブを右手で取って','red_cube','right'),
    ('Pick up the blue cube on the right with your left hand.','blue_cube','left'),
    ('Pick the red cube on the left using the right hand.','red_cube','right'),
    ('左手が右側の青を取って','blue_cube','left'),
])
def test_actor_is_distinct_from_source_location(world, text, object_id, arm):
    plan = AgentPlanner().plan(text, world)
    assert plan.goal['checks'] == [{'type':'holding','object':object_id,'arm':arm}]


@pytest.mark.parametrize('text,arm,target', [
    ('Put blue on the left side using the right hand.','right','left_side'),
    ('左手で青を右側に置いて','left','right_side'),
])
def test_actor_is_distinct_from_destination(world, text, arm, target):
    plan = AgentPlanner().plan(text, world)
    assert plan.arm == arm and plan.goal['target'] == target


@pytest.mark.parametrize('quantity', ['100 millimeters','100 millimetres','100 mm','10 centimeters','0.1 meters'])
def test_explicit_motion_units_are_converted_before_range_validation(world, quantity):
    plan = AgentPlanner().plan('Move the right arm upward by '+quantity+'.', world)
    step = next(s for s in plan.steps if s.skill=='move_relative')
    assert step.arm == 'right' and step.direction == 'up' and step.distance_cm == 10


@pytest.mark.parametrize('text', ['机を拭いてから青いキューブを掴んで','机を拭いた後に青いキューブを取って',
                                '机を拭いて。青いキューブを掴んで'])
def test_unknown_earlier_task_is_not_dropped_by_fast_path(world, text):
    with pytest.raises(PlannerProviderError):
        AgentPlanner().plan(text, world)


def test_failed_grounding_does_not_reclassify_a_procedure_as_unordered_goals(world, monkeypatch):
    planner = AgentPlanner()
    planner.enabled = planner.local_enabled = True
    stages = []
    def response(*args):
        stages.append(planner._interpret_meaning)
        return {'payload':{'actions':[{'op':'pick','object':'blue_cube','arm':'right'},
             {'op':'handover','object':'blue_cube','arm':'right','receiver':'left'},
             {'op':'place','object':'blue_cube','arm':'left','target':'red_cube','relation':'on'}]},'raw':'test'}
    monkeypatch.setattr(planner, '_ai_plan', response)
    plan = planner.plan('After picking up blue with the right hand, pass it to the left hand and put it on red.', world)
    assert stages == [False]
    assert [a['op'] for a in plan.program['actions']] == ['pick','handover','place']


def test_duplicated_final_object_is_rejected_before_model_or_motion(world, monkeypatch):
    planner = AgentPlanner()
    planner.enabled = True
    monkeypatch.setattr(planner, '_ai_plan', lambda *args: pytest.fail('Contradiction must not reach provider'))
    with pytest.raises(PlannerProviderError, match='two separate hands'):
        planner.plan('最後は青を左手と右手でそれぞれ一個ずつ持って', world)


def test_model_must_not_add_an_unrequested_registered_object(world, monkeypatch):
    # Valid schema and feasible geometry do not make a different task acceptable.
    world = deepcopy(world)
    for obj in world['objects']:
        x,y,z = obj['position']
        obj.update(type='cube', bounds=[[x-.023,y-.023,z-.023],[x+.023,y+.023,z+.023]])
    planner = AgentPlanner()
    planner.enabled = planner.local_enabled = True
    wrong = {'goals':[{'type':'holding','object':'blue_cube','arm':'right'},
                      {'type':'holding','object':'red_cube','arm':'left'}]}
    monkeypatch.setattr(planner, '_ai_plan', lambda *args: {'payload':wrong,'raw':'unrequested extra goal'})
    with pytest.raises(PlannerProviderError):
        planner.plan('青いやつを右手で保持した状態にしてもらえる？', world)
    assert 'unrequested' in planner._repair_feedback


def test_top_bottom_are_subject_roles_not_two_opposite_supports(world):
    expected = [{'type':'relative_placement','object':'blue_cube','target':'red_cube','relation':'on'}]
    assert relative_conditions('赤が下、青が上になるようにして', world) == expected
    assert relative_conditions('red is bottom, blue is top', world) == expected
    assert relative_conditions('青が下、赤が上になるようにして', world)[0]['object'] == 'red_cube'
    assert relative_conditions('最後は赤を下、青を上にしておいて。',world)==expected


def test_possessive_side_is_not_misread_as_vertical_support(world):
    assert relative_conditions("Put blue next to red, on red's right.",world)==[
        {'type':'relative_placement','object':'blue_cube','target':'red_cube','relation':'right_of'}]


def test_hold_request_and_preserved_object_are_separate_roles(world):
    plan=AgentPlanner().plan('赤には触らないで、青だけ右手に持ってて',world)
    assert plan.goal['checks']==[{'type':'holding','object':'blue_cube','arm':'right'}]


def test_unknown_requested_work_cannot_be_erased_by_using_a_final_goal(world, monkeypatch):
    world=deepcopy(world)
    for obj in world['objects']:
        x,y,z=obj['position']
        obj.update(type='cube',bounds=[[x-.023,y-.023,z-.023],[x+.023,y+.023,z+.023]])
    planner=AgentPlanner()
    planner.enabled=planner.local_enabled=True
    monkeypatch.setattr(planner,'_ai_plan',lambda *a:{'payload':{'goals':[
        {'type':'holding','object':'blue_cube','arm':'right'}]},'raw':'dropped wiping task'})
    audits=[]
    def audit(*args,**kwargs):
        audits.append(args)
        return '{"covered":false,"omitted_or_unsupported":["机を拭いて"],"reason":"No wiping controller"}'
    monkeypatch.setattr(planner,'_complete',audit)
    with pytest.raises(PlannerProviderError):
        planner.plan('青を右手で取って、そのあと机を拭いて',world)
    assert audits and 'INCOMPLETE_SEMANTIC_COVERAGE' in planner._repair_feedback
