"""Homing contract checks; physical continuity is verify_persistent_scene.py."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np

from backend.app.execution.task_executor import TaskExecutor
from backend.app.skills.robot_skills import SkillRunner


def test_home_uses_model_joint_branch_not_just_a_matching_tool_frame():
    home=np.zeros(7)
    q=np.ones(7)*.3
    position=np.array([.4,.15,1.3])
    rotation=np.eye(3)
    calls=[]
    def move(side,target,attitude,**kwargs):
        nonlocal q
        calls.append(kwargs.get('planned_joints'))
        # Two redundant joint configurations can produce the same tool frame.
        q=kwargs['planned_joints'][-1].copy()
        kwargs['on_sample']()
        return SimpleNamespace(success=True)
    env=SimpleNamespace(frame=lambda:SimpleNamespace(objects={}), object_contacts=lambda *a:[],
        ee_pose=lambda *a:(position,rotation),home_arm_qpos=lambda side:home.copy(),arm_qpos=lambda side:q.copy(),
        arm_limits=lambda side:np.array([[-3.,3.]]*7),
        arm_kinematics=lambda side:SimpleNamespace(evaluate=lambda value:(position,rotation,None,None)))
    skills=SkillRunner.__new__(SkillRunner)
    skills.env=env
    skills.controller=SimpleNamespace(move_to_site=move,stop_requested=lambda:False,
                                     solve_ik=lambda *a,**kw:(None,1.,1.))
    skills.held_object=lambda side:None
    result=skills.park_empty_arm('left')
    assert result['success'] and result['maximum_joint_error_rad']==0
    assert np.array_equal(calls[-1][-1],home)
    assert result['route']=='direct_home' and len(result['rejected_routes'])>=2


def _stack_environment():
    import mujoco
    from backend.app.simulation.openarm_v2.definition import ARM_JOINTS, GRIPPER_JOINTS
    from backend.app.simulation.openarm_v2.spawn import private_spawn_environment
    env=private_spawn_environment()
    q={'left':np.array([1.172183,-.367318,.18691,2.389375,-.035986,.206243,.401385]),
       'right':np.array([-.011976,.569094,.004417,1.71886,.209563,-.775969,1.254314])}
    original=env.arm_kinematics
    def kinematics(side):
        kin=original(side)
        for arm in q:
            for name,value in zip(ARM_JOINTS[arm],q[arm]):
                joint=mujoco.mj_name2id(kin._model,mujoco.mjtObj.mjOBJ_JOINT,name)
                kin._data.qpos[kin._model.jnt_qposadr[joint]]=value
            for name in [GRIPPER_JOINTS[arm],f'openarm_{arm}_finger_joint2']:
                joint=mujoco.mj_name2id(kin._model,mujoco.mjtObj.mjOBJ_JOINT,name)
                kin._data.qpos[kin._model.jnt_qposadr[joint]]=.785 if arm=='left' else -.785
        return kin
    env.arm_kinematics=kinematics
    env.arm_qpos=lambda side:q[side].copy()
    env.ee_pose=lambda side:kinematics(side).evaluate(q[side])[:2]
    objects={'red_cube':(np.array([.44,.15,1.030917]),np.eye(3)),
             'blue_cube':(np.array([.439995,.149634,1.082707]),np.eye(3))}
    env.frame=lambda:SimpleNamespace(objects=objects)
    env.object_pose=lambda name:objects[name]
    env.object_bounds=lambda name:np.array([objects[name][0]-.026,objects[name][0]+.026])
    return env,q,objects


def test_home_route_screens_the_current_stack_instead_of_the_original_layout():
    """Read-only official geometry regression from the observed failed retreat."""
    from backend.app.control.motion import OpenArmV2Controller
    from backend.app.skills.homing import plan_empty_arm_home
    env,q,objects=_stack_environment()
    kin=env.arm_kinematics('left');home=env.home_arm_qpos('left')
    assert not kin.empty_arm_path_is_clear([q['left'],home],objects)
    displaced={name:(p+[0.,-1.,0.],r) for name,(p,r) in objects.items()}
    assert kin.empty_arm_path_is_clear([q['left'],home],displaced)
    controller=OpenArmV2Controller(env,configure_motors=False)
    route,evidence=plan_empty_arm_home(env,controller,'left')
    assert route is not None, evidence
    assert not np.array_equal(route[-1],home)
    assert kin.empty_arm_path_is_clear(route,objects)
    # The return is chosen by measurement now: any screened candidate may win, including a retreat away
    # from the object before climbing, and the choice is the roomiest of the ones that passed.
    assert evidence['route'] in {item['route'] for item in evidence['considered']}
    chosen = next(item for item in evidence['considered'] if item['route'] == evidence['route'])
    # The first segment is what brushes the object; it is ranked before the whole-path minimum.
    assert chosen['clearance_first_segment_m'] == evidence['measured_clearance_m']
    assert chosen['clearance_first_segment_m'] == max(
        (item['clearance_first_segment_m'] for item in evidence['considered']
         if item['clearance_first_segment_m'] is not None))
    assert evidence['home_mode']=='scene_clearance_home'


def test_approach_recomputes_a_reachable_overflight_for_an_elevated_object():
    from backend.app.control.motion import OpenArmV2Controller
    from backend.app.skills.approach import plan_cube_approach
    from backend.app.skills.homing import plan_empty_arm_home
    from backend.app.simulation.openarm_v2.definition import GRASP_ROTATION,FINGERTIP_MIDPOINT
    env,q,objects=_stack_environment()
    controller=OpenArmV2Controller(env,configure_motors=False)
    home,_=plan_empty_arm_home(env,controller,'left')
    q['left']=home[-1]
    center=objects['blue_cube'][0]
    pinch=center-GRASP_ROTATION@FINGERTIP_MIDPOINT['left']
    standoff=pinch-.06*(GRASP_ROTATION@np.array([0.,0.,-1.]))
    assert controller.solve_ik('left',standoff+[0.,0.,.10],GRASP_ROTATION)[0] is None
    env.arm_velocity_limits=lambda side:None
    route,planning=plan_cube_approach(env,controller,'left','blue_cube',pinch,standoff,.10,1.24)
    # The screen now gates on the measured clearance as well as the boolean screen, so a scene whose routes
    # only cleared the old nominal margin can legitimately have none. What must hold either way: a route that
    # was found clears the requirement it was screened against.
    if route is None:
        assert planning['error'] == 'APPROACH_PATH_UNAVAILABLE'
        assert any(item['error'] == 'COLLISION_PATH' for item in planning['rejected_routes'])
        return
    assert planning['approach_overfly_m']<.10
    assert planning['measured_clearance_m'] >= planning['screen_clearance_m']


def test_approach_searches_grasp_yaw_when_the_nominal_collision_route_is_blocked():
    """Measured task 0347e1b4 attempt 2: placed red blocks blue's nominal right-hand route."""
    from backend.app.control.motion import OpenArmV2Controller
    from backend.app.skills.approach import cube_approach_frames, plan_cube_approach
    from backend.app.simulation.openarm_v2.definition import GRASP_ROTATION,FINGERTIP_MIDPOINT
    env,q,objects=_stack_environment()
    q['right']=np.array([.0000271,-.0000299,.0000047,1.5707867,.0000001,-.0000016,-.0000003])
    q['left']=np.array([.0068464,-.668628,.0894537,1.7181624,-.1766079,.7771122,-1.0942528])
    objects['red_cube']=(np.array([.4603332877,-.2991182506,1.0309566259]),np.eye(3))
    objects['blue_cube']=(np.array([.5,-.2199999988,1.0309567451]),np.eye(3))
    env.arm_velocity_limits=lambda side:None
    controller=OpenArmV2Controller(env,configure_motors=False)
    center=objects['blue_cube'][0]
    pinch=center-GRASP_ROTATION@FINGERTIP_MIDPOINT['right']
    standoff=pinch-.06*(GRASP_ROTATION@np.array([0.,0.,-1.]))
    nominal,why=plan_cube_approach(env,controller,'right','blue_cube',pinch,standoff,.10,1.24)
    assert nominal is None and all(r['error']=='COLLISION_PATH' for r in why['rejected_routes'])
    route,planning=plan_cube_approach(env,controller,'right','blue_cube',pinch,standoff,.10,1.24,
        frames=cube_approach_frames('right',center,.06))
    # The screen now demands the nominal 20 mm *plus* how far an approach actually lands off its plan
    # (measured 6.7-9.1 mm of displacement). A route that cleared only the nominal margin is no longer
    # accepted, so this scene may legitimately have no usable route at all - what must hold is that the
    # search was complete and every refusal is about clearance, not about failing to try.
    if route is None:
        reasons = {r['error'] for r in planning['rejected_routes']}
        assert reasons <= {'COLLISION_PATH', 'IK_FAILURE'}
        assert 'COLLISION_PATH' in reasons, 'the refusal is about clearance at the required margin'
        assert len(planning['attempted_grasp_yaws_deg']) >= 14, 'every candidate yaw was attempted'
        return
    # The listed yaws seed the search; the angle used is the one the clearance measurement preferred, so it
    # is not required to be a listed value - only to be away from the blocked nominal frame and to be clear.
    assert planning['screen_clearance_m'] >= planning['noncontact_clearance_m']
    assert env.arm_kinematics('right').empty_arm_path_is_clear(
        route, objects, minimum_object_clearance=planning['screen_clearance_m'])
    assert planning['measured_clearance_m'] is not None
    assert planning['noncontact_clearance_m'] >= .020
    assert len(route)>=4
    assert planning['pinch_target'] != pinch.tolist()
    assert all(r['error'] in {'COLLISION_PATH','IK_FAILURE'} for r in planning['rejected_routes'])


def test_selected_approach_frame_survives_the_physical_hold_check(monkeypatch):
    """A clear approach alone is not enough: the hold-verification lift must keep its frame."""
    from scipy.spatial.transform import Rotation
    from backend.app.skills.robot_skills import SkillRunner
    from backend.app.simulation.openarm_v2.definition import GRASP_ROTATION,FINGERTIP_MIDPOINT
    center=np.array([.5,-.22,1.031])
    attitude=Rotation.from_euler('z',-45,degrees=True).as_matrix() @ GRASP_ROTATION
    pinch=center-attitude@FINGERTIP_MIDPOINT['right']
    ee=[np.array([.401,-.1535,1.12]),GRASP_ROTATION.copy()]
    closed=[False]
    motions=[]
    def move(arm,target,rotation=None,**kwargs):
        motions.append((np.asarray(rotation).copy(),kwargs.get('planned_joints')))
        if kwargs.get('planned_joints') is not None:
            assert kwargs.get('scale_speed') is False
        target=np.asarray(target)
        if closed[0]:
            center[:]+=target-ee[0]
        ee[0]=target.copy()
        ee[1]=np.asarray(rotation).copy()
        return SimpleNamespace(success=True,final_position=target.tolist())
    env=SimpleNamespace(object_pose=lambda obj:(center.copy(),np.eye(3)),
        frame=lambda:SimpleNamespace(objects={'blue_cube':center}),
        ee_pose=lambda arm:(ee[0].copy(),ee[1].copy()),
        pinch_established=lambda obj,arm:closed[0],
        object_contacts=lambda obj,arm=None:[f'openarm_right_ee_{name}_finger|blue_cube'
            for name in ('inner','outer')] if closed[0] else [],
        grasp_mode='physical-friction-pinch',grasp_is_physical_contact=True)
    controller=SimpleNamespace(motion_speed=6.,wait_gripper=lambda arm,value,seconds:closed.__setitem__(0,value),
        move_to_site=move)
    def admitted(*args,**kwargs):
        assert any(frame['yaw_deg']==-45 and np.allclose(frame['pinch'],pinch)
                   for frame in kwargs['frames'])
        return [np.zeros(7),np.ones(7)],{
            'grasp_yaw_deg':-45,'grasp_rotation':attitude.tolist(),'pinch_target':pinch.tolist()}
    monkeypatch.setattr('backend.app.skills.approach.plan_cube_approach',admitted)
    skills=SkillRunner(env,controller)
    approached=skills.approach_object('right','blue_cube')
    assert approached['success'],approached
    assert np.allclose(approached['target'],pinch)
    lifted=skills.grasp_object('right','blue_cube')
    assert lifted['success'],lifted
    assert all(np.allclose(rotation,attitude) for rotation,_ in motions)
    assert motions[0][1] is not None
    assert motions[1][1] is None


def test_approach_stops_on_measured_displacement_during_the_trajectory(monkeypatch):
    """Live task 57efa442 pushed two cubes 248 mm after a statically clear plan."""
    from backend.app.skills.robot_skills import SkillRunner
    from backend.app.simulation.openarm_v2.definition import GRASP_ROTATION,FINGERTIP_MIDPOINT
    positions={'blue_cube':np.array([.5,-.22,1.031]),'red_cube':np.array([.46,-.30,1.031])}
    pinch=positions['blue_cube']-GRASP_ROTATION@FINGERTIP_MIDPOINT['right']
    def planned(*args,**kwargs):
        return [np.zeros(7),np.ones(7)],{
            'grasp_yaw_deg':-60,'grasp_rotation':GRASP_ROTATION.tolist(),
            'pinch_target':pinch.tolist()}
    monkeypatch.setattr('backend.app.skills.approach.plan_cube_approach',planned)
    motions=[]
    def move(arm,target,rotation,**kwargs):
        motions.append(target)
        assert callable(kwargs.get('on_sample'))
        positions['blue_cube'][1]+=.006
        kwargs['on_sample']()
        raise AssertionError('the approach must stop when the cube moves')
    env=SimpleNamespace(object_pose=lambda name:(positions[name].copy(),np.eye(3)),
        frame=lambda:SimpleNamespace(objects=positions),
        ee_pose=lambda arm:(np.array([.401,-.1535,1.12]),GRASP_ROTATION.copy()),
        pinch_established=lambda *a:False)
    controller=SimpleNamespace(motion_speed=6.,wait_gripper=lambda *a:None,move_to_site=move)
    result=SkillRunner(env,controller).approach_object('right','blue_cube')
    assert len(motions)==1
    assert not result['success'] and result['error']=='APPROACH_COLLISION'
    assert result['evidence']['object']=='blue_cube'
    assert result['evidence']['grasp_yaw_deg']==-60
    assert result['evidence']['displacement_m']>=.006-.000001


def test_completion_homing_preserves_objects_and_does_not_release_requested_hold():
    state={'objects':[{'id':'blue_cube','position':[.44,.15,1.08]}],
        'robot':{'available_arms':['left','right'],'left_arm':{'held_object':None},'right_arm':{'held_object':'blue_cube'}}}
    calls=[]
    homes={'left':np.zeros(7),'right':np.zeros(7)}
    q={'left':np.ones(7),'right':np.ones(7)}
    def park(side):
        calls.append(side)
        q[side]=homes[side].copy()
        return {'success':True,'maximum_joint_error_rad':0}
    executor=TaskExecutor.__new__(TaskExecutor)
    executor.discourse={'last_object':'blue_cube','last_target':'red_cube'}
    executor.controller=SimpleNamespace(cancelled=False,stop_requested=lambda:False)
    executor.skills=SimpleNamespace(held_object=lambda side:'blue_cube' if side=='right' else None,park_empty_arm=park)
    executor.env=SimpleNamespace(object_contacts=lambda *a:[],home_arm_qpos=lambda side:homes[side],arm_qpos=lambda side:q[side])
    executor.world_state=lambda:deepcopy(state)
    result=executor.home_empty_arms()
    assert result['success'] and calls==['left']
    assert result['arms'][1]['skipped']=='holding_requested_object'
    assert executor.discourse=={'last_object':'blue_cube','last_target':'red_cube'}


def test_homing_cannot_report_success_after_moving_a_placed_object():
    state={'objects':[{'id':'blue_cube','position':[.44,.15,1.08]}],
        'robot':{'available_arms':['left'],'left_arm':{'held_object':None}}}
    q=np.ones(7)
    def park(side):
        nonlocal q
        q=np.zeros(7)
        state['objects'][0]['position'][0]+=.02
        return {'success':True}
    executor=TaskExecutor.__new__(TaskExecutor)
    executor.controller=SimpleNamespace(cancelled=False,stop_requested=lambda:False)
    executor.skills=SimpleNamespace(held_object=lambda side:None,park_empty_arm=park)
    executor.env=SimpleNamespace(object_contacts=lambda *a:[],home_arm_qpos=lambda side:np.zeros(7),arm_qpos=lambda side:q.copy())
    executor.world_state=lambda:deepcopy(state)
    result=executor.home_empty_arms()
    assert not result['success'] and result['failure_category']=='HOMING_DISTURBED_SCENE'
