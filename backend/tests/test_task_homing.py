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
    assert evidence['route'] in {'free_attitude_lift_then_home','high_home'}
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
    assert route is not None,planning
    assert planning['approach_overfly_m']<.10


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
