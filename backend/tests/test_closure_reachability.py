"""A reachable open tool pose does not prove that the jaws can close there."""
import numpy as np

from backend.app.robots.collision_planner import PrivateCollisionQueries
from backend.app.robots.executor import ImportedRobotController
from backend.app.robots.urdf import URDFModel, URDFKinematics


class KinematicFixture:
    def __init__(self, source):
        self.model = URDFModel(source)
        chain = {'joints':['x','y','z'], 'end_link':'palm', 'tool_offset':[0.,0.,0.]}
        self.profile = {'urdf':str(source),'root_transform':np.eye(4).tolist(),'chains':{'left':chain},
                        'home':{'x':0.,'y':0.,'z':.2,'jaw':0.}}
        self._gripper_force_holds = set()
        self.collision_queries = PrivateCollisionQueries(self.profile)
        self.grippers = {'left':{'joint':'jaw','followers':{},'open_q':0.,'closed_q':1.,
            'pad_region':{'normal':[0.,1.,0.]},
            'sweep':[{'q':0.,'face_a':-.02,'face_b':.02},
                     {'q':1.,'face_a':.02,'face_b':.06}]}}

    def joint_snapshot(self):return {'x':0.,'y':0.,'z':.2,'jaw':0.}
    def arm_qpos(self, side):return np.array([0.,0.,.2])
    def home_arm_qpos(self, side):return self.arm_qpos(side)
    def arm_velocity_limits(self, side):return np.ones(3)
    def arm_limits(self, side):return np.array([[-.2,.2],[-.01,.1],[0.,.4]])
    def arm_kinematics(self, side):
        return URDFKinematics(self.model,self.profile['chains'][side],self.joint_snapshot())


def test_closure_checks_the_compensating_wrist_path_and_verification_lift(tmp_path):
    source = tmp_path/'cartesian.urdf'
    text = '<robot name="cartesian"><link name="base"/>'
    parent = 'base'
    for name,axis,lower,upper in [('x','1 0 0',-.2,.2),('y','0 1 0',-.01,.1),('z','0 0 1',0.,.4)]:
        child = 'palm' if name=='z' else 'carriage_'+name
        text += f'<link name="{child}"/><joint name="{name}" type="prismatic"><parent link="{parent}"/><child link="{child}"/><axis xyz="{axis}"/><limit lower="{lower}" upper="{upper}" effort="1" velocity="1"/></joint>'
        parent = child
    text += '<link name="finger"/><joint name="jaw" type="continuous"><parent link="palm"/><child link="finger"/><limit effort="1" velocity="1"/></joint></robot>'
    source.write_text(text)
    env = KinematicFixture(source)
    controller = ImportedRobotController(env,configure_motors=False)
    point = np.array([0.,0.,.2])
    opened,_,_ = controller.solve_ik('left',point,np.eye(3))
    assert opened is not None
    # Forty mm of wrist compensation exceeds the source Y limit, even though
    # the initial pose and finger joint are individually reachable.
    assert controller.plan_closure('left',point,np.eye(3),opened) is None
    env.grippers['left']['sweep'][1].update(face_a=-.015,face_b=.025)
    valid = controller.plan_closure('left',point,np.eye(3),opened)
    assert valid is not None and len(valid)==49
    # The fully-closed jaw can lift inside the Z limit, while a wider aperture
    # occupied by the cube needs extra wrist height and cannot lift there.
    g = env.grippers['left']
    g['pad_region']['normal']=[0.,0.,1.]
    g['sweep']=[{'q':0.,'face_a':-.03,'face_b':.03},
                {'q':.5,'face_a':-.04,'face_b':0.},
                {'q':1.,'face_a':-.02,'face_b':.01}]
    high = np.array([0.,0.,.36])
    assert controller.plan_closure('left',high,np.eye(3),opened) is not None
    g['cube_size']=.04
    assert controller.plan_closure('left',high,np.eye(3),opened) is None


def test_continuous_angle_uses_nearest_turn_without_changing_fractional_mimics(tmp_path):
    source=tmp_path/'turn.urdf'
    source.write_text('''<robot name="turn"><link name="base"/><link name="arm"/><link name="follower"/>
        <joint name="axis" type="continuous"><parent link="base"/><child link="arm"/><axis xyz="0 0 1"/></joint>
        <joint name="driven" type="continuous"><parent link="arm"/><child link="follower"/><axis xyz="0 0 1"/>
        <mimic joint="axis" multiplier="1"/></joint></robot>''')
    model=URDFModel(source)
    q=model.nearest_continuous(['axis'],[2*np.pi+.1],[0.])
    np.testing.assert_allclose(q,[.1],atol=1e-12)
    before=model.forward({'axis':2*np.pi+.1})[0]['follower']
    after=model.forward({'axis':q[0]})[0]['follower']
    np.testing.assert_allclose(after,before,atol=1e-12)
    source.write_text(source.read_text().replace('multiplier="1"','multiplier=".5"'))
    model=URDFModel(source)
    np.testing.assert_allclose(model.nearest_continuous(['axis'],[2*np.pi+.1],[0.]),[2*np.pi+.1])
