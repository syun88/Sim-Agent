"""An imported wheeled arm robot: it drives, and its arm's FK follows the body.

The Lime import only ever proved these two things with a script against one machine's package
(`scripts/verify_imported_robot.py`), so nothing pinned them: a wheeled base is the case where
`arm_kinematics` used to anchor on the spawn pose instead of on the settled body (the FK error
was 1.066 mm against a 0.1 mm gate), and where a diff drive has to actually move.

The package here is synthetic and lives in tmp_path, so the test needs no vendor checkout. Its
shape is what matters: a floor robot with two continuous wheels, a three-joint arm and the
collision declaration a manufacturer's CAD requires.
"""
import json
import pathlib

import numpy as np
import pytest

from backend.app.robots import registry
from backend.app.simulation import create_backend

URDF = '''<robot name="wheeled_arm">
  <link name="base_footprint"/>
  <link name="base_link">
    <inertial><origin xyz="0 0 0"/><mass value="2.0"/><inertia ixx=".02" ixy="0" ixz="0" iyy=".02" iyz="0" izz=".03"/></inertial>
    <collision><geometry><box size=".20 .20 .06"/></geometry></collision>
  </link>
  <joint name="base_joint" type="fixed"><parent link="base_footprint"/><child link="base_link"/>
    <origin xyz="0 0 .06"/></joint>
  <link name="wheel_left_link">
    <inertial><origin xyz="0 0 0"/><mass value=".2"/><inertia ixx=".001" ixy="0" ixz="0" iyy=".001" iyz="0" izz=".001"/></inertial>
    <collision><origin rpy="1.5708 0 0"/><geometry><cylinder radius=".06" length=".02"/></geometry></collision>
  </link>
  <joint name="wheel_left_joint" type="continuous"><parent link="base_link"/><child link="wheel_left_link"/>
    <origin xyz="0 .14 -.03"/><axis xyz="0 1 0"/><limit effort="3" velocity="8"/></joint>
  <link name="wheel_right_link">
    <inertial><origin xyz="0 0 0"/><mass value=".2"/><inertia ixx=".001" ixy="0" ixz="0" iyy=".001" iyz="0" izz=".001"/></inertial>
    <collision><origin rpy="1.5708 0 0"/><geometry><cylinder radius=".06" length=".02"/></geometry></collision>
  </link>
  <joint name="wheel_right_joint" type="continuous"><parent link="base_link"/><child link="wheel_right_link"/>
    <origin xyz="0 -.14 -.03"/><axis xyz="0 1 0"/><limit effort="3" velocity="8"/></joint>
  <link name="link1">
    <inertial><origin xyz=".05 0 0"/><mass value=".3"/><inertia ixx=".001" ixy="0" ixz="0" iyy=".001" iyz="0" izz=".001"/></inertial>
    <collision><origin xyz=".05 0 0"/><geometry><box size=".10 .04 .04"/></geometry></collision>
  </link>
  <joint name="joint1" type="revolute"><parent link="base_link"/><child link="link1"/>
    <origin xyz="0 0 .10"/><axis xyz="0 1 0"/><limit lower="-1.5" upper="1.5" effort="3" velocity="4"/></joint>
  <link name="link2">
    <inertial><origin xyz=".06 0 0"/><mass value=".25"/><inertia ixx=".001" ixy="0" ixz="0" iyy=".001" iyz="0" izz=".001"/></inertial>
    <collision><origin xyz=".06 0 0"/><geometry><box size=".12 .04 .04"/></geometry></collision>
  </link>
  <joint name="joint2" type="revolute"><parent link="link1"/><child link="link2"/>
    <origin xyz=".10 0 0"/><axis xyz="0 1 0"/><limit lower="-1.5" upper="1.5" effort="3" velocity="4"/></joint>
  <link name="link3">
    <inertial><origin xyz=".04 0 0"/><mass value=".2"/><inertia ixx=".001" ixy="0" ixz="0" iyy=".001" iyz="0" izz=".001"/></inertial>
    <collision><origin xyz=".04 0 0"/><geometry><box size=".08 .04 .04"/></geometry></collision>
  </link>
  <joint name="joint3" type="revolute"><parent link="link2"/><child link="link3"/>
    <origin xyz=".12 0 0"/><axis xyz="0 1 0"/><limit lower="-1.5" upper="1.5" effort="3" velocity="4"/></joint>
  <link name="tool"/>
  <joint name="tool_joint" type="fixed"><parent link="link3"/><child link="tool"/><origin xyz=".08 0 0"/></joint>
</robot>'''


@pytest.fixture(scope='module')
def wheeled(tmp_path_factory):
    source = tmp_path_factory.mktemp('wheeled') / 'wheeled_arm.urdf'
    source.write_text(URDF)
    # Every adjacent pair that carries collision geometry, which is what the import requires
    # before it will turn adjacent collision off wholesale.
    (source.parent / 'collision_config.json').write_text(json.dumps({'runtime_excluded_adjacent_pairs': [
        ['base_link', 'link1'], ['base_link', 'wheel_left_link'], ['base_link', 'wheel_right_link'],
        ['link1', 'link2'], ['link2', 'link3']]}))
    (source.parent / 'floor_objects.json').write_text(json.dumps({'objects': {
        'cube': {'position': [.22, 0., .16], 'support': {'size': [.08, .08, .12]}}}}))
    (source.parent / 'actuators.json').write_text(json.dumps({'joints': {
        'wheel_left_joint': {'preset': 'xm430_w210'}, 'wheel_right_joint': {'preset': 'xm430_w210'},
        'joint1': {'preset': 'xm430_w210'}, 'joint2': {'preset': 'xm430_w210'},
        'joint3': {'preset': 'xm430_w210'}}}))
    generator = tmp_path_factory.mktemp('generated')
    original = registry.GENERATED_ROOT
    registry.GENERATED_ROOT = generator
    try:
        profile = registry.import_robot(source)
        env = create_backend(robot_id=profile['id'])
        yield profile, env
        env.close()
    finally:
        registry.GENERATED_ROOT = original


def test_the_wheeled_arm_is_a_floor_robot_with_a_diff_drive(wheeled):
    profile, _ = wheeled
    assert profile['mobile_base']['drive'] == 'diff'
    assert profile['mobile_base']['wheel_joints'] == ['wheel_left_joint', 'wheel_right_joint']
    assert profile['chains']['left']['joints'] == ['joint1', 'joint2', 'joint3']


def test_driving_moves_the_body(wheeled):
    _, env = wheeled
    env.reset()
    before = env.base_pose()
    env.command_base_velocity(.25, 0., 0.)
    for _ in range(120):
        env.step(1)
    env.command_base_velocity(0., 0., 0.)
    for _ in range(20):
        env.step(1)
    after = env.base_pose()
    assert float(np.linalg.norm(np.asarray(after[:2]) - np.asarray(before[:2]))) > .05


def test_the_arms_fk_follows_the_driven_body(wheeled):
    """The defect this pins: FK anchored on the spawn pose reports a phantom offset after a drive."""
    _, env = wheeled
    env.reset()          # a spawn-anchored root, and no earlier test's base_pose() re-anchoring it
    q = env.arm_qpos('left')
    before = np.asarray(env.arm_kinematics('left').evaluate(q)[0])
    env.command_base_velocity(.25, 0., 0.)
    for _ in range(120):
        env.step(1)
    env.command_base_velocity(0., 0., 0.)
    for _ in range(20):
        env.step(1)
    # Same joint values, so the whole difference is the frame the FK is anchored on: with a
    # spawn-anchored root this vector stays put.
    after = np.asarray(env.arm_kinematics('left').evaluate(q)[0])
    assert float(np.linalg.norm((after - before)[:2])) > .05, 'the FK did not follow the body'
    # And read at one instant, FK and the body's own end pose must still agree after a drive;
    # verify_imported_robot.py only ever compares them at the spawn pose.
    now = env.arm_qpos('left')
    assert float(np.linalg.norm(np.asarray(env.ee_pose('left')[0])
                                - np.asarray(env.arm_kinematics('left').evaluate(now)[0]))) < .001


def test_an_authored_object_stands_on_its_support(wheeled):
    """An object placed above the floor needs a support, or it falls and nothing can reach it."""
    profile, env = wheeled
    entry = json.loads((pathlib.Path(profile['directory']) / 'spawn_validation.json').read_text())['objects']['cube']
    assert entry['support']['size'] == [.08, .08, .12], entry
    env.reset()
    env.step(120)
    z = float(np.asarray(env.object_pose('cube')[0])[2])
    assert z > .10, 'the cube fell to %.3f m: the support was not created' % z
