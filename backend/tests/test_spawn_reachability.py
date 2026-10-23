"""Source kinematics prove admission/rejection independently of physics execution."""
import json
from pathlib import Path

import numpy as np
import pytest

from backend.app.robots import registry
from backend.app.robots.spawn import ReachableDeskSpawns
from backend.app.robots.urdf import URDFKinematics
from backend.app.simulation.openarm_v2.spawn import validate_spawn_positions


@pytest.fixture(scope='module')
def official_kinematics():
    from backend.app.simulation.genesis_backend.environment import _OfficialModelMetadata
    from backend.app.simulation.openarm_v2.scene import resolved_scene_xml
    from backend.app.simulation.openarm_v2.definition import ARM_JOINTS, SPAWN_ARM_QPOS
    from backend.app.simulation.openarm_v2.kinematics import OfficialArmKinematics
    import mujoco
    meta = _OfficialModelMetadata(resolved_scene_xml())
    snapshot = meta.qpos_with_arm_pose(SPAWN_ARM_QPOS)
    class PrivateModel:
        def arm_kinematics(self, side):
            return OfficialArmKinematics(meta.model, snapshot, side)
        def arm_limits(self, side):
            return meta.model.jnt_range[[mujoco.mj_name2id(meta.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                                         for name in ARM_JOINTS[side]]]
        def home_arm_qpos(self, side):
            return SPAWN_ARM_QPOS.copy()
        def configure_motion_response(self, *args):
            raise AssertionError('Spawn IK must not configure live motors')
    return PrivateModel()


def test_official_default_and_random_spawns_include_precise_fingertip_approach_ik(official_kinematics):
    from backend.app.simulation.openarm_v2.scene import initial_object_positions
    for randomize in (False, True):
        report = validate_spawn_positions(official_kinematics, initial_object_positions(randomize))
        for entry in report['objects'].values():
            assert len(entry['checks']) == 3
            assert all(check['position_error'] < .003 and check['rotation_error'] < .03 for check in entry['checks'])
    from scripts.verify_pick_place_genesis import TASKS
    for task in TASKS.values():
        for x, y in task['positions']:
            validate_spawn_positions(official_kinematics, {task['object']: [x, y, 1.032]})
    from backend.app.simulation.genesis_backend.wine import BOTTLE_START
    assert len(validate_spawn_positions(official_kinematics, {'wine_bottle': BOTTLE_START})['objects']['wine_bottle']['checks']) == 2
    with pytest.raises(ValueError, match='IK reach'):
        validate_spawn_positions(official_kinematics, {'red_cube': [5., 0., 1.032]})


def test_urdf_random_spawns_stay_on_desk_separated_and_reachable(tmp_path, monkeypatch):
    source = tmp_path / 'cartesian.urdf'
    source.write_text('''<robot name="cartesian"><link name="base"/><link name="x"/><link name="y"/><link name="tool"/>
    <joint name="slide_x" type="prismatic"><parent link="base"/><child link="x"/><axis xyz="1 0 0"/><limit lower="-.4" upper=".4" effort="4" velocity="1"/></joint>
    <joint name="slide_y" type="prismatic"><parent link="x"/><child link="y"/><axis xyz="0 1 0"/><limit lower="-.4" upper=".4" effort="4" velocity="1"/></joint>
    <joint name="slide_z" type="prismatic"><parent link="y"/><child link="tool"/><axis xyz="0 0 1"/><limit lower="-.4" upper=".4" effort="4" velocity="1"/></joint></robot>''')
    monkeypatch.setattr(registry, 'GENERATED_ROOT', tmp_path / 'generated')
    profile = registry.import_robot(source)
    scene = json.loads((Path(profile['directory']) / 'scene.json').read_text())
    sampler = ReachableDeskSpawns(profile, scene)
    original = sampler.layout()['objects']
    assert len(original) == 2
    rng = np.random.default_rng(43);sampled=[]
    for _ in range(10):
        layout = sampler.layout(randomize=True, base=original, rng=rng)['objects']
        positions = [np.asarray(value['position']) for value in layout.values()]
        sampled.append({n:v['position'] for n,v in layout.items()})
        assert np.linalg.norm(positions[0][:2] - positions[1][:2]) >= .10
        for entry in layout.values():
            assert entry['position'][2] == pytest.approx(scene['table_top'] + .028)
            kin = URDFKinematics(sampler.model, profile['chains'][entry['arm']], profile['home'], profile['root_transform'])
            for check in entry['checks']:
                actual = kin.evaluate(np.asarray(check['q_rad']))[0]
                np.testing.assert_allclose(actual, check['target_position'], atol=.002)
    for name in original:
        assert np.ptp([v[name][0] for v in sampled])>.2
        assert np.ptp([v[name][1] for v in sampled])>.2
    a=sampler.layout(randomize=True,base=original,seed=71)
    b=sampler.layout(randomize=True,base=original,seed=71)
    assert a==b and a['randomization']['seed']==71
    only_one=sampler.layout(randomize=True,base={'red_cube':original['red_cube']},seed=71)
    assert set(only_one['objects'])=={'red_cube'} and 'blue_cube' in only_one['omitted']
    with pytest.raises(ValueError, match='reachable'):
        sampler.layout(object_positions={'red_cube': [4., 0., .788]})
    # Failing a broad search must be visible, never a silent return to a default.
    def only_original(point,side):
        for entry in original.values():
            if np.allclose(point,entry['position'],atol=1e-9):return entry
        raise ValueError('No new reachable position')
    monkeypatch.setattr(sampler,'validate',only_original)
    with pytest.raises(ValueError,match='No changed'):
        sampler.layout(randomize=True,base=original,seed=71,shuffle_sides=False)


def test_official_randomization_covers_broad_positions_and_reproduces_seeded_side_swaps(official_kinematics):
    from backend.app.simulation.openarm_v2.spawn import random_spawn_layout
    layouts=[random_spawn_layout(official_kinematics,seed=20261023+i)[1] for i in range(8)]
    for name in ('red_cube','blue_cube'):
        points=np.array([row['objects'][name]['position'] for row in layouts])
        assert np.ptp(points[:,0])>.2 and np.ptp(points[:,1])>.5
        assert len(np.unique(points,axis=0))==8
        assert {row['objects'][name]['arm'] for row in layouts}=={'left','right'}
    assert random_spawn_layout(official_kinematics,seed=20261023)[1]==layouts[0]
    _,fixed=random_spawn_layout(official_kinematics,seed=20261023,shuffle_sides=False)
    assert fixed['objects']['red_cube']['arm']=='left' and fixed['objects']['blue_cube']['arm']=='right'


def test_spawn_rejects_a_reachable_tool_pose_inside_the_robots_base(tmp_path):
    from backend.app.robots.urdf import URDFModel
    source = tmp_path/'obstructed.urdf'
    source.write_text('''<robot name="obstructed"><link name="base"><collision><origin xyz=".2 0 0"/>
    <geometry><box size=".06 .06 .06"/></geometry></collision></link><link name="x"/><link name="y"/>
    <link name="tool"><collision><geometry><sphere radius=".01"/></geometry></collision></link>
    <joint name="x" type="prismatic"><parent link="base"/><child link="x"/><axis xyz="1 0 0"/><limit lower="-.4" upper=".4" effort="4" velocity="1"/></joint>
    <joint name="y" type="prismatic"><parent link="x"/><child link="y"/><axis xyz="0 1 0"/><limit lower="-.4" upper=".4" effort="4" velocity="1"/></joint>
    <joint name="z" type="prismatic"><parent link="y"/><child link="tool"/><axis xyz="0 0 1"/><limit lower="0" upper=".4" effort="4" velocity="1"/></joint></robot>''')
    root = np.eye(4);root[2,3]=.786
    profile = {'urdf':str(source),'home':{'x':0.,'y':0.,'z':.1},'root_transform':root.tolist(),
               'chains':{'right':{'joints':['x','y','z'],'end_link':'tool','tool_offset':[0.,0.,0.]}}}
    sampler = ReachableDeskSpawns(profile,{'table_top':.76,'table_size':[1.6,1.4,.05]})
    sampler.validate([0.,0.,.788],'right')
    from backend.app.robots.urdf import solve_profile_ik
    assert solve_profile_ik(profile,'right',np.array([.2,0.,.786]),model=URDFModel(source))['success']
    with pytest.raises(ValueError,match='intersects'):
        sampler.validate([.2,0.,.788],'right')
    # A narrow workspace needs centimetre-scale alternatives around an obstructed
    # preferred pose; the former 10cm jumps skipped every usable position.
    import xml.etree.ElementTree as ET
    robot = ET.parse(source)
    robot.find('.//collision/origin').set('xyz','0 0 0')
    robot.find('.//collision/geometry/box').set('size','.004 .004 .004')
    for joint in ('x','y'):
        limit = robot.find(f"./joint[@name='{joint}']/limit")
        limit.set('lower','-.04');limit.set('upper','.04')
    robot.write(source)
    narrow = ReachableDeskSpawns(profile,{'table_top':.76,'table_size':[1.6,1.4,.05]})
    entry = narrow.layout()['objects']['red_cube']
    assert .012<np.linalg.norm(entry['position'][:2])<.04
