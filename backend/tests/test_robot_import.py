"""URDF import, private Jacobians and broker-only selection are separate gates."""
import json
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from backend.app.robots import registry
from backend.app.robots.urdf import URDFModel, URDFKinematics, solve_profile_ik
from backend.app.main import app
from backend.app.planning.agent_planner import AgentPlanner


def source_robot(tmp_path):
    root = tmp_path / 'source' / 'genesis'
    root.mkdir(parents=True)
    text = '''<robot name="test_arm"><link name="base"/><link name="link1"/><link name="tool"/>
    <joint name="turn" type="continuous"><parent link="base"/><child link="link1"/><axis xyz="0 0 1"/><limit effort="4" velocity="2"/></joint>
    <joint name="elbow" type="revolute"><parent link="link1"/><child link="tool"/><origin xyz=".3 0 0"/><axis xyz="0 1 0"/><limit lower="-1" upper="1" effort="3" velocity="1"/></joint></robot>'''
    path = root / 'robot.urdf'
    path.write_text(text)
    return path


def test_import_keeps_source_and_continuous_limits_and_generates_scene_ik(tmp_path, monkeypatch):
    source = source_robot(tmp_path)
    original = source.read_bytes()
    monkeypatch.setattr(registry, 'GENERATED_ROOT', tmp_path / 'generated')
    profile = registry.import_robot(source.parent.parent)
    assert source.read_bytes() == original
    assert profile['joints'][0]['lower'] is None and profile['joints'][0]['upper'] is None
    assert profile['chains']['left']['joints'] == ['turn', 'elbow']
    directory = Path(profile['directory'])
    assert json.loads((directory / 'scene.json').read_text())['openarm_cell'] is False
    assert json.loads((directory / 'scene.json').read_text())['adjacent_collision'] is True
    for name in ('profile.json', 'robot.urdf', 'scene.json', 'ik_seeds.json', 'ik_validation.json', 'solve_ik.py'):
        assert (directory / name).is_file()


def test_changed_mesh_invalidates_grasp_record_and_urdf_changes_keep_old_selection_working(tmp_path, monkeypatch):
    import trimesh
    source = source_robot(tmp_path)
    mesh = source.with_name('base.obj')
    trimesh.creation.box(extents=[.1, .1, .1]).export(mesh)
    source.write_text(source.read_text().replace('<link name="base"/>',
        '<link name="base"><collision><geometry><mesh filename="base.obj"/></geometry></collision></link>'))
    monkeypatch.setattr(registry, 'GENERATED_ROOT', tmp_path / 'generated')
    first = registry.import_robot(source)
    runtime = Path(first['directory']) / 'gripper_runtime.json'
    runtime.write_text(json.dumps({'configuration_sha256': first['configuration_sha256'], 'arms': {}}))
    trimesh.creation.box(extents=[.2, .1, .1]).export(mesh)
    second = registry.get_profile(first['id'])
    assert first['id'] == second['id']
    assert second['asset_sha256'] != first['asset_sha256']
    assert second['configuration_sha256'] != first['configuration_sha256']
    source.with_name('initial_state.json').write_text(json.dumps({'turn':.2}))
    changed_home = registry.get_profile(first['id'])
    assert changed_home['home']['turn']==.2
    assert changed_home['configuration_sha256']!=second['configuration_sha256']
    source.write_text(source.read_text().replace('effort="4"', 'effort="5"'))
    third = registry.get_profile(first['id'])
    assert third['id'] != first['id']
    assert registry.get_profile(first['id'])['id'] == third['id']
    assert len(registry.available_robots()) == 2


def test_private_analytic_jacobian_matches_numerical_derivative(tmp_path):
    model = URDFModel(source_robot(tmp_path))
    chain = {'joints': ['turn', 'elbow'], 'end_link': 'tool'}
    kin = URDFKinematics(model, chain, offset=[.1, .02, .03])
    q = np.array([.3, -.2])
    _, _, linear, _ = kin.evaluate(q)
    numerical = np.column_stack([(kin.evaluate(q + np.eye(2)[i] * 1e-6)[0] - kin.evaluate(q - np.eye(2)[i] * 1e-6)[0]) / 2e-6 for i in range(2)])
    np.testing.assert_allclose(linear, numerical, atol=1e-8)


def test_mimic_joint_contribution_is_included_in_private_jacobian(tmp_path):
    source = tmp_path / 'mimic.urdf'
    source.write_text('''<robot name="mimic"><link name="base"/><link name="a"/><link name="tool"/>
    <joint name="drive" type="continuous"><parent link="base"/><child link="a"/><axis xyz="0 0 1"/></joint>
    <joint name="coupled" type="continuous"><parent link="a"/><child link="tool"/><origin xyz=".3 0 0"/><axis xyz="0 0 1"/><mimic joint="drive" multiplier="-.5" offset=".1"/></joint></robot>''')
    model = URDFModel(source)
    kin = URDFKinematics(model, {'joints': ['drive'], 'end_link': 'tool'}, offset=[.2, .03, 0.])
    q = np.array([.4])
    numerical = (kin.evaluate(q + 1e-6)[0] - kin.evaluate(q - 1e-6)[0]) / 2e-6
    np.testing.assert_allclose(kin.evaluate(q)[2][:, 0], numerical, atol=1e-8)


def test_omitted_hand_uses_the_only_imported_chain(monkeypatch):
    monkeypatch.setenv('OPENARM_LOCAL_PLANNER', '0')
    for key in ('AI_AND_API_KEY', 'AI_AND_BASE_URL', 'AI_AND_MODEL'):
        monkeypatch.delenv(key, raising=False)
    state = {'robot': {'available_arms': ['right']}, 'objects': [], 'targets': [],
             'capabilities': {'skills': ['move_relative', 'return_home']}}
    assert AgentPlanner().plan('上に1cm動かして', state).steps[0].arm == 'right'


def test_robot_selection_is_queued_for_one_owner_and_never_constructs_physics(tmp_path, monkeypatch):
    source = source_robot(tmp_path)
    monkeypatch.setattr(registry, 'GENERATED_ROOT', tmp_path / 'generated')
    with TestClient(app) as client:
        result = client.post('/api/robots/import', json={'folder': str(source.parent)})
        assert result.status_code == 200
        robot_id = result.json()['id']
        selected = client.post('/api/robots/select', json={'robot_id': robot_id})
        assert selected.status_code == 200
        assert app.state.engine is None
        command = client.get('/api/genesis/next').json()
        assert command['action'] == 'select_robot' and command['robot_id'] == robot_id
        file = client.get(f'/api/robots/{robot_id}/files/scene.json')
        assert file.status_code == 200 and file.json()['openarm_cell'] is False
        assert client.get(f'/api/robots/{robot_id}/files/not-a-generated-file').status_code == 404
