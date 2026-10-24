"""URDF import, private Jacobians and broker-only selection are separate gates."""
import json
import xml.etree.ElementTree as ElementTree
from pathlib import Path

import numpy as np
import pytest
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


def test_a_package_declaring_adjacent_mesh_overlap_does_not_collide_them(tmp_path, monkeypatch):
    """A manufacturer URDF's adjacent meshes overlap; the runtime must be told, in data."""
    source = source_robot(tmp_path)
    (source.parent / 'collision_config.json').write_text(json.dumps(
        {'runtime_excluded_adjacent_pairs': [['base', 'link1'], ['link1', 'tool']]}))
    monkeypatch.setattr(registry, 'GENERATED_ROOT', tmp_path / 'generated')
    profile = registry.import_robot(source.parent.parent)
    scene = json.loads((Path(profile['directory']) / 'scene.json').read_text())
    assert profile['collision_config']['covers_adjacent_pairs'] is True
    assert scene['adjacent_collision'] is False
    assert scene['runtime_excluded_adjacent_pairs'] == [['base', 'link1'], ['link1', 'tool']]


def test_a_declaration_that_names_a_non_adjacent_pair_is_refused(tmp_path, monkeypatch):
    source = source_robot(tmp_path)
    (source.parent / 'collision_config.json').write_text(json.dumps(
        {'runtime_excluded_adjacent_pairs': [['base', 'tool']]}))
    monkeypatch.setattr(registry, 'GENERATED_ROOT', tmp_path / 'generated')
    with pytest.raises(ValueError, match='non-adjacent'):
        registry.import_robot(source.parent.parent)


def test_documented_overlaps_are_not_a_runtime_request(tmp_path, monkeypatch):
    """The Scorpion package documents its overlaps; only the runtime key may change behaviour."""
    source = source_robot(tmp_path)
    (source.parent / 'collision_config.json').write_text(json.dumps(
        {'excluded_adjacent_pairs': [['base', 'link1']]}))
    monkeypatch.setattr(registry, 'GENERATED_ROOT', tmp_path / 'generated')
    profile = registry.import_robot(source.parent.parent)
    assert json.loads((Path(profile['directory']) / 'scene.json').read_text())['adjacent_collision'] is True


def cartesian_jaw_robot(tmp_path):
    """A reachable three-axis arm with one rotary jaw, so the pinch frame has a home to be decided at."""
    root = tmp_path / 'jaw_source' / 'genesis'
    root.mkdir(parents=True)
    def jaw(link, y):
        return (f'<collision><origin xyz=".055 {y} 0"/><geometry>'
                f'<box size=".08 .008 .02"/></geometry></collision>')
    text = '<robot name="cartesian_jaw"><link name="base"/><link name="a"/><link name="b"/>'
    text += f'<link name="palm">{jaw("palm", -.022)}</link>'
    text += f'<link name="tool">{jaw("tool", .022)}</link>'
    parent = 'base'
    for name, child, axis in (('axis0', 'a', '1 0 0'), ('axis1', 'b', '0 1 0'), ('axis2', 'palm', '0 0 1')):
        text += (f'<joint name="{name}" type="prismatic"><parent link="{parent}"/><child link="{child}"/>'
                 f'<axis xyz="{axis}"/><limit lower="-.4" upper=".4" effort="5" velocity="1"/></joint>')
        parent = child
    text += ('<joint name="jaw" type="continuous"><parent link="palm"/><child link="tool"/>'
             '<axis xyz="0 0 1"/><limit effort="1" velocity="2"/></joint></robot>')
    path = root / 'robot.urdf'
    path.write_text(text)
    return path


def test_spawn_validation_resolves_one_jaw_frame_for_the_whole_pinch(tmp_path, monkeypatch):
    """Execution holds one jaw frame through the closure, so validation must not test several.

    A Codex review of this branch found _validated_pose re-deriving an achieved frame inside every
    checked_pose call: the approach, closure and lift checks could each pass holding a different
    orientation than execution can maintain, and the closure offsets read whichever frame the last
    check happened to use. The frame is decided once, at the grasp pose, and this pins that count.
    """
    from backend.app.robots import spawn as spawn_module
    from backend.app.robots.spawn import ReachableDeskSpawns
    monkeypatch.setattr(registry, 'GENERATED_ROOT', tmp_path / 'generated')
    profile = registry.import_robot(cartesian_jaw_robot(tmp_path).parent.parent)
    assert profile.get('grippers'), 'the fixture must carry a jaw for a pinch frame to be decided'
    scene = json.loads((Path(profile['directory']) / 'scene.json').read_text())
    sampler = ReachableDeskSpawns(profile, scene)
    point = np.array([0.0, 0.0, scene['table_top'] + sampler.half_size + sampler.clearance])
    # Force the underactuated path: the inferred frame cannot be held, the achieved one can. Only
    # the inferred request is refused, so every check after the first still runs - which is how a
    # per-check frame decision shows up as more than one resolution.
    inferred = np.asarray(profile['grippers']['left']['grasp_rotation'])
    original_ik = spawn_module.solve_profile_ik
    def inferred_frame_never_solves(*args, **kwargs):
        rotation = args[3] if len(args) > 3 else kwargs.get('rotation')
        result = original_ik(*args, **kwargs)
        if rotation is not None and np.allclose(np.asarray(rotation), inferred):
            return {**result, 'success': False}
        return result
    monkeypatch.setattr(spawn_module, 'solve_profile_ik', inferred_frame_never_solves)
    resolutions = []
    original_frame = sampler.achieved_attitude
    def counted(side, q, snapshot):
        resolutions.append(side)
        return original_frame(side, q, snapshot)
    sampler.achieved_attitude = counted
    try:
        sampler._validated_pose('left', point, 'arm-achieved')
    except ValueError:
        # The forced failure can leave this synthetic arm unable to pinch here; what is under test
        # is how many times the jaw frame is decided, not whether this fixture can grasp.
        pass
    assert resolutions == ['left'], resolutions


def overlapping_arm(tmp_path):
    """A package whose source meshes overlap at two adjacent joints, both with collision geometry."""
    root = tmp_path / 'overlap_source' / 'genesis'
    root.mkdir(parents=True)
    text = ('<robot name="overlap"><link name="base"><collision><geometry>'
            '<box size=".06 .06 .06"/></geometry></collision></link>'
            '<link name="link1"><collision><origin xyz="0 0 .02"/><geometry>'
            '<box size=".06 .06 .06"/></geometry></collision></link>'
            '<link name="tool"><collision><origin xyz="0 0 .04"/><geometry>'
            '<box size=".06 .06 .06"/></geometry></collision></link>'
            '<link name="tip"/>'
            '<joint name="j1" type="revolute"><parent link="base"/><child link="link1"/>'
            '<axis xyz="0 0 1"/><limit lower="-.5" upper=".5" effort="3" velocity="1"/></joint>'
            '<joint name="j2" type="revolute"><parent link="link1"/><child link="tool"/>'
            '<origin xyz=".1 0 0"/><axis xyz="0 1 0"/><limit lower="-.5" upper=".5" effort="3" velocity="1"/></joint>'
            '<joint name="j3" type="revolute"><parent link="tool"/><child link="tip"/>'
            '<axis xyz="0 0 1"/><limit lower="-.5" upper=".5" effort="3" velocity="1"/></joint></robot>')
    path = root / 'robot.urdf'
    path.write_text(text)
    return path


def test_a_partial_runtime_declaration_ships_an_engine_model(tmp_path, monkeypatch):
    """A partial declaration is allowed exactly when the package ships a model that excludes those pairs.

    Measured 2026-10-24: declaring one of two overlapping adjacent pairs used to be refused, because a
    URDF cannot filter individual pairs and telling the engine to drop adjacent collision wholesale
    while the private queries were silent on the declared pair left validation more permissive than
    physics. A package whose overlaps live in its authored visuals has no complete set to declare - no
    amount of collision-mesh preparation removes them - so the import shapes the private collision
    model into an engine model whose exclusions are the declaration itself. This test fails if that
    model stops carrying exactly the declared pairs.
    """
    source = overlapping_arm(tmp_path)
    (source.parent / 'collision_config.json').write_text(json.dumps(
        {'runtime_excluded_adjacent_pairs': [['base', 'link1']]}))
    monkeypatch.setattr(registry, 'GENERATED_ROOT', tmp_path / 'generated')
    profile = registry.import_robot(source.parent.parent)
    engine = profile.get('engine_model') or {}
    assert engine.get('morph') == 'mjcf', 'a partial declaration must force an engine model'
    root = ElementTree.parse(engine['file']).getroot()
    excluded = sorted(tuple(sorted((e.get('body1'), e.get('body2')))) for e in root.iter('exclude'))
    assert excluded == [('base', 'link1')], 'the engine model must exclude exactly the declared pairs'
    assert not any(geometry.get('name') == 'desk' for geometry in root.iter('geom'))
    bodies = {body.get('name') for body in root.iter('body')}
    assert {'base', 'link1', 'tool', 'tip'} <= bodies
    assert all(body.find('inertial') is not None for body in root.iter('body')), 'every body needs an inertial'
    assert sum(len(body.findall('joint')) for body in root.iter('body')) == 3
    assert profile['collision_config']['covers_adjacent_pairs'] is False
