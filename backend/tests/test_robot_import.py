"""URDF import, private Jacobians and broker-only selection are separate gates."""
import json
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


def test_a_partial_runtime_declaration_is_refused(tmp_path, monkeypatch):
    """A partial runtime set would make validation more permissive than the engine.

    Measured 2026-10-24: declaring one of two overlapping adjacent pairs left the engine with
    adjacent_collision=True while the private queries were silent on the declared pair, so a posture
    colliding only there passed validation and then collided in physics. The engine can only turn
    adjacent collision off wholesale, so a partial set cannot be enforced - the documentation-only
    key is the place for it.
    """
    source = overlapping_arm(tmp_path)
    (source.parent / 'collision_config.json').write_text(json.dumps(
        {'runtime_excluded_adjacent_pairs': [['base', 'link1']]}))
    monkeypatch.setattr(registry, 'GENERATED_ROOT', tmp_path / 'generated')
    with pytest.raises(ValueError, match='every collidable adjacent pair'):
        registry.import_robot(source.parent.parent)


def lime_source_robot(tmp_path):
    """The Lime's shape: a wheeled base carrying the 6-joint arm the profile shows."""
    root = tmp_path / 'source' / 'genesis'
    root.mkdir(parents=True)
    text = """<robot name="turtlebot3_lime">
    <link name="base_footprint"/><link name="base_link"/>
    <link name="wheel_left_link"/><link name="wheel_right_link"/><link name="link1"/>
    <joint name="base_joint" type="fixed"><parent link="base_footprint"/><child link="base_link"/><origin xyz="0 0 .01"/></joint>
    <joint name="wheel_left_joint" type="continuous"><parent link="base_link"/><child link="wheel_left_link"/><origin xyz="0 .1 0"/><axis xyz="0 1 0"/><limit effort="1" velocity="10"/></joint>
    <joint name="wheel_right_joint" type="continuous"><parent link="base_link"/><child link="wheel_right_link"/><origin xyz="0 -.1 0"/><axis xyz="0 1 0"/><limit effort="1" velocity="10"/></joint>
    <joint name="arm_mount_joint" type="fixed"><parent link="base_link"/><child link="link1"/><origin xyz=".08 0 .12"/></joint>""" + "".join(
        f'<link name="link{n + 1}"/><joint name="joint{n}" type="revolute">'
        f'<parent link="link{n}"/><child link="link{n + 1}"/><origin xyz="0 0 .1"/>'
        f'<axis xyz="0 1 0"/><limit lower="-1.5" upper="1.5" effort="3" velocity="1"/></joint>'
        for n in range(1, 7)) + """</robot>"""
    path = root / 'robot.urdf'
    path.write_text(text)
    return path


def test_a_wheeled_base_keeps_the_arm_chain_and_leaves_the_wheels_out(tmp_path, monkeypatch):
    """Measured on the real description: chains['left'] is joint1..joint6, wheels apart.

    The Lime is a wheeled base carrying a 6-joint arm, so an import that folded a wheel
    joint into the arm chain would drive the base while reaching, and one that lost the
    chain would leave the arm unusable.
    """
    source = lime_source_robot(tmp_path)
    monkeypatch.setattr(registry, 'GENERATED_ROOT', tmp_path / 'generated')
    profile = registry.import_robot(source.parent.parent)
    assert profile['root_link'] == 'base_footprint'
    assert profile['chains']['left']['joints'] == ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
    every_chain_joint = [j for chain in profile['chains'].values() for j in chain['joints']]
    assert not [j for j in every_chain_joint if 'wheel' in j]
    directory = Path(profile['directory'])
    assert json.loads((directory / 'scene.json').read_text())['openarm_cell'] is False
    assert profile['id'] in [r['id'] for r in registry.available_robots()]

def overlapping_robot(tmp_path):
    """Two adjacent links whose collision boxes sit in the same place, as manufacturer CAD does."""
    root = tmp_path / 'source' / 'genesis'
    root.mkdir(parents=True)
    text = '''<robot name="overlap_arm">
    <link name="base"><collision><geometry><box size=".10 .10 .10"/></geometry></collision></link>
    <link name="link1"><collision><geometry><box size=".10 .10 .10"/></geometry></collision></link>
    <link name="tool"/>
    <joint name="turn" type="continuous"><parent link="base"/><child link="link1"/><axis xyz="0 0 1"/><limit effort="4" velocity="2"/></joint>
    <joint name="elbow" type="revolute"><parent link="link1"/><child link="tool"/><origin xyz=".3 0 0"/><axis xyz="0 1 0"/><limit lower="-1" upper="1" effort="3" velocity="1"/></joint>
    </robot>'''
    path = root / 'robot.urdf'
    path.write_text(text)
    return path


def test_overlapping_adjacent_links_say_so_and_name_the_file_that_fixes_it(tmp_path, monkeypatch, capsys):
    source = overlapping_robot(tmp_path)
    monkeypatch.setattr(registry, 'GENERATED_ROOT', tmp_path / 'generated')
    profile = registry.import_robot(source)
    warnings = ' '.join(profile['warnings'])
    assert 'Adjacent links overlap' in warnings
    assert 'base|link1' in warnings and 'collision_config.json' in warnings
    assert 'WARNING:' in capsys.readouterr().out


def test_declaring_the_pairs_silences_the_overlap_warning(tmp_path, monkeypatch, capsys):
    source = overlapping_robot(tmp_path)
    (source.parent / 'collision_config.json').write_text(
        json.dumps({'runtime_excluded_adjacent_pairs': [['base', 'link1']]}))
    monkeypatch.setattr(registry, 'GENERATED_ROOT', tmp_path / 'generated')
    profile = registry.import_robot(source)
    assert not any('Adjacent links overlap' in warning for warning in profile['warnings'])
    assert 'WARNING:' not in capsys.readouterr().out


def test_an_actuator_file_keeps_the_import_warnings(tmp_path, monkeypatch):
    source = overlapping_robot(tmp_path)
    (source.parent / 'actuators.json').write_text(
        json.dumps({'joints': {'turn': {'preset': 'sts3215_c001_7_4v'}}}))
    monkeypatch.setattr(registry, 'GENERATED_ROOT', tmp_path / 'generated')
    profile = registry.import_robot(source)
    warnings = ' '.join(profile['warnings'])
    assert 'Adjacent links overlap' in warnings
    assert 'Explicit actuators.json supplies motor effort' in warnings
