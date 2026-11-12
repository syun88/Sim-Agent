"""A leaning search is the fallback when the quarter-turn search finds no corridor."""
from types import SimpleNamespace

import numpy as np

from backend.app.simulation.openarm_v2.definition import FINGERTIP_MIDPOINT, GRASP_ROTATION
from backend.app.skills.approach import two_axis_frames
from backend.app.skills.pinch_frames import (approach_axis, describe, pinch_frames,
                                             tilt_from_vertical_deg)


def test_every_candidate_keeps_the_pinch_on_the_object_centre():
    centre = np.array([0.45, 0.0, 1.09])
    frames = two_axis_frames('right', centre, 0.06)
    assert len(frames) > 14
    for frame in frames:
        assert np.allclose(frame['pinch'] + frame['rotation'] @ FINGERTIP_MIDPOINT['right'], centre, atol=1e-9)
        assert 'tilt_deg' in frame


def test_the_search_starts_at_the_nominal_attitude():
    first = next(pinch_frames())
    assert first[0] == 0. and first[1] == 0.
    assert np.allclose(first[2], GRASP_ROTATION)


def test_a_lean_changes_the_angle_the_hand_comes_down_at():
    nominal = next(pinch_frames())[2]
    leaning = next(pinch_frames(tilts=(50.,)))[2]
    assert tilt_from_vertical_deg(leaning) < tilt_from_vertical_deg(nominal)
    assert not np.allclose(approach_axis(leaning), approach_axis(nominal))
    assert set(describe(leaning)) == {'tilt_from_vertical_deg', 'approach_axis'}


def test_the_leaning_search_runs_only_when_the_quarter_turn_search_fails(monkeypatch):
    from backend.app.skills import robot_skills
    calls = []

    def planner(env, controller, arm, obj, pinch, standoff, overfly, staging, frames=None):
        calls.append({'tilt': any(frame.get('tilt_deg') for frame in frames or []),
                      'count': len(frames or [])})
        if calls[-1]['tilt']:
            return [np.zeros(7), np.ones(7)], {'grasp_yaw_deg': 0, 'grasp_tilt_deg': 20.,
                                               'grasp_rotation': GRASP_ROTATION.tolist(),
                                               'pinch_target': pinch.tolist()}
        return None, {'error': 'APPROACH_PATH_UNAVAILABLE', 'rejected_routes': []}

    monkeypatch.setattr('backend.app.skills.approach.plan_cube_approach', planner)
    cube = np.array([0.45, 0.0, 1.09])
    moves = []

    def move(arm, target, rotation=None, **kwargs):
        moves.append(kwargs)
        return SimpleNamespace(success=True, final_position=list(target))

    env = SimpleNamespace(
        object_pose=lambda name: (cube.copy(), np.eye(3)),
        object_bounds=lambda name: np.array([[.42, -.03, 1.06], [.48, .03, 1.12]]),
        frame=lambda: SimpleNamespace(objects={'blue_cube': cube}),
        ee_pose=lambda arm: (np.array([0.30, 0.0, 1.15]), np.eye(3)),
        arm_qpos=lambda arm: np.zeros(7), home_arm_qpos=lambda arm: np.zeros(7),
        arm_velocity_limits=lambda arm: None,
        arm_kinematics=lambda arm: SimpleNamespace(evaluate=lambda q: (cube, GRASP_ROTATION, None, None)),
        pinch_established=lambda obj, arm: False, object_contacts=lambda name, arm=None: [],
        step=lambda n: None, timestep=0.002, grasp_mode='physical-friction-pinch',
        gripper_position=lambda arm: 0.02, gripper_target=lambda arm, closed: 0.02,
        set_gripper=lambda arm, value: None)
    skills = robot_skills.SkillRunner(env, SimpleNamespace(motion_speed=6., target_rotations={},
                                                           move_to_site=move, wait_gripper=lambda *a: None))
    result = skills.approach_object('right', 'blue_cube')
    assert result['success'], result
    assert [call['tilt'] for call in calls] == [False, True]
    assert result['planning']['search'] == 'two_axis'
    assert result['planning']['grasp_tilt_deg'] == 20.
