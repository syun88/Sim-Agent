"""The lean is chosen by measuring how much room it leaves, not by taking a listed value."""
from types import SimpleNamespace

import numpy as np
import pytest

from backend.app.skills.approach import frame_at, plan_cube_approach
from backend.app.simulation.openarm_v2.definition import FINGERTIP_MIDPOINT, GRASP_ROTATION

CUBE = np.array([0.45, 0.0, 1.09])
BEST_TILT_DEG = 12.


def _tilt_of(rotation):
    """Recover the lean from a jaw rotation: R = Rz(azimuth) @ Ry(tilt) @ nominal."""
    lean = np.asarray(rotation, float) @ GRASP_ROTATION.T
    return -np.degrees(np.arcsin(np.clip(lean[2, 0], -1., 1.)))


def _scene():
    """A body whose only preference is a lean of 12 degrees, which no candidate list contains."""
    asked = {'target': CUBE.copy(), 'rotation': GRASP_ROTATION.copy()}
    kin = SimpleNamespace(
        evaluate=lambda q: (asked['target'].copy(), asked['rotation'].copy(), None, None),
        empty_arm_path_is_clear=lambda *a, **k: True,
        path_minimum_clearance=lambda waypoints, objects, **k: .045 - abs(_tilt_of(asked['rotation']) - BEST_TILT_DEG) * .0002)

    def solve_ik(arm, target, rotation, **kwargs):
        asked['target'] = np.asarray(target, float).copy()
        asked['rotation'] = np.asarray(rotation, float).copy()
        return np.zeros(7), 0.001, 0.0

    env = SimpleNamespace(
        arm_qpos=lambda arm: np.zeros(7), home_arm_qpos=lambda arm: np.zeros(7),
        ee_pose=lambda arm: (np.array([0.30, 0.0, 1.20]), np.eye(3)),
        arm_kinematics=lambda arm: kin, arm_velocity_limits=lambda arm: None,
        object_pose=lambda name: (CUBE.copy(), np.eye(3)),
        frame=lambda: SimpleNamespace(objects={'blue_cube': CUBE}))
    return env, SimpleNamespace(solve_ik=solve_ik), asked


def test_the_lean_is_derived_from_the_measurement():
    env, controller, asked = _scene()
    frames = [frame_at('right', CUBE, .06, 0., tilt_deg=0.), frame_at('right', CUBE, .06, 0., tilt_deg=35.)]
    route, planning = plan_cube_approach(env, controller, 'right', 'blue_cube',
                                         frames[0]['pinch'], frames[0]['standoff'], .10, 1.24, frames=frames)
    assert route is not None, planning
    assert planning['grasp_tilt_deg'] == pytest.approx(BEST_TILT_DEG, abs=3.)
    trace = next(item for item in planning['refinements'] if item['query'] == 'refine_grasp_tilt')
    assert trace['measured'] == len(trace['evaluations'])
    assert trace['best_value'] == planning['grasp_tilt_deg']
    # The gesture actually used carries that lean, not just the label.
    assert _tilt_of(planning['grasp_rotation']) == pytest.approx(BEST_TILT_DEG, abs=3.)
    assert _tilt_of(asked['rotation']) == pytest.approx(BEST_TILT_DEG, abs=3.)
    assert planning['pinch_target'] == pytest.approx(
        (CUBE - planning['grasp_rotation'] @ FINGERTIP_MIDPOINT['right']).tolist(), abs=1e-9)


def test_an_unmeasurable_lean_leaves_the_accepted_frame_alone():
    env, controller, asked = _scene()
    env.arm_kinematics = lambda arm: SimpleNamespace(
        evaluate=lambda q: (asked['target'].copy(), asked['rotation'].copy(), None, None),
        empty_arm_path_is_clear=lambda *a, **k: True)
    frames = [frame_at('right', CUBE, .06, 0., tilt_deg=0.)]
    route, planning = plan_cube_approach(env, controller, 'right', 'blue_cube',
                                         frames[0]['pinch'], frames[0]['standoff'], .10, 1.24, frames=frames)
    assert route is not None, planning
    assert planning['grasp_tilt_deg'] == 0.
    assert planning['refinements'] == []


def test_a_measured_win_the_screen_refuses_is_recorded_as_refused():
    """Live, a lean cleared the cubes by 21 mm and the table refused the route; the trace must say so."""
    env, controller, asked = _scene()
    # Only the first route may pass the screen: every refined candidate is blocked.
    calls = {'n': 0}

    def screen(*args, **kwargs):
        calls['n'] += 1
        return calls['n'] == 1
    env.arm_kinematics = lambda arm: SimpleNamespace(
        evaluate=lambda q: (asked['target'].copy(), asked['rotation'].copy(), None, None),
        empty_arm_path_is_clear=screen,
        path_minimum_clearance=lambda waypoints, objects, **k: .045 - abs(_tilt_of(asked['rotation']) - BEST_TILT_DEG) * .0002)
    frames = [frame_at('right', CUBE, .06, 0., tilt_deg=0.)]
    route, planning = plan_cube_approach(env, controller, 'right', 'blue_cube',
                                         frames[0]['pinch'], frames[0]['standoff'], .10, 1.24, frames=frames)
    assert route is not None, planning
    refused = [item for item in planning['refinements'] if not item['accepted']]
    reasons = {item.get('rejected_by') for item in refused}
    assert refused and reasons <= {'COLLISION_PATH', 'NOT_BETTER_THAN_ACCEPTED', 'IK_FAILURE', 'UNMEASURED'}
    assert 'COLLISION_PATH' in reasons, 'the screen is what refused the measured win'
    assert planning['grasp_tilt_deg'] == 0.
    assert not any(item['accepted'] for item in planning['refinements'])
