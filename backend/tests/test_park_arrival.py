"""A parked arm must land on the waypoint it planned, and must not sweep through geometry unmeasured."""
from types import SimpleNamespace

import numpy as np

from backend.app.skills.robot_skills import SkillRunner

HOME = np.zeros(7)


def _skills(*, arrived_at=HOME, gap=0.03, move_success=True):
    """An arm that reports success while ending somewhere else, or passing through something."""
    planned = {'joints': None}

    def evaluate(value):
        return np.asarray(value, float)[:3], np.eye(3), None, None

    def move(side, target, attitude, **kwargs):
        planned['joints'] = np.asarray(kwargs['planned_joints'][-1], float)
        if 'on_sample' in kwargs:
            kwargs['on_sample']()
        return SimpleNamespace(success=move_success)

    kin = SimpleNamespace(
        evaluate=evaluate,
        empty_arm_path_is_clear=lambda *a, **k: True,
        path_clearance_detail=lambda route, objects, **k: {'gap_m': gap, 'pair': ['finger', 'cube'],
                                                           'sample': 1, 'samples': 2})
    env = SimpleNamespace(
        frame=lambda: SimpleNamespace(objects={'red_cube': np.array([0.45, 0.0, 1.05])}),
        object_contacts=lambda *a: [], object_pose=lambda name: (np.array([0.45, 0.0, 1.05]), np.eye(3)),
        object_bounds=lambda name: np.array([[.43, -.02, 1.03], [.47, .02, 1.07]]),
        ee_pose=lambda *a: (np.array([0.45, 0.0, 1.05]), np.eye(3)),
        home_arm_qpos=lambda side: HOME.copy(),
        arm_qpos=lambda side: np.asarray(arrived_at, float).copy(),
        arm_limits=lambda side: np.array([[-3., 3.]] * 7),
        arm_kinematics=lambda side: kin)
    skills = SkillRunner.__new__(SkillRunner)
    skills.env = env
    skills.controller = SimpleNamespace(move_to_site=move, stop_requested=lambda: False,
                                        solve_ik=lambda *a, **kw: (None, 1., 1.))
    skills.held_object = lambda side: None
    return skills


def test_an_arm_that_ends_where_it_planned_is_parked():
    skills = _skills(arrived_at=HOME)
    result = skills.park_empty_arm('left')
    assert result['success'] is True


def test_an_arm_that_stops_short_is_reported_as_short_stop_not_as_a_collision():
    result = _skills(arrived_at=np.full(7, .3)).park_empty_arm('left')
    assert result['success'] is False
    assert result['error'] == 'HOME_SHORT_STOP'
    evidence = result['evidence']
    assert evidence['deviation_m'] > .005
    assert evidence['arrived_joints'] and evidence['planned_joints']
    assert evidence['traversed']['gap_m'] >= 0, 'nothing was touched; the report must not blame a collision'


def test_a_sweep_that_passes_through_geometry_is_reported_before_any_displacement():
    result = _skills(gap=-0.004).park_empty_arm('left')
    assert result['success'] is False
    assert result['error'] == 'PARK_PATH_UNCLEAR'
    assert result['evidence']['traversed']['gap_m'] < 0
