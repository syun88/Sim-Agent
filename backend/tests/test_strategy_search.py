"""The stack strategy must come from queries about the body, not from a remembered constant."""
from types import SimpleNamespace

import numpy as np

from backend.app.planning import strategy_search
from backend.app.planning.capability_queries import catalog, pinch_point, reachable
from backend.app.planning.strategy_search import plan_stack_on_held_support

SUPPORT = np.array([0.44, 0.15, 1.031])
BOUNDS = np.array([[0.417, 0.127, 1.005], [0.463, 0.173, 1.057]])


def _env(clearance=-0.0405):
    return SimpleNamespace(
        object_pose=lambda name: (SUPPORT.copy(), np.eye(3)),
        object_bounds=lambda name: BOUNDS.copy(),
        arm_kinematics=None,
        arm_qpos=lambda arm: np.zeros(7),
    )


def _controller(reachable_arms=('left', 'right')):
    def solve(arm, target, rotation, **kw):
        return (np.zeros(7), 0.004, 0.0) if arm in reachable_arms else (None, 1.0, 1.0)
    return SimpleNamespace(solve_ik=solve)


def test_the_catalog_is_the_planner_visible_question_list():
    names = {entry['name'] for entry in catalog()}
    assert names == {'reachable_pose', 'pad_blocks_support_face'}
    assert all(entry['returns'] for entry in catalog())


def test_a_held_support_whose_fingers_cover_its_face_stops_the_search(monkeypatch):
    monkeypatch.setattr(strategy_search, 'pad_blocks_support_face',
                        lambda *a, **k: {'blocks': True, 'clearance_m': -0.0405})
    result = plan_stack_on_held_support(_env(), _controller(), 'left', 'support')
    assert result['ok'] is False
    assert result['reason'] == 'SUPPORT_FACE_BLOCKED'
    assert result['mechanism'] == 'HAND_GEOMETRY'
    assert result['clearance_m'] == -0.0405
    # The refusal is one query, not a search: the fingers are the blocker whatever the staging point.
    assert len(result['evaluations']) == 1


def test_the_search_prefers_a_staging_point_on_the_placing_arms_side(monkeypatch):
    monkeypatch.setattr(strategy_search, 'pad_blocks_support_face', lambda *a, **k: {'blocks': False})
    result = plan_stack_on_held_support(_env(), _controller(), 'left', 'support')
    assert result['ok'] is True
    assert result['staging'][1] > 0, 'a left-hand placement stages on the left side of the table'
    assert result['placer_yaw_deg'] in strategy_search.ATTITUDE_YAWS_DEG


def test_a_scene_where_the_placing_arm_cannot_reach_refuses_with_the_measurements(monkeypatch):
    monkeypatch.setattr(strategy_search, 'pad_blocks_support_face', lambda *a, **k: {'blocks': False})
    result = plan_stack_on_held_support(_env(), _controller(reachable_arms=('right',)), 'left', 'support')
    assert result['ok'] is False
    assert result['reason'] == 'NO_REACHABLE_STACK_POSE'
    assert any(item['query'] == 'reachable_pose' and item['arm'] == 'left' for item in result['evaluations'])
    # The holder was reachable; it is the placing arm that could not get there in any attitude.
    assert any(item['query'] == 'reachable_pose' and item['arm'] == 'right'
               and item.get('ok') is True for item in result['evaluations'])
    assert all(item.get('ok') is not True for item in result['evaluations'] if item['arm'] == 'left')


def test_an_unmeasurable_body_falls_back_instead_of_refusing(monkeypatch):
    monkeypatch.setattr(strategy_search, 'pad_blocks_support_face', lambda *a, **k: {'blocks': False})
    no_solver = SimpleNamespace()
    result = plan_stack_on_held_support(_env(), no_solver, 'left', 'support')
    assert result['ok'] is None and result['reason'] == 'UNMEASURED'
    assert result['evaluations'], 'the attempts are still recorded'


def test_a_reach_query_reports_a_missing_solver_as_unknown():
    assert reachable(_env(), SimpleNamespace(), 'left', np.zeros(3))['ok'] is None


def test_the_pinch_point_puts_the_pads_on_the_object_center():
    from backend.app.simulation.openarm_v2.definition import FINGERTIP_MIDPOINT
    from backend.app.planning.capability_queries import attitude
    centre = np.array([0.45, 0.0, 1.10])
    offset = pinch_point('left', centre) - centre
    # The offset is the fingertip midpoint turned by the grasp attitude: a yaw rotates which way the
    # pads face, so the control point moves with it.
    assert np.allclose(offset, -attitude(0.) @ FINGERTIP_MIDPOINT['left'], atol=1e-9)
    assert not np.allclose(pinch_point('left', centre, 90.) - centre, offset, atol=1e-6)
