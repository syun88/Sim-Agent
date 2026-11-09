"""A placement the arm cannot reach inside the requested region: find a reachable point, or refuse."""
import numpy as np
import pytest

from backend.app.skills.robot_skills import SkillRunner

SUPPORT = np.array([0.45, 0.0, 1.03])
#: A region wider than the object (a zone, or a large support face): offsetting inside it is legitimate.
REGION = np.array([[0.40, -0.05, 1.00], [0.50, 0.05, 1.06]])
HELD = np.array([[0.43, -0.02, 1.01], [0.47, 0.02, 1.05]])
OFFSET = SUPPORT - np.array([0.40, 0.0, 1.15])


def _runner(accept):
    """`accept(distance)` decides whether IK reaches the requested object point."""
    def solve(arm, target, rotation, **kwargs):
        # The carry commands the end effector held off by the grasp offset, so the object point is the aim
        # plus that offset: the fake judges the point that was actually asked for.
        point = np.asarray(target, float) + OFFSET
        distance = float(np.linalg.norm(point[:2] - SUPPORT[:2]))
        return np.zeros(7), 0.001 if accept(distance) else 0.05, 0.0

    env = type('E', (), {})()
    env.object_pose = lambda name: (SUPPORT.copy(), np.eye(3))
    env.object_bounds = lambda name: (HELD if name == 'red_cube' else REGION).copy()
    env.ee_pose = lambda arm: (np.array([0.40, 0.0, 1.15]), np.eye(3))
    env.frame = lambda: type('F', (), {'objects': {'red_cube': SUPPORT.copy()},
                                       'zones': {'blue_zone': np.array([0.45, 0.0, 1.0])}})()
    skills = SkillRunner.__new__(SkillRunner)
    skills.env = env
    skills.controller = type('C', (), {'solve_ik': staticmethod(solve), 'target_rotations': {}})()
    return skills


def test_the_requested_point_is_used_when_it_is_reachable():
    skills = _runner(lambda distance: True)
    desired, search = skills._reachable_destination('red_cube', 'blue_zone', 'on', 'left')
    assert search['chosen'] == 'requested'
    assert np.allclose(desired, search['requested'])


def test_an_unreachable_exact_point_is_replaced_inside_the_region_and_recorded():
    skills = _runner(lambda distance: 0.004 < distance <= 0.02)
    desired, search = skills._reachable_destination('red_cube', 'blue_zone', 'on', 'left')
    assert search['chosen'] == 'offset_inside_region'
    assert search['chosen_point'] == pytest.approx(list(desired))
    assert len(search['considered']) > 1, 'every candidate that was tried is in the evidence'
    # The chosen point stayed inside the requested region.
    assert REGION[0][0] <= desired[0] <= REGION[1][0]
    assert REGION[0][1] <= desired[1] <= REGION[1][1]


def test_nothing_reachable_inside_the_region_is_reported_not_substituted():
    skills = _runner(lambda distance: False)
    desired, search = skills._reachable_destination('red_cube', 'blue_zone', 'on', 'left')
    assert search['chosen'] == 'unreachable'
    assert np.allclose(desired, search['requested']), 'the request is not silently moved'


def test_a_region_no_wider_than_the_object_offers_no_alternative():
    """Stacking onto a cube of the same size: there is nowhere else on the face to put it."""
    skills = _runner(lambda distance: False)
    skills.env.object_bounds = lambda name: HELD.copy()
    desired, search = skills._reachable_destination('red_cube', 'blue_cube', 'on', 'left')
    assert search['chosen'] == 'unreachable'
    assert len(search['considered']) == 1


def test_the_region_measures_whether_the_landing_error_fits_not_a_global_tolerance():
    """A zone has tens of mm of freedom; a cube stacked on a cube has only its support half-extent."""
    from backend.app.skills import motion_accuracy

    motion_accuracy.reset()
    for _ in range(3):
        motion_accuracy.record('left', 0.020, source='carry')
    skills = _runner(lambda distance: True)
    skills.controller.solve_ik = staticmethod(
        lambda arm, target, rotation, **kwargs: (np.zeros(7), 0.008, 0.0))
    _desired, search = skills._reachable_destination('red_cube', 'blue_zone', 'on', 'left')
    assert search['expected_carry_error_m'] == pytest.approx(0.020)
    feasibility = search['feasibility']
    assert feasibility['required_m'] == pytest.approx(0.03, abs=.002), 'zone freedom = half zone - half object'
    assert feasibility['estimated_landing_error_m'] == pytest.approx(0.028)
    assert feasibility['feasible'] is True, '28 mm of landing error still fits a 30 mm region'
    assert search['chosen'] == 'requested'


def test_stacking_onto_a_same_size_cube_does_not_have_that_room():
    from backend.app.skills import motion_accuracy

    motion_accuracy.reset()
    for _ in range(3):
        motion_accuracy.record('left', 0.020, source='carry')
    skills = _runner(lambda distance: True)
    skills.controller.solve_ik = staticmethod(
        lambda arm, target, rotation, **kwargs: (np.zeros(7), 0.008, 0.0))
    skills.env.object_bounds = lambda name: HELD.copy()      # support the same size as the held cube
    _desired, search = skills._reachable_destination('red_cube', 'blue_cube', 'on', 'left')
    feasibility = search['feasibility']
    assert feasibility['required_m'] == pytest.approx(0.02, abs=.002)
    assert feasibility['feasible'] is False, '28 mm cannot be guaranteed onto a 20 mm support half-extent'


def test_a_zone_placement_searches_inside_the_zone_instead_of_carrying_to_the_centre():
    """Measured 2026-11-09: zone placements failed 10 of 12 because this path had no search at all."""
    from types import SimpleNamespace
    from backend.app.skills.robot_skills import RELEASE_HEIGHT

    zone = np.array([0.48, 0.18, 1.0])
    carried = []

    def solve(arm, target, rotation, **kwargs):
        point = np.asarray(target, float) + OFFSET
        offset = float(np.linalg.norm(point[:2] - zone[:2]))
        return np.zeros(7), 0.001 if offset > 0.004 else 0.05, 0.0

    def move(arm, target, rotation, **kwargs):
        carried.append(np.asarray(target, float).copy())
        return SimpleNamespace(success=True, final_position=list(target))

    env = type('E', (), {})()
    env.object_pose = lambda name: (SUPPORT.copy(), np.eye(3))
    env.object_bounds = lambda name: (HELD if name == 'red_cube' else REGION).copy()
    env.ee_pose = lambda arm: (np.array([0.40, 0.0, 1.15]), np.eye(3))
    env.zone_position = lambda name: zone.copy()
    env.frame = lambda: type('F', (), {'objects': {'red_cube': SUPPORT.copy()}, 'zones': {'blue_zone': zone}})()
    skills = SkillRunner.__new__(SkillRunner)
    skills.env = env
    skills.held_object = lambda arm: 'red_cube'
    skills.controller = type('C', (), {'solve_ik': staticmethod(solve), 'target_rotations': {},
                                       'move_to_site': staticmethod(move), 'motion_speed': 1.})()
    skills._require_hold = lambda *a, **k: None
    skills._align_for_placement = lambda *a, **k: None
    skills._carry_to = lambda arm, obj, desired, **k: {'object': obj, 'position': np.asarray(desired).tolist(),
                                                      'position_error': 0.0}
    result = skills._move_to_zone('left', 'blue_zone')
    search = result['destination_search']
    assert search['chosen'] == 'offset_inside_region'
    chosen = np.asarray(search['chosen_point'], float)
    assert np.linalg.norm(chosen - np.asarray(search['requested'], float)) > 1e-6, \
        'the zone centre was not reachable; a point inside it was'
    assert REGION[0][0] <= chosen[0] <= REGION[1][0] and REGION[0][1] <= chosen[1] <= REGION[1][1]
