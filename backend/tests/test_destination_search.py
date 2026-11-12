"""A placement the arm cannot reach inside the requested region: find a reachable point, or refuse."""
from types import SimpleNamespace

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
    env.arm_limits = lambda arm: np.array([[-3., 3.]] * 7)
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
    zone_region = np.stack([zone - [0.05, 0.05, 0.], zone + [0.05, 0.05, 0.]])
    env.object_bounds = lambda name: (HELD if name == 'red_cube' else zone_region).copy()
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
    assert zone_region[0][0] <= chosen[0] <= zone_region[1][0]
    assert zone_region[0][1] <= chosen[1] <= zone_region[1][1]


def test_a_destination_that_needs_a_joint_limit_is_not_commandable():
    """Measured 2026-11-09: the carries that stopped 27 mm short were at the edge of the workspace."""
    from backend.app.skills import robot_skills

    skills = _runner(lambda distance: True)
    # Solve "successfully" but into the last 1 degree of a joint: IK says yes, the servo cannot hold it.
    skills.controller.solve_ik = staticmethod(
        lambda arm, target, rotation, **kwargs: (np.array([3.0 - 0.017, 0., 0., 0., 0., 0., 0.]), 0.001, 0.0))
    skills.env.arm_limits = lambda arm: np.array([[-3., 3.]] + [[-3., 3.]] * 6)
    _desired, search = skills._reachable_destination('red_cube', 'blue_zone', 'on', 'left')
    assert search['considered'][0]['joint_limit_margin_rad'] == pytest.approx(0.017, abs=1e-3)
    assert search['chosen'] != 'requested', 'a pose on a joint limit is not a placement this arm can hold'


def test_a_destination_with_room_left_in_the_joints_is_commandable():
    skills = _runner(lambda distance: True)
    skills.controller.solve_ik = staticmethod(
        lambda arm, target, rotation, **kwargs: (np.array([0.5, 0., 0., 0., 0., 0., 0.]), 0.001, 0.0))
    skills.env.arm_limits = lambda arm: np.array([[-3., 3.]] + [[-3., 3.]] * 6)
    _desired, search = skills._reachable_destination('red_cube', 'blue_zone', 'on', 'left')
    assert search['considered'][0]['joint_limit_margin_rad'] == pytest.approx(2.5)
    assert search['chosen'] == 'requested'


def test_the_configuration_the_search_accepted_is_what_the_carry_starts_from():
    """Measured 2026-11-09: without seeding, the motion re-solved to a local solution 27 mm short."""
    skills = _runner(lambda distance: True)
    accepted = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    skills.controller.solve_ik = staticmethod(
        lambda arm, target, rotation, **kwargs: (accepted.copy(), 0.001, 0.0))
    _desired, search = skills._reachable_destination('red_cube', 'blue_zone', 'on', 'left')
    assert search['chosen_joints'] == pytest.approx(accepted.tolist())

    seen = {}
    carried = {'object': SUPPORT.copy()}

    def move(arm, target, rotation, **kwargs):
        seen['q_init'] = kwargs.get('q_init')
        # The carried object ends up where the carry aimed, so the placement itself can be judged.
        offset = OFFSET.copy()
        carried['object'] = np.asarray(target, float) + offset
        return SimpleNamespace(success=True, error=None, position_error=0.0, rotation_error=0.)

    skills.controller.move_to_site = move
    skills.controller.motion_speed = 6.
    skills.env.object_pose = lambda name: (carried['object'].copy(), np.eye(3))
    skills.env.frame = lambda: type('F', (), {'objects': {'red_cube': carried['object'].copy()}, 'zones': {}})()
    skills._require_hold = lambda *a, **k: None
    target_point = np.asarray(search.get('chosen_point') or search['requested'], float)
    skills._carry_to('left', 'red_cube', target_point, q_init=search['chosen_joints'])
    assert seen['q_init'] == pytest.approx(accepted.tolist())


def test_a_pose_that_would_need_a_joint_limit_is_treated_as_unusable_not_unknown():
    """Measured 2026-11-09: it was called unmeasurable, the search gave up, and the carry failed 28 mm short."""
    skills = _runner(lambda distance: True)
    # Solves, but into the last degree of a joint: known to be unusable rather than unknown.
    skills.controller.solve_ik = staticmethod(
        lambda arm, target, rotation, **kwargs: (np.array([3.0 - 0.01, 0., 0., 0., 0., 0., 0.]), 0.001, 0.0))
    skills.env.arm_limits = lambda arm: np.array([[-3., 3.]] + [[-3., 3.]] * 6)
    _desired, search = skills._reachable_destination('red_cube', 'blue_zone', 'on', 'left')
    assert search['considered'][0]['blocked_by_joint_limit'] is True
    assert search['chosen'] in {'unreachable', 'offset_inside_region'}
    assert search['chosen'] != 'unmeasured', 'a measured-but-unusable pose is not an unknown'


def test_no_ik_solution_is_unreachable_not_unknown():
    """Measured 2026-11-09: 'no solution' was read as unknown, so a point 254 mm out of reach was commanded."""
    skills = _runner(lambda distance: True)
    skills.controller.solve_ik = staticmethod(lambda arm, target, rotation, **kwargs: (None, None, 0.))
    _desired, search = skills._reachable_destination('red_cube', 'blue_zone', 'on', 'left')
    assert search['chosen'] != 'unmeasured', 'a solver that says "no" has answered the question'
    assert search['chosen'] in {'unreachable', 'offset_inside_region'}


def test_a_posture_that_would_reach_the_zone_is_offered_as_staging():
    """The zone may be unreachable from here and reachable from somewhere else; that is worth knowing."""
    skills = _runner(lambda distance: True)
    home = np.zeros(7)
    good = np.array([0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    skills.env.home_arm_qpos = lambda arm: home.copy()
    skills.env.arm_limits = lambda arm: np.array([[-3., 3.]] * 7)
    skills.env.arm_qpos = lambda arm: np.full(7, 3.0 - 0.01)   # arm parked on a limit: cannot reach

    def solve(arm, target, rotation, **kwargs):
        seed = kwargs.get('q_init')
        if seed is not None and np.allclose(seed, home):
            return home.copy(), 0.002, 0.0
        if seed is not None and np.allclose(seed, good):
            return good.copy(), 0.002, 0.0
        return None, None, 0.0

    skills.controller.solve_ik = staticmethod(solve)
    _desired, search = skills._reachable_destination('red_cube', 'blue_zone', 'on', 'left')
    assert search['chosen'] == 'unreachable'
    assert search['stage_posture'] is not None, 'a reachable posture exists and must be reported'
