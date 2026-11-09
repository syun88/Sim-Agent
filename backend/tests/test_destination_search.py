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
    env.frame = lambda: type('F', (), {'objects': {'red_cube': SUPPORT.copy()}})()
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
