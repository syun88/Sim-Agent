"""Getting home is a search that prefers the roomiest route, including retreating away first."""
from types import SimpleNamespace

import numpy as np
import pytest

from backend.app.skills.homing import plan_empty_arm_home

HOME = np.zeros(7)


def _scene(object_position):
    """A body whose hands sit next to an object, with home above and behind it."""
    state = {'pose': np.array([0.50, 0.00, 1.09]), 'clearance_by_pose': {}}

    class Kin:
        def evaluate(self, q):
            return state['pose'].copy(), np.eye(3), None, None

        def empty_arm_path_is_clear(self, route, objects, **kwargs):
            return True

        def path_minimum_clearance(self, route, objects, **kwargs):
            # Room depends on where the route starts: starting right next to the object is tightest.
            first = np.asarray(route[0], float).sum()
            return state['clearance_by_pose'].get(round(float(first), 6), .01)

    def solve_ik(arm, target, rotation, q_init=None):
        state['pose'] = np.asarray(target, float).copy()
        return np.full(7, float(np.asarray(target, float).sum())), 0.001, 0.0

    env = SimpleNamespace(
        arm_kinematics=lambda arm: Kin(),
        arm_qpos=lambda arm: HOME.copy(), home_arm_qpos=lambda arm: HOME.copy(),
        ee_pose=lambda arm: (np.array([0.50, 0.00, 1.09]), np.eye(3)),
        object_pose=lambda name: (np.asarray(object_position, float), np.eye(3)),
        object_bounds=lambda name: np.array([np.asarray(object_position, float) - .02,
                                             np.asarray(object_position, float) + .02]),
        frame=lambda: SimpleNamespace(objects={'red_cube': np.asarray(object_position, float)}),
        # Shape is (joints, [min, max]): the planner unpacks one pair per joint.
        arm_limits=lambda arm: np.array([[-3., 3.]] * 7))
    controller = SimpleNamespace(solve_ik=solve_ik, stop_requested=lambda: False)
    return env, controller, state


#: The hand sits beside the object, not inside it: a retreating direction has to exist to be searched.
CUBE_BESIDE = [0.52, 0.012, 1.06]


def test_a_retreat_away_from_the_object_is_a_candidate():
    env, controller, state = _scene(CUBE_BESIDE)
    route, evidence = plan_empty_arm_home(env, controller, 'left')
    assert route is not None, evidence
    names = {item['route'] for item in evidence['considered']}
    assert any(name.startswith('retreat_away_from_nearest') for name in names)


def test_the_roomiest_screened_route_is_the_one_used():
    env, controller, state = _scene(CUBE_BESIDE)
    route, evidence = plan_empty_arm_home(env, controller, 'left')
    assert route is not None, evidence
    chosen = next(item for item in evidence['considered'] if item['route'] == evidence['route'])
    assert chosen['clearance_first_segment_m'] == evidence['measured_clearance_m']
    assert chosen['clearance_first_segment_m'] == max(
        item['clearance_first_segment_m'] for item in evidence['considered'])


def test_every_rejected_candidate_says_why():
    env, controller, state = _scene(CUBE_BESIDE)

    class Blocked:
        def evaluate(self, q):
            return state['pose'].copy(), np.eye(3), None, None

        def empty_arm_path_is_clear(self, *args, **kwargs):
            return False

        def path_minimum_clearance(self, *args, **kwargs):
            return -.01
    env.arm_kinematics = lambda arm: Blocked()
    route, evidence = plan_empty_arm_home(env, controller, 'left')
    assert route is None
    assert evidence['error'] == 'HOME_PATH_UNAVAILABLE'
    reasons = {item['error'] for item in evidence['rejected_routes']}
    # Both refusals are named: a candidate that could not be solved and one that was screened out.
    assert reasons <= {'COLLISION_PATH', 'IK_FAILURE'}
    assert 'COLLISION_PATH' in reasons
    assert evidence['considered'] == []
