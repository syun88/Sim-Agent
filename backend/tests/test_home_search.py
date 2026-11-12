"""Getting home is a search that prefers the roomiest route, including retreating away first."""
from types import SimpleNamespace

import numpy as np
import pytest

from backend.app.skills.homing import plan_empty_arm_home

HOME = np.zeros(7)


def _scene(object_position):
    """A body whose joint configuration determines its pose, so "reachability" means something.

    An earlier fake returned one fixed pose for every configuration, which made every target look equally
    reachable (and the canonical-posture search look broken). The mapping below is deliberately linear and
    trivially invertible: the fixture is about which candidates are *tried and ranked*, not about IK.
    """
    state = {'clearance_by_pose': {}, 'pose': np.asarray(object_position, float)}

    def q_to_pose(q):
        q = np.asarray(q, float)
        return np.array([0.40 + q[0] * 0.1, 0.0 + q[1] * 0.1, 1.00 + q[2] * 0.1])

    def pose_to_q(point):
        point = np.asarray(point, float)
        return np.array([(point[0] - 0.40) / 0.1, point[1] / 0.1, (point[2] - 1.00) / 0.1, 0., 0., 0., 0.])

    class Kin:
        def evaluate(self, q):
            return q_to_pose(q), np.eye(3), None, None

        def empty_arm_path_is_clear(self, route, objects, **kwargs):
            return True

        def path_minimum_clearance(self, route, objects, **kwargs):
            first = np.asarray(route[0], float).sum()
            return state['clearance_by_pose'].get(round(float(first), 6), .01)

    def solve_ik(arm, target, rotation, q_init=None):
        return pose_to_q(target), 0.001, 0.0

    env = SimpleNamespace(
        arm_kinematics=lambda arm: Kin(),
        arm_qpos=lambda arm: HOME.copy(), home_arm_qpos=lambda arm: HOME.copy(),
        ee_pose=lambda arm: (q_to_pose(HOME), np.eye(3)),
        object_pose=lambda name: (np.asarray(object_position, float), np.eye(3)),
        object_bounds=lambda name: np.array([np.asarray(object_position, float) - .02,
                                             np.asarray(object_position, float) + .02]),
        frame=lambda: SimpleNamespace(objects={'red_cube': np.asarray(object_position, float)}),
        arm_limits=lambda arm: np.array([[-3., 3.]] * 7))
    controller = SimpleNamespace(solve_ik=solve_ik, stop_requested=lambda: False)
    return env, controller, state


CUBE_BESIDE = [0.52, 0.012, 1.06]


def test_a_retreat_away_from_the_object_is_a_candidate():
    env, controller, state = _scene(CUBE_BESIDE)
    route, evidence = plan_empty_arm_home(env, controller, 'left')
    assert route is not None, evidence
    names = {item['route'] for item in evidence['considered']}
    assert any('retreat_away_from_nearest' in name for name in names), \
        f'a retreat away from the object must be tried; tried: {sorted(names)[:6]}'


def test_the_roomiest_screened_route_is_the_one_used():
    env, controller, state = _scene(CUBE_BESIDE)
    route, evidence = plan_empty_arm_home(env, controller, 'left')
    assert route is not None, evidence
    chosen = next(item for item in evidence['considered'] if item['route'] == evidence['route'])
    assert chosen['clearance_first_segment_m'] == evidence['measured_clearance_m']
    measured = [item['clearance_first_segment_m'] for item in evidence['considered']
                if item['clearance_first_segment_m'] is not None]
    assert chosen['clearance_first_segment_m'] == max(measured)


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
    # Every refusal is named. Which mix appears depends on which candidates the body can solve and screen; the
    # requirement is that a reason is attached to each, never a silent drop.
    assert reasons and reasons <= {'COLLISION_PATH', 'IK_FAILURE'}
    assert evidence['considered'] == []
