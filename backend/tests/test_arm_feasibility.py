"""The hand the instruction names is measured before any motion, and the alternative is named."""
from types import SimpleNamespace

import numpy as np

from backend.app.planning.preflight import arm_feasibility


def _env(cube, *, right_error, left_error):
    """Two arms whose IK residual at this cube's pinch point is fixed per arm."""
    def solve(arm, target, rotation, **kwargs):
        error = right_error if arm == 'right' else left_error
        return (np.zeros(7) if error <= .012 else None), error, 0.

    env = SimpleNamespace(object_pose=lambda name: (np.asarray(cube, float), np.eye(3)),
                          frame=lambda: SimpleNamespace(objects={name: cube for name in ('red_cube',)},
                                                       zones={}))
    controller = SimpleNamespace(solve_ik=solve)
    return env, controller


def test_an_arm_that_cannot_reach_is_reported_with_the_one_that_can():
    """Measured 2026-11-09: a right-hand instruction for a cube at y=+0.264 failed twelve attempts."""
    env, controller = _env([0.54, 0.264, 1.031], right_error=.06, left_error=.002)
    rows = arm_feasibility(env, controller, [{'op': 'pick', 'object': 'red_cube', 'arm': 'right'}])
    row = rows[0]
    assert row['verdict'] == 'REQUESTED_ARM_UNREACHABLE'
    assert row['reachable_arms'] == ['left']
    assert row['alternatives']['right']['reachable_yaws'] == 0
    assert row['alternatives']['left']['reachable_yaws'] > 0


def test_a_reachable_request_is_not_refused():
    env, controller = _env([0.45, 0.0, 1.03], right_error=.002, left_error=.002)
    rows = arm_feasibility(env, controller, [{'op': 'pick', 'object': 'red_cube', 'arm': 'right'}])
    assert rows[0]['verdict'] == 'REQUESTED_ARM_REACHABLE'


def test_neither_arm_reaching_is_not_blamed_on_the_hand_that_was_asked_for():
    env, controller = _env([0.9, 0.9, 1.03], right_error=.08, left_error=.09)
    rows = arm_feasibility(env, controller, [{'op': 'pick', 'object': 'red_cube', 'arm': 'right'}])
    row = rows[0]
    assert row['requested_reachable'] is False
    assert row['reachable_arms'] == []
    assert row['verdict'] == 'UNMEASURED', 'the refusal must be reserved for "the other hand can"'


def test_an_unmeasurable_arm_says_so_instead_of_guessing():
    def broken(arm, target, rotation, **kwargs):
        raise RuntimeError('no ik')
    env = SimpleNamespace(object_pose=lambda name: (np.array([0.45, 0., 1.03]), np.eye(3)),
                          frame=lambda: SimpleNamespace(objects={'red_cube': [0.45, 0., 1.03]}, zones={}))
    rows = arm_feasibility(env, SimpleNamespace(solve_ik=broken),
                           [{'op': 'pick', 'object': 'red_cube', 'arm': 'left'}])
    assert rows[0]['verdict'] == 'UNMEASURED'
    answer = rows[0]['alternatives']['left']
    assert answer['asked_yaws'] == 0, 'nothing was measurable, which is not the same as reachable'
    assert 'RuntimeError' in (answer.get('unknown') or '')


def test_the_arm_question_is_asked_with_the_approach_planner_not_only_the_ik():
    """Measured 2026-11-09: an IK probe called the hand reachable while the approach search found no route."""
    import numpy as np
    from backend.app.planning import preflight
    from backend.app.planning.capability_queries import approach_feasible

    env, controller = _env([0.45, 0.0, 1.03], right_error=.002, left_error=.002)

    def screened(env_, controller_, arm, obj):
        return {'feasible': arm == 'left', 'reason': None if arm == 'left' else 'APPROACH_PATH_UNAVAILABLE'}
    original = approach_feasible
    try:
        import backend.app.planning.capability_queries as queries
        queries.approach_feasible = screened
        rows = preflight.arm_feasibility(env, controller, [{'op': 'pick', 'object': 'red_cube', 'arm': 'right'}])
    finally:
        queries.approach_feasible = original
    row = rows[0]
    assert row['screen']['right']['feasible'] is False
    assert row['verdict'] == 'REQUESTED_ARM_UNREACHABLE'
    assert row['reachable_arms'] == ['left']


def test_a_body_that_cannot_answer_the_route_question_is_unknown_not_unreachable():
    from types import SimpleNamespace
    from backend.app.planning.capability_queries import approach_feasible

    class Bare:
        def object_pose(self, name):
            return np.array([0.45, 0., 1.03]), np.eye(3)
    answer = approach_feasible(Bare(), SimpleNamespace(), 'left', 'red_cube')
    assert answer['feasible'] is None and 'unknown' in answer
