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


def test_a_place_also_asks_whether_the_arm_can_reach_what_it_will_pick_up():
    """Zone placements compile to a single `place` action; the reach question still applies."""
    env, controller = _env([0.54, 0.264, 1.031], right_error=.06, left_error=.002)
    rows = arm_feasibility(env, controller, [{'op': 'place', 'object': 'red_cube', 'arm': 'right',
                                              'target': 'blue_zone'}])
    assert rows and rows[0]['verdict'] == 'REQUESTED_ARM_UNREACHABLE'


def test_an_inferred_hand_is_switched_not_refused():
    """Measured 2026-11-09: '青いキューブを赤いキューブの上に置いて' was refused for the inferred left hand
    while the right hand had a route."""
    from types import SimpleNamespace
    from backend.app.agent.task_agent import _switch_plan_arm

    action = SimpleNamespace(arm='left', receiver=None)
    step = SimpleNamespace(arm='left', receiver=None)
    plan = SimpleNamespace(program=SimpleNamespace(actions=[action]), steps=[step])
    _switch_plan_arm(plan, 'left', 'right')
    assert action.arm == 'right' and step.arm == 'right'


def test_the_refusal_is_reserved_for_a_hand_the_instruction_named():
    from backend.app.planning.language_fallback import arm_mentions

    assert arm_mentions('青いキューブを右手で青いゾーンに置いて') == ['right']
    assert arm_mentions('青いキューブを赤いキューブの上に置いて') == []


def test_the_layout_answer_does_not_depend_on_where_the_arm_is_standing():
    """Root cause, measured 2026-11-09: from-here answered False/True/True for one layout depending on posture."""
    from backend.app.planning.capability_queries import approach_feasible, clear_feasibility_cache
    from backend.app.control.motion import OpenArmV2Controller
    from backend.tests.test_task_homing import _stack_environment

    env, q, _objects = _stack_environment()
    env.arm_velocity_limits = lambda side: None
    controller = OpenArmV2Controller(env, configure_motors=False)
    answers = []
    for posture in (np.zeros(7),
                    np.array([1.17, -.37, .19, 2.39, -.04, .21, .40]),
                    np.array([.01, -.67, .09, 1.72, -.18, .78, -1.09])):
        clear_feasibility_cache()
        q['left'] = posture
        answers.append(approach_feasible(env, controller, 'left', 'blue_cube').get('feasible'))
    assert answers[0] is not None
    assert len(set(answers)) == 1, f'the same layout gave different answers: {answers}'


def test_the_canonical_postures_are_a_fixed_set():
    from backend.app.planning.capability_queries import canonical_postures

    env = type('E', (), {'home_arm_qpos': lambda self, arm: np.zeros(7),
                         'arm_limits': lambda self, arm: np.array([[-3., 3.]] * 7)})()
    first = canonical_postures(env, 'left')
    second = canonical_postures(env, 'left')
    assert len(first) >= 2
    assert all(np.allclose(a, b) for a, b in zip(first, second))
    assert np.allclose(first[0], np.zeros(7)), 'home is always part of the set'


def test_an_inferred_hand_is_switched_even_when_refusal_is_off(monkeypatch):
    """Refusing was measured as harmful; switching an unnamed hand costs nothing, so it always runs."""
    from backend.app.planning import measurement_flags

    monkeypatch.setattr(measurement_flags, 'enabled', lambda name: False)
    rows = [{'verdict': 'REQUESTED_ARM_UNREACHABLE', 'arm': 'right', 'object': 'red_cube',
             'reachable_arms': ['left'], 'screen': {'left': {'feasible': True}}}]
    explicit = set()          # no hand named in the instruction
    switchable = [row for row in rows
                  if row.get('verdict') == 'REQUESTED_ARM_UNREACHABLE'
                  and row['arm'] not in explicit and row.get('reachable_arms')]
    assert [row['reachable_arms'][0] for row in switchable] == ['left']


def test_the_bounded_route_question_is_cheaper_but_same_shape():
    """A reset admission runs on every scene, so the question can be given a budget."""
    import inspect
    from backend.app.planning.capability_queries import approach_feasible

    params = inspect.signature(approach_feasible).parameters
    assert 'max_postures' in params and 'max_frames' in params
