"""One owner answers "where can this hand go", and it answers the same way every time."""
from types import SimpleNamespace

import numpy as np

from backend.app.planning import body_solver


def _body(*, residual=0.002, solution=None, limits=None):
    limits = np.array([[-3., 3.]] * 7) if limits is None else limits
    config = np.zeros(7) if solution is None else np.asarray(solution, float)

    def solve(arm, target, rotation, **kwargs):
        # A body whose solve does not depend on the seed: the answer is about the pose, not the starting point.
        return config.copy(), residual, 0.

    env = SimpleNamespace(home_arm_qpos=lambda arm: np.zeros(7), arm_limits=lambda arm: limits)
    return env, SimpleNamespace(solve_ik=solve), config


def test_the_same_question_twice_gives_the_same_answer():
    body_solver.clear_cache()
    env, controller, _ = _body()
    first = body_solver.solve_pose(env, controller, 'left', [0.45, 0.0, 1.05], np.eye(3))
    second = body_solver.solve_pose(env, controller, 'left', [0.45, 0.0, 1.05], np.eye(3))
    assert first['ok'] == second['ok']
    assert first['solution'] == second['solution']
    assert second.get('cached') is True


def test_the_answer_does_not_depend_on_where_the_arm_is_standing():
    body_solver.clear_cache()
    env, controller, config = _body()
    asked = [body_solver.solve_pose(env, controller, 'left', [0.45, 0.0, 1.05], np.eye(3),
                                    posture='canonical') for _ in range(3)]
    assert len({(item['ok'], tuple(item['solution'])) for item in asked}) == 1


def test_a_pose_on_a_joint_limit_is_not_feasible():
    body_solver.clear_cache()
    env, controller, _ = _body(solution=np.array([3.0 - 0.01, 0., 0., 0., 0., 0., 0.]))
    answer = body_solver.solve_pose(env, controller, 'left', [0.45, 0.0, 1.05], np.eye(3))
    assert answer['ok'] is False
    assert answer['joint_limit_margin_rad'] is not None and answer['joint_limit_margin_rad'] < body_solver.LIMIT_MARGIN_RAD


def test_the_capability_query_and_the_solver_agree_on_the_same_pose():
    body_solver.clear_cache()
    from backend.app.planning.capability_queries import reachable
    env, controller, _ = _body()
    solver = body_solver.solve_pose(env, controller, 'left', [0.45, 0.0, 1.05], np.eye(3))
    query = reachable(env, controller, 'left', [0.45, 0.0, 1.05], 0.)
    assert query['ok'] == solver['ok']
    assert query['position_error_m'] == solver['residual_m']


def test_a_solver_that_cannot_answer_says_unknown_rather_than_reachable():
    env = SimpleNamespace(home_arm_qpos=lambda arm: np.zeros(7), arm_limits=lambda arm: np.array([[-3., 3.]] * 7))
    body_solver.clear_cache()
    answer = body_solver.solve_pose(env, SimpleNamespace(), 'left', [0.45, 0.0, 1.05], np.eye(3))
    assert answer['ok'] is None and 'unknown' in answer
