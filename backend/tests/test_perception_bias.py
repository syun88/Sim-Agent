"""A measurement harness that biases what the planner reads, keeping the truth for verification."""
import numpy as np

from backend.app.simulation import perception


def _env(pose=(0.45, 0.0, 1.05)):
    position = np.asarray(pose, float)
    return type('E', (), {
        'object_pose': lambda self, name: (position.copy(), np.eye(3)),
        'frame': lambda self: type('F', (), {'objects': {'red_cube': position.copy()}})(),
        'zone_position': lambda self, name: np.zeros(3),
    })()


def test_the_bias_is_exactly_the_requested_size_and_leaves_the_truth_reachable():
    env = _env()
    perception.install(env, 0.010, seed=3)
    biased, _ = env.object_pose('red_cube')
    truth, _ = perception.truth_pose(env, 'red_cube')
    assert np.linalg.norm(biased - truth) == np.testing.approx(0.010, abs=1e-6) if hasattr(np.testing, 'approx') \
        else abs(np.linalg.norm(biased - truth) - 0.010) < 1e-6
    assert np.allclose(truth, [0.45, 0.0, 1.05])


def test_the_bias_is_fixed_per_object_not_noise():
    env = _env()
    perception.install(env, 0.020, seed=7)
    first, _ = env.object_pose('red_cube')
    second, _ = env.object_pose('red_cube')
    assert np.allclose(first, second), 'a calibration bias does not average out between samples'


def test_verification_reads_the_truth_while_the_planner_reads_the_bias():
    env = _env()
    perception.install(env, 0.030, seed=1)
    assert not np.allclose(env.object_pose('red_cube')[0], [0.45, 0.0, 1.05])
    view = perception.truth_view(env)
    assert np.allclose(view.object_pose('red_cube')[0], [0.45, 0.0, 1.05])
    assert view.zone_position('blue_zone').shape == (3,), 'everything else delegates to the body'


def test_without_a_bias_the_view_is_the_body_itself():
    env = _env()
    assert perception.truth_view(env) is env
    assert perception.describe(env) is None
