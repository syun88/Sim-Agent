"""A rotation has two arcs. Both must end at the target, and they must sweep differently."""
import numpy as np
from scipy.spatial.transform import Rotation

from backend.app.skills.reorientation import arc_samples


def test_every_arc_ends_at_the_target():
    start = Rotation.from_rotvec([0.2, -0.1, 0.3]).as_matrix()
    target = Rotation.from_rotvec([0.1, 1.8, -0.4]).as_matrix()
    for mode in ('short', 'long', 'direct'):
        samples = arc_samples(start, target, mode)
        assert samples, mode
        assert np.allclose(samples[-1], target, atol=1e-8), mode


def test_the_complementary_arc_sweeps_the_other_way():
    start = Rotation.from_rotvec([0.2, -0.1, 0.3]).as_matrix()
    target = Rotation.from_rotvec([0.1, 1.8, -0.4]).as_matrix()
    def travelled(mode):
        step = Rotation.from_matrix(start).inv() * Rotation.from_matrix(arc_samples(start, target, mode)[3])
        return float(np.linalg.norm(step.as_rotvec()))
    assert travelled('long') > travelled('short')


def test_the_direct_move_is_a_single_attitude():
    start = np.eye(3)
    target = Rotation.from_rotvec([0., 0., 1.2]).as_matrix()
    samples = arc_samples(start, target, 'direct')
    assert len(samples) == 1
    assert np.allclose(samples[0], target, atol=1e-8)


def test_an_unchanged_attitude_needs_no_arc():
    start = Rotation.from_rotvec([0.3, 0.2, -0.1]).as_matrix()
    assert len(arc_samples(start, start, 'short')) == 1
