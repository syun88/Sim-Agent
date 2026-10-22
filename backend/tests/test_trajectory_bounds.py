"""Velocity limits must hold between, as well as at, spline knots."""
import numpy as np
from scipy.interpolate import PchipInterpolator

from backend.app.control.motion import maximum_spline_slope, joint_spline


def test_velocity_bound_covers_dense_quintic_samples_and_does_not_grow_with_sampling():
    rng = np.random.default_rng(21)
    points = np.cumsum(rng.uniform(-.8, .8, (12, 7)), axis=0)
    curve = PchipInterpolator(np.linspace(0., 1., len(points)), points, axis=0)
    bound = maximum_spline_slope(curve)
    t = np.linspace(0., 1., 20001)
    alpha = t**3*(10.-15.*t+6.*t*t)
    velocity = abs(curve.derivative()(alpha)*(30.*t*t*(1.-t)**2)[:, None])
    assert np.all(velocity.max(axis=0) <= 1.875*bound+1e-9)
    np.testing.assert_allclose(bound, abs(curve.derivative()(t)).max(axis=0), rtol=2e-5)
    for count in (3, 30, 300):
        straight = PchipInterpolator(np.linspace(0., 1., count), np.linspace([0., 0.], [1., -.4], count), axis=0)
        np.testing.assert_allclose(maximum_spline_slope(straight), [1., .4], atol=1e-10)


def test_dense_short_segments_do_not_slow_down_a_single_long_segment():
    points = np.array([[0.],[.001],[.002],[1.],[1.001],[1.002]])
    uniform = joint_spline(points)
    allocated = joint_spline(points,np.array([1.]))
    assert maximum_spline_slope(allocated)[0]<maximum_spline_slope(uniform)[0]/4.
    np.testing.assert_allclose(allocated(allocated.x),points,atol=1e-12)
    np.testing.assert_allclose(maximum_spline_slope(allocated),[1.002],atol=1e-9)
