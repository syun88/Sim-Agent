"""How precisely a placement has to land is measured from the region it lands on."""
from types import SimpleNamespace

import numpy as np
import pytest

from backend.app.skills import placement_accuracy as accuracy

CUBE = np.array([[0.43, -0.02, 1.01], [0.47, 0.02, 1.05]])         # 40 mm cube
ZONE = np.array([[0.425, -0.055, 1.00], [0.535, 0.055, 1.002]])     # 110 mm zone


def _env(bounds, zones=('blue_zone', 'green_zone')):
    frame = SimpleNamespace(zones={name: [0., 0., 1.] for name in zones if name in bounds})
    return SimpleNamespace(object_bounds=lambda name: np.asarray(bounds[name], float), frame=lambda: frame)


def test_a_zone_is_as_wide_as_the_model_says_minus_the_object():
    answer = accuracy.required_accuracy(_env({'red_cube': CUBE, 'blue_zone': ZONE}), 'red_cube', 'blue_zone', 'in')
    # 55 mm half zone - 20 mm half object = 35 mm of freedom.
    assert answer['required_m'] == pytest.approx(0.035, abs=.001)
    assert answer['basis']['region_half_m'] == pytest.approx([0.055, 0.055])


def test_stacking_is_limited_by_the_support_face_not_by_the_object_width():
    answer = accuracy.required_accuracy(_env({'red_cube': CUBE, 'blue_cube': CUBE}), 'red_cube', 'blue_cube', 'on')
    assert answer['required_m'] == pytest.approx(0.020, abs=.001)
    assert 'support_half_m' in answer['basis']


def test_an_unmeasurable_region_is_unknown_rather_than_permissive():
    def broken(name):
        raise KeyError(name)
    env = SimpleNamespace(object_bounds=broken)
    answer = accuracy.required_accuracy(env, 'red_cube', 'blue_zone', 'in')
    assert answer['required_m'] is None and answer['unknown']


def test_feasibility_uses_the_probe_plus_how_far_the_body_lands():
    env = _env({'red_cube': CUBE, 'blue_zone': ZONE})
    # 8 mm probe + 20 mm landing = 28 mm <= 35 mm of freedom.
    ok = accuracy.feasibility(env, 'red_cube', 'blue_zone', 'in', probe_residual_m=.008, expected_error_m=.020)
    assert ok['feasible'] is True and ok['estimated_landing_error_m'] == pytest.approx(.028)
    # The same body stacking onto a cube: 28 mm > 20 mm of support.
    bad = accuracy.feasibility(_env({'red_cube': CUBE, 'blue_cube': CUBE}), 'red_cube', 'blue_cube', 'on',
                               probe_residual_m=.008, expected_error_m=.020)
    assert bad['feasible'] is False


def test_without_a_probe_or_a_requirement_the_answer_is_unknown_not_false():
    env = _env({'red_cube': CUBE, 'blue_zone': ZONE})
    assert accuracy.feasibility(env, 'red_cube', 'blue_zone', 'in', probe_residual_m=None,
                                expected_error_m=.02)['feasible'] is None
    broken = SimpleNamespace(object_bounds=lambda name: (_ for _ in ()).throw(KeyError(name)))
    assert accuracy.feasibility(broken, 'red_cube', 'blue_zone', 'in', probe_residual_m=.008,
                                expected_error_m=.02)['feasible'] is None
