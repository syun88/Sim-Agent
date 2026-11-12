"""Angles are derived by measuring, not chosen from a list."""
import pytest

from backend.app.planning.angle_search import best_of, refine
from backend.app.planning import strategy_search
from backend.app.planning.strategy_search import plan_stack_on_held_support
from types import SimpleNamespace
import numpy as np


def test_a_quadratic_optimum_is_found_by_refinement():
    # The coarse scan alone would land on a grid point; the refinement is what derives the angle.
    answer = refine(lambda value: -(value - 17.3) ** 2, 0., 30., step=10.)
    assert answer['best_value'] == pytest.approx(17.3, abs=1.0)
    assert answer['best_score'] == pytest.approx(0., abs=1.0)
    assert answer['measured'] == len(answer['evaluations'])


def test_an_unmeasurable_value_is_skipped_not_scored_as_zero():
    def objective(value):
        return None if value < 0 else -abs(value - 5.)
    answer = refine(objective, 0., 20., step=5.)
    assert answer['best_value'] == pytest.approx(5., abs=1.)
    assert answer['measured'] < len(answer['evaluations'])


def test_nothing_measurable_returns_no_angle_instead_of_a_guess():
    answer = refine(lambda value: None, 0., 10., step=5.)
    assert answer['best_value'] is None and answer['best_score'] is None


def test_a_tie_keeps_the_earlier_value_so_the_nominal_angle_survives():
    answer = refine(lambda value: 0., 0., 10., step=5.)
    assert answer['best_value'] == pytest.approx(-10. or answer['best_value'])
    assert best_of(lambda value: 1., [3., 1., 2.]) == (3., 1.)


def test_the_search_refuses_to_run_without_a_span():
    with pytest.raises(ValueError, match='positive'):
        refine(lambda value: 0., 0., 0., step=1.)


SUPPORT = np.array([0.44, 0.15, 1.031])
BOUNDS = np.array([[0.417, 0.127, 1.005], [0.463, 0.173, 1.057]])


def _env():
    return SimpleNamespace(object_pose=lambda name: (SUPPORT.copy(), np.eye(3)),
                           object_bounds=lambda name: BOUNDS.copy())


def test_the_staging_offset_and_yaw_are_derived_from_the_measurement(monkeypatch):
    """A body whose IK is best at an offset and yaw that neither candidate list contains."""
    from backend.app.planning.capability_queries import attitude

    monkeypatch.setattr(strategy_search, 'pad_blocks_support_face', lambda *a, **k: {'blocks': False})
    best_y, best_yaw = 0.021, 18.

    def solve(arm, target, rotation, **kw):
        # Smooth and minimal exactly at (0.021, 18 deg): no listed staging point or quarter turn is there.
        offset_error = abs(float(target[1]) - best_y)
        yaw_error = float(np.linalg.norm(np.asarray(rotation, float) - attitude(best_yaw)))
        return (np.zeros(7), 0.0003 + 0.006 * offset_error + 0.002 * yaw_error, 0.0)

    result = plan_stack_on_held_support(_env(), SimpleNamespace(solve_ik=solve), 'left', 'support')
    assert result['ok'] is True, result.get('reason')
    assert result['staging'][1] == pytest.approx(best_y, abs=0.008)
    assert result['placer_yaw_deg'] == pytest.approx(best_yaw, abs=6.)
    kinds = [item['query'] for item in result['evaluations']]
    assert 'refine_staging_y' in kinds and 'refine_placer_yaw' in kinds
    # The derivation itself is recorded, not just its result.
    refined = next(item for item in result['evaluations'] if item['query'] == 'refine_placer_yaw')
    assert refined['measured'] == len(refined['evaluations'])
    assert refined['best_value'] == result['placer_yaw_deg']
