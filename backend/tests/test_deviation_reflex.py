"""The deviation reflex: it watches a leg, and it only stops a leg that had already left the plan."""
import numpy as np
import pytest

from backend.app.control import reflex
from backend.app.skills import motion_accuracy


class FakeKinematics:
    """Positions are the joint vector itself, so a deviation is exactly the joint offset."""

    def evaluate(self, joints):
        return np.asarray(joints, float), np.eye(3), None, None


def test_the_threshold_follows_the_measured_error_for_the_phase():
    motion_accuracy.reset()
    approach = reflex.threshold_m('left', 'approach')      # floor 8 mm -> 22 mm
    park = reflex.threshold_m('left', 'park_arrival')      # floor 35 mm -> 62.5 mm
    assert approach < park
    assert approach == pytest.approx(min(0.12, max(0.01, 0.008 * 1.5 + 0.01)))
    motion_accuracy.reset()


def test_a_transient_departure_does_not_stop_the_leg():
    watcher = reflex.DeviationReflex('left', 'approach', threshold_m=0.02)
    planned = np.zeros(7)
    far = np.full(7, 0.05)
    assert watcher.observe(planned, far, 0.5, FakeKinematics()) is None, 'one sample is a transient'
    assert watcher.observe(planned, far, 0.5, FakeKinematics()) is None


def test_a_persistent_departure_is_reported_with_its_numbers():
    watcher = reflex.DeviationReflex('left', 'approach', threshold_m=0.02)
    planned = np.zeros(7)
    far = np.full(7, 0.05)
    reported = None
    for _ in range(reflex.CONSECUTIVE_SAMPLES):
        reported = watcher.observe(planned, far, 0.5, FakeKinematics())
    assert reported is not None
    assert reported['reason'] == 'MOTION_DEVIATION'
    assert reported['deviation_m'] == pytest.approx(0.05 * np.sqrt(7))
    assert reported['threshold_m'] == pytest.approx(0.02)
    assert reported['phase'] == 'approach'


def test_the_start_of_a_leg_is_not_watched():
    watcher = reflex.DeviationReflex('left', 'approach', threshold_m=0.02)
    for _ in range(reflex.CONSECUTIVE_SAMPLES + 2):
        assert watcher.observe(np.zeros(7), np.full(7, 0.05), 0.05, FakeKinematics()) is None


def test_a_leg_within_threshold_reports_a_summary_rather_than_a_violation():
    watcher = reflex.DeviationReflex('right', 'carry', threshold_m=0.05)
    for _ in range(4):
        assert watcher.observe(np.zeros(7), np.full(7, 0.001), 0.6, FakeKinematics()) is None
    summary = watcher.evidence()
    assert summary['samples'] == 4 and summary['worst_m'] < 0.05 and summary['acted'] is False


def test_the_reflex_can_be_switched_off(monkeypatch):
    monkeypatch.setenv('OPENARM_DEVIATION_REFLEX', 'off')
    assert reflex.enabled() is False
    monkeypatch.setenv('OPENARM_DEVIATION_REFLEX', '1')
    assert reflex.enabled() is True


def test_the_threshold_can_be_forced_so_the_guard_can_be_observed_failing(monkeypatch):
    monkeypatch.setenv('OPENARM_DEVIATION_REFLEX_MM', '3')
    assert reflex.threshold_m('left', 'park_arrival') == pytest.approx(0.003)


def test_the_applied_threshold_is_the_forced_one(monkeypatch):
    monkeypatch.setenv('OPENARM_DEVIATION_REFLEX_MM', '2')
    watcher = reflex.DeviationReflex('left', 'approach')
    assert watcher.threshold_m == pytest.approx(0.002)


def test_a_spike_is_distinguished_from_a_hold():
    """A leg that reaches far but never holds it must not look like a leg that drifted steadily."""
    watcher = reflex.DeviationReflex('left', 'park_arrival', threshold_m=0.02)
    watcher.observe(np.zeros(7), np.full(7, 0.05), 0.5, FakeKinematics())
    watcher.observe(np.zeros(7), np.zeros(7), 0.6, FakeKinematics())
    watcher.observe(np.zeros(7), np.full(7, 0.05), 0.7, FakeKinematics())
    summary = watcher.evidence()
    assert summary['max_run'] == 1
    assert summary['worst_fraction'] == pytest.approx(0.5), 'the first sample at the worst depth'
    assert summary['acted'] is False
