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
    path = [np.zeros(7)]
    far = np.full(7, 0.05)
    assert watcher.observe(path, far, 0.5, FakeKinematics()) is None, 'one sample is a transient'
    assert watcher.observe(path, far, 0.5, FakeKinematics()) is None


def test_a_persistent_departure_is_reported_with_its_numbers():
    watcher = reflex.DeviationReflex('left', 'approach', threshold_m=0.02)
    path = [np.zeros(7)]
    far = np.full(7, 0.05)
    reported = None
    for _ in range(reflex.CONSECUTIVE_SAMPLES):
        reported = watcher.observe(path, far, 0.5, FakeKinematics())
    assert reported is not None
    assert reported['reason'] == 'MOTION_DEVIATION'
    assert reported['deviation_m'] == pytest.approx(0.05 * np.sqrt(7))
    assert reported['threshold_m'] == pytest.approx(0.02)
    assert reported['phase'] == 'approach'


def test_the_start_of_a_leg_is_not_watched():
    watcher = reflex.DeviationReflex('left', 'approach', threshold_m=0.02)
    for _ in range(reflex.CONSECUTIVE_SAMPLES + 2):
        assert watcher.observe([np.zeros(7)], np.full(7, 0.05), 0.05, FakeKinematics()) is None


def test_a_leg_within_threshold_reports_a_summary_rather_than_a_violation():
    watcher = reflex.DeviationReflex('right', 'carry', threshold_m=0.05)
    for _ in range(4):
        assert watcher.observe([np.zeros(7)], np.full(7, 0.001), 0.6, FakeKinematics()) is None
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
    watcher.observe([np.zeros(7)], np.full(7, 0.05), 0.5, FakeKinematics())
    watcher.observe([np.zeros(7)], np.zeros(7), 0.6, FakeKinematics())
    watcher.observe([np.zeros(7)], np.full(7, 0.05), 0.7, FakeKinematics())
    summary = watcher.evidence()
    assert summary['max_run'] == 1
    assert summary['worst_fraction'] == pytest.approx(0.5), 'the first sample at the worst depth'
    assert summary['acted'] is False


def test_a_short_leg_cannot_be_required_to_hold_three_samples():
    """The park legs produce two samples; demanding three made the reflex blind where drift was worst."""
    short = reflex.DeviationReflex('left', 'park_arrival', threshold_m=0.02, expected_samples=2)
    assert short.required == 2
    long = reflex.DeviationReflex('left', 'park_arrival', threshold_m=0.02, expected_samples=40)
    assert long.required == reflex.CONSECUTIVE_SAMPLES


def test_two_samples_over_threshold_act_on_a_two_sample_leg():
    watcher = reflex.DeviationReflex('left', 'park_arrival', threshold_m=0.02, expected_samples=2)
    path, far = [np.zeros(7)], np.full(7, 0.05)
    assert watcher.observe(path, far, 0.5, FakeKinematics()) is None
    reported = watcher.observe(path, far, 0.7, FakeKinematics())
    assert reported is not None and reported['reason'] == 'MOTION_DEVIATION'


def test_the_required_run_matches_the_samples_the_leg_will_produce():
    """A park leg yields two samples; the requirement must be two, not three."""
    watcher = reflex.DeviationReflex('left', 'park_arrival', threshold_m=0.02, expected_samples=2)
    path, far = [np.zeros(7)], np.full(7, 0.05)
    first = watcher.observe(path, far, 0.4, FakeKinematics())
    second = watcher.observe(path, far, 0.9, FakeKinematics())
    assert first is None and second is not None


def test_the_reorientation_tries_both_arcs_and_the_direct_move():
    """A wrist limit on one arc must not refuse a reorientation the other way round can make."""
    import inspect
    from backend.app.skills import robot_skills

    source = inspect.getsource(robot_skills.SkillRunner._align_for_placement)
    assert "'short_arc'" in source and "'long_arc'" in source and "'direct'" in source


def test_being_late_on_the_path_is_not_drift():
    """A fast large move that trails its setpoint is still on the path, and must not be stopped for it."""
    watcher = reflex.DeviationReflex('left', 'move_to_site', threshold_m=0.02, expected_samples=6)
    # The path runs from 0 to 1 in joint 0; the arm is at 0.4 the whole time, which is *on* the path.
    path = [np.array([value] + [0.] * 6) for value in np.arange(0., 1. + 4e-3, 4e-3)]
    actual = np.array([0.4] + [0.] * 6)
    for _ in range(6):
        assert watcher.observe(path, actual, 0.9, FakeKinematics()) is None
    assert watcher.worst_m < reflex.PATH_SPACING_M, watcher.worst_m


def test_the_receiver_standoff_is_searched_from_comfortable_to_close():
    """The standoff was a constant, and a receiver at the edge of its reach could not solve there."""
    from backend.app.skills.robot_skills import RECEIVER_STANDOFF_CANDIDATES_M

    assert RECEIVER_STANDOFF_CANDIDATES_M[0] == 0.12, 'the comfortable distance is still tried first'
    assert list(RECEIVER_STANDOFF_CANDIDATES_M) == sorted(RECEIVER_STANDOFF_CANDIDATES_M, reverse=True)
    assert all(0. < distance < 0.12 for distance in RECEIVER_STANDOFF_CANDIDATES_M[1:])


def test_an_axial_slide_is_allowed_and_a_lateral_shove_is_not():
    """An offset grasp slides one pad along the face; a shove across the face is still a collision."""
    from backend.app.skills import robot_skills

    assert robot_skills.APPROACH_SLIDE_TOLERANCE > robot_skills.APPROACH_DISTURBANCE_TOLERANCE
    source = inspect_source(robot_skills.SkillRunner._approach)
    assert 'lateral > APPROACH_DISTURBANCE_TOLERANCE' in source
    assert 'axial > APPROACH_SLIDE_TOLERANCE' in source


def inspect_source(function):
    import inspect
    return inspect.getsource(function)
