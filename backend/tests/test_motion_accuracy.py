"""A target is only commanded when its clearance beats this body's measured positioning error."""
import numpy as np

from backend.app.skills import motion_accuracy
from backend.app.skills.homing import plan_empty_arm_home


def test_the_required_clearance_comes_from_the_measured_error():
    motion_accuracy.reset()
    before = motion_accuracy.required_clearance('left')
    motion_accuracy.record('left', 0.05)
    after = motion_accuracy.required_clearance('left')
    assert after > before
    assert after >= 0.05, 'a target must be clearer than the error it has to absorb'
    assert after <= 0.12, 'a body that lands badly everywhere does not make home unreachable by definition'


def test_without_recordings_the_measured_default_is_used_not_an_optimistic_zero():
    motion_accuracy.reset()
    assert motion_accuracy.observed_error('left') == motion_accuracy.DEFAULT_OBSERVED_ERROR_M
    assert motion_accuracy.required_clearance('left') > motion_accuracy.DEFAULT_OBSERVED_ERROR_M


class _Kin:
    """Home is 'clear' only if the caller demands less than `actual_gap`."""

    def __init__(self, actual_gap):
        self.actual_gap = actual_gap
        self.thresholds = []
        # Paired with solve_ik: the IK that just "solved" is the pose this arm reports, so a candidate
        # route is accepted on its own terms and the test is about the clearance threshold, not IK.
        self.pose = np.array([0.5, 0.0, 1.2])

    def evaluate(self, q):
        return self.pose.copy(), np.eye(3), None, None

    def empty_arm_path_is_clear(self, route, objects, *, minimum_object_clearance=0., **kwargs):
        self.thresholds.append(minimum_object_clearance)
        return minimum_object_clearance <= self.actual_gap

    def path_minimum_clearance(self, *args, **kwargs):
        return self.actual_gap


def _plan(kin):
    from types import SimpleNamespace
    env = SimpleNamespace(
        arm_kinematics=lambda arm: kin,
        arm_qpos=lambda arm: np.zeros(7), home_arm_qpos=lambda arm: np.zeros(7),
        ee_pose=lambda arm: (np.array([0.5, 0.0, 1.2]), np.eye(3)),
        object_pose=lambda name: (np.array([0.5, 0.0, 1.05]), np.eye(3)),
        object_bounds=lambda name: np.array([[0.48, -0.02, 1.03], [0.52, 0.02, 1.07]]),
        frame=lambda: SimpleNamespace(objects={'red_cube': np.array([0.5, 0.0, 1.05])}),
        arm_limits=lambda arm: np.array([[-3., 3.]] * 7))
    def solve_ik(arm, target, rotation, **kwargs):
        kin.pose = np.asarray(target, float).copy()
        return np.zeros(7), 0.001, 0.0
    controller = SimpleNamespace(solve_ik=solve_ik, stop_requested=lambda: False)
    return env, controller


def test_a_cube_at_the_old_30mm_margin_no_longer_counts_as_clear_home():
    """Measured 2026-11-09: 30.8 mm was called clear and the arm still brushed the cube."""
    motion_accuracy.reset()
    kin = _Kin(actual_gap=0.0308)
    env, controller = _plan(kin)
    route, evidence = plan_empty_arm_home(env, controller, 'left')
    assert route is not None, evidence
    assert evidence['home_mode'] == 'scene_clearance_home'
    assert evidence['home_clearance_required_m'] > 0.0308
    # The arm parks above the home frame instead of descending into the cube's neighbourhood.
    assert 'lift_then_home' in {item['route'] for item in evidence['considered']}


def test_a_home_that_is_genuinely_clear_is_still_used():
    motion_accuracy.reset()
    kin = _Kin(actual_gap=0.20)
    env, controller = _plan(kin)
    route, evidence = plan_empty_arm_home(env, controller, 'left')
    assert route is not None, evidence
    assert evidence['home_mode'] == 'model_home'


def test_a_recorded_larger_error_raises_the_bar_for_the_next_attempt():
    motion_accuracy.reset()
    env, controller = _plan(_Kin(actual_gap=0.0308))
    first, _ = plan_empty_arm_home(env, controller, 'left')
    motion_accuracy.record('left', 0.09)
    second_kin = _Kin(actual_gap=0.055)
    env, controller = _plan(second_kin)
    route, evidence = plan_empty_arm_home(env, controller, 'left')
    assert evidence['home_clearance_required_m'] >= 0.09
    assert evidence['home_mode'] == 'scene_clearance_home'
