"""A target is only commanded when its clearance beats this body's measured positioning error."""
import numpy as np

from backend.app.skills import motion_accuracy
from backend.app.skills.homing import plan_empty_arm_home


def test_the_required_clearance_comes_from_the_measured_error():
    motion_accuracy.reset()
    before = motion_accuracy.required_clearance('left')
    motion_accuracy.record('left', 0.05, source='move_to_site')
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
        # A linear joint-to-pose map, so a route can actually be solved through a high waypoint: the test is
        # about which clearance decides the home target, not about IK.
        self.pose = np.array([0.5, 0.0, 1.2])

    def evaluate(self, q):
        q = np.asarray(q, float)
        return np.array([0.40 + q[0] * 0.1, q[1] * 0.1, 1.00 + q[2] * 0.1]), np.eye(3), None, None

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
        target = np.asarray(target, float)
        return np.array([(target[0] - 0.40) / 0.1, target[1] / 0.1, (target[2] - 1.00) / 0.1,
                         0., 0., 0., 0.]), 0.001, 0.0
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
    # The arm parks above a frame instead of descending into the cube's neighbourhood; which frame is a
    # measured choice now, so the requirement is that the route ends above rather than in a posture.
    assert any('park_above' in item['route'] for item in evidence['considered'])
    assert evidence.get('park_target') is None, 'a park-above route does not claim a posture'


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
    motion_accuracy.record('left', 0.09, source='move_to_site')
    second_kin = _Kin(actual_gap=0.055)
    env, controller = _plan(second_kin)
    route, evidence = plan_empty_arm_home(env, controller, 'left')
    assert evidence['home_clearance_required_m'] >= 0.09
    assert evidence['home_mode'] == 'scene_clearance_home'


def test_a_move_that_stopped_in_contact_does_not_raise_the_home_bar():
    """A grasp legitimately stops short. That is contact, not the free-space error a home target absorbs."""
    motion_accuracy.reset()
    baseline = motion_accuracy.required_clearance('left')
    motion_accuracy.record('left', 0.10, source='move_blocked')
    assert motion_accuracy.required_clearance('left') == baseline


def test_the_controller_records_every_landing_with_its_source():
    from types import SimpleNamespace
    from backend.app.control.motion import OpenArmV2Controller

    motion_accuracy.reset()
    env = SimpleNamespace(
        arm_qpos=lambda side: np.zeros(7), ee_pose=lambda side: (np.array([0.5, 0.0, 1.2]), np.eye(3)),
        arm_kinematics=lambda side: SimpleNamespace(evaluate=lambda q: (np.array([0.5, 0.0, 1.2]), np.eye(3),
                                                                        None, None)),
        arm_limits=lambda side: np.array([[-3., 3.]] * 7),
        configure_motion_response=lambda *a, **k: None, timestep=.002, set_arm=lambda *a, **k: None,
        step=lambda *a, **k: None, arm_velocity_limits=lambda side: None,
        motion_position_tolerance=.012, motion_rotation_tolerance=.15)
    controller = OpenArmV2Controller(env, configure_motors=False)
    controller.move_to_site('left', np.array([0.5, 0.0, 1.2]), np.eye(3))
    recorded = motion_accuracy.samples('left')
    assert recorded and recorded[-1]['source'] in {'move_to_site', 'move_blocked'}


def test_ordinary_moves_never_lower_the_bar_below_the_measured_baseline():
    """Free-space moves land within ~12 mm; that must not make the near-object 35 mm case 'clear' again."""
    motion_accuracy.reset()
    baseline = motion_accuracy.required_clearance('left')
    motion_accuracy.record('left', 0.012, source='move_to_site')
    assert motion_accuracy.required_clearance('left') == baseline
    motion_accuracy.record('left', 0.05, source='move_to_site')
    assert motion_accuracy.required_clearance('left') > baseline


def test_one_blocked_carry_does_not_disable_every_later_placement():
    """A 50 mm blocked carry is an outlier, not the body's usual accuracy."""
    motion_accuracy.reset()
    for value in (0.006, 0.008, 0.010):
        motion_accuracy.record('left', value, source='carry')
    motion_accuracy.record('left', 0.050, source='carry')
    assert motion_accuracy.expected_error('left', 'carry') == motion_accuracy.PHASE_DEFAULTS['carry']


def test_one_recording_is_not_an_estimate():
    motion_accuracy.reset()
    motion_accuracy.record('left', 0.080, source='carry')
    assert motion_accuracy.expected_error('left', 'carry') == motion_accuracy.PHASE_DEFAULTS['carry']


def test_the_phase_expectation_follows_a_consistently_worse_body():
    motion_accuracy.reset()
    for value in (0.020, 0.022, 0.024):
        motion_accuracy.record('left', value, source='carry')
    assert motion_accuracy.expected_error('left', 'carry') == 0.022
