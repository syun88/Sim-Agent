"""A safeguard can be switched off, so "it helped" can be measured instead of assumed."""
import numpy as np

from backend.app.planning import measurement_flags as flags


def test_defaults_follow_the_measurement(monkeypatch):
    """Harmful or insufficiently validated refusals are off by default."""
    for name in flags.FLAGS:
        monkeypatch.delenv(f'OPENARM_MEASURE_{name.upper()}', raising=False)
    state = flags.describe()
    assert state['arm_preflight'] is False, 'measured harmful: it refused tasks the robot could do'
    assert state['requested_hand_preflight'] is False
    assert all(value for name, value in state.items()
               if name not in {'arm_preflight', 'requested_hand_preflight'})


def test_a_flag_can_be_switched_off(monkeypatch):
    monkeypatch.delenv('OPENARM_MEASURE_APPROACH_MARGIN', raising=False)
    monkeypatch.setenv('OPENARM_MEASURE_DESTINATION_SEARCH', '0')
    assert flags.enabled('destination_search') is False
    assert flags.enabled('approach_margin') is True, 'flags are independent'


def test_an_unknown_flag_is_an_error_not_a_silent_default():
    try:
        flags.enabled('invented')
    except KeyError as error:
        assert 'unknown measurement flag' in str(error)
    else:
        raise AssertionError('an unknown flag name must not resolve')


def test_the_approach_screen_drops_the_measured_error_when_switched_off(monkeypatch):
    from types import SimpleNamespace
    from backend.app.skills import approach, motion_accuracy

    motion_accuracy.reset()
    seen = {}

    def check(route, objects, **kwargs):
        seen['clearance'] = kwargs.get('minimum_object_clearance')
        return True

    from backend.app.simulation.openarm_v2.definition import GRASP_ROTATION
    pose = {'point': np.array([0.45, 0.0, 1.05])}
    kin = SimpleNamespace(evaluate=lambda q: (pose['point'].copy(), GRASP_ROTATION.copy(), None, None),
                          empty_arm_path_is_clear=check)
    env = SimpleNamespace(arm_qpos=lambda a: np.zeros(7), home_arm_qpos=lambda a: np.zeros(7),
                          ee_pose=lambda a: (np.array([0.40, 0.0, 1.20]), np.eye(3)),
                          arm_kinematics=lambda a: kin, arm_velocity_limits=lambda a: None,
                          object_pose=lambda n: (np.array([0.45, 0.0, 1.05]), np.eye(3)),
                          frame=lambda: SimpleNamespace(objects={'blue_cube': np.array([0.45, 0.0, 1.05])}))

    def solve_ik(arm, target, rotation, **kwargs):
        pose['point'] = np.asarray(target, float).copy()
        return np.zeros(7), 0.001, 0.0

    controller = SimpleNamespace(solve_ik=solve_ik)
    approach.plan_cube_approach(env, controller, 'left', 'blue_cube', np.array([0.45, 0., 1.05]),
                                np.array([0.45, 0., 1.11]), .10, 1.24)
    with_measured = seen['clearance']
    monkeypatch.setenv('OPENARM_MEASURE_APPROACH_MARGIN', '0')
    approach.plan_cube_approach(env, controller, 'left', 'blue_cube', np.array([0.45, 0., 1.05]),
                                np.array([0.45, 0., 1.11]), .10, 1.24)
    assert seen['clearance'] == approach.NONCONTACT_CLEARANCE_M
    assert with_measured > approach.NONCONTACT_CLEARANCE_M
