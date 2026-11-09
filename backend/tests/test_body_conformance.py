"""A backend is only usable if it answers the operations the skills drive it with, as documented."""
from types import SimpleNamespace

import numpy as np

from backend.app.simulation import conformance


class _Kinematics:
    def evaluate(self, q):
        return np.zeros(3), np.eye(3), None, None


def _body(**overrides):
    """A minimal conforming body; overrides break one aspect at a time."""
    frame = SimpleNamespace(objects={'red_cube': [0.4, 0.1, 1.03]}, zones={'blue_zone': [0.4, 0.2, 1.0]},
                            ee={'left': [0.3, 0.0, 1.2]}, grippers={'left': .02}, contacts=[], sim_time=0.)
    fields = {
        'frame': lambda: frame,
        'object_pose': lambda name: (np.array([0.4, 0.1, 1.03]), np.eye(3)),
        'object_bounds': lambda name: np.array([[0.38, 0.08, 1.01], [0.42, 0.12, 1.05]]),
        'object_contacts': lambda name, arm: [],
        'object_linear_speed': lambda name: 0.0,
        'ee_pose': lambda arm: (np.array([0.3, 0.0, 1.2]), np.eye(3)),
        'arm_qpos': lambda arm: np.zeros(7),
        'home_arm_qpos': lambda arm: np.zeros(7),
        'arm_limits': lambda arm: np.array([[-3., 3.]] * 7),
        'arm_kinematics': lambda arm: _Kinematics(),
        'step': lambda n: None,
    }
    fields.update(overrides)
    body = SimpleNamespace(**fields)
    if 'step' not in overrides:
        def step(n):
            frame.sim_time += n * .002
        body.step = step
    return body


def test_a_conforming_body_reports_no_problems():
    assert conformance.check(_body()) == []


def test_a_missing_operation_is_named_with_what_it_had_to_return():
    problems = conformance.check(_body(ee_pose=None))
    assert any(item['operation'] == 'ee_pose' and item['problem'] == 'missing' for item in problems)


def test_a_wrong_shape_is_caught_not_rounded_off():
    problems = conformance.check(_body(object_pose=lambda name: (np.zeros(2), np.eye(3))))
    assert any(item['operation'] == 'object_pose' and item['problem'] == 'wrong_shape' for item in problems)


def test_a_body_whose_step_does_nothing_cannot_report_motion_it_did_not_make():
    problems = conformance.check(_body(step=lambda n: None))
    assert any(item['operation'] == 'step' and item['problem'] == 'did_not_advance' for item in problems)


def test_forward_kinematics_has_to_evaluate_a_configuration():
    broken = SimpleNamespace(evaluate=None)
    problems = conformance.check(_body(arm_kinematics=lambda arm: broken))
    assert any(item['operation'] == 'arm_kinematics' and item['problem'] == 'no_evaluate' for item in problems)


def test_the_contract_can_be_read_without_a_body():
    rows = conformance.describe()
    operations = {row['operation'] for row in rows}
    assert {'frame', 'object_pose', 'arm_kinematics', 'step'} <= operations
    assert all(row.get('kind') in {'read_only', 'command'} for row in rows)
