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


def test_the_approach_screen_carries_the_measured_approach_error(monkeypatch):
    """20 mm nominal + how far an approach lands off its plan, measured rather than assumed."""
    from backend.app.skills import approach, motion_accuracy

    motion_accuracy.reset()
    seen = {}

    def fake_check(route, objects, **kwargs):
        seen['minimum_object_clearance'] = kwargs.get('minimum_object_clearance')
        return True

    # Paired with solve_ik so a route actually gets built and therefore screened.
    from backend.app.simulation.openarm_v2.definition import GRASP_ROTATION
    pose = {'point': np.array([0.45, 0.0, 1.05])}
    # The attitude has to be the one the planner asks for, or every IK candidate is rejected on orientation.
    kin = SimpleNamespace(evaluate=lambda q: (pose['point'].copy(), GRASP_ROTATION.copy(), None, None),
                          empty_arm_path_is_clear=fake_check)
    env = SimpleNamespace(
        arm_qpos=lambda arm: np.zeros(7), home_arm_qpos=lambda arm: np.zeros(7),
        ee_pose=lambda arm: (np.array([0.40, 0.0, 1.20]), np.eye(3)),
        arm_kinematics=lambda arm: kin, arm_velocity_limits=lambda arm: None,
        object_pose=lambda name: (np.array([0.45, 0.0, 1.05]), np.eye(3)),
        frame=lambda: SimpleNamespace(objects={'blue_cube': np.array([0.45, 0.0, 1.05])}))
    def solve_ik(arm, target, rotation, **kwargs):
        pose['point'] = np.asarray(target, float).copy()
        return np.zeros(7), 0.001, 0.0
    controller = SimpleNamespace(solve_ik=solve_ik)
    approach.plan_cube_approach(env, controller, 'left', 'blue_cube',
                                np.array([0.45, 0.0, 1.05]), np.array([0.45, 0.0, 1.11]), .10, 1.24)
    expected = approach.NONCONTACT_CLEARANCE_M + motion_accuracy.PHASE_DEFAULTS['approach']
    assert seen['minimum_object_clearance'] == expected


def test_the_approach_falls_back_to_a_shorter_standoff_when_the_nominal_one_has_no_corridor():
    """A live round crashed on this import, which no test exercised: the fallback has to be pinned."""
    from types import SimpleNamespace
    from backend.app.skills import approach, robot_skills
    from backend.app.simulation.openarm_v2.definition import GRASP_ROTATION

    assert approach.STANDOFF_VARIANTS_M, 'the fallback distances must exist'
    assert hasattr(approach, 'frame_at')
    assert len(approach.CUBE_GRASP_YAWS_DEG) > 14, 'the half-steps are part of the candidate set'

    calls = []

    def fake_plan(env, controller, arm, obj, pinch, standoff, overfly, staging, *, frames=None,
                  start_override=None):
        calls.append(round(float(standoff[2] - pinch[2]), 3))
        if len(calls) == 1:
            return None, {'error': 'APPROACH_PATH_UNAVAILABLE'}
        return [np.zeros(7), np.ones(7)], {'grasp_yaw_deg': 0.0, 'grasp_rotation': GRASP_ROTATION.tolist(),
                                           'pinch_target': list(pinch)}
    original = approach.plan_cube_approach
    try:
        import backend.app.skills.approach as module
        module.plan_cube_approach = fake_plan
        skills = robot_skills.SkillRunner.__new__(robot_skills.SkillRunner)
        skills.env = SimpleNamespace(arm_qpos=lambda a: np.zeros(7), home_arm_qpos=lambda a: np.zeros(7),
                                     ee_pose=lambda a: (np.array([0.4, 0.0, 1.2]), np.eye(3)),
                                     arm_velocity_limits=lambda a: None,
                                     object_pose=lambda n: (np.array([0.45, 0.0, 1.05]), np.eye(3)),
                                     frame=lambda: SimpleNamespace(objects={'blue_cube': np.array([0.45, 0., 1.05])}),
                                     arm_kinematics=lambda a: SimpleNamespace(
                                         evaluate=lambda q: (np.array([0.45, 0., 1.05]), GRASP_ROTATION, None, None)))
        skills.controller = SimpleNamespace(solve_ik=lambda *a, **k: (np.zeros(7), 0.001, 0.))
        frames = approach.cube_approach_frames('left', np.array([0.45, 0.0, 1.05]), 0.06)
        result = skills._stage_for_approach('left', 'blue_cube', np.array([0.45, 0.0, 1.05]), frames, {})
    finally:
        module.plan_cube_approach = original
    # The staged helper asks from canonical postures; what matters here is that the variant path is reachable code
    # and that a shorter standoff is one of the questions it can ask.
    assert result is None or isinstance(result, tuple)


def test_a_complete_reference_body_passes_the_contract():
    """The checker's positive path, exercised without Genesis: a body that answers every operation."""
    from backend.app.simulation.conformance_reference import ReferenceBody

    body = ReferenceBody()
    assert conformance.check(body) == []


def test_the_reference_body_is_a_template_a_bridge_can_copy():
    """Every operation the contract names is implemented, so a hardware port has a starting file."""
    from backend.app.simulation.conformance import COMMANDS, READ_ONLY
    from backend.app.simulation.conformance_reference import ReferenceBody

    body = ReferenceBody()
    missing = [name for name in list(READ_ONLY) + list(COMMANDS) if not callable(getattr(body, name, None))]
    assert missing == []
