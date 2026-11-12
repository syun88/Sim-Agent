"""Preflight answers the geometrical questions before motion, and admits when it cannot."""
from types import SimpleNamespace

import numpy as np

from backend.app.planning import preflight
from backend.app.planning.task_program import TaskProgram

PLAN = TaskProgram(actions=[{'op': 'place', 'object': 'red_cube', 'arm': 'left',
                             'target': 'blue_cube', 'relation': 'on'}])


def test_a_plan_with_no_relative_placement_asks_nothing():
    plan = TaskProgram(actions=[{'op': 'place', 'object': 'red_cube', 'arm': 'left', 'target': 'left_side'}])
    answer = preflight.run_preflight(None, None, plan.model_dump())
    assert answer['available'] is True and answer['questions'] == []


def test_without_an_environment_it_reports_unavailable_rather_than_clear():
    answer = preflight.run_preflight(None, None, PLAN.model_dump())
    assert answer['available'] is False and 'no environment' in answer['reason']


def test_it_asks_pad_and_reachability_questions_without_moving(monkeypatch):
    monkeypatch.setattr(preflight, 'pad_blocks_support_face',
                        lambda env, arm, support: {'blocks': True, 'clearance_m': -0.0405})
    monkeypatch.setattr(preflight, 'reachable',
                        lambda env, controller, arm, target, yaw=0.: {'ok': yaw == 90., 'yaw_deg': yaw})
    env = SimpleNamespace(object_pose=lambda name: (np.array([0.45, 0.0, 1.05]), np.eye(3)),
                          frame=lambda: SimpleNamespace(objects={'blue_cube': np.zeros(3)}))
    answer = preflight.run_preflight(env, SimpleNamespace(), PLAN.model_dump())
    row = answer['questions'][0]
    assert row['arm'] == 'left' and row['target'] == 'blue_cube'
    assert row['pad_blocks_face'] is True and row['pad_clearance_m'] == -0.0405
    assert row['reachable_yaws'] == 1 and row['reachable_asked'] == 4


def test_an_unmeasurable_pad_clearance_never_returns_infinity():
    """Measured 2026-11-08: `inf` in the evidence aborted 11 of 24 live tasks at JSON serialisation."""
    import json
    from backend.app.skills.pad_clearance import support_face_clearance
    from backend.app.simulation.openarm_v2.spawn import private_spawn_environment

    env = private_spawn_environment()
    env.arm_kinematics = lambda side: (_ for _ in ()).throw(RuntimeError('no model'))
    measured = support_face_clearance(env, 'right', np.zeros(3), 0.026, 1.05)
    assert measured['clearance_m'] is None
    json.dumps(measured, allow_nan=False)


def test_preflight_sanitises_a_non_finite_measurement(monkeypatch):
    import json
    monkeypatch.setattr(preflight, 'pad_blocks_support_face',
                        lambda env, arm, support: {'blocks': None, 'clearance_m': float('inf')})
    monkeypatch.setattr(preflight, 'reachable', lambda *a, **k: {'ok': True, 'position_error_m': float('nan')})
    env = SimpleNamespace(object_pose=lambda name: (np.zeros(3), np.eye(3)),
                          frame=lambda: SimpleNamespace(objects={'blue_cube': np.zeros(3)}))
    answer = preflight.run_preflight(env, SimpleNamespace(), PLAN.model_dump())
    row = answer['questions'][0]
    assert row['pad_clearance_m'] is None
    json.dumps(answer, allow_nan=False)
