"""A carry is judged by what the placement needs, not only by the controller's own tolerance."""
from types import SimpleNamespace

import numpy as np
import pytest

from backend.app.skills.robot_skills import SkillRunner

DESIRED = np.array([0.48, 0.18, 1.14])


def _runner(*, achieved_offset, controller_success):
    """An arm that lands `achieved_offset` away from the requested pose."""
    object_position = DESIRED + np.array([achieved_offset, 0., 0.])

    def move(arm, target, rotation, **kwargs):
        return SimpleNamespace(success=controller_success, error=None,
                               position_error=abs(achieved_offset), rotation_error=0.)

    env = SimpleNamespace(
        object_pose=lambda name: (object_position.copy(), np.eye(3)),
        ee_pose=lambda arm: (object_position + [0.05, 0., -0.12], np.eye(3)),
        frame=lambda: SimpleNamespace(objects={'red_cube': object_position.copy()}),
        object_contacts=lambda *a, **k: ['finger'], arm_qpos=lambda arm: np.zeros(7))
    skills = SkillRunner.__new__(SkillRunner)
    skills.env = env
    # motion_speed != 1 skips the Cartesian waypoint pre-pass and goes straight to the judged carry.
    skills.controller = SimpleNamespace(move_to_site=move, motion_speed=6., target_rotations={})
    skills._require_hold = lambda *a, **k: None
    skills._motion_evidence = lambda result, **k: {'phase': k.get('phase')}
    return skills


def test_a_short_carry_inside_the_task_tolerance_is_a_placement():
    """Measured 2026-11-09: the arm stopped 27 mm short of a zone centre; the zone allows 35 mm."""
    skills = _runner(achieved_offset=0.027, controller_success=False)
    result = skills._carry_to('left', 'red_cube', DESIRED, allow_error_m=0.035)
    assert result['accepted_within_task_tolerance_m'] == pytest.approx(0.035)
    assert result['position_error'] == pytest.approx(0.027)


def test_the_same_short_carry_without_a_task_tolerance_still_fails():
    skills = _runner(achieved_offset=0.027, controller_success=False)
    try:
        skills._carry_to('left', 'red_cube', DESIRED)
    except Exception as error:
        assert error.category in {'IK_FAILURE', 'MOTION_BLOCKED'}
        assert error.evidence['achieved_offset_m'] == pytest.approx(0.027)
    else:
        raise AssertionError('a 27 mm shortfall must not pass silently when the task asks for precision')


def test_a_shortfall_beyond_the_task_tolerance_is_still_a_failure():
    skills = _runner(achieved_offset=0.040, controller_success=False)
    try:
        skills._carry_to('left', 'red_cube', DESIRED, allow_error_m=0.035)
    except Exception as error:
        assert error.evidence['task_tolerance_m'] == pytest.approx(0.035)
        assert error.evidence['achieved_offset_m'] == pytest.approx(0.040)
    else:
        raise AssertionError('40 mm does not fit a 35 mm region')


def test_the_carry_records_where_the_object_went_not_only_where_it_stopped():
    """A final residual cannot tell "stopped 27 mm short" from "never tracked the command"."""
    skills = _runner(achieved_offset=0.027, controller_success=False)
    # Two guard samples: the object approaches, then stops moving while still short.
    positions = [[0.40, 0.18, 1.14], [0.453, 0.18, 1.14]]
    state = {'i': 0}

    def object_pose(name):
        return np.asarray(positions[min(state['i'], len(positions) - 1)], float), np.eye(3)

    def move(arm, target, rotation, **kwargs):
        state['i'] += 1
        if 'on_sample' in kwargs:
            kwargs['on_sample']()
        return SimpleNamespace(success=False, error=None, position_error=0.027, rotation_error=0.)
    skills.env.object_pose = object_pose
    skills.env.frame = lambda: SimpleNamespace(objects={'red_cube': np.asarray(positions[-1], float)})
    skills.controller.move_to_site = move
    try:
        skills._carry_to('left', 'red_cube', DESIRED)
    except Exception as error:
        trace = error.evidence['carry_trace']
        assert trace['samples'] >= 1
        assert trace['final_m'] == pytest.approx(0.027, abs=.002)
        assert trace['movement_at_end_m'] is not None
    else:
        raise AssertionError('the carry must fail without a task tolerance')
