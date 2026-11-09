"""The approach transit is driven slowly while the servo keeps the response it was configured with."""
from types import SimpleNamespace

import numpy as np
import pytest

from backend.app.skills import robot_skills
from backend.app.skills.robot_skills import APPROACH_TRAJECTORY_SPEED, SkillRunner


def _grasping_runner():
    from backend.app.simulation.openarm_v2.definition import GRASP_ROTATION, FINGERTIP_MIDPOINT
    cube = np.array([0.45, 0.0, 1.09])
    calls = []

    def move(arm, target, rotation=None, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(success=True, final_position=list(target))

    env = SimpleNamespace(
        object_pose=lambda name: (cube.copy(), np.eye(3)),
        object_bounds=lambda name: np.array([[.42, -.03, 1.06], [.48, .03, 1.12]]),
        frame=lambda: SimpleNamespace(objects={'blue_cube': cube}),
        ee_pose=lambda arm: (np.array([0.30, 0.0, 1.15]), np.eye(3)),
        arm_qpos=lambda arm: np.zeros(7), home_arm_qpos=lambda arm: np.zeros(7),
        arm_velocity_limits=lambda arm: None,
        arm_kinematics=lambda arm: SimpleNamespace(evaluate=lambda q: (np.array([0.45, 0.0, 1.09]), GRASP_ROTATION, None, None)),
        pinch_established=lambda obj, arm: False,
        object_contacts=lambda name, arm=None: [],
        step=lambda n: None, timestep=0.002, grasp_mode='physical-friction-pinch',
        gripper_position=lambda arm: 0.02, gripper_target=lambda arm, closed: 0.02,
        set_gripper=lambda arm, value: None)

    def admitted(*args, **kwargs):
        return [np.zeros(7), np.ones(7)], {'grasp_yaw_deg': 0, 'grasp_rotation': GRASP_ROTATION.tolist(),
                                           'pinch_target': (cube - GRASP_ROTATION @ FINGERTIP_MIDPOINT['right']).tolist()}
    return env, move, calls, admitted


def test_the_cube_approach_transit_is_driven_below_full_speed(monkeypatch):
    env, move, calls, admitted = _grasping_runner()
    monkeypatch.setattr('backend.app.skills.approach.plan_cube_approach', admitted)
    skills = SkillRunner(env, SimpleNamespace(motion_speed=6., target_rotations={}, move_to_site=move,
                                              wait_gripper=lambda *a: None))
    result = skills.approach_object('right', 'blue_cube')
    assert result['success'], result
    assert calls and calls[0]['trajectory_speed'] == APPROACH_TRAJECTORY_SPEED
    assert APPROACH_TRAJECTORY_SPEED < 1., 'the whole point is a slower transit than the carry'


def test_a_negative_trajectory_speed_is_refused():
    from backend.app.control.motion import OpenArmV2Controller
    controller = OpenArmV2Controller.__new__(OpenArmV2Controller)
    controller.motion_speed = 6.
    controller.env = SimpleNamespace(configure_motion_response=lambda s: None, arm_qpos=lambda side: np.zeros(7))
    with pytest.raises(ValueError, match='positive'):
        controller.move_to_site('left', np.zeros(3), np.eye(3), trajectory_speed=-1.)
