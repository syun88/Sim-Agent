"""Physical motor tracking, calibrated timing and interruption regressions."""
import numpy as np
import pytest

from backend.app.control.motion import OpenArmV2Controller


def test_transport_response_retains_force_limits_and_finger_gains(env):
    entity = env._entity
    env.configure_motion_response(1.)
    fingers = list(env._gripper_dofs.values())
    before_fingers = entity.get_dofs_act_gain(fingers).clone()
    before_forces = [value.clone() for value in entity.get_dofs_force_range()]
    q_before = env.arm_qpos("left")
    env.configure_motion_response(6.)
    np.testing.assert_array_equal(env.arm_qpos("left"), q_before)
    np.testing.assert_array_equal(entity.get_dofs_act_gain(fingers).cpu(), before_fingers.cpu())
    for before, after in zip(before_forces, entity.get_dofs_force_range()):
        np.testing.assert_array_equal(before.cpu(), after.cpu())
    env.configure_motion_response(1.)
    for side, dofs in env._dofs.items():
        np.testing.assert_allclose(entity.get_dofs_act_gain(dofs).cpu(), env._act_gains[side])
        for actual, original in zip(entity.get_dofs_act_bias(dofs)[1:], env._act_biases[side]):
            np.testing.assert_allclose(actual.cpu(), original)


def test_calibrated_motion_keeps_its_simulation_duration(env):
    controller = OpenArmV2Controller(env, motion_speed=6.)
    position, rotation = env.ee_pose("left")
    before = env._sim_step
    result = controller.move_to_site("left", position, rotation, duration=.12, scale_speed=False)
    assert result.success
    assert result.trajectory_duration == pytest.approx(.12)
    assert (env._sim_step - before) * env.timestep >= .12
    assert env._motion_response_speed == 1.


def test_interrupted_trajectory_clears_velocity_reference(env, monkeypatch):
    controller = OpenArmV2Controller(env, motion_speed=6.)
    original_set_arm = env.set_arm
    commands = []

    def record(side, target, velocities=None):
        commands.append((np.asarray(target).copy(), velocities))
        original_set_arm(side, target, velocities)

    monkeypatch.setattr(env, "set_arm", record)
    initial_step = env._sim_step

    def interrupt():
        if env._sim_step - initial_step >= 20:
            raise InterruptedError("requested stop")

    env.step_observer = interrupt
    try:
        with pytest.raises(InterruptedError):
            controller.move_to_site("left", np.array([.36, .15, 1.18]))
    finally:
        env.step_observer = None
    assert any(velocity is not None and np.linalg.norm(velocity) > 0. for _, velocity in commands)
    assert commands[-1][1] is None
    np.testing.assert_allclose(commands[-1][0], env.arm_qpos("left"))
