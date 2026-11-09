"""The clearance query answers with a number, so a search has something to rank."""
import numpy as np
import pytest

from backend.app.control.motion import OpenArmV2Controller
from backend.app.simulation.openarm_v2.definition import FINGERTIP_MIDPOINT, GRASP_ROTATION
from backend.tests.test_task_homing import _stack_environment


def _case():
    env, q, objects = _stack_environment()
    return env, OpenArmV2Controller(env, configure_motors=False), objects


def _pinch_q(env, controller, objects, name, lift=0.):
    """A joint configuration whose hand sits at a cube's pinch point (lifted by `lift`)."""
    centre = objects[name][0] + np.array([0., 0., lift])
    target = centre - GRASP_ROTATION @ FINGERTIP_MIDPOINT['left']
    q, error, _ = controller.solve_ik('left', target, GRASP_ROTATION)
    if q is None or error > .004:
        pytest.skip('the calibrated frame cannot reach this cube in the fixture')
    return q


def test_the_scalar_agrees_with_the_screen_on_whether_the_arm_is_inside_something():
    env, controller, objects = _case()
    kin = env.arm_kinematics('left')
    home = env.home_arm_qpos('left')
    for path in ([home, home], [env.arm_qpos('left'), home]):
        gap = kin.path_minimum_clearance(path, objects)
        assert gap is not None
        assert kin.empty_arm_path_is_clear(path, objects) is (gap > -0.0005)


def test_moving_a_cube_out_of_the_way_increases_the_measured_room():
    env, controller, objects = _case()
    kin = env.arm_kinematics('left')
    at_cube = _pinch_q(env, controller, objects, 'blue_cube')
    before = kin.path_minimum_clearance([at_cube, at_cube], objects)
    displaced = {name: (position + np.array([0., -1., 0.]), rotation)
                 for name, (position, rotation) in objects.items()}
    after = kin.path_minimum_clearance([at_cube, at_cube], displaced)
    assert before is not None and after is not None
    assert after > before, 'the query has to see the object the search is ranking against'


def test_a_scene_without_movable_objects_is_unknown_not_clear():
    env, controller, objects = _case()
    kin = env.arm_kinematics('left')
    assert kin.path_minimum_clearance([env.arm_qpos('left'), env.home_arm_qpos('left')], {}) is None
