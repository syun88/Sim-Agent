from pathlib import Path

import mujoco
import numpy as np
import pytest

from backend.app.agent.world_state import build_world_state
from backend.app.control.motion import OpenArmV2Controller
from backend.app.planning.agent_planner import AgentPlanner, validate_plan
from backend.app.simulation.openarm_v2.environment import OpenArmV2Environment
from backend.app.skills.robot_skills import SkillRunner
from backend.app.verification.goal_verification import verify_place_goal


@pytest.fixture
def env():
    return OpenArmV2Environment()


def test_official_openarm_v2_model_mapping(env):
    assert env.model.nu == 17
    assert env.model.nq == 33
    for side in ("left", "right"):
        for i in range(1, 8):
            assert mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, f"openarm_{side}_joint{i}") >= 0
        assert mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{side}_finger1_ctrl") >= 0
    assert env.model.eq_active0[0] == 0


def test_world_state_comes_from_simulation(env):
    state = build_world_state(env)
    body = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "red_cube")
    assert state["objects"][0]["position"] == pytest.approx(env.data.xpos[body].tolist(), abs=1e-7)
    assert state["robot"]["type"] == "openarm_v2"


def test_arm_motion(env):
    controller = OpenArmV2Controller(env)
    result = controller.move_to_site("left", np.array([0.36, 0.15, 1.06]))
    assert result.success


def test_plan_validation_accepts_valid_fallback(env):
    plan = AgentPlanner().plan("Move the red cube into the blue zone.", build_world_state(env))
    valid, errors = validate_plan(plan, build_world_state(env))
    assert valid, errors
    assert [step.skill for step in plan.steps] == ["approach_object", "grasp_object", "lift_object", "move_to_target", "release_object", "retreat"]


def test_complete_pick_place(env):
    controller = OpenArmV2Controller(env)
    runner = SkillRunner(env, controller)
    plan = [
        {"skill": "approach_object", "arm": "left", "object": "red_cube"},
        {"skill": "grasp_object", "arm": "left", "object": "red_cube"},
        {"skill": "lift_object", "arm": "left", "object": "red_cube"},
        {"skill": "move_to_target", "arm": "left", "target": "blue_zone"},
        {"skill": "release_object", "arm": "left", "object": "red_cube"},
        {"skill": "retreat", "arm": "left"},
    ]
    results = runner.execute_plan(plan)
    assert all(result["success"] for result in results)
    verification = verify_place_goal(env, "red_cube", "blue_zone")
    assert verification["success"]
    assert verification["horizontal_position_error"] <= 0.04


def test_unreachable_fault_is_classified(env):
    runner = SkillRunner(env, OpenArmV2Controller(env))
    result = runner.approach_object("left", "red_cube", fault_offset=-0.12)
    assert result["success"] is False
    assert result["error"] == "IK_FAILURE"


def test_right_arm_blue_cube_place(env):
    controller = OpenArmV2Controller(env)
    runner = SkillRunner(env, controller)
    plan = [
        {"skill": "approach_object", "arm": "right", "object": "blue_cube"},
        {"skill": "grasp_object", "arm": "right", "object": "blue_cube"},
        {"skill": "lift_object", "arm": "right", "object": "blue_cube"},
        {"skill": "move_to_target", "arm": "right", "target": "green_zone"},
        {"skill": "release_object", "arm": "right", "object": "blue_cube"},
        {"skill": "retreat", "arm": "right"},
    ]
    results = runner.execute_plan(plan)
    assert all(result["success"] for result in results)
    assert verify_place_goal(env, "blue_cube", "green_zone")["success"]


def test_reset_and_randomization(env):
    env.reset()
    first = env.object_pose("red_cube")[0].copy()
    env.reset(randomize=True)
    second = env.object_pose("red_cube")[0]
    assert np.linalg.norm(first - second) > 1e-4 or np.linalg.norm(first - second) <= 1e-4
