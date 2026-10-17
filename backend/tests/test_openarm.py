from pathlib import Path

import mujoco
import numpy as np
import pytest

from backend.app.agent.world_state import build_world_state
from backend.app.control.motion import OpenArmV2Controller
from backend.app.planning.agent_planner import AgentPlanner, validate_plan
from backend.app.simulation.openarm_v2.definition import (
    ARM_JOINTS,
    OBJECTS,
    SPAWN_ARM_QPOS,
    ZONES,
)
from backend.app.simulation.openarm_v2.scene import initial_object_positions, resolved_scene_xml
from backend.app.skills.robot_skills import SkillRunner
from backend.app.verification.goal_verification import verify_place_goal

PICK_AND_PLACE = [
    {"skill": "approach_object", "arm": "left", "object": "red_cube"},
    {"skill": "grasp_object", "arm": "left", "object": "red_cube"},
    {"skill": "lift_object", "arm": "left", "object": "red_cube"},
    {"skill": "move_to_target", "arm": "left", "target": "blue_zone"},
    {"skill": "release_object", "arm": "left", "object": "red_cube"},
    {"skill": "retreat", "arm": "left"},
]


def test_scene_carries_the_official_openarm_v2_model():
    """The scene the engine loads is the official Cell model, with its structure intact."""
    model = mujoco.MjModel.from_xml_string(resolved_scene_xml())
    assert model.nu == 17
    assert model.nq == 33
    for side in ("left", "right"):
        for i in range(1, 8):
            assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"openarm_{side}_joint{i}") >= 0
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{side}_finger1_ctrl") >= 0
    # The only equalities are the official Cell's two finger mimics, both active. The scene declares
    # no grasp welds: a grasp is the fingers' physical friction pinch, so no constraint that could
    # move an object is declared at all, and an engine that builds every equality (Genesis) builds
    # exactly the official two.
    assert list(model.eq_active0) == [1, 1]


def test_starts_in_the_documented_scene(fresh_engine_state):
    """A freshly constructed engine is already in the documented spawn pose, before any `reset`.

    The server builds one backend at startup and never resets it, so a task that arrives first must
    not begin from the model's zero configuration: the arm would sweep across the table on its way to
    the first waypoint.  The documented spawn pose is `SPAWN_ARM_QPOS`.
    """
    documented = initial_object_positions()
    assert set(fresh_engine_state["objects"]) == set(documented)
    for name, position in documented.items():
        assert fresh_engine_state["objects"][name] == pytest.approx(position, abs=1e-6)
    assert set(fresh_engine_state["zones"]) == set(ZONES)
    for side in ("left", "right"):
        assert fresh_engine_state[f"arm_{side}"] == pytest.approx(SPAWN_ARM_QPOS, abs=1e-5)
    # The chosen spawn pose is the official model keyframe, so it must still match the model.
    model = mujoco.MjModel.from_xml_string(resolved_scene_xml())
    for side in ("left", "right"):
        names = ARM_JOINTS[side]
        keyframe = np.array(
            [
                model.key_qpos[0][
                    model.jnt_qposadr[
                        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                    ]
                ]
                for name in names
            ]
        )
        assert fresh_engine_state[f"arm_{side}"] == pytest.approx(keyframe, abs=1e-5)


def test_genesis_builds_the_official_model(genesis_env):
    """What Genesis World constructed from that scene: the same joints, only the active equalities."""
    summary = genesis_env.model_summary()
    assert summary["positions"] == 33
    assert summary["equalities"] == 2
    assert summary["objects"] == sorted(OBJECTS)
    for side, joints in ARM_JOINTS.items():
        assert len(genesis_env.arm_qpos(side)) == len(joints) == 7


ROOT = Path(__file__).resolve().parents[2]


def test_scene_has_no_machine_specific_paths():
    """The demo scene resolves the official assets at load time instead of hard-coding a machine."""
    text = (ROOT / "assets" / "scenes" / "openarm_v2_demo.xml").read_text()
    assert "/Users/" not in text
    assert "@OPENARM_MUJOCO_V2_CELL_XML@" in text


def test_world_state_comes_from_simulation(env):
    state = build_world_state(env)
    position, _ = env.object_pose("red_cube")
    assert state["objects"][0]["position"] == pytest.approx(position.tolist(), abs=1e-6)
    assert state["robot"]["type"] == "openarm_v2"


def test_arm_motion(env, controller):
    result = controller.move_to_site("left", np.array([0.36, 0.15, 1.06]))
    assert result.success


def test_plan_validation_accepts_valid_fallback(env):
    plan = AgentPlanner().plan("Move the red cube into the blue zone.", build_world_state(env))
    valid, errors = validate_plan(plan, build_world_state(env))
    assert valid, errors
    assert [step.skill for step in plan.steps] == [step["skill"] for step in PICK_AND_PLACE]
    assert [step["skill"] for step in PICK_AND_PLACE] == ["approach_object", "grasp_object", "lift_object", "move_to_target", "release_object", "retreat"]


def test_complete_pick_place(env, controller):
    runner = SkillRunner(env, controller)
    results = runner.execute_plan(PICK_AND_PLACE)
    # The whole skill sequence executes on every engine.
    assert all(result["success"] for result in results), results
    approach, grasp = results[0], results[1]
    # The approach must not shove the cube: it is measured, and a shove fails the skill.
    assert approach["object_displacement"] < 0.005, approach
    # The grasp is a physical pinch: both fingers contact the cube, the cube follows a
    # verification lift, and the engine reports its grasp as physical contact.
    assert grasp["grasp_mode"] == env.grasp_mode
    assert grasp["grasp_is_physical_contact"] is True
    fingers = grasp["pinch_contacts"]
    assert any("inner" in finger for finger in fingers) and any("outer" in finger for finger in fingers), grasp
    assert grasp["object_rose"] >= 0.02, grasp
    verification = verify_place_goal(env, "red_cube", "blue_zone")
    assert verification["success"], verification
    assert verification["horizontal_position_error"] <= 0.04
    # Released means no gripper finger touches the object any more.
    assert verification["released"] is True


def test_unreachable_fault_is_classified(env, controller):
    runner = SkillRunner(env, controller)
    result = runner.approach_object("left", "red_cube", fault_offset=-0.12)
    assert result["success"] is False
    assert result["error"] == "IK_FAILURE"


def test_grasp_fails_without_physical_contact(env, controller):
    """A grasp with nothing between the fingers is a failure, not a success.

    The gripper approaches the red cube, then the plan asks it to grasp the *blue* cube: the
    fingers close on air, the contact check finds no pinch, and the skill must fail. This is
    the guard that a grasp can never be reported as successful without physical contact.
    """
    runner = SkillRunner(env, controller)
    approach = runner.approach_object("left", "red_cube")
    assert approach["success"], approach
    result = runner.grasp_object("left", "blue_cube")
    assert result["success"] is False
    assert result["error"] == "GRASP_FAILED"


def test_right_arm_blue_cube_place(env, controller):
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
    assert all(result["success"] for result in results), results
    assert verify_place_goal(env, "blue_cube", "green_zone")["success"]


def test_reset_restores_the_documented_scene(env):
    env.reset()
    placed = env.object_pose("red_cube")[0].copy()
    env.reset(randomize=True)
    moved = env.object_pose("red_cube")[0]
    assert np.linalg.norm(placed - moved) > 1e-3, "randomization must move the cube"
    env.reset()
    assert env.object_pose("red_cube")[0] == pytest.approx(
        initial_object_positions()["red_cube"], abs=1e-6
    )
