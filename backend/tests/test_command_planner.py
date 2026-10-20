from __future__ import annotations

import pytest

from backend.app.planning.agent_planner import AgentPlanner, PlannerProviderError, validate_plan

WORLD_STATE = {
    "objects": [{"id": "red_cube"}, {"id": "blue_cube"}],
    "targets": [{"id": "blue_zone"}, {"id": "green_zone"}],
}


def _plan_and_validate(instruction: str):
    plan = AgentPlanner().plan(instruction, WORLD_STATE)
    valid, errors = validate_plan(plan, WORLD_STATE)
    assert valid, errors
    return plan


def test_japanese_gripper_open_command_is_direct():
    plan = _plan_and_validate("グリッパーを開いてください")
    assert plan.goal["type"] == "direct_command"
    assert plan.goal["action"] == "open_gripper"
    assert [step.skill for step in plan.steps] == ["open_gripper"]


def test_gripper_close_command_is_direct():
    plan = _plan_and_validate("Close the left hand.")
    assert plan.goal["action"] == "close_gripper"
    assert plan.arm == "left"
    assert [step.skill for step in plan.steps] == ["close_gripper"]


def test_relative_arm_command_uses_direction_and_distance():
    plan = _plan_and_validate("右アームを右に10cm動かしてください")
    assert plan.arm == "right"
    assert [step.skill for step in plan.steps] == ["move_relative"]
    assert plan.steps[0].direction == "right"
    assert plan.steps[0].distance_cm == 10


def test_object_relative_command_goes_through_a_physical_grasp():
    plan = _plan_and_validate("赤いキューブを右に5cm動かしてください")
    assert plan.goal["type"] == "direct_command"
    assert plan.goal["action"] == "grasp_and_move"
    assert [step.skill for step in plan.steps] == [
        "approach_object", "grasp_object", "lift_object", "move_relative"
    ]
    assert plan.steps[-1].direction == "right"
    assert plan.steps[-1].distance_cm == 5


def test_japanese_pick_and_place_target_is_parsed():
    plan = _plan_and_validate("赤いキューブを緑のゾーンに入れてください")
    assert plan.goal["type"] == "place_in_zone"
    assert plan.goal["object"] == "red_cube"
    assert plan.goal["target"] == "green_zone"
    assert [step.skill for step in plan.steps] == [
        "approach_object", "grasp_object", "lift_object", "move_to_target", "release_object", "retreat"
    ]


def test_unknown_object_fails_loudly_when_local_planner_is_off():
    with pytest.raises(PlannerProviderError):
        AgentPlanner().plan("Move the invisible cube.", WORLD_STATE)


def test_both_cubes_can_be_selected_by_plural_instruction():
    plan = _plan_and_validate("pick red and blue cube")
    assert plan.goal["type"] == "direct_command"
    assert plan.goal["action"] == "multi_pick"
    assert plan.goal["objects"] == ["red_cube", "blue_cube"]
    assert [step.skill for step in plan.steps] == [
        "approach_object", "grasp_object", "lift_object",
        "approach_object", "grasp_object", "lift_object",
    ]


def test_count_only_instruction_selects_every_cube():
    plan = _plan_and_validate("pick two cube")
    assert plan.goal["action"] == "multi_pick"
    assert plan.goal["objects"] == ["red_cube", "blue_cube"]


def test_repeated_place_goal_is_not_reexecuted_when_already_satisfied():
    state = {
        "objects": [{"id": "red_cube", "position": [0.40, 0.00, 1.03]}],
        "targets": [{"id": "blue_zone", "position": [0.41, 0.00, 1.02]}],
    }
    plan = AgentPlanner().plan("Move the red cube into the blue zone.", state)
    valid, errors = validate_plan(plan, state)
    assert valid, errors
    assert plan.goal["type"] == "already_at_target"
    assert [step.skill for step in plan.steps] == ["return_home"]
