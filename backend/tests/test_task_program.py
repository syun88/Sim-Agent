from __future__ import annotations

import pytest

from backend.app.planning.agent_planner import AgentPlanner, PlannerProviderError
from backend.app.planning.task_program import GroundingError, TaskAction, TaskProgram, compile_program

WORLD = {"objects": [{"id": "red_cube", "position": [.44, .15, 1.031]}, {"id": "blue_cube", "position": [.50, -.22, 1.031]}],
         "targets": [{"id": "blue_zone", "position": [.48, .18, 1.008]}, {"id": "green_zone", "position": [.48, -.2, 1.008]}]}
EXAMPLE = "右手がBlue Cubeを取って左手に渡した後に置いて、その左手にあるRED CUBEを掴んで右手がそれを取って右手のところに置く"


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    for key in ("AI_AND_API_KEY", "AI_AND_BASE_URL", "AI_AND_MODEL", "OPENARM_LOCAL_PLANNER"):
        monkeypatch.delenv(key, raising=False)


def test_full_user_example_preserves_both_handovers_and_both_final_goals():
    plan = AgentPlanner().plan(EXAMPLE, WORLD)
    actions = plan.program["actions"]
    assert [a["op"] for a in actions] == ["pick", "handover", "place", "pick", "handover", "place"]
    assert [(a["object"], a["arm"], a["receiver"]) for a in actions if a["op"] == "handover"] == [
        ("blue_cube", "right", "left"), ("red_cube", "left", "right")]
    assert [(c["object"], c["target"]) for c in plan.goal["checks"]] == [("blue_cube", "left_side"), ("red_cube", "right_side")]
    assert sum(s.skill == "handover_object" for s in plan.steps) == 2
    assert plan.assumptions  # Unspecified first placement is explicit and visible.


def test_english_sequential_objects_do_not_collapse_to_the_first_cube():
    plan = AgentPlanner().plan("Pick the blue cube with the right hand, then pass it to the left hand, then place it on the left side. Pick the red cube with the left hand, then pass it to the right hand, then place it on the right side.", WORLD)
    assert len(plan.program["actions"]) == 6
    assert plan.goal["checks"][1]["object"] == "red_cube"


def test_occupied_receiving_hand_is_rejected_before_execution():
    actions = [TaskAction(op="pick", object="red_cube", arm="left"), TaskAction(op="pick", object="blue_cube", arm="right"), TaskAction(op="handover", object="blue_cube", arm="right", receiver="left")]
    with pytest.raises(GroundingError, match="cannot receive"):
        compile_program(TaskProgram(actions=actions), WORLD)


def test_a_hand_cannot_silently_drop_the_first_object_to_pick_the_second():
    with pytest.raises(GroundingError, match="already holds"):
        compile_program(TaskProgram(actions=[TaskAction(op="pick", object=o, arm="left") for o in ("red_cube", "blue_cube")]), WORLD)


def test_followup_it_uses_observed_held_object_without_reopening_the_gripper():
    state = {**WORLD, "robot": {"right_arm": {"held_object": "blue_cube"}, "left_arm": {"held_object": None}}}
    plan = AgentPlanner().plan("それを左側に置く", state)
    assert plan.goal["object"] == "blue_cube"
    assert plan.arm == "right"
    assert [s.skill for s in plan.steps] == ["move_to_target", "release_object", "retreat"]


def test_arm_selection_follows_the_measured_object_side_after_a_prior_task():
    state = {**WORLD, "objects": [{"id": "blue_cube", "position": [.44, .25, 1.03]}]}
    assert AgentPlanner().plan("青を掴んで", state).arm == "left"


@pytest.mark.parametrize("text", ["黄色のキューブを掴んで", "Pick the red and yellow cube", "pick one cube", "左手に渡す"])
def test_unsupported_or_ambiguous_objects_are_not_replaced_with_a_default_task(text):
    with pytest.raises(PlannerProviderError):
        AgentPlanner().plan(text, WORLD)


def test_provider_repair_is_validated_and_cannot_invent_an_object(monkeypatch):
    p = AgentPlanner()
    p.enabled, p.local_enabled = True, True
    calls = []
    def provider(*args):
        calls.append(p._repair_feedback)
        payload = {"actions": [{"op": "pick", "object": "yellow_cube" if len(calls) == 1 else "blue_cube", "arm": "right"}]}
        return {"payload": payload, "raw": "validated test response"}
    monkeypatch.setattr(p, "_ai_plan", provider)
    plan = p.plan("take the azure one please", WORLD)
    assert len(calls) == 2 and "yellow_cube" in calls[1]
    assert plan.provider == "local_ollama" and plan.goal["object"] == "blue_cube"


def test_model_cannot_drop_half_of_an_explicit_multi_action_request(monkeypatch):
    p = AgentPlanner()
    p.enabled, p.local_enabled = True, True
    monkeypatch.setattr(p, "_ai_plan", lambda *a: {"payload": {"actions": [{"op": "pick", "object": "blue_cube", "arm": "right"}]}, "raw": "incomplete"})
    plan = p.plan(EXAMPLE, WORLD)
    assert plan.provider == "deterministic_fallback"
    assert plan.provider_error == "PROVIDER_INVALID_PROGRAM"
    assert len(plan.program["actions"]) == 6


def test_provider_exceptions_do_not_disclose_credentials(monkeypatch):
    p = AgentPlanner()
    p.enabled, p.local_enabled = True, True
    def fail(*a):
        raise RuntimeError("secret-api-key-123")
    monkeypatch.setattr(p, "_ai_plan", fail)
    plan = p.plan(EXAMPLE, WORLD)
    assert "secret-api-key" not in plan.model_dump_json()
    assert plan.provider_error == "PROVIDER_UNAVAILABLE"


def test_cross_prompt_reference_uses_last_manipulated_object_even_after_release():
    state = {**WORLD, "discourse": {"last_object": "blue_cube", "last_arm": "left", "last_target": "left_side"}}
    plan = AgentPlanner().plan("それを右側に置く", state)
    assert plan.goal["object"] == "blue_cube" and plan.goal["target"] == "right_side"


def test_a_low_level_lift_without_a_grasp_is_invalid():
    from backend.app.models.schemas import AgentPlan, PlanStep
    from backend.app.planning.agent_planner import validate_plan
    plan = AgentPlan(goal={"type": "direct_command"}, steps=[PlanStep(skill="lift_object", object="red_cube")])
    valid, errors = validate_plan(plan, WORLD)
    assert not valid and any("does not hold" in error for error in errors)


def test_model_can_explicitly_reject_an_unsupported_task_without_making_up_actions(monkeypatch):
    p = AgentPlanner()
    p.enabled, p.local_enabled = True, True
    monkeypatch.setattr(p, "_ai_plan", lambda *a: {"payload": {"actions": [], "error": "Stacking is not implemented"}, "raw": "test"})
    with pytest.raises(PlannerProviderError, match="Stacking") as error:
        p.plan("stack both cubes into a tower", WORLD)
    assert error.value.category == "UNSUPPORTED_TASK"


def test_object_and_zone_ids_do_not_turn_a_zone_color_into_an_extra_cube():
    plan = AgentPlanner().plan("Move red_cube to blue_zone", WORLD)
    assert plan.goal["object"] == "red_cube" and plan.goal["target"] == "blue_zone"
    assert len(plan.program["actions"]) == 1
