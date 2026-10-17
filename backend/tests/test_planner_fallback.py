"""Guards for the planner's fallback contract.

Written after a real failure: running the suite with a verbatim copy of `.env.example` in place
made `test_plan_validation_accepts_valid_fallback` crash with `UnboundLocalError`, because the
planner treated the file's `replace-*` placeholders as real credentials, called the placeholder
endpoint, and - when the provider failed - reached for a `payload` the except branch never set.
These tests pin both halves of the fix, without Genesis and without the network.
"""

from __future__ import annotations

from backend.app.models.schemas import AgentPlan
from backend.app.planning.agent_planner import AgentPlanner, PlannerProviderError, validate_plan

PICK_AND_PLACE_SKILLS = [
    "approach_object", "grasp_object", "lift_object",
    "move_to_target", "release_object", "retreat",
]

WORLD_STATE = {
    "objects": [{"id": "red_cube"}, {"id": "blue_cube"}],
    "targets": [{"id": "blue_zone"}, {"id": "green_zone"}],
}

INSTRUCTION = "Move the red cube into the blue zone."


def _assert_valid_fallback_plan(plan: AgentPlan) -> None:
    assert plan.provider == "deterministic_fallback"
    assert [step.skill for step in plan.steps] == PICK_AND_PLACE_SKILLS
    valid, errors = validate_plan(plan, WORLD_STATE)
    assert valid, errors


def test_placeholder_credentials_leave_the_planner_disabled(monkeypatch):
    """The exact values `.env.example` ships must disable the planner, not configure it."""
    monkeypatch.setenv("AI_AND_API_KEY", "replace-if-available")
    monkeypatch.setenv("AI_AND_BASE_URL", "https://replace-with-official-ai-and-endpoint/v1")
    monkeypatch.setenv("AI_AND_MODEL", "replace-with-approved-model")
    planner = AgentPlanner()
    assert planner.enabled is False
    _assert_valid_fallback_plan(planner.plan(INSTRUCTION, WORLD_STATE))


def test_provider_failure_falls_back_to_the_deterministic_plan(monkeypatch):
    """An enabled planner whose provider fails must return the deterministic plan, not crash."""
    monkeypatch.setenv("AI_AND_API_KEY", "sk-not-a-real-key")
    monkeypatch.setenv("AI_AND_BASE_URL", "https://planner-fallback-test.invalid/v1")
    monkeypatch.setenv("AI_AND_MODEL", "test-model")
    planner = AgentPlanner()
    assert planner.enabled is True

    def _fail(self, instruction, state, memory):
        raise PlannerProviderError("PROVIDER_ERROR", "connection refused")

    monkeypatch.setattr(AgentPlanner, "_ai_plan", _fail)
    plan = planner.plan(INSTRUCTION, WORLD_STATE)
    _assert_valid_fallback_plan(plan)
    assert plan.raw_response is not None and plan.raw_response.startswith("AI provider failed")
