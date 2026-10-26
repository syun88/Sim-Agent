"""Guards for the per-request provenance ledger.

Written because the same question kept being unanswerable: whether a task was done by the model or by
the deterministic grammar. The planner answered a single direct command from the grammar and recorded
nothing, so a request that consulted no model at all was indistinguishable from one where the model
agreed - `provider` alone says deterministic_fallback in both cases. These tests pin that the answer
path, the reason for skipping, and the model's own verdict are visible per request, and that the two
new modes (capability, shadow) cannot silently change what runs.
"""

from __future__ import annotations

import pytest

from backend.app.models.schemas import AgentPlan
from backend.app.planning.agent_planner import PROVENANCE_FIELDS, AgentPlanner, program_signature

LEDGER_FIELDS = {"provider", "model", "provider_error", "model_called", "skipped_because",
                 "capability_mode", "fallback_verdict", "model_verdict", "agree", "shadow"}

WORLD_STATE = {
    "objects": [{"id": "red_cube"}, {"id": "blue_cube"}],
    "targets": [{"id": "blue_zone"}, {"id": "green_zone"}],
}

INSTRUCTION = "Move the red cube into the blue zone."
OTHER = "Move the blue cube into the green zone."


def _disable(monkeypatch):
    for name in ("AI_AND_API_KEY", "AI_AND_BASE_URL", "AI_AND_MODEL", "OPENARM_LOCAL_PLANNER"):
        monkeypatch.delenv(name, raising=False)


def _local(monkeypatch):
    _disable(monkeypatch)
    monkeypatch.setenv("OPENARM_LOCAL_PLANNER", "1")


def test_a_deterministic_answer_records_that_no_model_was_consulted(monkeypatch):
    _disable(monkeypatch)
    plan = AgentPlanner().plan(INSTRUCTION, WORLD_STATE)
    assert plan.provider == "deterministic_fallback"
    assert plan.model_called is False
    assert plan.skipped_because == "provider_not_configured"
    assert plan.fallback_verdict["ok"] is True and plan.fallback_verdict["actions"] > 0
    assert plan.model_verdict["ok"] is False
    assert plan.agree is None
    assert plan.shadow is None


def test_a_skipped_model_says_which_shortcut_skipped_it(monkeypatch):
    """`exact_grammar_fast_path` and `single_direct_command` are different reasons; keep them apart."""
    _local(monkeypatch)
    monkeypatch.setenv("OPENARM_FAST_LANGUAGE", "1")

    def _must_not_run(self, instruction, state, memory):
        raise AssertionError("the fast language path must not call the provider")

    monkeypatch.setattr(AgentPlanner, "_ai_plan", _must_not_run)
    plan = AgentPlanner().plan(INSTRUCTION, WORLD_STATE)
    assert plan.model_called is False
    assert plan.skipped_because == "exact_grammar_fast_path"


def test_capability_mode_asks_the_model_even_when_the_rules_parse_the_sentence(monkeypatch):
    _local(monkeypatch)
    monkeypatch.setenv("OPENARM_FAST_LANGUAGE", "1")
    monkeypatch.setenv("OPENARM_CAPABILITY_MODE", "1")
    expected = AgentPlanner().plan(INSTRUCTION, WORLD_STATE)  # disabled run, for its program
    seen = []

    def _same(self, instruction, state, memory):
        seen.append(instruction)
        return {"payload": expected.program, "raw": "shadowed"}

    monkeypatch.setattr(AgentPlanner, "_ai_plan", _same)
    plan = AgentPlanner().plan(INSTRUCTION, WORLD_STATE)
    assert seen == [INSTRUCTION], "capability mode must ask the provider exactly once"
    assert plan.model_called is True
    assert plan.capability_mode is True
    assert plan.skipped_because is None, "a called model was not skipped"
    assert plan.provider == "local_ollama"
    assert plan.model_verdict["ok"] is True
    assert plan.agree is True
    assert program_signature(plan) == program_signature(expected)


def test_capability_mode_keeps_and_labels_the_deterministic_answer_when_the_model_differs(monkeypatch):
    """A model that changes the requested objects is refused, and the ledger says so."""
    _local(monkeypatch)
    monkeypatch.setenv("OPENARM_CAPABILITY_MODE", "1")
    expected = AgentPlanner().plan(INSTRUCTION, WORLD_STATE)
    other = AgentPlanner().plan(OTHER, WORLD_STATE)

    monkeypatch.setattr(AgentPlanner, "_ai_plan",
                        lambda self, i, s, m: {"payload": other.program, "raw": "model"})
    plan = AgentPlanner().plan(INSTRUCTION, WORLD_STATE)
    assert plan.model_called is True
    assert plan.model_verdict["ok"] is False
    assert plan.model_verdict["error"] == "PROVIDER_INVALID_PROGRAM"
    assert program_signature(plan) == program_signature(expected), "the labelled fallback is what runs"
    assert plan.fallback_verdict["ok"] is True


def test_shadow_mode_records_the_model_without_changing_what_runs(monkeypatch):
    _local(monkeypatch)
    monkeypatch.setenv("OPENARM_FAST_LANGUAGE", "1")
    monkeypatch.setenv("OPENARM_SHADOW_PLAN", "1")
    expected = AgentPlanner().plan(INSTRUCTION, WORLD_STATE)

    monkeypatch.setattr(AgentPlanner, "_ai_plan",
                        lambda self, i, s, m: {"payload": expected.program, "raw": "shadow"})
    plan = AgentPlanner().plan(INSTRUCTION, WORLD_STATE)
    assert plan.model_called is False, "a shadow call is not an execution path"
    assert plan.provider == "deterministic_fallback"
    assert plan.shadow["attempted"] is True and plan.shadow["ok"] is True
    assert plan.shadow["agree"] is True
    assert program_signature(plan) == program_signature(expected)


def test_a_failing_shadow_call_cannot_fail_the_request(monkeypatch):
    _local(monkeypatch)
    monkeypatch.setenv("OPENARM_FAST_LANGUAGE", "1")
    monkeypatch.setenv("OPENARM_SHADOW_PLAN", "1")
    expected = AgentPlanner().plan(INSTRUCTION, WORLD_STATE)

    def _explode(self, instruction, state, memory):
        raise RuntimeError("provider went away")

    monkeypatch.setattr(AgentPlanner, "_ai_plan", _explode)
    plan = AgentPlanner().plan(INSTRUCTION, WORLD_STATE)
    assert plan.shadow["ok"] is False and "provider went away" in plan.shadow["error"]
    assert program_signature(plan) == program_signature(expected)


def test_a_replanned_continuation_carries_the_whole_ledger():
    """A recovery rebuilds the steps through the compiler; it must not lose who planned the task."""
    assert LEDGER_FIELDS <= set(PROVENANCE_FIELDS), "a provenance field was left out of the carry-over list"
    assert LEDGER_FIELDS <= set(AgentPlan.model_fields), "a provenance field is missing from the response model"


def test_a_placement_cannot_stand_in_for_a_requested_throw(monkeypatch):
    """The model proposed carrying the cube onto the other one for '投げて当てて' and it passed as success."""
    from backend.app.planning.agent_planner import missing_obligations

    _local(monkeypatch)
    monkeypatch.setenv("OPENARM_CAPABILITY_MODE", "1")
    placement = AgentPlanner().plan("青いキューブを赤いキューブの上に置いて", WORLD_STATE)
    assert missing_obligations("青いキューブを赤いキューブの上に置いて", placement) == []
    assert missing_obligations("青いキューブを上に投げて赤いキューブに当てて", placement) == ["throw_object"]

    monkeypatch.setattr(AgentPlanner, "_ai_plan",
                        lambda self, i, s, m: {"payload": placement.program, "raw": "model"})
    with pytest.raises(Exception):
        AgentPlanner().plan("青いキューブを上に投げて赤いキューブに当てて", WORLD_STATE)
