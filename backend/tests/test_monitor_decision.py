"""Guards for the typed-decision monitor.

The monitor answers questions whose answers are options, not prose, so it cannot produce a malformed
answer - only an illegal one. That is the property these tests pin: an illegal or weak judgement is
ignored and the deterministic path is used, and a legal one is honoured and labelled, so a reader can
always tell which component chose a recovery.
"""

from __future__ import annotations

import pytest

from backend.app.planning import decision_model
from backend.app.planning.agent_planner import AgentPlanner

STATE = {
    "objects": [{"id": "red_cube", "type": "cube", "position": [-.05, 0., .773], "held_by": []},
                {"id": "blue_cube", "type": "cube", "position": [.05, 0., .773], "held_by": []}],
    "targets": [], "robot": {"left_arm": {"held_object": None}, "right_arm": {"held_object": None},
                             "available_arms": ["left"]},
}
FAILURE = {"category": "COLLISION_PATH", "detail": "path blocked near the table"}
LEGAL = ["recover_in_place", "reset_and_retry"]


def _verdict(monkeypatch, choice, confidence):
    payload = {"answers": {"next": {"type": "choice", "choice": choice,
                                    "confidence": confidence, "probabilities": {}}}}
    monkeypatch.setattr(decision_model, "ask", lambda *a, **k: payload)


def test_no_server_configured_means_no_judgement(monkeypatch):
    monkeypatch.delenv("OPENARM_DECISION_URL", raising=False)
    assert decision_model.available() is False
    questions = {"next": {"type": "choice", "instructions": "?", "criteria": {"a": "A"}}}
    assert decision_model.ask("state", questions) is None


def test_an_unreachable_server_returns_none_instead_of_raising(monkeypatch):
    monkeypatch.setenv("OPENARM_DECISION_URL", "http://127.0.0.1:1")
    questions = {"next": {"type": "noul", "instructions": "?"}}
    assert decision_model.ask("state", questions, timeout=.5) is None


def test_a_legal_verdict_is_used_and_labelled(monkeypatch):
    _verdict(monkeypatch, "reset_and_retry", .91)
    planner = AgentPlanner()
    assessment = planner.assess_failure("置いて", STATE, FAILURE, LEGAL)
    assert assessment["decision"] == "reset_and_retry"
    assert assessment["provider"] == "local_decision"
    assert assessment["confidence"] == pytest.approx(.91)


def test_an_illegal_verdict_is_ignored(monkeypatch):
    """Measured on d1-omni-600M: on a one-armed robot it chose alternate_arm with 0.80 confidence."""
    _verdict(monkeypatch, "alternate_arm", .80)
    planner = AgentPlanner()
    planner.enabled = planner.local_enabled = False
    assessment = planner.assess_failure("置いて", STATE, FAILURE, LEGAL)
    assert assessment["provider"] == "deterministic_fallback"
    assert assessment["decision"] in LEGAL


def test_a_low_confidence_verdict_is_ignored(monkeypatch):
    _verdict(monkeypatch, "reset_and_retry", .31)
    planner = AgentPlanner()
    planner.enabled = planner.local_enabled = False
    assessment = planner.assess_failure("置いて", STATE, FAILURE, LEGAL)
    assert assessment["provider"] == "deterministic_fallback"
    assert assessment["monitor"]["used"] is False, "an overruled judgement must still be reported"
    assert assessment["monitor"]["confidence"] == pytest.approx(.31)


def test_a_single_legal_option_does_not_ask_the_monitor(monkeypatch):
    seen = []
    monkeypatch.setattr(decision_model, "ask", lambda *a, **k: seen.append(1))
    planner = AgentPlanner()
    planner.enabled = planner.local_enabled = False
    planner.assess_failure("置いて", STATE, FAILURE, ["recover_in_place"])
    assert seen == [], "a question with one possible answer costs a forward pass and teaches nothing"
