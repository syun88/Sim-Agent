"""Guards for the split between what the scene answers and what a model is asked.

Measured 2026-10-27: the local decision model counted cubes correctly but did not respond to swapping their
sides. These guards keep the consequence: geometry is derived, and only judgements are asked.
"""

from __future__ import annotations

from backend.app.agent.task_judgement import QUESTIONS, fact_packet, geometry_in_questions, relations


def _state():
    return {
        "objects": [
            {"id": "red_cube", "type": "cube", "color": "red", "position": [-0.05, 0.0, 0.773],
             "held_by": [], "touching": ["box_baselink"], "bounds": [[0, 0, 0]], "rotation": [[1, 0, 0]] * 3},
            {"id": "blue_cube", "type": "cube", "color": "blue", "position": [0.05, 0.10, 0.773],
             "held_by": [], "touching": ["box_baselink"]},
        ],
        "robot": {"left_arm": {"held_object": None}, "right_arm": {"held_object": "red_cube"}},
        "capabilities": {"skills": ["grasp_object", "throw_object"], "navigation": False},
    }


def test_the_packet_derives_relations_instead_of_asking_about_them():
    packet = fact_packet(_state())
    pairs = {(row["object"], row["reference"]): row["relation"] for row in packet["relations"]}
    # +Y is left, +X is forward: blue is to the left and in front of red.
    assert pairs[("red_cube", "blue_cube")] == ["front_of", "left_of"]
    assert pairs[("blue_cube", "red_cube")] == ["behind", "right_of"]
    assert packet["hands"] == {"left": None, "right": "red_cube"}
    assert packet["objects"][0]["touching"] == ["box_baselink"]


def test_the_packet_carries_no_evidence_the_planner_does_not_need():
    packet = fact_packet(_state())
    assert "bounds" not in packet["objects"][0] and "rotation" not in packet["objects"][0]
    assert packet["derived_by"].startswith("engine state")


def test_no_question_asks_what_the_scene_already_answers():
    assert geometry_in_questions() == [], "a spatial question was added back; the facts already answer it"


def test_every_question_says_why_a_model_call_is_justified():
    for name, spec in QUESTIONS.items():
        assert spec.get("why"), f"{name} does not justify the forward pass"
        assert spec["type"] in {"noul", "choice", "score"}


def test_a_spatial_question_is_detected_when_someone_adds_one():
    bad = {"side": {"type": "choice", "instructions": "Is the red cube left of the blue cube?",
                    "criteria": {"left": "left", "right": "right"}}}
    assert geometry_in_questions(bad) == ["side"]


def test_relations_ignore_objects_without_positions():
    assert relations([{"id": "a"}, {"id": "b", "position": [0, 0, 0]}]) == []


def test_trust_requires_a_measurement_and_names_the_model():
    """Calibrated 2026-10-27: omni-600M scored 5/11 overall, d1-3B 9/11 - three of three on "which cause
    blocks this step" and two of two on ambiguity, four of six on outcome. Only measured questions may be
    trusted, and the record must say which model produced the number."""
    trusted = [name for name, spec in QUESTIONS.items() if spec.get("trusted")]
    assert trusted == ["next_step_blocked", "reference_ambiguous"], trusted
    for name, spec in QUESTIONS.items():
        assert "trusted" in spec and "measured" in spec, f"{name} is missing its record"
        assert "d1-3B" in spec["measured"] or "omni-600M" in spec["measured"], f"{name} names no model"
    assert QUESTIONS["outcome_achieved"]["trusted"] is False, "4/6 is not a basis for acting"


def test_the_calibration_script_exists_and_uses_the_labels_the_state_gives():
    from pathlib import Path

    script = Path(__file__).resolve().parents[2] / "scripts" / "calibrate_decision_model.py"
    text = script.read_text()
    assert "OUTCOME_CASES" in text and "BLOCK_CASES" in text
    assert "fact_packet" in text, "calibration must ask over the same packet the app sends"
