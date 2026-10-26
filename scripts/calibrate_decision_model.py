"""Calibrate the decision model's questions against labels the simulator already knows.

Every judgement the app asks is a question with a ground truth we can construct: whether a requested outcome
has been reached, why the next step cannot proceed, whether a reference is ambiguous. The state gives the
label; the model gives an answer; the difference is the number that decides whether a question may influence a
plan or must stay advisory.

Run against a live decision server:
    scripts/serve_models.sh d1
    /data/d1-venv/bin/python scripts/calibrate_decision_model.py --out /tmp/d1_calibration.json
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.app.agent.task_judgement import QUESTIONS, fact_packet  # noqa: E402


def cube(cid, colour, position, held_by=(), touching=("box_baselink",)):
    return {"id": cid, "type": "cube", "color": colour, "position": list(position),
            "held_by": list(held_by), "touching": list(touching), "graspable": True}


def state(objects, left=None, right=None):
    return {"objects": objects,
            "robot": {"left_arm": {"held_object": left}, "right_arm": {"held_object": right}},
            "capabilities": {"skills": ["grasp_object", "place_object", "throw_object", "move_object"]}}


# Each case: the instruction, the observed state, and the label a human would give.
OUTCOME_CASES = [
    ("赤いキューブを青いキューブの隣に置いて", "on the table apart", False,
     state([cube("red_cube", "red", (-.05, .05, .773)), cube("blue_cube", "blue", (.05, .0, .773))])),
    ("赤いキューブを青いキューブの隣に置いて", "parked beside the blue cube", True,
     state([cube("red_cube", "red", (-.05, .08, .773)), cube("blue_cube", "blue", (-.05, .005, .773))])),
    ("青いキューブを赤いキューブの上に置いて", "stacked", True,
     state([cube("red_cube", "red", (-.05, .0, .773)), cube("blue_cube", "blue", (-.05, .0, .799))])),
    ("青いキューブを赤いキューブの上に置いて", "both on the table", False,
     state([cube("red_cube", "red", (-.05, .0, .773)), cube("blue_cube", "blue", (.05, .0, .773))])),
    ("青いキューブを掴んで", "blue is in the left hand", True,
     state([cube("red_cube", "red", (-.05, .0, .773)), cube("blue_cube", "blue", (.05, .0, .78), ("left",))],
           left="blue_cube")),
    ("青いキューブを掴んで", "hands are empty", False,
     state([cube("red_cube", "red", (-.05, .0, .773)), cube("blue_cube", "blue", (.05, .0, .773))])),
]

BLOCK_CASES = [
    ("grasp red_cube with the left hand", "the left hand is holding blue_cube", "hand_occupied",
     state([cube("red_cube", "red", (-.05, .0, .773)), cube("blue_cube", "blue", (.05, .0, .78), ("left",))],
           left="blue_cube")),
    ("place blue_cube on red_cube", "blue_cube already rests on red_cube", "target_occupied",
     state([cube("red_cube", "red", (-.05, .0, .773)), cube("blue_cube", "blue", (-.05, .0, .799), touching=("red_cube",))])),
    ("grasp red_cube", "red_cube sits at (0.9, 0.9), outside this arm's reach", "out_of_reach",
     state([cube("red_cube", "red", (.9, .9, .773))])),
]

AMBIGUOUS_CASES = [
    ("take the blue cube", "there are two blue cubes", True,
     state([cube("blue_cube", "blue", (-.05, .0, .773)), cube("blue_cube_2", "blue", (.05, .05, .773))])),
    ("take the red cube", "there is exactly one red cube", False,
     state([cube("red_cube", "red", (-.05, .0, .773)), cube("blue_cube", "blue", (.05, .0, .773))])),
]


class InlineModel:
    """Load the model in this process and answer like the server does.

    Comparing models needs one thing the HTTP server cannot give: a different checkpoint. Loading here keeps
    the questions and the labels identical to the scored set, so the only variable is the model.
    """

    def __init__(self, repo: str):
        import torch
        from transformers import AutoModel
        self.model = AutoModel.from_pretrained(repo, trust_remote_code=True, dtype=torch.bfloat16).eval()

    def ask(self, packet: dict, questions: dict, timeout: float = 30.) -> dict:
        import json as _json
        return self.model.system_one(_json.dumps(packet, ensure_ascii=False), questions)["answers"]


def ask(base_url: str, packet: dict, questions: dict, timeout: float = 30.) -> dict:
    body = json.dumps({"state": json.dumps(packet, ensure_ascii=False), "questions": questions}).encode()
    request = urllib.request.Request(base_url + "/decide", data=body,
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())["answers"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:11435")
    parser.add_argument("--out", default="/tmp/d1_calibration.json")
    parser.add_argument("--inline", help="load this checkpoint here instead of calling a server")
    args = parser.parse_args()
    inline = InlineModel(args.inline) if args.inline else None
    report = {"model": args.inline or args.url, "noul_means": "P(YES) (measured separately)", "questions": {}}

    def score(name, rows):
        correct, detail = 0, []
        for row in rows:
            instruction, description, label, world = row
            packet = fact_packet(world)
            text = f"Instruction: {instruction}. Observed situation: {description}."
            question = {name: {key: value for key, value in QUESTIONS[name].items() if key != "why"}}
            payload = {"facts": packet, "instruction": instruction, "situation": description}
            answers = (inline.ask(payload, question) if inline is not None
                       else ask(args.url, payload, question))
            answer = answers.get(name) or {}
            if name == "next_step_blocked":
                predicted = answer.get("choice")
                agree = predicted == label
                confidence = float(answer.get("confidence") or 0.)
            else:
                p_yes = float(answer.get("noul", 0.))
                predicted = p_yes >= .5
                agree = predicted == label
                confidence = max(p_yes, 1. - p_yes)
            correct += int(agree)
            detail.append({"instruction": instruction, "situation": description, "label": label,
                           "predicted": predicted, "confidence": round(confidence, 3), "correct": bool(agree),
                           "prompt_chars": len(text)})
        report["questions"][name] = {"cases": len(rows), "correct": correct,
                                     "accuracy": round(correct / max(len(rows), 1), 3), "detail": detail}
        print(f'{name:20s} {correct}/{len(rows)} correct', flush=True)
        for row in detail:
            mark = 'ok ' if row["correct"] else 'MISS'
            print(f'   {mark} label={row["label"]} predicted={row["predicted"]} '
                  f'confidence={row["confidence"]:.2f} :: {row["situation"]}', flush=True)

    score("outcome_achieved", OUTCOME_CASES)
    score("next_step_blocked", BLOCK_CASES)
    score("reference_ambiguous", AMBIGUOUS_CASES)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"report -> {args.out}")
    total = sum(q["correct"] for q in report["questions"].values())
    n = sum(q["cases"] for q in report["questions"].values())
    print(f"overall {total}/{n}")


if __name__ == "__main__":
    main()
