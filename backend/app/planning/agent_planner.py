from __future__ import annotations

import json
import os
import re
import time
from typing import Any

from backend.app.models.schemas import AgentPlan


class PlannerProviderError(Exception):
    def __init__(self, category: str, message: str):
        super().__init__(message)
        self.category = category


class AgentPlanner:
    """LLM planner with a deterministic OpenArm v2 skill fallback."""

    def __init__(self):
        self.api_key = os.getenv("AI_AND_API_KEY", "")
        self.base_url = os.getenv("AI_AND_BASE_URL", "")
        self.model = os.getenv("AI_AND_MODEL", "")
        self.enabled = bool(self.api_key and self.base_url and self.model)
        self.last_error: str | None = None

    def _extract_json(self, text: str) -> dict[str, Any]:
        match = re.search(r"\{.*\}", text, re.S)
        if not match:
            raise PlannerProviderError("PROVIDER_ERROR", "No JSON object in response")
        return json.loads(match.group(0))

    def _ai_plan(self, instruction: str, state: dict, memory: list[dict]) -> dict:
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=10.0)
        object_summary = [
            {"id": item["id"], "position": [round(v, 4) for v in item["position"]]}
            for item in state["objects"]
        ]
        target_summary = [{"id": item["id"]} for item in state["targets"]]
        memory_summary = [
            {
                "instruction": item.get("instruction"),
                "success": item.get("success"),
                "failure_category": item.get("failure_category"),
            }
            for item in memory[-3:]
        ]
        prompt = {
            "instruction": instruction,
            "objects": object_summary,
            "targets": target_summary,
            "past_experiences": memory_summary,
            "required_schema": {
                "goal": {"type": "place_in_zone", "object": "<object_id>", "target": "<target_id>"},
                "arm": "left|right",
                "steps": [
                    {"skill": "<skill_name>", "arm": "<arm>", "object": "<object_id|null>", "target": "<target_id|null>"}
                ],
            },
            "allowed_skills": {
                "approach_object": {"requires": "object", "moves": "gripper to a grasp approach pose near the object"},
                "grasp_object": {"requires": "object", "closes": "the gripper and starts the grasp hold"},
                "lift_object": {"requires": "object", "raises": "the gripper by 10 cm"},
                "move_to_target": {"requires": "target", "moves": "the gripper above the target zone"},
                "release_object": {"requires": "object", "opens": "the gripper and releases the grasp hold"},
                "retreat": {"requires": None, "moves": "the gripper back and up"},
                "return_home": {"requires": None, "moves": "the arm to its home pose"},
            },
            "rules": [
                "Use the exact object_id and target_id from this prompt.",
                "Plan a pick and place: approach → grasp → lift → move_to_target → release → retreat.",
                "Do not invent skills or object names.",
                "Return only JSON matching required_schema. No code, no markdown.",
            ],
        }
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a physical robot task planner for an OpenArm v2 bimanual arm. "
                        "You must return only valid JSON matching required_schema. Do not generate code, markdown, or explanations."
                    ),
                },
                {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
            ],
            response_format={"type": "json_object"},
            max_tokens=800,
            temperature=0.1,
        )
        content = response.choices[0].message.content or ""
        return {"payload": self._extract_json(content), "raw": content, "usage": getattr(response, "usage", None)}

    def plan(self, instruction: str, state: dict, memory: list[dict] | None = None) -> AgentPlan:
        started = time.perf_counter()
        memory = memory or []
        self.last_error = None
        objects = {item["id"] for item in state["objects"]}
        targets = {item["id"] for item in state["targets"]}
        lowered = instruction.lower()
        object_candidates = []
        for name in objects:
            color = name.split("_")[0]
            index = lowered.find(color)
            if index >= 0:
                object_candidates.append((index, name))
        obj = min(object_candidates)[1] if object_candidates else None
        target_candidates = []
        for name in targets:
            color = name.split("_")[0]
            index = lowered.find(color)
            if index >= 0:
                target_candidates.append((index, name))
        non_object_targets = [candidate for candidate in target_candidates if candidate[1].split("_")[0] != obj.split("_")[0]]
        target = min(non_object_targets or target_candidates)[1] if target_candidates else None
        if obj is None:
            raise PlannerProviderError("UNKNOWN_OBJECT", "No known cube named in instruction")
        if target is None:
            target = "blue_zone"
        arm = "right" if obj == "blue_cube" else "left"
        raw = None
        usage = None
        if any(item.get("success") for item in memory):
            approach_offset = 0.06
            memory_decision = "Reusing the remembered successful approach offset (60 mm)."
        elif memory:
            approach_offset = 0.06
            memory_decision = "Avoiding the previously recorded failure; using the safe approach offset."
        else:
            approach_offset = 0.055
            memory_decision = "No related experience; using the exploratory approach offset."
        if self.enabled:
            try:
                ai = self._ai_plan(instruction, state, memory)
                payload = ai["payload"]
                raw = ai["raw"]
                usage = {"input": getattr(ai.get("usage"), "prompt_tokens", None), "output": getattr(ai.get("usage"), "completion_tokens", None)}
            except Exception as exc:
                error_detail = f"{type(exc).__name__}: {str(exc).splitlines()[0]}"
                self.last_error = error_detail
                raw = f"AI provider failed: {error_detail}"
        else:
            payload = {
                "goal": {"type": "place_in_zone", "object": obj, "target": target},
                "arm": arm,
                "steps": [
                    {"skill": "approach_object", "arm": arm, "object": obj, "approach_offset": approach_offset},
                    {"skill": "grasp_object", "arm": arm, "object": obj},
                    {"skill": "lift_object", "arm": arm, "object": obj},
                    {"skill": "move_to_target", "arm": arm, "target": target},
                    {"skill": "release_object", "arm": arm, "object": obj},
                    {"skill": "retreat", "arm": arm},
                ],
            }
        ai_succeeded = self.enabled and raw is not None and not raw.startswith("AI provider failed")
        provider = self.model if ai_succeeded else "deterministic_fallback"
        plan = AgentPlan(
            goal=payload.get("goal", {"type": "place_in_zone", "object": obj, "target": target}),
            arm=payload.get("arm", "left"),
            steps=payload.get("steps", []),
            provider=provider,
            memory_used=memory,
            memory_decision=memory_decision,
            planning_latency=time.perf_counter() - started,
            raw_response=raw,
            model=self.model if ai_succeeded else None,
            provider_error=self.last_error,
        )
        return plan


def validate_plan(plan: AgentPlan, state: dict) -> tuple[bool, list[str]]:
    errors: list[str] = []
    objects = {item["id"] for item in state["objects"]}
    targets = {item["id"] for item in state["targets"]}
    allowed = {"approach_object", "grasp_object", "lift_object", "move_to_target", "release_object", "retreat", "return_home"}
    if len(plan.steps) > 20:
        errors.append("Plan exceeds maximum length")
    if not plan.steps:
        errors.append("Plan is empty")
    if not plan.goal.get("object") or plan.goal.get("object") not in objects:
        errors.append(f"Unknown goal object: {plan.goal.get('object')}")
    if not plan.goal.get("target") or plan.goal.get("target") not in targets:
        errors.append(f"Unknown goal target: {plan.goal.get('target')}")
    if plan.goal.get("type") != "place_in_zone":
        errors.append(f"Unsupported goal type: {plan.goal.get('type')}")
    for i, step in enumerate(plan.steps):
        if step.skill not in allowed:
            errors.append(f"Unknown skill at step {i}: {step.skill}")
        if step.skill in {"approach_object", "grasp_object", "lift_object", "release_object"}:
            if step.object not in objects:
                errors.append(f"Step {i} ({step.skill}) requires a known object, got: {step.object}")
        if step.skill == "move_to_target":
            if step.target not in targets:
                errors.append(f"Step {i} (move_to_target) requires a known target, got: {step.target}")
        if step.arm not in {"left", "right"}:
            errors.append(f"Step {i} has unsupported arm: {step.arm}")
    skills = [step.skill for step in plan.steps]
    expected = ["approach_object", "grasp_object", "lift_object", "move_to_target", "release_object"]
    for required in expected:
        if required not in skills:
            errors.append(f"Plan is missing required skill: {required}")
    if "grasp_object" in skills and "release_object" in skills:
        if skills.index("grasp_object") >= skills.index("release_object"):
            errors.append("grasp_object must precede release_object")
    if plan.arm not in {"left", "right"}:
        errors.append("Unsupported arm")
    return not errors, errors
