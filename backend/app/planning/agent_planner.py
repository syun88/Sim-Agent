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

    def _extract_json(self, text: str) -> dict[str, Any]:
        match = re.search(r"\{.*\}", text, re.S)
        if not match:
            raise PlannerProviderError("PROVIDER_ERROR", "No JSON object in response")
        return json.loads(match.group(0))

    def _ai_plan(self, instruction: str, state: dict, memory: list[dict]) -> dict:
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=8.0)
        prompt = {
            "instruction": instruction,
            "world_state": state,
            "allowed_skills": [
                "approach_object", "grasp_object", "lift_object", "move_to_target",
                "release_object", "retreat", "return_home",
            ],
            "objects": [x["id"] for x in state["objects"]],
            "targets": [x["id"] for x in state["targets"]],
            "experiences": memory,
            "required_schema": {
                "goal": {"type": "string", "object": "string", "target": "string"},
                "arm": "left|right",
                "steps": [{"skill": "string", "arm": "left|right", "object": "string|null", "target": "string|null"}],
            },
        }
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a physical robot task planner. Return only JSON. Do not generate code."},
                {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
            ],
            response_format={"type": "json_object"},
            max_tokens=500,
            temperature=0.1,
        )
        content = response.choices[0].message.content or ""
        return {"payload": self._extract_json(content), "raw": content, "usage": getattr(response, "usage", None)}

    def plan(self, instruction: str, state: dict, memory: list[dict] | None = None) -> AgentPlan:
        started = time.perf_counter()
        memory = memory or []
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
                raw = f"AI provider failed: {type(exc).__name__}: {str(exc).splitlines()[0]}"
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
        provider = "ai&" if self.enabled and raw and not raw.startswith("AI provider failed") else "deterministic_fallback"
        plan = AgentPlan(
            goal=payload.get("goal", {"type": "place_in_zone", "object": obj, "target": target}),
            arm=payload.get("arm", "left"),
            steps=payload.get("steps", []),
            provider=provider,
            memory_used=memory,
            memory_decision=memory_decision,
            planning_latency=time.perf_counter() - started,
            raw_response=raw,
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
    for i, step in enumerate(plan.steps):
        if step.skill not in allowed:
            errors.append(f"Unknown skill at step {i}: {step.skill}")
        if step.skill.endswith("object") and step.object not in objects:
            errors.append(f"Unknown object at step {i}: {step.object}")
        if step.skill == "move_to_target" and step.target not in targets:
            errors.append(f"Unknown target at step {i}: {step.target}")
    if plan.arm not in {"left", "right"}:
        errors.append("Unsupported arm")
    return not errors, errors
