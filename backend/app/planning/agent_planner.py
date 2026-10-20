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


def _credential(name: str) -> str:
    """An environment value that is really a credential.

    `.env.example` marks its placeholders with a `replace-` prefix. A verbatim copy of that
    file must behave like missing credentials, not like a configured provider.
    """
    value = os.getenv(name, "").strip()
    return "" if value.startswith("replace-") else value


class AgentPlanner:
    """LLM planner with deterministic OpenArm v2 skill fallback.

    `OPENARM_LOCAL_PLANNER=1` may point the OpenAI-compatible planner at a local Ollama
    model. This does not use or store a secret; Ollama's local endpoint accepts a dummy
    key. If planning fails, the labeled deterministic plan is used.
    """

    def __init__(self):
        self.api_key = _credential("AI_AND_API_KEY")
        self.base_url = _credential("AI_AND_BASE_URL")
        self.model = _credential("AI_AND_MODEL")
        self.local_model = os.getenv("OPENARM_OLLAMA_MODEL", "").strip()
        self.local_enabled = os.getenv("OPENARM_LOCAL_PLANNER", "").lower() in {"1", "true", "yes"} and bool(self.local_model)
        self.enabled = bool(self.api_key and self.base_url and self.model) or self.local_enabled

    def _extract_json(self, text: str) -> dict[str, Any]:
        match = re.search(r"\{.*\}", text, re.S)
        if not match:
            raise PlannerProviderError("PROVIDER_ERROR", "No JSON object in response")
        return json.loads(match.group(0))

    def _ai_plan(self, instruction: str, state: dict, memory: list[dict]) -> dict:
        from openai import OpenAI

        remote = bool(self.api_key and self.base_url and self.model)
        base_url = self.base_url if remote else "http://127.0.0.1:11434/v1"
        model = self.model if remote else self.local_model
        api_key = self.api_key if remote else "local"
        client = OpenAI(api_key=api_key, base_url=base_url, timeout=20.0)
        prompt = {
            "instruction": instruction,
            "world_state": state,
            "allowed_skills": [
                "approach_object", "grasp_object", "lift_object", "move_to_target",
                "release_object", "retreat", "return_home", "open_gripper",
                "close_gripper", "move_relative",
            ],
            "objects": [x["id"] for x in state["objects"]],
            "targets": [x["id"] for x in state["targets"]],
            "experiences": memory,
            "required_schema": {
                "goal": {"type": "string", "object": "string", "target": "string"},
                "arm": "left|right",
                "steps": [
                    {
                        "skill": "string",
                        "arm": "left|right",
                        "object": "string|null",
                        "target": "string|null",
                        "direction": "up|down|left|right|forward|back|null",
                        "distance_cm": "number|null",
                    }
                ],
            },
            "rules": [
                "Use only objects and targets present in world_state.",
                "For move_relative, distance_cm must be 1-30.",
                "A pick-and-place plan is approach_object, grasp_object, lift_object, move_to_target, release_object, retreat.",
            ],
        }
        response = client.chat.completions.create(
            model=model,
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
        objects = self._find_objects(instruction, state)
        obj = objects[0] if objects else None
        target = self._find_target(instruction, state, obj)
        arm = self._arm_for(instruction, obj)
        raw = None
        deterministic_payload = None
        deterministic_error = None
        try:
            deterministic_payload = self._deterministic_plan(instruction, state, obj, target, arm, objects)
        except PlannerProviderError as exc:
            deterministic_error = exc
        if any(item.get("success") for item in memory):
            memory_decision = "Related success found in memory; the grasp alignment itself is measured from the official model, so the plan is unchanged."
        elif memory:
            memory_decision = "Related failure found in memory; the grasp alignment itself is measured from the official model, so the plan is unchanged."
        else:
            memory_decision = "No related experience; the grasp alignment is measured from the official model."
        payload = None
        if self.enabled and not (deterministic_payload and deterministic_payload["goal"]["type"] == "direct_command"):
            try:
                ai = self._ai_plan(instruction, state, memory)
                payload = ai["payload"]
                raw = ai["raw"]
            except Exception as exc:
                raw = f"AI provider failed: {type(exc).__name__}: {str(exc).splitlines()[0]}"
        if payload is None and deterministic_payload is not None:
            payload = deterministic_payload
        if payload is None and deterministic_error is not None:
            raise deterministic_error
        using_remote = bool(self.api_key and self.base_url and self.model)
        provider = "deterministic_fallback"
        if raw and not raw.startswith("AI provider failed"):
            provider = "ai&" if using_remote else "local_ollama"
        return AgentPlan(
            goal=payload.get("goal", {"type": "place_in_zone", "object": obj, "target": target}),
            arm=payload.get("arm", "left"),
            steps=payload.get("steps", []),
            provider=provider,
            model=self.model if using_remote else (self.local_model or None),
            memory_used=memory,
            memory_decision=memory_decision,
            planning_latency=time.perf_counter() - started,
            raw_response=raw,
        )

    @staticmethod
    def _find_object(instruction: str, state: dict) -> str | None:
        objects = AgentPlanner._find_objects(instruction, state)
        return objects[0] if objects else None

    @staticmethod
    def _find_objects(instruction: str, state: dict) -> list[str]:
        lowered = instruction.lower()
        aliases = {
            "red_cube": (r"\bred cube\b", r"\bred\b(?!\s+(?:zone|target))", r"赤(?!い?ゾーン)"),
            "blue_cube": (r"\bblue cube\b", r"\bblue\b(?!\s+(?:zone|target))", r"青(?!い?ゾーン)"),
        }
        object_ids = {item["id"] for item in state["objects"]}
        candidates = []
        for name, values in aliases.items():
            if name not in object_ids:
                continue
            for pattern in values:
                match = re.search(pattern, lowered)
                if match:
                    candidates.append((match.start(), match.end(), name))
        ranked = sorted(candidates)
        return list(dict.fromkeys(item[2] for item in ranked))

    @staticmethod
    def _plural_requested(instruction: str) -> bool:
        return bool(re.search(
            r"\bboth\b|\ball\b|\b(?:1|2|two|three)\b|すべて|両方|2つ|3つ",
            instruction.lower(),
        ))

    @staticmethod
    def _find_target(instruction: str, state: dict, obj: str | None) -> str | None:
        lowered = instruction.lower()
        aliases = {
            "blue_zone": ("blue zone", "blue target", "青", "あお"),
            "green_zone": ("green zone", "green target", "緑", "みどり"),
        }
        target_ids = {item["id"] for item in state["targets"]}
        candidates = []
        for name, values in aliases.items():
            if name not in target_ids:
                continue
            for value in values:
                index = lowered.find(value)
                if index >= 0:
                    candidates.append((index, len(value), name))
        if not candidates:
            return None
        obj_color = obj.split("_")[0] if obj else None
        different = [item for item in candidates if item[2].split("_")[0] != obj_color]
        return min(different or candidates)[2]

    @staticmethod
    def _arm_for(instruction: str, obj: str | None) -> str:
        lowered = instruction.lower()
        if re.search(r"右アーム|右手|right arm|right hand", lowered):
            return "right"
        if re.search(r"左アーム|左手|left arm|left hand", lowered):
            return "left"
        return "right" if obj == "blue_cube" else "left"

    @staticmethod
    def _direction(instruction: str) -> str | None:
        lowered = instruction.lower()
        patterns = {
            "up": r"\bup\b|上に|上へ",
            "down": r"\bdown\b|下に|下へ",
            "right": r"\bright\b(?!\s+(?:arm|hand))|右に|右へ",
            "left": r"\bleft\b(?!\s+(?:arm|hand))|左に|左へ",
            "forward": r"\bforward\b|\bfront\b|前[にへ]|前方",
            "back": r"\bback(?:ward)?s?\b|後[ろにへ]|奥に",
        }
        for direction, pattern in patterns.items():
            if re.search(pattern, lowered):
                return direction
        return None

    @staticmethod
    def _distance_cm(instruction: str) -> float:
        match = re.search(r"(\d+(?:\.\d+)?)\s*(mm|ミリ|cm|センチ|m|メートル)?", instruction.lower())
        if not match:
            return 5.0
        value = float(match.group(1))
        unit = match.group(2)
        if unit in {"mm", "ミリ"}:
            return value / 10.0
        if unit in {"m", "メートル"}:
            return value * 100.0
        return value

    def _deterministic_plan(
        self,
        instruction: str,
        state: dict,
        obj: str | None,
        target: str | None,
        arm: str,
        objects: list[str] | None = None,
    ) -> dict[str, Any]:
        lowered = instruction.lower()
        objects = objects or ([obj] if obj else [])
        if self._plural_requested(lowered) and not objects:
            objects = [item["id"] for item in state["objects"]]
        has_gripper = bool(re.search(r"グリッパー|ハンド|手|gripper|hand", lowered))
        if has_gripper and re.search(r"開|open", lowered):
            return self._direct_plan(arm, "open_gripper")
        if has_gripper and re.search(r"閉|close", lowered):
            return self._direct_plan(arm, "close_gripper")
        if re.search(r"初期姿勢|ホーム|home", lowered):
            return self._direct_plan(arm, "return_home")

        direction = self._direction(instruction)
        distance = self._distance_cm(instruction)
        if obj and direction:
            return {
                "goal": {
                    "type": "direct_command", "action": "grasp_and_move", "object": obj,
                    "direction": direction, "distance_cm": str(distance), "arm": arm,
                },
                "arm": arm,
                "steps": [
                    {"skill": "approach_object", "arm": arm, "object": obj},
                    {"skill": "grasp_object", "arm": arm, "object": obj},
                    {"skill": "lift_object", "arm": arm, "object": obj},
                    {"skill": "move_relative", "arm": arm, "direction": direction, "distance_cm": distance},
                ],
            }
        if direction and re.search(r"アーム|腕|ハンド|手|arm|hand|gripper", lowered):
            return {
                "goal": {
                    "type": "direct_command", "action": "move_relative",
                    "direction": direction, "distance_cm": str(distance), "arm": arm,
                },
                "arm": arm,
                "steps": [{"skill": "move_relative", "arm": arm, "direction": direction, "distance_cm": distance}],
            }
        if len(objects) > 1:
            steps = []
            for object_id in objects:
                object_arm = self._arm_for("", object_id)
                steps.extend([
                    {"skill": "approach_object", "arm": object_arm, "object": object_id},
                    {"skill": "grasp_object", "arm": object_arm, "object": object_id},
                    {"skill": "lift_object", "arm": object_arm, "object": object_id},
                ])
                if target:
                    steps.extend([
                        {"skill": "move_to_target", "arm": object_arm, "target": target},
                        {"skill": "release_object", "arm": object_arm, "object": object_id},
                        {"skill": "retreat", "arm": object_arm},
                    ])
            action = "multi_place" if target else "multi_pick"
            goal: dict[str, Any] = {"type": "direct_command", "action": action, "objects": objects}
            if target:
                goal["target"] = target
            return {"goal": goal, "arm": arm, "steps": steps}
        if obj and target and self._already_at_target(state, obj, target):
            return {
                "goal": {
                    "type": "already_at_target", "object": obj, "target": target, "arm": arm,
                },
                "arm": arm,
                "steps": [{"skill": "return_home", "arm": arm}],
            }
        if obj and re.search(r"近づ|アプローチ|approach", lowered):
            return {
                "goal": {"type": "direct_command", "action": "approach", "object": obj, "arm": arm},
                "arm": arm,
                "steps": [{"skill": "approach_object", "arm": arm, "object": obj}],
            }
        if obj and re.search(r"掴|つか|握|grasp|pick", lowered):
            return {
                "goal": {"type": "direct_command", "action": "grasp", "object": obj, "arm": arm},
                "arm": arm,
                "steps": [
                    {"skill": "approach_object", "arm": arm, "object": obj},
                    {"skill": "grasp_object", "arm": arm, "object": obj},
                    {"skill": "lift_object", "arm": arm, "object": obj},
                ],
            }
        if obj is None:
            raise PlannerProviderError("UNKNOWN_OBJECT", "No known object named in instruction")
        if target is None:
            target = "blue_zone"
        return {
            "goal": {"type": "place_in_zone", "object": obj, "target": target},
            "arm": arm,
            "steps": [
                {"skill": "approach_object", "arm": arm, "object": obj},
                {"skill": "grasp_object", "arm": arm, "object": obj},
                {"skill": "lift_object", "arm": arm, "object": obj},
                {"skill": "move_to_target", "arm": arm, "target": target},
                {"skill": "release_object", "arm": arm, "object": obj},
                {"skill": "retreat", "arm": arm},
            ],
        }

    @staticmethod
    def _already_at_target(state: dict, obj: str, target: str) -> bool:
        object_position = next((item.get("position") for item in state.get("objects", []) if item.get("id") == obj), None)
        target_position = next((item.get("position") for item in state.get("targets", []) if item.get("id") == target), None)
        if not object_position or not target_position:
            return False
        horizontal = sum((float(object_position[axis]) - float(target_position[axis])) ** 2 for axis in (0, 1)) ** 0.5
        vertical = abs(float(object_position[2]) - float(target_position[2]))
        return horizontal <= 0.04 and vertical <= 0.05

    @staticmethod
    def _direct_plan(arm: str, action: str) -> dict[str, Any]:
        return {
            "goal": {"type": "direct_command", "action": action, "arm": arm},
            "arm": arm,
            "steps": [{"skill": action, "arm": arm}],
        }


def validate_plan(plan: AgentPlan, state: dict) -> tuple[bool, list[str]]:
    errors: list[str] = []
    objects = {item["id"] for item in state["objects"]}
    targets = {item["id"] for item in state["targets"]}
    allowed = {
        "approach_object", "grasp_object", "lift_object", "move_to_target",
        "release_object", "retreat", "return_home", "open_gripper",
        "close_gripper", "move_relative",
    }
    object_skills = {"approach_object", "grasp_object", "lift_object", "release_object"}
    if len(plan.steps) > 20:
        errors.append("Plan exceeds maximum length")
    if not plan.steps:
        errors.append("Plan is empty")
    for i, step in enumerate(plan.steps):
        if step.skill not in allowed:
            errors.append(f"Unknown skill at step {i}: {step.skill}")
        if step.skill in object_skills and step.object not in objects:
            errors.append(f"Unknown object at step {i}: {step.object}")
        if step.skill == "move_to_target" and step.target not in targets:
            errors.append(f"Unknown target at step {i}: {step.target}")
        if step.skill == "move_relative":
            if step.direction not in {"up", "down", "left", "right", "forward", "back"}:
                errors.append(f"Invalid move direction at step {i}: {step.direction}")
            if step.distance_cm is None or not 1 <= step.distance_cm <= 30:
                errors.append(f"Invalid move distance at step {i}: {step.distance_cm}")
    if plan.arm not in {"left", "right"}:
        errors.append("Unsupported arm")
    return not errors, errors
