from __future__ import annotations

import json
import os
import re
import time

from backend.app.models.schemas import AgentPlan
from backend.app.planning.language_fallback import named_objects, parse_fallback
from backend.app.planning.task_program import GroundingError, TABLE_TARGETS, TaskProgram, compile_program


class PlannerProviderError(Exception):
    def __init__(self, category: str, message: str):
        super().__init__(message)
        self.category = category


def _credential(name: str) -> str:
    value = os.getenv(name, "").strip()
    return "" if value.startswith("replace-") else value


class AgentPlanner:
    """Language -> typed intentions -> grounded/validated skills, without fine tuning.

    Both providers use the same task language. Unsupported language is an explicit failure,
    rather than an unrelated default pick-and-place. Provider output never drives motors.
    """

    def __init__(self):
        self.api_key = _credential("AI_AND_API_KEY")
        self.base_url = _credential("AI_AND_BASE_URL")
        self.model = _credential("AI_AND_MODEL")
        self.local_model = os.getenv("OPENARM_OLLAMA_MODEL", "qwen3-vl:4b").strip()
        self.local_enabled = os.getenv("OPENARM_LOCAL_PLANNER", "").lower() in {"1", "true", "yes"} and bool(self.local_model)
        self.enabled = bool(self.api_key and self.base_url and self.model) or self.local_enabled
        self._repair_feedback: str | None = None
        self._ollama_raw_chatml: bool | None = None

    @property
    def configured_provider(self) -> str:
        if self.api_key and self.base_url and self.model:
            return "ai&"
        return "local_ollama" if self.local_enabled else "deterministic_fallback"

    def _extract_json(self, text: str) -> dict:
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
        match = re.search(r"\{.*\}", text, re.S)
        if not match:
            raise PlannerProviderError("PROVIDER_ERROR", "No JSON object in response")
        return json.loads(match.group())

    def _ai_plan(self, instruction: str, state: dict, memory: list[dict]) -> dict:
        system = (
            "Translate informal Japanese, English or mixed language into an ordered OpenArm task program. "
            "Return JSON with actions and assumptions. Preserve EVERY task, its order, object, hand and destination. "
            "Operations: pick (grasp and hold), handover (giver arm + receiver), place (put on table), "
            "approach, move (direction + distance_cm), open, close, home. "
            "A request to move a cube INTO a zone means place. A request just to pick means hold. "
            "Use red_cube and blue_cube; destinations blue_zone, green_zone, left_side, right_side, center. "
            "Object 'it' means the last mentioned object; held:left/held:right mean what that hand actually holds. "
            "Optional arm null means current holder or nearer arm. A hand can hold one cube. "
            "For handover, arm is the GIVER and receiver is the destination hand. Read hand_occupancy. "
            "If right holds blue and user says give it to left, arm MUST be right, receiver left. Never use the same arm twice. "
            "If another hand takes a held cube, use handover, never a second independent table pick. "
            "If destination is omitted on place, leave target null (the holder's side). "
            "Never invent objects, destinations, geometry, joint angles or successful outcomes. "
            "If the FULL task is unsupported or ambiguous, return actions:[] and error with a short explanation; never guess a partial task. "
            "Example: 右手で青を取って左に渡して置いて、左で赤を取り右へ渡して右側に置く => "
            '{"actions":[{"op":"pick","object":"blue_cube","arm":"right"},'
            '{"op":"handover","object":"it","arm":"right","receiver":"left"},'
            '{"op":"place","object":"it","arm":"left","target":"left_side"},'
            '{"op":"pick","object":"red_cube","arm":"left"},'
            '{"op":"handover","object":"it","arm":"left","receiver":"right"},'
            '{"op":"place","object":"it","arm":"right","target":"right_side"}],"assumptions":[]}'
        )
        user = {"instruction": instruction, "world_state": state, "relevant_experience": memory[-3:]}
        occupancy = {s: state.get("robot", {}).get(f"{s}_arm", {}).get("held_object") for s in ("left", "right")}
        user["hand_occupancy"] = occupancy
        held_objects = {obj for obj in occupancy.values() if obj}
        user["pronoun_it"] = state.get("discourse", {}).get("last_object") or (next(iter(held_objects)) if len(held_objects) == 1 else None)
        if self.configured_provider == "ai&":
            user["output_schema"] = TaskProgram.model_json_schema()
        if self._repair_feedback:
            user["previous_validation_error"] = self._repair_feedback
        messages = [{"role": "system", "content": system}, {"role": "user", "content": json.dumps(user, ensure_ascii=False)}]
        if self.configured_provider == "local_ollama":
            # Ollama's native schema decoder constrains the entire program, rather than just
            # asking a small local model to produce something JSON-shaped.
            import httpx
            if self._ollama_raw_chatml is None:
                info = httpx.post("http://127.0.0.1:11434/api/show", json={"model": self.local_model}, timeout=5).json()
                # Imported Qwen GGUFs can have '{{ .Prompt }}' as their Ollama template.
                # /api/chat then omits the role delimiters and ignores think=false;
                # the installed model spent its entire output budget in thinking. Supply
                # Qwen's role format only for this positively identified template defect.
                self._ollama_raw_chatml = (info.get("model_info", {}).get("general.architecture") == "qwen3vl"
                                           and info.get("template", "").strip() == "{{ .Prompt }}")
            request = {"model": self.local_model, "stream": False,
                       "format": TaskProgram.model_json_schema(),
                       "options": {"temperature": 0, "num_predict": 1600, "num_ctx": 8192,
                                   "stop": ["<|im_end|>", "<|endoftext|>"]}}
            if self._ollama_raw_chatml:
                request.update(raw=True, prompt="".join(
                    f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages
                ) + "<|im_start|>assistant\n<think>\n\n</think>\n")
                endpoint = "generate"
            else:
                request.update(messages=messages, think=False)
                endpoint = "chat"
            response = httpx.post(f"http://127.0.0.1:11434/api/{endpoint}", timeout=60.0, json=request)
            response.raise_for_status()
            content = response.json()["response"] if self._ollama_raw_chatml else response.json()["message"]["content"]
        else:
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=35.0, max_retries=0)
            response = client.chat.completions.create(
                model=self.model, messages=messages, response_format={"type": "json_object"},
                max_tokens=1800, temperature=0,
            )
            content = response.choices[0].message.content or ""
        return {"payload": self._extract_json(content), "raw": content}

    def plan(self, instruction: str, state: dict, memory: list[dict] | None = None) -> AgentPlan:
        started = time.perf_counter()
        memory = memory or []
        fallback = None
        fallback_error = None
        try:
            fallback = compile_program(parse_fallback(instruction, state), state)
        except (ValueError, KeyError) as exc:
            fallback_error = str(exc)
            if fallback_error.startswith("This scene has only"):
                raise PlannerProviderError("UNKNOWN_OBJECT", fallback_error) from None
        plan = None
        provider_error = None
        raw = None
        self._repair_feedback = None
        # Single direct commands have an exact grammar. Multi-action instructions and
        # unrecognised language use the language model, then the same deterministic compiler.
        use_ai = self.enabled and not (fallback and len(fallback.program["actions"]) == 1 and fallback.goal["type"] == "direct_command")
        if use_ai:
            for _ in range(2):
                try:
                    response = self._ai_plan(instruction, state, memory)
                    program = TaskProgram.model_validate(response["payload"])
                    if program.error:
                        if fallback:
                            provider_error = "PROVIDER_UNSUPPORTED_LANGUAGE"
                            break
                        raise PlannerProviderError("UNSUPPORTED_TASK", program.error)
                    candidate = compile_program(program, state)
                    if fallback:
                        # When the explicit grammar understood the ENTIRE request, the model
                        # may not drop a handover, swap a colour, or change an explicit arm.
                        def signature(p):
                            # Compare physical intentions after compilation. 'Pick then
                            # hand over' and an auto-acquiring handover are equivalent;
                            # an already-held pick may also be an explicit no-op.
                            return [(s.skill, s.object, s.arm, s.receiver, s.target, s.direction, s.distance_cm) for s in p.steps]
                        if signature(candidate) != signature(fallback):
                            raise GroundingError("Program changes explicit actions/hands in instruction. Expected: " + json.dumps(fallback.program, ensure_ascii=False))
                    valid, errors = validate_plan(candidate, state)
                    if not valid:
                        raise GroundingError("; ".join(errors))
                    plan = candidate
                    plan.provider = self.configured_provider
                    provider_error = None
                    raw = response["raw"]
                    break
                except (ValueError, KeyError) as exc:
                    self._repair_feedback = str(exc)[:3000]
                    if fallback:
                        self._repair_feedback += "\nThe complete explicit instruction grounds to: " + json.dumps(fallback.program, ensure_ascii=False)
                    provider_error = "PROVIDER_INVALID_PROGRAM"
                except PlannerProviderError as exc:
                    if exc.category == "UNSUPPORTED_TASK":
                        raise
                    provider_error = "PROVIDER_UNAVAILABLE"
                    break
                except Exception:
                    # Provider exception strings can include URLs, request data or secrets.
                    provider_error = "PROVIDER_UNAVAILABLE"
                    break
            if plan is None:
                raw = f"AI provider failed: {provider_error}"
        if plan is None:
            plan = fallback
        if plan is None:
            raise PlannerProviderError("INSTRUCTION_NOT_UNDERSTOOD", f"Cannot safely ground the full instruction: {fallback_error or provider_error}")
        plan.provider_error = provider_error
        plan.raw_response = raw
        plan.model = self.model if self.configured_provider == "ai&" else self.local_model if self.local_enabled else None
        plan.memory_used = memory[-3:]
        plan.memory_decision = "Use live geometry and contacts; past experience never overrides physical preconditions."
        plan.planning_latency = time.perf_counter() - started
        # Keep the established idempotent place behaviour, but verify it physically at execution.
        if plan.goal["type"] == "place_in_zone" and self._already_at_target(state, plan.goal["object"], plan.goal["target"]):
            from backend.app.models.schemas import PlanStep
            plan.goal["type"] = "already_at_target"
            plan.steps = [PlanStep(skill="return_home", arm=plan.arm)]
        return plan

    @staticmethod
    def _find_objects(instruction: str, state: dict) -> list[str]:
        return named_objects(instruction, state)

    @staticmethod
    def _find_object(instruction: str, state: dict) -> str | None:
        return next(iter(named_objects(instruction, state)), None)

    @staticmethod
    def _already_at_target(state: dict, obj: str, target: str) -> bool:
        a = next((x.get("position") for x in state["objects"] if x["id"] == obj), None)
        b = next((x.get("position") for x in state["targets"] if x["id"] == target), None)
        if a is None or b is None:
            return False
        return sum((a[i] - b[i]) ** 2 for i in (0, 1)) ** .5 <= .04 and abs(a[2] - b[2]) <= .05 and not any(state.get("robot", {}).get(f"{s}_arm", {}).get("held_object") == obj for s in ("left", "right"))


def validate_plan(plan: AgentPlan, state: dict) -> tuple[bool, list[str]]:
    from backend.app.planning.task_program import observed_holds
    errors = []
    objects = {x["id"] for x in state["objects"]}
    targets = {x["id"] for x in state["targets"]} | set(TABLE_TARGETS)
    object_skills = {"approach_object", "grasp_object", "lift_object", "release_object", "handover_object"}
    held = observed_holds(state)
    approached = {}
    if not plan.steps or len(plan.steps) > 144:
        errors.append("Plan must contain 1–144 compiled skills")
    for i, step in enumerate(plan.steps):
        if step.skill in object_skills and step.object not in objects:
            errors.append(f"Unknown object at step {i}: {step.object}")
        if step.skill == "move_to_target" and step.target not in targets:
            errors.append(f"Unknown target at step {i}: {step.target}")
        if step.skill == "handover_object" and (step.receiver not in {"left", "right"} or step.receiver == step.arm):
            errors.append(f"Handover needs two hands at step {i}")
        if step.skill == "move_relative" and (step.direction is None or step.distance_cm is None or not 1 <= step.distance_cm <= 30):
            errors.append(f"Invalid relative motion at step {i}")
        if step.skill == "approach_object":
            if held[step.arm]:
                errors.append(f"Step {i}: {step.arm} hand is occupied")
            approached[step.arm] = step.object
        elif step.skill == "grasp_object":
            if held[step.arm] or approached.get(step.arm) != step.object:
                errors.append(f"Step {i}: grasp needs an empty hand aligned with the object")
            held[step.arm] = step.object
        elif step.skill in {"lift_object", "release_object", "handover_object"}:
            if held[step.arm] != step.object:
                errors.append(f"Step {i}: {step.arm} does not hold {step.object}")
            if step.skill == "release_object":
                held[step.arm] = None
            elif step.skill == "handover_object" and step.receiver in held:
                if held[step.receiver]:
                    errors.append(f"Step {i}: receiving hand is occupied")
                held[step.receiver], held[step.arm] = step.object, None
        elif step.skill == "move_to_target" and not held[step.arm]:
            errors.append(f"Step {i}: transport needs a physical hold")
        elif step.skill == "open_gripper":
            held[step.arm] = None
        elif step.skill == "return_home" and held[step.arm]:
            errors.append(f"Step {i}: place the held object before homing")
    return not errors, errors
