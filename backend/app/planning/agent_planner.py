from __future__ import annotations

import json
import os
import re
import time
import unicodedata
import numpy as np

from backend.app.models.schemas import AgentPlan
from backend.app.planning.language_fallback import arm_mentions, named_objects, parse_fallback
from backend.app.planning.task_program import GroundingError, TABLE_TARGETS, TaskProgram, compile_program
from backend.app.planning.spatial_language import validate_instruction_relations, placement_references, preservation_spans


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
        self.exact_grammar_fast_path = os.getenv("OPENARM_FAST_LANGUAGE", "0").strip() == "1"
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
            "Translate informal Japanese, English or mixed language into an ordered task program for the robot in world_state. "
            "Return JSON with actions and assumptions. Preserve EVERY task, its order, object, hand and destination. "
            "Operations: pick (grasp and hold), handover (giver arm + receiver), place (put on table), "
            "approach, move (direction + distance_cm), open, close, home, throw (upward cube toss + height_cm), "
            "pour (wine_bottle into wine_glass, volume_ml 5 to 35, default 20). "
            "For imported robots, read capabilities.skills and detected grippers; never invent an unavailable skill or gripper. "
            "When the user omits a hand, prefer the object's admitted_grasp_arm if present: its spawn approach passed IK. "
            "Operation joint uses joint=<exact URDF joint name>,position_deg=<requested rotary angle>. "
            "Honor negations and preservation: 'do not pass' forbids a handover; 'leave red where it is' is not a placement. "
            "A request to move a cube INTO a zone means place. A request just to pick means hold. "
            "Use only objects in world_state: red_cube, blue_cube, and when present wine_bottle, wine_glass. "
            "Destinations blue_zone, green_zone, left_side, right_side, center, wine_glass for pour. "
            "For a placement relative to a cube, target is that cube's ID and relation is one of "
            "on, left_of, right_of, front_of, behind, near. 'On', 'on top', 'stack', '上に乗せる', "
            "'重ねる' mean place with relation=on. A support/reference noun is NEVER another pick command. "
            "For 'blue on red', object=blue_cube,target=red_cube,relation=on. "
            "In '左側の赤いキューブの上', left describes red's location; the destination is ON red, not left_side. "
            "World +Y is left, -Y is right, +X is forward. Spatial relations use observed object positions. "
            "The glass is a fixed receiving vessel, not graspable. Throw only cubes, upward; default free-flight rise 3 cm. "
            "Putting wine into a glass means pour liquid, never place the bottle in the glass. Unspecified small pour means 20 mL. "
            "Pour automatically returns the bottle upright to its starting table position and releases it; no extra place action is needed for that. "
            "Object 'it' means the last mentioned object; held:left/held:right mean what that hand actually holds. "
            "Optional arm null means current holder or nearer arm. A hand can hold one cube. "
            "For handover, arm is the GIVER and receiver is the destination hand. Read hand_occupancy. "
            "If right holds blue and user says give it to left, arm MUST be right, receiver left. Never use the same arm twice. "
            "If another hand takes a held cube, use handover, never a second independent table pick. "
            "If destination is omitted on place, leave target null (the holder's side). "
            "Never invent objects, destinations, geometry, joint angles or successful outcomes. "
            "Return only operations requested in the CURRENT instruction. Do not add a second object's pick, "
            "an extra handover, or a destination absent from it. Resolve each clause independently, in order. "
            "A single place action automatically acquires its object; a handover automatically acquires if needed. "
            "If the FULL task is unsupported or ambiguous, return actions:[] and error with a short explanation; never guess a partial task."
        )
        # Setup diagnostics include many numerical IK candidates. They belong in the
        # evidence API, not the language context where they waste the model budget.
        language_state = {key: value for key, value in state.items() if key != 'spawn_validation'}
        user = {"instruction": instruction, "world_state": language_state, "relevant_experience": memory[-3:]}
        user["literal_spatial_constraints"] = [{"target": item["target"], "relation": item["relation"],
                                                "source": instruction[slice(*item["span"])]}
                                               for item in placement_references(instruction, state)]
        user["leave_in_place"] = [name for _, _, name in preservation_spans(instruction, state)]
        occupancy = {s: state.get("robot", {}).get(f"{s}_arm", {}).get("held_object") for s in ("left", "right")}
        user["hand_occupancy"] = occupancy
        held_objects = {obj for obj in occupancy.values() if obj}
        user["pronoun_it"] = state.get("discourse", {}).get("last_object") or (next(iter(held_objects)) if len(held_objects) == 1 else None)
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
            validate_instruction_relations(fallback, instruction, state)
        except (ValueError, KeyError) as exc:
            fallback_error = str(exc)
            if fallback_error.startswith("This scene has only"):
                raise PlannerProviderError("UNKNOWN_OBJECT", fallback_error) from None
        plan = None
        provider_error = None
        raw = None
        self._repair_feedback = None
        # The launcher can immediately execute a fully parsed/grounded request.
        # Unrecognised language still uses the model and the same validation.
        # This avoids a model round trip just to reconfirm an identical program.
        use_ai = self.enabled and not (fallback and (self.exact_grammar_fast_path or (
            len(fallback.program["actions"]) == 1 and fallback.goal["type"] == "direct_command")))
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
                    normalized = unicodedata.normalize("NFKC", instruction)
                    amounts = re.findall(r"(-?\d+(?:\.\d+)?)\s*(?:ml(?![a-z])|ミリリットル)", normalized, re.I)
                    pours = [a for a in program.actions if a.op == "pour"]
                    if amounts:
                        if len(amounts) != len(pours):
                            raise GroundingError("Preserve each explicitly requested mL amount as one ordered pour")
                        for action, amount in zip(pours, amounts):
                            action.volume_ml = float(amount)
                    if not arm_mentions(instruction):
                        # A language model's arbitrary default must not pick the
                        # far arm for the bottle. Let observed ownership/geometry
                        # choose when the human did not specify a hand.
                        for action in program.actions:
                            if action.op in {"pour", "place"} and action.object in {"wine_bottle", "it"}:
                                action.arm = None
                    if not re.search(r"[左右](?:手|腕|アーム)|[左右]で|\b(?:left|right)\s+(?:hand|arm)\b", instruction, re.I):
                        for action in program.actions:
                            if action.op == "place" and action.relation:
                                action.arm = None
                    # Only the grounding compiler can explain concrete choices. Model
                    # statements such as "blue is held" are not observed scene facts.
                    program.assumptions = []
                    # Revalidate anchored quantities, including the supported range.
                    program = TaskProgram.model_validate(program.model_dump())
                    candidate = compile_program(program, state)
                    validate_instruction_relations(candidate, instruction, state)
                    if fallback:
                        # When the explicit grammar understood the ENTIRE request, the model
                        # may not drop a handover, swap a colour, or change an explicit arm.
                        def signature(p):
                            # Compare physical intentions after compilation. 'Pick then
                            # hand over' and an auto-acquiring handover are equivalent;
                            # an already-held pick may also be an explicit no-op.
                            return [(s.skill, s.object, s.arm, s.receiver, s.target, s.direction, s.distance_cm,
                                     s.height_cm, s.volume_ml, s.relation, s.joint, s.position_deg) for s in p.steps]
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
            detail = str(fallback_error or provider_error)
            category = "ROBOT_CAPABILITY" if "ROBOT_CAPABILITY:" in detail else "INSTRUCTION_NOT_UNDERSTOOD"
            raise PlannerProviderError(category, f"Cannot ground the full instruction: {detail}")
        plan.provider_error = provider_error
        plan.raw_response = raw
        plan.model = (self.model if plan.provider == "ai&" else
                      self.local_model if plan.provider == "local_ollama" else None)
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
    object_skills = {"approach_object", "grasp_object", "lift_object", "release_object", "handover_object", "throw_object", "pour_liquid", "move_to_object"}
    held = observed_holds(state)
    approached = {}
    if not plan.steps or len(plan.steps) > 144:
        errors.append("Plan must contain 1–144 compiled skills")
    for i, step in enumerate(plan.steps):
        allowed = state.get("capabilities", {}).get("skills")
        if allowed is not None and step.skill not in allowed:
            errors.append(f"Selected robot does not support {step.skill}")
        if step.arm not in state.get("robot", {}).get("available_arms", ["left", "right"]):
            errors.append(f"Selected robot has no {step.arm} chain")
        if step.skill in object_skills and step.object not in objects:
            errors.append(f"Unknown object at step {i}: {step.object}")
        if step.skill == "move_to_target" and step.target not in targets:
            errors.append(f"Unknown target at step {i}: {step.target}")
        if step.skill == "move_to_object" and (step.target not in objects or step.target == step.object or not step.relation):
            errors.append(f"Invalid object-relative placement at step {i}")
        if step.skill == "handover_object" and (step.receiver not in {"left", "right"} or step.receiver == step.arm):
            errors.append(f"Handover needs two hands at step {i}")
        if step.skill == "move_relative" and (step.direction is None or step.distance_cm is None or not 1 <= step.distance_cm <= 30):
            errors.append(f"Invalid relative motion at step {i}")
        if step.skill == 'move_joint':
            specs = {j['name']: j for j in state.get('robot', {}).get('joints', [])}
            chain = state.get('robot', {}).get('chains', {}).get(step.arm, {}).get('joints', [])
            g = state.get('robot', {}).get('grippers', {}).get(step.arm, {})
            chain = [*chain, *([g['joint']] if g.get('joint') else [])]
            spec = specs.get(step.joint)
            if spec is None or step.joint not in chain or step.position_deg is None or not np.isfinite(step.position_deg):
                errors.append(f'Invalid imported joint at step {i}')
            elif spec['type'] == 'prismatic':
                errors.append(f'Joint {step.joint} is linear; degree commands are unsupported')
            elif spec.get('lower') is not None and not spec['lower'] <= np.deg2rad(step.position_deg) <= spec['upper']:
                errors.append(f'Joint limit exceeded at step {i}')
        if step.skill == "approach_object":
            if held[step.arm]:
                errors.append(f"Step {i}: {step.arm} hand is occupied")
            approached[step.arm] = step.object
        elif step.skill == "grasp_object":
            if held[step.arm] or approached.get(step.arm) != step.object:
                errors.append(f"Step {i}: grasp needs an empty hand aligned with the object")
            held[step.arm] = step.object
        elif step.skill in {"lift_object", "release_object", "handover_object", "throw_object", "pour_liquid"}:
            if held[step.arm] != step.object:
                errors.append(f"Step {i}: {step.arm} does not hold {step.object}")
            if step.skill in {"release_object", "throw_object", "pour_liquid"}:
                held[step.arm] = None
            elif step.skill == "handover_object" and step.receiver in held:
                if held[step.receiver]:
                    errors.append(f"Step {i}: receiving hand is occupied")
                held[step.receiver], held[step.arm] = step.object, None
        elif step.skill in {"move_to_target", "move_to_object"} and not held[step.arm]:
            errors.append(f"Step {i}: transport needs a physical hold")
        elif step.skill == "move_to_object" and held[step.arm] != step.object:
            errors.append(f"Step {i}: relative placement must move the actually held object")
        elif step.skill == "open_gripper":
            held[step.arm] = None
        elif step.skill == "return_home" and held[step.arm]:
            errors.append(f"Step {i}: place the held object before homing")
    return not errors, errors
