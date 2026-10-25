from __future__ import annotations

import json
import os
import re
import time
import unicodedata
import numpy as np

from backend.app.models.schemas import AgentPlan
from backend.app.planning.language_fallback import VERBS, arm_mentions, motion_distance_cm, named_objects, parse_fallback
from backend.app.planning.task_program import ActionProposal, GroundingError, InstructionMeaning, TABLE_TARGETS, TaskProgram, compile_program
from backend.app.planning.spatial_language import validate_instruction_relations, placement_references, preservation_spans, strip_preservation
from backend.app.planning.goal_language import holding_conditions, relative_conditions, unsupported_group_request
from backend.app.planning.language_schema import meaning_schema
from backend.app.planning.preparation import compile_prepared_program
from backend.app.agent.scene_reasoning import SceneReferenceError, bind_scene_references


class PlannerProviderError(Exception):
    def __init__(self, category: str, message: str):
        super().__init__(message)
        self.category = category


def _credential(name: str) -> str:
    value = os.getenv(name, "").strip()
    return "" if value.startswith("replace-") else value


#: What a re-planned continuation must carry over from the plan it continues. A recovery that rebuilds
#: the steps through ``compile_prepared_program`` constructs a fresh ``AgentPlan``, so without this the
#: ledger of a model-planned task reads as "no model was consulted" the moment it recovers - measured
#: 2026-10-25, where the top-level plan of a task the model had planned showed ``model_called=False``
#: while the attempt's own plan showed ``True``. One list, so a new provenance field cannot be
#: half-wired.
PROVENANCE_FIELDS = ('provider', 'model', 'provider_error', 'model_called', 'skipped_because',
                     'capability_mode', 'fallback_verdict', 'model_verdict', 'agree', 'shadow')


#: Words by which a model states that something *happened*, rather than what it decided. An
#: assumption is a recorded choice; an outcome claim is unobserved evidence and is discarded.
_OUTCOME_WORDS = re.compile(r'\b(?:success(?:ful(?:ly)?)?|succeeded|completed?|accomplished|verified|'
                            r'achieved|done|finished)\b|成功|完了|できました|済み', re.I)


def _is_choice_not_outcome(text: str) -> bool:
    """Keep an assumption that records a decision; drop one that claims an observed result."""
    return not _OUTCOME_WORDS.search(text)


def program_signature(plan) -> list:
    """The physical intentions a compiled program carries, for comparing two of them.

    After compilation: 'pick then hand over' and an auto-acquiring handover are equivalent, and an
    already-held pick may be an explicit no-op, so raw action text cannot be compared directly.
    """
    return [(s.skill, s.object, s.arm, s.receiver, s.target, s.direction, s.distance_cm,
             s.height_cm, s.volume_ml, s.relation, s.joint, s.position_deg) for s in plan.steps]


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
        # Capability mode stops the planner from answering out of the deterministic grammar: the model is
        # asked even when the sentence parses, and the rules are demoted to a labelled fallback. Shadow
        # mode asks the model anyway after the deterministic path has answered, and records only.
        self.capability_mode = os.getenv("OPENARM_CAPABILITY_MODE", "0").strip() == "1"
        self.shadow_mode = os.getenv("OPENARM_SHADOW_PLAN", "0").strip() == "1"
        self._repair_feedback: str | None = None
        self._ollama_raw_chatml: bool | None = None
        self._interpret_meaning = False

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
            "Return one JSON envelope with mode, items, assumptions and error. Never return separate actions/goals arrays or keep_in_place. "
            "Mode actions: items are ordered TaskAction objects with op. "
            "Mode error: items is empty and error explains why. Preserve EVERY requested task, order, object, hand and destination. "
            "Use mode actions for the complete requested sequence. Preservation is checked independently from source text and physical state. "
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
            "Refuse with mode error ONLY when the requested action is physically unsupported: an object "
            "not in world_state, a skill this robot does not have, or impossible geometry. Never refuse "
            "because the instruction is vague. If objects, hands or destinations are unspecified but the "
            "observed scene admits a sensible interpretation, choose concrete ones from world_state and "
            "record EVERY choice you made in assumptions, one short Japanese line each (which object, "
            "which destination, which hand, and why the scene admits it). 'Tidy up' means place each "
            "graspable cube on a distinct free destination. An assumption on the record is required; a "
            "silent guess is not allowed, and neither is inventing an object that is not observed."
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
        schema = ActionProposal.model_json_schema()
        if self._interpret_meaning:
            # Do not put action examples/recipes in this context: a small model
            # otherwise copies two impossible handovers instead of stating the
            # desired ownership. Planning the preparation is the solver's job.
            system = (
                'Read the instruction and identify its meaning, without proposing any robot procedure. '
                'Return interpretation final_state if the user describes how objects should end up, '
                'an arrangement, ownership, or asks to exchange what the hands contain without prescribing steps. '
                'For final_state, express only desired predicates: holding(object,arm), placement(object,target), '
                'relative_placement(object,target,relation,distance_cm optional). Relations on,left_of,right_of,front_of,behind,near. '
                'On another cube ALWAYS means type relative_placement, target that cube ID and relation on, never placement in a coloured zone. '
                'required_relative_conditions gives literal object/reference bindings: preserve them; a support is not another goal subject. '
                'A request to exchange held cubes means each ends in the opposite hand. Read actual hand_occupancy. '
                'Observed hand_occupancy describes the CURRENT state, never the requested final state. '
                'required_final_holders gives literal bindings from the instruction; preserve them exactly. '
                'Never try to describe temporary steps. The solver can free occupied hands using table buffers. '
                'Use interpretation ordered_actions with empty goals when the user prescribes a temporal sequence, '
                'an explicit manipulation procedure, or asks for a primitive pick, grasp, approach, movement, throw, pour, open, close or joint angle. '
                'A requested final holder is not a prescribed manipulation procedure. Arm belongs only to holding goals, not placement goals. '
                'Do not reduce requested ordered actions to their final states. '
                'keep_in_place lists objects explicitly required to stay where they are. '
                'Return the minimum requested goals. Registered objects are available references, not a list of tasks. '
                'Never add a holding or placement goal for an object the user did not ask to manipulate, EXCEPT when '
                'the user asks for the scene to be arranged, tidied or cleaned: then include each graspable object and '
                'record the choice. '
                'A negative clause adds a constraint, never a positive goal. '
                'Use only registered objects and targets. A reference cube is a target, not another manipulated subject. '
                'An unspecified but feasible outcome is NOT unsupported. If the instruction does not say which objects, '
                'which destinations or which hands, choose concrete ones from objects and targets (for tidying, one '
                'distinct free target per graspable object) and write EVERY choice into assumptions, one short Japanese '
                'line each. Unsupported means the physical action cannot be done by this robot: an unknown object, a '
                'skill it does not have, or impossible geometry - never mere vagueness. '
                'Return JSON matching the supplied schema, no invented object poses or successful outcomes.'
            )
            user = {'instruction': strip_preservation(instruction,state), 'objects': [
                {key: obj[key] for key in ('id', 'type', 'graspable', 'position') if key in obj}
                for obj in state['objects']], 'targets': state['targets'], 'hand_occupancy': occupancy,
                'available_arms': state.get('robot', {}).get('available_arms', ['left', 'right']),
                'discourse': state.get('discourse', {}), 'literal_spatial_constraints': user['literal_spatial_constraints'],
                'leave_in_place': user['leave_in_place']}
            user['coordinate_frame'] = '+Y is left, -Y is right, +X is forward. Object location is not the acting hand.'
            schema = meaning_schema(state)
        user["output_schema"] = schema
        user['required_final_holders'] = holding_conditions(instruction, state)
        user['required_relative_conditions'] = relative_conditions(instruction, state)
        if self._interpret_meaning and user['required_relative_conditions']:
            # Do not offer unrelated coloured zones to a model interpreting an
            # explicitly object-relative request. The compiler still validates IDs.
            user['targets'] = [t for t in state['targets'] if t['id'] in instruction]
        if self._repair_feedback:
            user["previous_validation_error"] = self._repair_feedback
        messages = [{"role": "system", "content": system}, {"role": "user", "content": json.dumps(user, ensure_ascii=False)}]
        content = self._complete(messages, schema)
        payload = self._extract_json(content)
        if self._interpret_meaning:
            for goal in payload.get('goals',[]):
                # "on" is a support/contact predicate, not a configurable gap.
                # Some backends ignore decoder bounds and emit a redundant 0.
                if goal.get('type')=='relative_placement' and goal.get('relation')=='on' and goal.get('distance_cm')==0:
                    goal.pop('distance_cm')
            meaning = InstructionMeaning.model_validate(payload)
            if meaning.interpretation == 'ordered_actions':
                return {'needs_actions': True, 'raw': content}
            program = (TaskProgram(error=meaning.error) if meaning.interpretation == 'unsupported' else
                       TaskProgram(goals=meaning.goals, keep_in_place=meaning.keep_in_place))
            return {'payload': program.model_dump(), 'raw': content}
        proposal = ActionProposal.model_validate(payload)
        return {"payload": proposal.task_program().model_dump(), "raw": content}

    def _complete(self, messages, schema, *, timeout=20., max_tokens=1600):
        # Timeouts measured on this host, 2026-10-25, with the local model on the Radeon RX 7800 XT
        # through ROCm: a one-word reply takes 1.8 s cold (including loading 3.6 GB) and 0.12 s warm,
        # and a 220-token planning-shaped generation takes 2.8 s. The previous 60 s and 25 s were sized
        # for CPU inference, where the same one-word reply took 43 s. 20 s and 12 s keep several times
        # the measured margin while a stalled provider falls back faster instead of stalling a task.
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
                       "format": schema,
                       "options": {"temperature": 0, "num_predict": max_tokens, "num_ctx": 8192,
                                   "stop": ["<|im_end|>", "<|endoftext|>"]}}
            if self._ollama_raw_chatml:
                request.update(raw=True, prompt="".join(
                    f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages
                ) + "<|im_start|>assistant\n<think>\n\n</think>\n")
                endpoint = "generate"
            else:
                request.update(messages=messages, think=False)
                endpoint = "chat"
            response = httpx.post(f"http://127.0.0.1:11434/api/{endpoint}", timeout=timeout, json=request)
            response.raise_for_status()
            content = response.json()["response"] if self._ollama_raw_chatml else response.json()["message"]["content"]
        else:
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=35.0, max_retries=0)
            response = client.chat.completions.create(
                model=self.model, messages=messages, response_format={"type": "json_object"},
                max_tokens=max_tokens, temperature=0,
            )
            content = response.choices[0].message.content or ""
        return content

    def assess_failure(self, instruction, state, failure, legal_actions):
        """Choose among physically admitted recovery operations, never a new goal."""
        if not self.enabled:
            return {'decision':legal_actions[0], 'reason':'現在の状態で実行可能な復旧手順を選びます。',
                    'provider':'deterministic_fallback'}
        schema = {'type':'object','properties':{
            'decision':{'enum':legal_actions},'reason':{'type':'string'},
            'requirements':{'type':'array','items':{'type':'string'},'maxItems':5}},
            'required':['decision','reason'],'additionalProperties':False}
        facts = {'hand_occupancy':{s:state.get('robot',{}).get(s+'_arm',{}).get('held_object') for s in ('left','right')},
                 'objects':[{k:o.get(k) for k in ('id','position','held_by','touching','linear_speed_m_s')} for o in state['objects']]}
        messages = [{'role':'system','content':
            'Diagnose a robot task failure from measured facts. Choose exactly one legal recovery. '
            'resume means continue uncompleted operations from the observed state, without replaying completed steps. '
            'recover_in_place means open/retreat only empty contacting grippers, return free arms home, '
            'then replan from the SAME unchanged scene. alternate_arm uses another available hand only when the user omitted a hand. '
            'reset_and_retry means visibly restart the simulation scene and retry the SAME bound object/task, discarding progress. '
            'Reset is a last resort; it is legal only after in-place alternatives are exhausted and the caller records an unrecoverable scene condition. '
            'Prefer resume if meaningful progress exists and its remaining plan is valid; ordinary grasp, IK or path failures do not justify resetting the scene. '
            'Explain the observed obstacle and required next preparation in Japanese. Reasons for failure are hypotheses; do not invent observations, tools, successful grasps or capabilities.'},
            {'role':'user','content':json.dumps({'instruction':instruction,'observed':facts,'failure':failure,
                'legal_recoveries':legal_actions,'output_schema':schema},ensure_ascii=False)}]
        try:
            assessment = self._extract_json(self._complete(messages,schema,timeout=12.,max_tokens=450))
            if assessment.get('decision') not in legal_actions or not isinstance(assessment.get('reason'),str):
                raise ValueError('Invalid recovery choice')
            return {**assessment,'provider':self.configured_provider}
        except Exception:
            return {'decision':legal_actions[0],'reason':'モデルによる復旧案を検証できないため、観測状態と実行条件から復旧します。',
                    'provider':'deterministic_fallback','provider_error':'RECOVERY_REASONING_UNAVAILABLE'}

    def _check_coverage(self, instruction, candidate, state):
        """Independent semantic audit for compound clauses the grammar could not cover."""
        clauses = [c.strip() for c in re.split(r'(?<=て)から|(?<=た)後(?:に)?|そのあと|その後|それから|[。;；]|\bthen\b',
            strip_preservation(instruction,state),flags=re.I) if c.strip(' 、,.!?！？')]
        unknown = [c for c in clauses if not VERBS.search(c) and not re.search(r'ないで|\b(?:do not|don.t|never)\b',c,re.I)]
        unknown = [c for c in unknown if not re.fullmatch(
            r'(?:手順|やり方|方法|途中(?:の置き方|の手順)?)(?:は|を)?(?:全部)?(?:任せる|任せます|任せた)[.!。\s]*'
            r'|(?:work out|figure out|choose)\s+(?:the\s+)?(?:intermediate steps|steps|procedure)(?:\s+yourself)?[.!\s]*',c,re.I)]
        if len(clauses)<2 or not unknown:
            return None
        if not any(VERBS.search(c) for c in clauses) and not re.search(r'てから|た後|そのあと|その後|それから|\bthen\b',instruction,re.I):
            # Pure final-state prose plus "work out the steps" is not an
            # unhandled sequence of physical verbs.
            return None
        schema = {'type':'object','properties':{'covered':{'type':'boolean'},
            'omitted_or_unsupported':{'type':'array','items':{'type':'string'}},'reason':{'type':'string'}},
            'required':['covered','omitted_or_unsupported','reason'],'additionalProperties':False}
        system = ('Judge whether EACH isolated requested clause can be implemented with the listed robot controllers. '
            'Return covered=false if ANY clause requires unsupported physical work. '
            'Available operations only: pick/hold a known cube, handover, place, approach, move an arm, '
            'open/close/home, upward throw, wine pour, named rotary joint. There is no wiping, rubbing, cleaning, '
            'cutting or arbitrary new contact controller. Negated/protected clauses are constraints, not extra actions. '
            'Do not replace an unsupported interaction by grasping or moving an object. '
            'List each unsupported clause verbatim in omitted_or_unsupported. Return JSON only.')
        output = self._extract_json(self._complete([{'role':'system','content':system},
            {'role':'user','content':json.dumps({'isolated_clauses':unknown,'capabilities':state.get('capabilities',{}),
                'output_schema':schema},ensure_ascii=False)}],timeout=12.,max_tokens=450))
        if output.get('covered') is not True or output.get('omitted_or_unsupported') != []:
            raise GroundingError('INCOMPLETE_SEMANTIC_COVERAGE: '+str(output.get('reason','uncovered requested work')))
        return output

    def _shadow_plan(self, instruction, state, memory, fallback):
        """Ask the model what it would have said, on a request the rules already answered. Record only.

        The deterministic plan is what runs: this cannot change a step, a hand, or a verdict. That is
        the point - it measures the model's contribution on requests that never used it, which is the
        only way to see whether rules and model agree where the rules were confident.
        """
        saved = (self._interpret_meaning, self._repair_feedback)
        try:
            self._interpret_meaning = False
            self._repair_feedback = None
            response = self._ai_plan(instruction, state, memory)
            if response.get('needs_actions'):
                return {'attempted': True, 'ok': False, 'error': 'PROVIDER_REQUESTED_ACTIONS'}
            shadow_plan = compile_prepared_program(TaskProgram.model_validate(response['payload']), state)
            return {'attempted': True, 'ok': True, 'steps': len(shadow_plan.steps),
                    'deterministic_steps': None if fallback is None else len(fallback.steps),
                    'agree': (program_signature(shadow_plan) == program_signature(fallback)
                              if fallback is not None else None)}
        except Exception as exc:
            # A shadow call must never fail the request it is shadowing.
            return {'attempted': True, 'ok': False, 'error': f'{type(exc).__name__}: {str(exc)[:200]}'}
        finally:
            self._interpret_meaning, self._repair_feedback = saved

    def plan(self, instruction: str, state: dict, memory: list[dict] | None = None) -> AgentPlan:
        started = time.perf_counter()
        try:
            observed_binding = bind_scene_references(instruction,state)
            instruction = observed_binding.instruction
        except SceneReferenceError as error:
            raise PlannerProviderError('REFERENCE_NOT_PRESENT',str(error)) from None
        if unsupported_group_request(instruction):
            raise PlannerProviderError('UNSUPPORTED_TASK', 'GROUP_GRASP_UNSUPPORTED: one-hand grouped or simultaneous multi-cube acquisition has no executable contact contract')
        literal_holders = holding_conditions(instruction, state)
        for name in {c['object'] for c in literal_holders}:
            if len({c['arm'] for c in literal_holders if c['object']==name}) > 1:
                raise PlannerProviderError('AMBIGUOUS_INSTRUCTION', 'One registered object cannot finish in two separate hands: '+name)
        memory = memory or []
        fallback = None
        fallback_source = None
        fallback_error = None
        try:
            fallback_source = parse_fallback(instruction, state)
            fallback = compile_prepared_program(fallback_source, state)
            validate_instruction_relations(fallback, instruction, state)
        except (ValueError, KeyError) as exc:
            fallback_error = str(exc)
            if fallback_error.startswith("This scene has only"):
                raise PlannerProviderError("UNKNOWN_OBJECT", fallback_error) from None
        plan = None
        model_plan = None
        provider_error = None
        raw = None
        self._repair_feedback = None
        # A failed parse/grounding does not turn a procedure into a final state.
        # In particular 'after picking ... pass ... put ...' and numeric arm
        # movement need action interpretation even when the fallback cannot parse.
        temporal = bool(re.search(r'\b(?:after|then|before)\b|てから|た後|渡して|その後', instruction, re.I))
        verbs = [m.lastgroup for m in VERBS.finditer(instruction)]
        procedural = (temporal and len(verbs)>1) or ('move' in verbs and motion_distance_cm(instruction) is not None)
        self._interpret_meaning = not ((fallback_source and fallback_source.actions) or procedural)
        language_stages = []
        # Why the model is not asked, stated rather than left implicit. The previous behaviour answered a
        # single direct command from the grammar and recorded nothing, so a request that used no model at
        # all looked exactly like one where the model agreed.
        if not self.enabled:
            skipped_because = "provider_not_configured"
        elif fallback and self.exact_grammar_fast_path:
            skipped_because = "exact_grammar_fast_path"
        elif fallback and len(fallback.program["actions"]) == 1 and fallback.goal["type"] == "direct_command":
            skipped_because = "single_direct_command"
        else:
            skipped_because = None
        fallback_verdict = {"ok": fallback is not None, "actions": len(fallback.program["actions"]) if fallback else 0,
                            "goal": fallback.goal["type"] if fallback else None, "error": fallback_error}
        # The launcher can immediately execute a fully parsed/grounded request.
        # Unrecognised language still uses the model and the same validation.
        # This avoids a model round trip just to reconfirm an identical program.
        use_ai = self.enabled and (self.capability_mode or skipped_because is None)
        shadow = self._shadow_plan(instruction, state, memory, fallback) if (self.shadow_mode and self.enabled
                                                                              and not use_ai) else None
        if use_ai:
            for _ in range(3 if self._interpret_meaning else 2):
                try:
                    language_stages.append('meaning' if self._interpret_meaning else 'actions')
                    response = self._ai_plan(instruction, state, memory)
                    if response.get('needs_actions'):
                        self._interpret_meaning = False
                        self._repair_feedback = None
                        continue
                    program = TaskProgram.model_validate(response["payload"])
                    binding_repairs = []
                    if program.error:
                        if fallback:
                            provider_error = "PROVIDER_UNSUPPORTED_LANGUAGE"
                            break
                        raise PlannerProviderError("UNSUPPORTED_TASK", program.error)
                    if program.goals:
                        if fallback_source and fallback_source.actions:
                            raise GroundingError('Keep the explicitly understood ordered actions; do not replace them by final goals')
                        program.keep_in_place = sorted(set(program.keep_in_place) |
                            {name for _, _, name in preservation_spans(instruction, state)})
                        mentioned = set(named_objects(instruction, state))
                        represented = {g.object for g in program.goals} | {g.target for g in program.goals}
                        if mentioned - represented - set(program.keep_in_place):
                            raise GroundingError('Final goals omit explicitly mentioned objects: ' + str(sorted(mentioned-represented-set(program.keep_in_place))))
                        if mentioned:
                            extras = {g.object for g in program.goals} - mentioned
                            if extras:
                                raise GroundingError('Do not invent tasks for unrequested objects: '+str(sorted(extras)))
                        conditions = holding_conditions(instruction, state)
                        required = {(c['object'], c['arm']) for c in conditions}
                        for obj in {obj for obj, _ in required}:
                            requested = {arm for name, arm in required if name == obj}
                            if len(requested) != 1:
                                raise GroundingError('Conflicting literal final holders for ' + obj)
                            goal = next((g for g in program.goals if g.object == obj and g.type == 'holding'), None)
                            if goal is not None and goal.arm not in requested:
                                # Like literal mL quantities, an unambiguous
                                # source binding is authoritative over model bias.
                                target_arm = next(iter(requested))
                                binding_repairs.append({'object': obj, 'model_arm': goal.arm, 'requested_arm': target_arm,
                                    'source': next(c['source'] for c in conditions if c['object'] == obj)})
                                goal.arm = target_arm
                        actual = {(g.object, g.arm) for g in program.goals if g.type == 'holding'}
                        if required-actual:
                            raise GroundingError('Preserve requested FINAL holders, not observed current ownership: ' +
                                json.dumps(sorted(required), ensure_ascii=False))
                        relative = relative_conditions(instruction, state)
                        for requested_relation in relative:
                            for goal in program.goals:
                                if (goal.type=='relative_placement' and goal.object==requested_relation['target']
                                        and goal.target==requested_relation['object']):
                                    binding_repairs.append({'type':'relative_roles','model_object':goal.object,
                                        'model_target':goal.target,'requested':requested_relation})
                                    goal.object,goal.target,goal.relation = (requested_relation[k] for k in ('object','target','relation'))
                        expected_relative = {(r['object'],r['target'],r['relation']) for r in relative}
                        actual_relative = {(g.object,g.target,g.relation) for g in program.goals if g.type=='relative_placement'}
                        references_only = {r['target'] for r in relative}-{r['object'] for r in relative}
                        if expected_relative-actual_relative or any(g.object in references_only for g in program.goals):
                            raise GroundingError('Use the requested final relative goals; do not manipulate support-only objects: '+json.dumps(relative, ensure_ascii=False))
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
                    # The compiler explains concrete choices from the scene; the model's own choices are
                    # kept and labelled, because they are the record of what it assumed. They were
                    # previously discarded outright, which left an ambiguous instruction either refused
                    # or executed with no trace of the interpretation behind it.
                    model_assumptions = [a.strip() for a in (program.assumptions or [])
                                         if isinstance(a, str) and a.strip() and _is_choice_not_outcome(a)][:5]
                    program.assumptions = []
                    # Revalidate anchored quantities, including the supported range.
                    program = TaskProgram.model_validate(program.model_dump())
                    candidate = compile_prepared_program(program, state)
                    if model_assumptions:
                        candidate.assumptions = [f'モデルが選んだ前提: {a}' for a in model_assumptions] + candidate.assumptions
                        candidate.planning_evidence = {**(candidate.planning_evidence or {}),
                            'model_assumptions': model_assumptions}
                    if binding_repairs:
                        candidate.planning_evidence = {**(candidate.planning_evidence or {}),
                            'semantic_binding_repairs': binding_repairs}
                        candidate.assumptions.append('文章で指定された物体・保持者・支持物の対応を優先しました。補正内容は実行ログに記録します。')
                    validate_instruction_relations(candidate, instruction, state)
                    if not fallback_source:
                        audit = self._check_coverage(instruction,candidate,state)
                        if audit:
                            candidate.planning_evidence = {**(candidate.planning_evidence or {}),'semantic_coverage_audit':audit}
                    if fallback:
                        # When the explicit grammar understood the ENTIRE request, the model
                        # may not drop a handover, swap a colour, or change an explicit arm.
                        def signature(p):
                            # Compare physical intentions after compilation. 'Pick then
                            # hand over' and an auto-acquiring handover are equivalent;
                            # an already-held pick may also be an explicit no-op.
                            return program_signature(p)
                        if signature(candidate) != signature(fallback):
                            raise GroundingError("Program changes explicit actions/hands in instruction. Expected: " + json.dumps(fallback.program, ensure_ascii=False))
                    valid, errors = validate_plan(candidate, state)
                    if not valid:
                        raise GroundingError("; ".join(errors))
                    plan = candidate
                    model_plan = candidate
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
            # Report the failure of the path that actually ran. When the model was asked, its own
            # validation error is why there is no plan - the grammar's parse error is background, kept in
            # fallback_verdict. Without this a scene the solver could not satisfy was reported as
            # "instruction not understood", which points a reader at the wrong component.
            model_detail = self._repair_feedback if use_ai else None
            detail = str(model_detail or fallback_error or provider_error)
            if "ROBOT_CAPABILITY:" in detail:
                category = "ROBOT_CAPABILITY"
            elif detail.startswith("GOAL_UNREACHABLE"):
                category = "GOAL_UNREACHABLE"
            else:
                category = "INSTRUCTION_NOT_UNDERSTOOD"
            raise PlannerProviderError(category, f"Cannot ground the full instruction: {detail[:600]}")
        plan.provider_error = provider_error
        if observed_binding.decisions:
            plan.planning_evidence = {**(plan.planning_evidence or {}),'scene_bindings':observed_binding.decisions}
            plan.assumptions += [d['explanation'] for d in observed_binding.decisions]
        plan.raw_response = raw
        plan.model_called = bool(use_ai)
        # 'skipped_because' answers one question only: why the model was not called.
        plan.skipped_because = None if use_ai else skipped_because
        plan.capability_mode = self.capability_mode
        plan.fallback_verdict = fallback_verdict
        plan.model_verdict = {"ok": model_plan is not None,
                              "actions": len(model_plan.steps) if model_plan is not None else 0,
                              "error": provider_error}
        plan.agree = (program_signature(model_plan) == program_signature(fallback)
                      if model_plan is not None and fallback is not None else None)
        plan.shadow = shadow
        plan.model = (self.model if plan.provider == "ai&" else
                      self.local_model if plan.provider == "local_ollama" else None)
        plan.memory_used = memory[-3:]
        plan.memory_decision = "Use live geometry and contacts; past experience never overrides physical preconditions."
        plan.planning_latency = time.perf_counter() - started
        if language_stages:
            plan.planning_evidence = {**(plan.planning_evidence or {}), 'language_stages': language_stages}
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
        elif step.skill in {"return_home", "park_empty_arm"} and held[step.arm]:
            errors.append(f"Step {i}: place the held object before homing")
    return not errors, errors
