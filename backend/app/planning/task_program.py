"""Typed, grounded task language. A language model chooses intentions, never joint poses.

The compiler simulates hand occupancy before *any* motion. The executor subsequently checks
the same transitions against Genesis contacts; a compiler's belief is not physical evidence.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from backend.app.models.schemas import AgentPlan, PlanStep

SIDES = {"left", "right"}
TABLE_TARGETS = {
    "left_side": (0.46, 0.30, 1.008),
    "right_side": (0.46, -0.30, 1.008),
    "center": (0.50, 0.0, 1.008),
}


class TaskAction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    op: Literal["pick", "handover", "place", "approach", "move", "open", "close", "home", "throw", "pour", "joint"]
    object: str | None = None
    arm: Literal["left", "right"] | None = None
    receiver: Literal["left", "right"] | None = None
    target: str | None = None
    relation: Literal["on", "left_of", "right_of", "front_of", "behind", "near"] | None = None
    direction: Literal["up", "down", "left", "right", "forward", "back"] | None = None
    distance_cm: float | None = Field(default=None, ge=1, le=30)
    height_cm: float | None = Field(default=None, ge=1, le=10)
    volume_ml: float | None = Field(default=None, ge=5, le=35)
    joint: str | None = None
    position_deg: float | None = Field(default=None, allow_inf_nan=False)


class TaskGoal(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["holding", "placement", "relative_placement"]
    object: str
    arm: Literal["left", "right"] | None = None
    target: str | None = None
    relation: Literal["on", "left_of", "right_of", "front_of", "behind", "near"] | None = None
    distance_cm: float | None = Field(default=None, ge=1, le=30)

    @model_validator(mode="after")
    def required_arguments(self):
        if self.type == "holding":
            if self.arm is None or self.target is not None or self.relation is not None or self.distance_cm is not None:
                raise ValueError("A holding goal needs one arm and no placement arguments")
        elif self.type == "placement":
            if self.target is None or self.arm is not None or self.relation is not None or self.distance_cm is not None:
                raise ValueError("A placement goal needs a table target and no relation")
        elif self.target is None or self.relation is None or self.arm is not None:
            raise ValueError("A relative placement goal needs a reference object and relation")
        return self


class TaskProgram(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={'oneOf': [
        {'required': ['actions'], 'properties': {'actions': {'minItems': 1}, 'goals': {'maxItems': 0}, 'error': {'type': 'null'}}},
        {'required': ['goals'], 'properties': {'goals': {'minItems': 1}, 'actions': {'maxItems': 0}, 'error': {'type': 'null'}}},
        {'required': ['error'], 'properties': {'error': {'type': 'string', 'minLength': 1}, 'actions': {'maxItems': 0}, 'goals': {'maxItems': 0}}},
    ]})
    actions: list[TaskAction] = Field(default_factory=list, max_length=24)
    goals: list[TaskGoal] = Field(default_factory=list, max_length=12)
    keep_in_place: list[str] = Field(default_factory=list, max_length=12)
    assumptions: list[str] = Field(default_factory=list, max_length=12)
    error: str | None = Field(default=None, max_length=300)

    @model_validator(mode="after")
    def actions_or_error(self):
        if sum(map(bool, (self.actions, self.goals, self.error))) != 1:
            raise ValueError("Return exactly one of actions, final goals, or an error; never a guessed partial task")
        if self.keep_in_place and not self.goals:
            raise ValueError("keep_in_place is a goal-program constraint")
        return self


class GroundingError(ValueError):
    def __init__(self, message, *, preparation=None):
        super().__init__(message)
        self.preparation = preparation


class ActionProposal(BaseModel):
    """One model output array, with a declared interpretation of its items.

    In the inspected Ollama setup, oneOf alone did not prevent responses from
    filling both optional actions/goals arrays. Each language stage now owns one
    array and item type, while the compiler-facing TaskProgram stays explicit.
    """
    model_config = ConfigDict(extra='forbid')
    mode: Literal['actions', 'error']
    items: list[TaskAction] = Field(default_factory=list, max_length=24)
    assumptions: list[str] = Field(default_factory=list, max_length=12)
    error: str | None = Field(default=None, max_length=300)

    def task_program(self) -> TaskProgram:
        if self.mode == 'error':
            if self.items or not self.error:
                raise ValueError('Error mode needs an error and empty items')
            return TaskProgram(error=self.error)
        if not self.items or self.error:
            raise ValueError('Action mode requires action items and no error')
        return TaskProgram(actions=self.items, assumptions=self.assumptions)


class InstructionMeaning(BaseModel):
    """Extract end-state meaning without exposing action recipes to the model."""
    model_config = ConfigDict(extra='forbid')
    interpretation: Literal['final_state', 'ordered_actions', 'unsupported']
    goals: list[TaskGoal] = Field(default_factory=list, max_length=12)
    keep_in_place: list[str] = Field(default_factory=list, max_length=12)
    error: str | None = Field(default=None, max_length=300)

    @model_validator(mode='after')
    def consistent_mode(self):
        if self.interpretation == 'final_state':
            if not self.goals or self.error:
                raise ValueError('Final-state interpretation needs goals and no error')
        elif self.goals or (self.keep_in_place and self.interpretation!='ordered_actions'):
            raise ValueError('Only final_state may contain goals; preservation may also constrain ordered actions')
        elif (self.interpretation == 'unsupported') != bool(self.error):
            raise ValueError('Only unsupported needs an error')
        return self


def observed_holds(state: dict) -> dict[str, str | None]:
    robot = state.get("robot", {})
    return {side: robot.get(f"{side}_arm", {}).get("held_object") for side in SIDES}


def compile_program(program: TaskProgram, state: dict) -> AgentPlan:
    if program.error:
        raise GroundingError(program.error)
    if program.goals:
        from backend.app.planning.goal_search import compile_goal_program
        return compile_goal_program(program, state)
    objects = {item["id"]: item for item in state["objects"]}
    targets = {item["id"] for item in state["targets"]} | set(TABLE_TARGETS)
    held = observed_holds(state)
    discourse = state.get("discourse", {})
    last_object: str | None = discourse.get("last_object")
    last_arm: str = discourse.get("last_arm") or "left"
    available_arms = state.get('robot', {}).get('available_arms', ['left', 'right'])
    if last_arm not in available_arms:
        last_arm = available_arms[0]
    steps: list[PlanStep] = []
    checks: dict[str, dict] = {obj: {"type": "holding", "object": obj, "arm": side} for side, obj in held.items() if obj}
    canonical: list[TaskAction] = []
    assumptions = list(program.assumptions)
    remaining_wine = state.get("fluid", {}).get("bottle_ml") if state.get("fluid") else None

    def emit(skill: str, arm: str, index: int, **kwargs):
        allowed = state.get("capabilities", {}).get("skills")
        if allowed is not None and skill not in allowed:
            raise GroundingError(f"The selected robot does not offer {skill}; it offers "
                                 f"{', '.join(sorted(allowed)) if allowed else 'nothing yet'}")
        available = state.get("robot", {}).get("available_arms")
        if available is not None and arm not in available:
            raise GroundingError(f"The selected robot has arms {available}, not {arm}")
        steps.append(PlanStep(skill=skill, arm=arm, action_index=index, **kwargs))

    def owner(obj: str) -> str | None:
        owners = [side for side, item in held.items() if item == obj]
        if len(owners) > 1:
            raise GroundingError(f"{obj} touches both hands; finish/recover the handover first")
        return owners[0] if owners else None

    def auto_arm(obj: str) -> str:
        grippers = state.get("robot", {}).get("grippers")
        admitted=objects[obj].get('admitted_grasp_arm')
        if admitted in available_arms and (grippers is None or admitted in grippers):return admitted
        if grippers and len(grippers) == 1:
            return next(iter(grippers))
        if len(available_arms) == 1:
            return available_arms[0]
        position = objects[obj].get("position")
        return ("left" if position[1] >= 0 else "right") if position else ("right" if obj == "blue_cube" else "left")

    def acquire(obj: str, arm: str, index: int, for_handover: bool = False):
        if not objects[obj].get("graspable", True):
            raise GroundingError(f"ROBOT_CAPABILITY: {obj} is fixed or this robot has no detected opposed actuated jaws")
        grippers = state.get("robot", {}).get("grippers")
        if grippers is not None and arm not in grippers:
            raise GroundingError(f"ROBOT_CAPABILITY: no opposed actuated jaws were detected for {arm}")
        if held[arm] and held[arm] != obj:
            raise GroundingError(f"{arm} hand already holds {held[arm]}; place it before picking {obj}",
                preparation={'kind':'free_hand','arm':arm,'object':held[arm],'action_index':index,
                             'resolved_action':{**program.actions[index].model_dump(),'object':obj,'arm':arm}})
        source = owner(obj)
        if source == arm:
            return
        if source:
            if obj == "wine_bottle":
                raise GroundingError("Bottle handover is not supported; use the current holder")
            emit("handover_object", source, index, object=obj, receiver=arm)
            held[source] = None
        else:
            handover_next = for_handover or program.actions[index].op == "handover" or any((a.op == "handover" and a.object in {obj, "it", "それ", None}) or (a.op == "pick" and a.arm and a.arm != arm and a.object in {obj, "it", "それ"}) for a in program.actions[index + 1:])
            for skill in (("approach_object", "grasp_object") if obj == "wine_bottle" else ("approach_object", "grasp_object", "lift_object")):
                emit(skill, arm, index, object=obj, grasp_offset_x=-.015 if skill == "approach_object" and handover_next else 0.)
        held[arm] = obj

    for index, original in enumerate(program.actions):
        action = original.model_copy(deep=True)
        if action.op != "handover":
            # Small models sometimes fill an irrelevant recipient on a pour.
            # It must not change discourse ownership after a completed action.
            action.receiver = None
        obj = action.object
        if obj in {"it", "that", "それ", "そのキューブ", "same"}:
            obj = last_object
        elif obj in {"held:left", "held:right"}:
            requested_hand = obj.split(':')[1]
            obj = held[requested_hand]
            if obj is None:
                raise GroundingError(f'REFERENCE_NOT_PRESENT: {requested_hand} hand is empty; specify or acquire a known object before using held:{requested_hand}')
        elif obj in {"left_cube", "right_cube"}:
            side = obj.split("_")[0]
            candidates = [v for v in objects.values() if "position" in v and v.get('type','cube')=='cube']
            obj = (max if side == "left" else min)(candidates, key=lambda x: x["position"][1])["id"] if candidates else None
        needs_object = action.op in {"pick", "handover", "place", "approach", "throw", "pour"}
        if obj is None and needs_object:
            obj = held.get(action.arm) if action.arm else None
            unique_held = list({item for item in held.values() if item})
            obj = obj or last_object or (unique_held[0] if len(unique_held) == 1 else None)
        if needs_object and obj not in objects:
            raise GroundingError(f"Action {index + 1}: specify a known object ({', '.join(objects)}); reference {obj!r} is ambiguous")
        if obj is not None and obj not in objects:
            raise GroundingError(f"Unknown object: {obj}")
        arm = action.arm or ((owner(obj) or auto_arm(obj)) if obj else last_arm)
        if action.op == "place" and action.relation and action.arm is None and action.target in objects:
            arm = auto_arm(action.target)
        if action.op != "approach":
            checks.pop(f"approach:{arm}", None)
        action.object, action.arm = obj, arm
        if action.relation and action.op != "place":
            raise GroundingError("An object-relative relation belongs to a placement action")
        if action.op == "pick":
            previous_owner = owner(obj)
            acquire(obj, arm, index)
            if previous_owner and previous_owner != arm:
                action.op, action.arm, action.receiver = "handover", previous_owner, arm
                last_arm = arm
            checks[obj] = {"type": "holding", "object": obj, "arm": arm}
        elif action.op == "handover":
            if obj == "wine_bottle":
                raise GroundingError("Bottle handover is not supported; pour with the current holder")
            receiver = action.receiver
            if receiver not in SIDES or receiver == arm:
                raise GroundingError(f"Action {index + 1}: handover needs two different hands")
            acquire(obj, arm, index)
            if held[receiver]:
                raise GroundingError(f"{receiver} hand holds {held[receiver]}; cannot receive {obj}",
                    preparation={'kind':'free_hand','arm':receiver,'object':held[receiver],'action_index':index,
                                 'resolved_action':action.model_dump()})
            emit("handover_object", arm, index, object=obj, receiver=receiver)
            held[arm], held[receiver] = None, obj
            checks[obj] = {"type": "holding", "object": obj, "arm": receiver}
            last_arm = receiver
        elif action.op == "place":
            if obj == "wine_bottle":
                if (action.target is None and canonical and canonical[-1].op == "pour"
                        and canonical[-1].object == obj):
                    # Pour is an atomic serve-and-return skill. A following
                    # unspecified 'put the bottle back' is already fulfilled.
                    assumptions.append(f"Action {index + 1}: bottle return is included in the preceding pour")
                    canonical.append(action)
                    continue
                raise GroundingError("Use pour for the bottle; it includes returning it upright to its starting place")
            target = action.target or f"{arm}_side"
            if target in {"there", "そこ", "same_target"}:
                target = discourse.get("last_target")
            if action.target is None:
                assumptions.append(f"Action {index + 1}: unspecified placement means the {arm} side of the table")
            if action.relation:
                if target not in objects or target == obj:
                    raise GroundingError("Relative placement needs a different known reference object")
                if objects[obj].get("type", "cube") != "cube" or objects[target].get("type", "cube") != "cube":
                    raise GroundingError("Relative placement currently requires two cubes")
                if owner(target):
                    raise GroundingError("Place the reference cube on the table before placing another cube relative to it")
            elif target not in targets:
                raise GroundingError(f"Unknown placement target: {target}")
            if action.relation and owner(obj) is None and auto_arm(obj) != arm:
                source_arm = auto_arm(obj)
                acquire(obj, source_arm, index, for_handover=True)
                assumptions.append(f"Action {index + 1}: acquire with {source_arm}, then transfer to {arm} near the reference")
            acquire(obj, arm, index)
            movement = ("move_to_object", {"object": obj, "target": target, "relation": action.relation,
                                           "distance_cm": action.distance_cm}) if action.relation else ("move_to_target", {"target": target})
            for skill, args in (movement, ("release_object", {"object": obj}), ("retreat", {})):
                emit(skill, arm, index, **args)
            held[arm] = None
            checks[obj] = {"type": "relative_placement" if action.relation else "placement", "object": obj, "target": target}
            if action.relation:
                checks[obj].update(relation=action.relation, distance_cm=action.distance_cm, arm=arm)
            action.target = target
        elif action.op == "approach":
            if held[arm]:
                raise GroundingError(f"{arm} hand must be empty before approach")
            emit("approach_object", arm, index, object=obj)
            checks[f"approach:{arm}"] = {"type": "approach", "object": obj, "arm": arm}
        elif action.op == "pour":
            if obj != "wine_bottle" or action.target not in {None, "wine_glass"} or "wine_glass" not in objects:
                raise GroundingError("Pour requires the wine bottle and receiving glass")
            action.target = "wine_glass"
            action.volume_ml = action.volume_ml or 20.
            if remaining_wine is not None and action.volume_ml + 2. > remaining_wine:
                raise GroundingError("The requested amount exceeds the bottle's available wine")
            if remaining_wine is not None:
                remaining_wine -= action.volume_ml
            acquire(obj, arm, index)
            emit("pour_liquid", arm, index, object=obj, target="wine_glass", volume_ml=action.volume_ml)
            held[arm] = None
            checks.pop(obj, None)
            checks.pop("wine_glass", None)
            checks[f"pour:{index}"] = {"type": "pour", "object": obj, "target": "wine_glass", "volume_ml": action.volume_ml, "action_index": index}
        elif action.op == "throw":
            if objects[obj].get("type") != "cube":
                raise GroundingError("Only a cube can be thrown in this scene")
            if action.direction not in {None, "up"}:
                raise GroundingError("Only upward throws are supported")
            acquire(obj, arm, index)
            action.direction = "up"
            action.height_cm = action.height_cm or 3.
            emit("throw_object", arm, index, object=obj, height_cm=action.height_cm)
            held[arm] = None
            checks.pop(obj, None)
            checks[f"throw:{index}"] = {"type": "throw", "object": obj, "arm": arm,
                                       "min_height": action.height_cm / 100., "action_index": index}
        elif action.op == "move":
            if not action.direction or action.distance_cm is None:
                raise GroundingError("Relative motion requires a direction and distance (1–30 cm)")
            if obj:
                acquire(obj, arm, index)
                checks[obj] = {"type": "holding", "object": obj, "arm": arm}
            emit("move_relative", arm, index, direction=action.direction, distance_cm=action.distance_cm)
        elif action.op == "joint":
            joints = {j["name"]: j for j in state.get("robot", {}).get("joints", [])}
            if action.joint not in joints or action.position_deg is None:
                raise GroundingError("Name an imported robot joint and its target angle in degrees")
            spec = joints[action.joint]
            if spec.get("type") == "prismatic":
                raise GroundingError("Degree commands apply to rotary joints; this joint is prismatic")
            import math
            radians = math.radians(action.position_deg)
            if ((spec.get("lower") is not None and radians < spec["lower"])
                    or (spec.get("upper") is not None and radians > spec["upper"])):
                raise GroundingError("The target exceeds this URDF joint's limits")
            arm = next((side for side, chain in state["robot"]["chains"].items() if action.joint in chain["joints"]), None)
            if arm is None:
                arm = next((side for side, g in state["robot"].get("grippers", {}).items() if action.joint == g["joint"]), None)
            if arm is None:
                raise GroundingError("This joint is not in an exposed arm or gripper")
            action.arm = arm
            emit("move_joint", arm, index, joint=action.joint, position_deg=action.position_deg)
        else:
            skill = {"open": "open_gripper", "close": "close_gripper", "home": "return_home"}[action.op]
            if action.op == "home" and held[arm]:
                raise GroundingError(f"Place {held[arm]} before homing the {arm} arm")
            if action.op == "open" and held[arm]:
                released = held[arm]
                checks[released] = {"type": "released", "object": released}
                held[arm] = None
            emit(skill, arm, index)
        canonical.append(action)
        if obj:
            last_object = obj
        if action.op != "handover":
            last_arm = arm
    if not steps:
        # An already-held pick remains measurable; do not open and reacquire it.
        steps = [PlanStep(skill="lift_object", arm=last_arm, object=last_object)]
    first = canonical[0]
    goal: dict = {"type": "task_sequence", "checks": list(checks.values())}
    if len(canonical) == 1:
        if first.op == "place" and not first.relation:
            goal.update(type="place_in_zone", object=first.object, target=first.target)
        elif first.op == "place":
            goal.update(type="direct_command", action="relative_placement", object=first.object, target=first.target,
                        relation=first.relation)
        else:
            labels = {"pick": "grasp", "approach": "approach", "move": "grasp_and_move" if first.object else "move_relative",
                      "open": "open_gripper", "close": "close_gripper", "home": "return_home", "handover": "handover",
                      "throw": "throw", "pour": "pour", "joint": "joint"}
            goal.update(type="direct_command", action=labels[first.op], arm=first.arm)
            if first.object:
                goal["object"] = first.object
    elif all(a.op == "pick" for a in canonical):
        goal.update(type="direct_command", action="multi_pick", objects=[a.object for a in canonical])
    return AgentPlan(goal=goal, arm=first.arm, steps=steps,
                     program=TaskProgram(actions=canonical, assumptions=assumptions).model_dump(), assumptions=assumptions)
