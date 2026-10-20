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
    op: Literal["pick", "handover", "place", "approach", "move", "open", "close", "home"]
    object: str | None = None
    arm: Literal["left", "right"] | None = None
    receiver: Literal["left", "right"] | None = None
    target: str | None = None
    direction: Literal["up", "down", "left", "right", "forward", "back"] | None = None
    distance_cm: float | None = Field(default=None, ge=1, le=30)


class TaskProgram(BaseModel):
    model_config = ConfigDict(extra="forbid")
    actions: list[TaskAction] = Field(default_factory=list, max_length=24)
    assumptions: list[str] = Field(default_factory=list, max_length=12)
    error: str | None = Field(default=None, max_length=300)

    @model_validator(mode="after")
    def actions_or_error(self):
        if bool(self.actions) == bool(self.error):
            raise ValueError("Return either a nonempty action list or an error, never a guessed partial task")
        return self


class GroundingError(ValueError):
    pass


def observed_holds(state: dict) -> dict[str, str | None]:
    robot = state.get("robot", {})
    return {side: robot.get(f"{side}_arm", {}).get("held_object") for side in SIDES}


def compile_program(program: TaskProgram, state: dict) -> AgentPlan:
    if program.error:
        raise GroundingError(program.error)
    objects = {item["id"]: item for item in state["objects"]}
    targets = {item["id"] for item in state["targets"]} | set(TABLE_TARGETS)
    held = observed_holds(state)
    discourse = state.get("discourse", {})
    last_object: str | None = discourse.get("last_object")
    last_arm: str = discourse.get("last_arm") or "left"
    steps: list[PlanStep] = []
    checks: dict[str, dict] = {obj: {"type": "holding", "object": obj, "arm": side} for side, obj in held.items() if obj}
    canonical: list[TaskAction] = []
    assumptions = list(program.assumptions)

    def emit(skill: str, arm: str, index: int, **kwargs):
        steps.append(PlanStep(skill=skill, arm=arm, action_index=index, **kwargs))

    def owner(obj: str) -> str | None:
        owners = [side for side, item in held.items() if item == obj]
        if len(owners) > 1:
            raise GroundingError(f"{obj} touches both hands; finish/recover the handover first")
        return owners[0] if owners else None

    def auto_arm(obj: str) -> str:
        position = objects[obj].get("position")
        return ("left" if position[1] >= 0 else "right") if position else ("right" if obj == "blue_cube" else "left")

    def acquire(obj: str, arm: str, index: int):
        if held[arm] and held[arm] != obj:
            raise GroundingError(f"{arm} hand already holds {held[arm]}; place it before picking {obj}")
        source = owner(obj)
        if source == arm:
            return
        if source:
            emit("handover_object", source, index, object=obj, receiver=arm)
            held[source] = None
        else:
            handover_next = program.actions[index].op == "handover" or any((a.op == "handover" and a.object in {obj, "it", "それ", None}) or (a.op == "pick" and a.arm and a.arm != arm and a.object in {obj, "it", "それ"}) for a in program.actions[index + 1:])
            for skill in ("approach_object", "grasp_object", "lift_object"):
                emit(skill, arm, index, object=obj, grasp_offset_x=-.015 if skill == "approach_object" and handover_next else 0.)
        held[arm] = obj

    for index, original in enumerate(program.actions):
        action = original.model_copy(deep=True)
        obj = action.object
        if obj in {"it", "that", "それ", "そのキューブ", "same"}:
            obj = last_object
        elif obj in {"held:left", "held:right"}:
            obj = held[obj.split(":")[1]]
        elif obj in {"left_cube", "right_cube"}:
            side = obj.split("_")[0]
            candidates = [v for v in objects.values() if "position" in v]
            obj = (max if side == "left" else min)(candidates, key=lambda x: x["position"][1])["id"] if candidates else None
        needs_object = action.op in {"pick", "handover", "place", "approach"}
        if obj is None and needs_object:
            obj = held.get(action.arm) if action.arm else None
            unique_held = list({item for item in held.values() if item})
            obj = obj or last_object or (unique_held[0] if len(unique_held) == 1 else None)
        if needs_object and obj not in objects:
            raise GroundingError(f"Action {index + 1}: specify a known cube (red_cube or blue_cube); reference {obj!r} is ambiguous")
        if obj is not None and obj not in objects:
            raise GroundingError(f"Unknown object: {obj}")
        arm = action.arm or ((owner(obj) or auto_arm(obj)) if obj else last_arm)
        if action.op != "approach":
            checks.pop(f"approach:{arm}", None)
        action.object, action.arm = obj, arm
        if action.op == "pick":
            previous_owner = owner(obj)
            acquire(obj, arm, index)
            if previous_owner and previous_owner != arm:
                action.op, action.arm, action.receiver = "handover", previous_owner, arm
                last_arm = arm
            checks[obj] = {"type": "holding", "object": obj, "arm": arm}
        elif action.op == "handover":
            receiver = action.receiver
            if receiver not in SIDES or receiver == arm:
                raise GroundingError(f"Action {index + 1}: handover needs two different hands")
            acquire(obj, arm, index)
            if held[receiver]:
                raise GroundingError(f"{receiver} hand holds {held[receiver]}; cannot receive {obj}")
            emit("handover_object", arm, index, object=obj, receiver=receiver)
            held[arm], held[receiver] = None, obj
            checks[obj] = {"type": "holding", "object": obj, "arm": receiver}
            last_arm = receiver
        elif action.op == "place":
            target = action.target or f"{arm}_side"
            if target in {"there", "そこ", "same_target"}:
                target = discourse.get("last_target")
            if action.target is None:
                assumptions.append(f"Action {index + 1}: unspecified placement means the {arm} side of the table")
            if target not in targets:
                raise GroundingError(f"Unknown placement target: {target}")
            acquire(obj, arm, index)
            for skill, args in (("move_to_target", {"target": target}), ("release_object", {"object": obj}), ("retreat", {})):
                emit(skill, arm, index, **args)
            held[arm] = None
            checks[obj] = {"type": "placement", "object": obj, "target": target}
            action.target = target
        elif action.op == "approach":
            if held[arm]:
                raise GroundingError(f"{arm} hand must be empty before approach")
            emit("approach_object", arm, index, object=obj)
            checks[f"approach:{arm}"] = {"type": "approach", "object": obj, "arm": arm}
        elif action.op == "move":
            if not action.direction or action.distance_cm is None:
                raise GroundingError("Relative motion requires a direction and distance (1–30 cm)")
            if obj:
                acquire(obj, arm, index)
                checks[obj] = {"type": "holding", "object": obj, "arm": arm}
            emit("move_relative", arm, index, direction=action.direction, distance_cm=action.distance_cm)
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
        if first.op == "place":
            goal.update(type="place_in_zone", object=first.object, target=first.target)
        else:
            labels = {"pick": "grasp", "approach": "approach", "move": "grasp_and_move" if first.object else "move_relative",
                      "open": "open_gripper", "close": "close_gripper", "home": "return_home", "handover": "handover"}
            goal.update(type="direct_command", action=labels[first.op], arm=first.arm)
            if first.object:
                goal["object"] = first.object
    elif all(a.op == "pick" for a in canonical):
        goal.update(type="direct_command", action="multi_pick", objects=[a.object for a in canonical])
    return AgentPlan(goal=goal, arm=first.arm, steps=steps,
                     program=TaskProgram(actions=canonical, assumptions=assumptions).model_dump(), assumptions=assumptions)
