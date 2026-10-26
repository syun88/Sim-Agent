from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field

SkillName = Literal[
    "approach_object", "grasp_object", "lift_object", "move_to_target",
    "release_object", "retreat", "return_home", "open_gripper",
    "close_gripper", "move_relative",
    "handover_object",
    "throw_object", "pour_liquid", "move_to_object", "move_joint",
    "observe_scene",
    "park_empty_arm", "navigate_to",
]
ArmName = Literal["left", "right"]
MoveDirection = Literal["up", "down", "left", "right", "forward", "back"]
ExecutionState = Literal[
    "IDLE", "PLANNING", "PREPARING_SCENE", "VALIDATING", "EXECUTING", "VERIFYING",
    "SUCCEEDED", "FAILED", "REPLANNING", "HOMING", "STOPPED",
]


class PlanStep(BaseModel):
    skill: SkillName
    arm: ArmName = "left"
    object: str | None = None
    target: str | None = None
    relation: Literal["on", "left_of", "right_of", "front_of", "behind", "near"] | None = None
    fault_offset: float | None = None
    direction: MoveDirection | None = None
    distance_cm: float | None = Field(default=None, gt=0, le=30)
    receiver: ArmName | None = None
    action_index: int | None = None
    grasp_offset_x: float = Field(default=0., ge=-.02, le=.02)
    height_cm: float | None = Field(default=None, ge=1, le=10)
    volume_ml: float | None = Field(default=None, ge=5, le=35)
    stroke_cm: float | None = Field(default=None, ge=2, le=30)
    release_speed_m_s: float | None = Field(default=None, ge=.2, le=3.)
    flight_time_s: float | None = Field(default=None, ge=.10, le=1.0)
    back_off_cm: float | None = Field(default=None, ge=0., le=20.)
    joint: str | None = None
    position_deg: float | None = Field(default=None, allow_inf_nan=False)
    target_xy: list[float] | None = Field(default=None, min_length=2, max_length=2)
    target_yaw: float | None = Field(default=None, allow_inf_nan=False)


class AgentPlan(BaseModel):
    goal: dict[str, Any]
    arm: ArmName = "left"
    steps: list[PlanStep]
    provider: str = "deterministic_fallback"
    model: str | None = None
    provider_error: str | None = None
    memory_used: list[dict] = Field(default_factory=list)
    memory_decision: str | None = None
    planning_latency: float | None = None
    raw_response: str | None = None
    program: dict[str, Any] | None = None
    goal_program: dict[str, Any] | None = None
    planning_evidence: dict[str, Any] | None = None
    assumptions: list[str] = Field(default_factory=list)
    # Provenance ledger. Which path actually answered a request, and why, must be readable per request:
    # a provider name alone cannot tell "the model agreed" from "the model was never consulted".
    model_called: bool = False
    skipped_because: str | None = None
    capability_mode: bool = False
    fallback_verdict: dict[str, Any] | None = None
    model_verdict: dict[str, Any] | None = None
    agree: bool | None = None
    shadow: dict[str, Any] | None = None


class TaskRequest(BaseModel):
    instruction: str = Field(min_length=3, max_length=2000)
    randomize: bool = False
    seed: int | None = Field(default=None,ge=0,le=2**32-1)
    shuffle_sides: bool = True
    force_failure: bool = False


class TaskResponse(BaseModel):
    task_id: str
    status: ExecutionState
    message: str
