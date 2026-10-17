from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field

SkillName = Literal[
    "approach_object", "grasp_object", "lift_object", "move_to_target",
    "release_object", "retreat", "return_home",
]
ArmName = Literal["left", "right"]
ExecutionState = Literal[
    "IDLE", "PLANNING", "VALIDATING", "EXECUTING", "VERIFYING",
    "SUCCEEDED", "FAILED", "REPLANNING", "STOPPED",
]


class PlanStep(BaseModel):
    skill: SkillName
    arm: ArmName = "left"
    object: str | None = None
    target: str | None = None
    fault_offset: float | None = None
    approach_offset: float | None = None


class AgentPlan(BaseModel):
    goal: dict[str, str]
    arm: ArmName = "left"
    steps: list[PlanStep]
    provider: str = "deterministic_fallback"
    model: str | None = None
    provider_error: str | None = None
    memory_used: list[dict] = Field(default_factory=list)
    memory_decision: str | None = None
    planning_latency: float | None = None
    raw_response: str | None = None


class TaskRequest(BaseModel):
    instruction: str = Field(min_length=3, max_length=500)
    randomize: bool = False
    force_failure: bool = False


class TaskResponse(BaseModel):
    task_id: str
    status: ExecutionState
    message: str
