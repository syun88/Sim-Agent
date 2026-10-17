from __future__ import annotations

import asyncio
import time
import uuid
from typing import Awaitable, Callable

from backend.app.agent.world_state import build_world_state
from backend.app.control.motion import OpenArmV2Controller
from backend.app.memory.graph_memory import MemoryStore
from backend.app.models.schemas import AgentPlan, TaskRequest
from backend.app.planning.agent_planner import AgentPlanner, validate_plan
from backend.app.simulation.openarm_v2.environment import OpenArmV2Environment
from backend.app.skills.robot_skills import SkillRunner
from backend.app.verification.goal_verification import verify_place_goal


class ExecutionEngine:
    def __init__(self, env: OpenArmV2Environment, memory: MemoryStore):
        self.env = env
        self.memory = memory
        self.controller = OpenArmV2Controller(env)
        self.skills = SkillRunner(env, self.controller)
        self.planner = AgentPlanner()
        self.events: list[dict] = []
        self.executions: list[dict] = []
        self.current: dict | None = None
        self.state = "IDLE"
        self._task: asyncio.Task | None = None
        self._listeners: list[Callable[[dict], None]] = []

    def emit(self, kind: str, **payload):
        event = {"timestamp": time.time(), "kind": kind, "state": self.state, **payload}
        self.events.append(event)
        if len(self.events) > 500:
            self.events = self.events[-500:]
        for listener in list(self._listeners):
            try:
                listener(event)
            except Exception:
                pass

    def subscribe(self, listener: Callable[[dict], None]):
        self._listeners.append(listener)
        return lambda: self._listeners.remove(listener)

    def start(self, request: TaskRequest) -> str:
        if self._task and not self._task.done():
            raise RuntimeError("Another task is active")
        task_id = str(uuid.uuid4())
        self.current = {
            "task_id": task_id,
            "execution_id": str(uuid.uuid4()),
            "instruction": request.instruction,
            "randomize": request.randomize,
            "force_failure": request.force_failure,
            "retry_count": 0,
            "status": "PLANNING",
        }
        self.state = "PLANNING"
        if request.randomize:
            self.env.reset(randomize=True)
        self._task = asyncio.create_task(self._run(request))
        return task_id

    def stop(self) -> None:
        if self._task and not self._task.done():
            self.controller.cancel()
            self.state = "STOPPED"
            self.emit("task_stopped")

    async def _run(self, request: TaskRequest):
        assert self.current
        started = time.perf_counter()
        self.emit("task_started", task_id=self.current["task_id"], instruction=request.instruction)
        try:
            attempts = []
            success = False
            for attempt in range(3):
                self.current["attempt"] = attempt + 1
                state = build_world_state(self.env)
                self.current["initial_world_state"] = state
                self.state = "PLANNING"
                self.emit("memory_retrieval_started")
                obj = state["objects"][0]["id"]
                memory = self.memory.retrieve("place_in_zone", obj)
                self.emit("memory_retrieved", count=len(memory), memories=memory[-3:])
                plan = self.planner.plan(request.instruction, state, memory)
                if attempt == 0 and request.force_failure and plan.steps:
                    plan.steps[0].fault_offset = -0.12
                    self.emit("controlled_fault_injected", detail="First attempt uses an unreachable approach offset")
                self.state = "VALIDATING"
                self.emit("plan_generated", plan=plan.model_dump())
                valid, errors = validate_plan(plan, state)
                self.emit("plan_validated", valid=valid, errors=errors)
                if not valid:
                    attempts.append({"attempt": attempt + 1, "status": "FAILED", "error": "INVALID_PLAN", "details": errors})
                    break
                self.state = "EXECUTING"
                loop = asyncio.get_running_loop()
                progress = lambda phase: self.emit("skill_progress", phase=phase, attempt=attempt + 1)
                results = await loop.run_in_executor(None, lambda: self.skills.execute_plan(plan.model_dump()["steps"], progress))
                self.emit("skills_completed", results=results)
                failed_skill = next((r for r in results if not r.get("success", False)), None)
                goal = plan.goal
                obj_id = goal.get("object") or (state["objects"][0]["id"])
                target_id = goal.get("target") or "blue_zone"
                self.state = "VERIFYING"
                verification = verify_place_goal(self.env, obj_id, target_id)
                success = verification["success"] and not (request.force_failure and attempt == 0)
                self.emit("verification_completed", verification=verification, ai_claimed_success=success)
                attempts.append({
                    "attempt": attempt + 1,
                    "plan": plan.model_dump(),
                    "skills": results,
                    "verification": verification,
                    "status": "SUCCEEDED" if success else "FAILED",
                })
                if success:
                    break
                if attempt < 2:
                    self.state = "REPLANNING"
                    self.emit("replanning", failed_skill=failed_skill, verification=verification)
                    self.controller.cancelled = False
                    self.env.reset(randomize=attempt == 1)
            self.state = "SUCCEEDED" if success else "FAILED"
            final_state = build_world_state(self.env)
            record = {
                "task_id": self.current["task_id"],
                "execution_id": self.current["execution_id"],
                "instruction": request.instruction,
                "goal_type": "place_in_zone",
                "object_ids": [o["id"] for o in final_state["objects"]],
                "skills": [r.get("skill") for a in attempts for r in a.get("skills", [])],
                "success": success,
                "failure_category": None if success else attempts[-1].get("verification", {}).get("failure_category", "SIMULATION_ERROR"),
                "attempts": attempts,
                "retry_count": len(attempts) - 1,
                "total_duration": time.perf_counter() - started,
                "final_world_state": final_state,
                "planner_provider": self.planner.provider if hasattr(self.planner, "provider") else None,
            }
            self.executions.append(record)
            self.memory.save_execution(record)
            self.emit("task_finished", status=self.state, record=record)
        except Exception as exc:
            self.state = "FAILED"
            self.emit("task_error", error=f"{type(exc).__name__}: {str(exc).splitlines()[0]}")
        finally:
            self.controller.cancelled = False
