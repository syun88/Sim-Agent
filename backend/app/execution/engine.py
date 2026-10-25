from __future__ import annotations

import asyncio
import time
import uuid
from typing import Callable

from backend.app.execution.task_executor import TaskExecutor
from backend.app.memory.graph_memory import MemoryStore
from backend.app.models.schemas import TaskRequest
from backend.app.planning.agent_planner import AgentPlanner
from backend.app.simulation.base import SimulationBackend


class ExecutionEngine:
    def __init__(self, env: SimulationBackend, memory: MemoryStore):
        self.env = env
        self.memory = memory
        self.executor = TaskExecutor(env)
        self.controller = self.executor.controller
        self.skills = self.executor.skills
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
            self.executor.reset_context()
        self._task = asyncio.create_task(self._run(request))
        return task_id

    def stop(self) -> None:
        if self._task and not self._task.done():
            self.controller.cancel()
            self.state = "STOPPED"
            self.emit("task_stopped")

    async def _run(self, request: TaskRequest):
        assert self.current
        self.emit("task_started", task_id=self.current["task_id"], instruction=request.instruction)
        executor = self.executor
        try:
            state = executor.world_state()
            loop = asyncio.get_running_loop()
            plan = await loop.run_in_executor(None, lambda: self.planner.plan(request.instruction, state, []))
            self.state = "VALIDATING"
            self.emit("plan_generated", plan=plan.model_dump())
            self.state = "EXECUTING"
            def emit(kind, **payload):
                if kind == "verifying":
                    self.state = "VERIFYING"
                self.emit(kind, **payload)
            result = await loop.run_in_executor(None, lambda: executor.run(plan, emit=emit, should_stop=lambda: self.state == "STOPPED", force_failure=request.force_failure))
            self.state = result["status"]
            record = {**self.current, **result, "goal_type": plan.goal["type"],
                      "object_ids": [o["id"] for o in result["final_world_state"]["objects"]],
                      "failure_category": result["verification"]["failure_category"],
                      "attempts": [{"plan": result["plan"], "skills": result["skills"], "verification": result["verification"], "status": result["status"]}],
                      "skills": [s["skill"] for s in result["skills"]], "planner_provider": plan.provider}
            self.executions.append(record)
            self.memory.save_execution(record)
            self.emit("task_finished", status=self.state, record=record)
        except Exception as exc:
            self.state = "FAILED"
            self.emit("task_error", error=f"Execution stopped: {type(exc).__name__}")
        finally:
            self.controller.cancelled = False
