from __future__ import annotations

import base64
import io
import os
from typing import Any
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

from backend.app.agent.world_state import build_world_state
from backend.app.execution.engine import ExecutionEngine
from backend.app.memory.graph_memory import MemoryStore
from backend.app.models.schemas import TaskRequest, TaskResponse
from backend.app.planning.agent_planner import AgentPlanner
from backend.app.simulation import ENGINE_DESCRIPTION, create_backend

load_dotenv(Path(__file__).resolve().parents[2] / ".env")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.memory = MemoryStore()
    # The standalone Genesis viewer process (run_genesis_gui.py) creates the
    # simulation environment. This backend only queues instructions and stores
    # results, so it never opens a second Genesis scene or window.
    app.state.engine = None
    app.state.latest_events = []
    app.state.executions: list[dict] = []
    yield
    app.state.memory.close()


app = FastAPI(title="Embodied AI Agent — OpenArm v2", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"], allow_methods=["*"], allow_headers=["*"])

simple_page = Path(__file__).resolve().parent / "static" / "simple.html"

# Simple queue that bridges web instructions to the standalone Genesis viewer process.
genesis_command_queue: list[str] = []


@app.get("/simple", include_in_schema=False)
def simple_commands():
    """A minimal command-only page for operators who do not need the full dashboard."""
    return FileResponse(simple_page)


class Health(BaseModel):
    status: str
    robot: str
    simulation: str
    simulation_engine: str
    simulation_description: str
    grasp_mode: str
    provider: str
    memory: str
    agent: str


@app.get("/api/health", response_model=Health)
def health() -> Health:
    return Health(
        status="ok",
        robot="OpenArm v2",
        simulation="Genesis viewer process",
        simulation_engine="genesis",
        simulation_description=ENGINE_DESCRIPTION,
        grasp_mode="kinematic-grasp-hold",
        provider="ai&" if AgentPlanner().enabled else "Fallback",
        memory="Connected" if app.state.memory.available else "Fallback",
        agent="IDLE",
    )


@app.get("/api/system/status")
def system_status():
    return health().model_dump()


@app.get("/api/scene")
def scene():
    return build_world_state(app.state.env)


@app.post("/api/scene/reset")
def reset_scene(randomize: bool = False):
    # The standalone Genesis viewer performs the reset; the backend just records it.
    return {"status": "ok", "randomized": randomize}


@app.post("/api/tasks", response_model=TaskResponse)
async def create_task(request: TaskRequest):
    import uuid
    task_id = str(uuid.uuid4())
    genesis_command_queue.append(request.instruction)
    app.state.executions.append({
        "task_id": task_id,
        "instruction": request.instruction,
        "status": "QUEUED_FOR_GENESIS_VIEWER",
    })
    return TaskResponse(task_id=task_id, status="PLANNING", message="Task queued for Genesis viewer")


@app.get("/api/genesis/next", include_in_schema=False)
def genesis_next_command():
    """Return the next pending instruction for the standalone Genesis viewer process."""
    if genesis_command_queue:
        return {"instruction": genesis_command_queue.pop(0)}
    return {"instruction": None}


@app.post("/api/genesis/result", include_in_schema=False)
def genesis_result(result: dict[str, Any]):
    """Accept execution results from the standalone Genesis viewer process."""
    for record in reversed(app.state.executions):
        if record["instruction"] == result.get("instruction") and record["status"] == "QUEUED_FOR_GENESIS_VIEWER":
            record["status"] = "SUCCEEDED" if result.get("success") else "FAILED"
            break
    return {"status": "ok", "received": result}


@app.get("/api/tasks/{task_id}")
def get_task(task_id: str):
    for record in reversed(app.state.executions):
        if record["task_id"] == task_id:
            return record
    raise HTTPException(404, "Task not found")


@app.post("/api/tasks/{task_id}/stop")
def stop_task(task_id: str):
    engine: ExecutionEngine = app.state.engine
    if engine.current and engine.current.get("task_id") == task_id:
        engine.stop()
        return {"status": "stopping"}
    raise HTTPException(404, "Active task not found")


@app.get("/api/executions")
def executions():
    return app.state.executions


@app.get("/api/memory")
def memory():
    store = app.state.memory
    return {"available": store.available, "last_error": store.last_error, "history": store.history()}


@app.post("/api/memory/reconnect")
def reconnect_memory():
    store: MemoryStore = app.state.memory
    store._connect()
    return {"available": store.available, "last_error": store.last_error}


@app.get("/api/config/status")
def config_status():
    return {
        "simulation_engine": "genesis",
        "ai_provider": "ai&" if AgentPlanner().enabled else "Fallback",
        "ai_model": os.getenv("AI_AND_MODEL", ""),
        "ai_base_url_configured": bool(os.getenv("AI_AND_BASE_URL")),
        "neo4j_uri_configured": bool(os.getenv("NEO4J_URI")),
        "neo4j": "Connected" if app.state.memory.available else "Fallback",
        "neo4j_last_error": app.state.memory.last_error,
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    # The Genesis viewer renders in its own process. The dashboard image feed is
    # served by the full React app's own polling of /api/executions; the
    # lightweight /simple page does not need a WebSocket at all.
    await websocket.send_json({
        "type": "info",
        "message": "Genesis viewer renders in its own process; no simulation frame feed here.",
    })
    await websocket.close()


import asyncio


frontend_dist = Path(__file__).resolve().parents[2] / "frontend" / "dist"
if frontend_dist.exists():
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")
