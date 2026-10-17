from __future__ import annotations

import base64
import io
import os
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
from backend.app.simulation import available_backends, create_backend

load_dotenv(Path(__file__).resolve().parents[2] / ".env")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.env = create_backend()
    app.state.memory = MemoryStore()
    app.state.engine = ExecutionEngine(app.state.env, app.state.memory)
    app.state.engine.subscribe(lambda event: app.state.latest_events.append(event))
    app.state.latest_events = list(app.state.engine.events)
    yield
    app.state.memory.close()


app = FastAPI(title="Embodied AI Agent — OpenArm v2", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"], allow_methods=["*"], allow_headers=["*"])


class Health(BaseModel):
    status: str
    robot: str
    simulation: str
    simulation_engine: str
    simulation_authoritative: bool
    grasp_mode: str
    provider: str
    memory: str
    agent: str


@app.get("/api/health", response_model=Health)
def health() -> Health:
    engine: ExecutionEngine = app.state.engine
    env = app.state.env
    return Health(
        status="ok",
        robot="OpenArm v2",
        simulation="Connected",
        simulation_engine=env.name,
        simulation_authoritative=bool(next(item["authoritative"] for item in available_backends() if item["name"] == env.name)),
        grasp_mode=env.grasp_mode,
        provider="ai&" if engine.planner.enabled else "Fallback",
        memory="Connected" if app.state.memory.available else "Fallback",
        agent=engine.state,
    )


@app.get("/api/system/status")
def system_status():
    return health().model_dump()


@app.get("/api/backends")
def backends():
    """Describe the available engines and which one this process selected."""
    return {"active": app.state.env.name, "backends": available_backends()}


@app.get("/api/scene")
def scene():
    return build_world_state(app.state.env)


@app.post("/api/scene/reset")
def reset_scene(randomize: bool = False):
    engine: ExecutionEngine = app.state.engine
    if engine._task and not engine._task.done():
        raise HTTPException(409, "Cannot reset while a task is active")
    app.state.env.reset(randomize=randomize)
    return {"status": "ok", "randomized": randomize}


@app.post("/api/tasks", response_model=TaskResponse)
async def create_task(request: TaskRequest):
    engine: ExecutionEngine = app.state.engine
    try:
        task_id = engine.start(request)
        return TaskResponse(task_id=task_id, status=engine.state, message="Task accepted")
    except RuntimeError as exc:
        raise HTTPException(409, str(exc))


@app.get("/api/tasks/{task_id}")
def get_task(task_id: str):
    engine: ExecutionEngine = app.state.engine
    for record in reversed(engine.executions):
        if record["task_id"] == task_id:
            return record
    if engine.current and engine.current.get("task_id") == task_id:
        return engine.current
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
    return app.state.engine.executions


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
    engine: ExecutionEngine = app.state.engine
    env = app.state.env
    return {
        "simulation_engine": env.name,
        "simulation_backend_env": os.getenv("SIM_BACKEND", "mujoco"),
        "ai_provider": "ai&" if engine.planner.enabled else "Fallback",
        "ai_model": os.getenv("AI_AND_MODEL", ""),
        "ai_base_url_configured": bool(os.getenv("AI_AND_BASE_URL")),
        "neo4j_uri_configured": bool(os.getenv("NEO4J_URI")),
        "neo4j": "Connected" if app.state.memory.available else "Fallback",
        "neo4j_last_error": app.state.memory.last_error,
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    queue: list[dict] = []
    unsubscribe = app.state.engine.subscribe(lambda event: queue.append(event))
    try:
        while True:
            state = build_world_state(app.state.env)
            frame = app.state.env.frame()
            rendered = app.state.env.render(720, 480)
            image = None
            if rendered is not None:
                buffer = io.BytesIO()
                Image.fromarray(rendered).save(buffer, format="JPEG", quality=80)
                image = "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()
            payload = {
                "type": "simulation_frame",
                "engine": app.state.env.name,
                "grasp_mode": app.state.env.grasp_mode,
                "state": state,
                "sim_time": frame.sim_time,
                "qpos": frame.qpos,
                "contacts": frame.contacts,
                "events": queue[:30],
                "agent_state": app.state.engine.state,
                "current": app.state.engine.current,
                "image": image,
            }
            queue.clear()
            await websocket.send_json(payload)
            await asyncio.sleep(0.1)
    except WebSocketDisconnect:
        pass
    finally:
        unsubscribe()


import asyncio


frontend_dist = Path(__file__).resolve().parents[2] / "frontend" / "dist"
if frontend_dist.exists():
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")
