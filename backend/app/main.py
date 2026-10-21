from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.app.memory.graph_memory import MemoryStore
from backend.app.models.schemas import TaskRequest, TaskResponse
from backend.app.planning.agent_planner import AgentPlanner
from backend.app.simulation import ENGINE_DESCRIPTION

load_dotenv(Path(__file__).resolve().parents[2] / '.env')
genesis_command_queue: list[dict] = []
TERMINAL = {'SUCCEEDED', 'FAILED', 'STOPPED'}


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.memory = MemoryStore()
    app.state.engine = None
    app.state.executions = []
    app.state.world_state = None
    app.state.viewer = {}
    genesis_command_queue.clear()
    yield
    app.state.memory.close()


app = FastAPI(title='OpenArm v2 — Language tasks on Genesis', version='2.0.0', lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=['http://localhost:5173', 'http://127.0.0.1:5173'], allow_methods=['*'], allow_headers=['*'])
simple_page = Path(__file__).resolve().parent / 'static' / 'simple.html'


def _record(task_id: str) -> dict:
    for record in reversed(app.state.executions):
        if record['task_id'] == task_id:
            return record
    raise HTTPException(404, 'Task not found')


def _online() -> bool:
    return app.state.world_state is not None and time.monotonic() - app.state.viewer.get('last_seen', -100) < 10


def _owns_scene(viewer_id: str | None):
    if app.state.viewer and app.state.viewer.get('viewer_id') != viewer_id:
        raise HTTPException(409, 'This command belongs to the registered Genesis viewer')


@app.get('/simple', include_in_schema=False)
def simple_commands():
    return FileResponse(simple_page)


@app.get('/api/health')
def health():
    active = next((r for r in reversed(app.state.executions) if r['status'] not in TERMINAL), None)
    return {'status': 'ok', 'robot': 'OpenArm v2', 'simulation': 'Genesis viewer process',
            'simulation_engine': 'genesis', 'simulation_description': ENGINE_DESCRIPTION,
            'grasp_mode': 'physical-friction-pinch', 'provider': app.state.viewer.get('provider', AgentPlanner().configured_provider),
            'memory': 'Connected' if app.state.memory.available else 'Fallback',
            'agent': active['status'] if active else 'IDLE', 'viewer_online': _online(),
            'viewer_starting': app.state.world_state is None and time.monotonic() < app.state.viewer.get('starting_until', 0)}


@app.get('/api/system/status')
def system_status():
    return health()


@app.get('/api/scene')
def scene():
    if app.state.world_state is None:
        raise HTTPException(503, 'Waiting for the Genesis viewer to publish its measured scene')
    return app.state.world_state


def _enqueue(command: dict) -> dict:
    if len(genesis_command_queue) >= 32:
        raise HTTPException(429, 'Command queue is full')
    task_id = str(uuid.uuid4())
    command['task_id'] = task_id
    genesis_command_queue.append(command)
    record = {'task_id': task_id, 'instruction': command.get('instruction', 'Scene reset'),
              'status': 'QUEUED_FOR_GENESIS_VIEWER', 'events': [], 'created_at': time.time()}
    app.state.executions.append(record)
    return record


@app.post('/api/scene/reset')
def reset_scene(randomize: bool = False):
    record = _enqueue({'action': 'reset', 'randomize': randomize})
    return {'status': 'queued', 'randomized': randomize, 'task_id': record['task_id']}


@app.post('/api/tasks', response_model=TaskResponse)
async def create_task(request: TaskRequest):
    record = _enqueue({'action': 'task', **request.model_dump()})
    return TaskResponse(task_id=record['task_id'], status='PLANNING', message='Task queued for Genesis viewer')


@app.post('/api/genesis/register', include_in_schema=False)
def register_viewer(body: dict):
    viewer_id = body.get('viewer_id')
    if not viewer_id:
        raise HTTPException(400, 'viewer_id is required')
    lease_live = _online() or time.monotonic() < app.state.viewer.get('starting_until', 0)
    if lease_live and app.state.viewer.get('viewer_id') != viewer_id:
        raise HTTPException(409, 'A Genesis viewer already owns this scene')
    app.state.viewer = {'viewer_id': viewer_id, 'last_seen': time.monotonic(), 'starting_until': time.monotonic() + 300,
                        'provider': body.get('provider', 'deterministic_fallback')}
    app.state.world_state = None
    return {'status': 'registered'}


@app.get('/api/genesis/next', include_in_schema=False)
def genesis_next_command(viewer_id: str | None = None):
    """Destructive queue read, reserved for the one registered simulation owner."""
    _owns_scene(viewer_id)
    while genesis_command_queue:
        command = genesis_command_queue.pop(0)
        record = _record(command['task_id'])
        if record['status'] == 'STOPPED':
            continue
        record['status'] = 'PLANNING'
        return command
    return {'action': None}


@app.get('/api/genesis/control/{task_id}', include_in_schema=False)
def control(task_id: str):
    return {'stop': _record(task_id)['status'] in {'STOP_REQUESTED', 'STOPPED'}}


@app.post('/api/genesis/state', include_in_schema=False)
def viewer_state(body: dict):
    if app.state.viewer.get('viewer_id') != body.get('viewer_id'):
        raise HTTPException(409, 'Viewer registration changed; this process must stop')
    app.state.viewer['last_seen'] = time.monotonic()
    app.state.viewer['starting_until'] = 0
    if body.get('world_state'):
        app.state.world_state = body['world_state']
    if body.get('provider'):
        app.state.viewer['provider'] = body['provider']
    task_id = body.get('task_id')
    if task_id:
        record = _record(task_id)
        if record['status'] not in TERMINAL | {'STOP_REQUESTED'}:
            record['status'] = body.get('status', record['status'])
        for field in ('plan', 'step_index', 'step', 'result', 'detail'):
            if field in body:
                record[field] = body[field]
        if body.get('event'):
            record['events'] = (record['events'] + [{'kind': body['event'], 'time': time.time(), 'step_index': body.get('step_index')}])[-200:]
    return {'status': 'ok'}


@app.post('/api/genesis/result', include_in_schema=False)
def genesis_result(result: dict[str, Any]):
    _owns_scene(result.get('viewer_id'))
    record = _record(result.get('task_id', ''))
    for field in ('plan', 'skills', 'verification', 'detail', 'retry_count', 'total_duration', 'initial_world_state', 'final_world_state'):
        if field in result:
            record[field] = result[field]
    stopped = record['status'] == 'STOP_REQUESTED' or result.get('status') == 'STOPPED'
    record['status'] = 'STOPPED' if stopped else 'SUCCEEDED' if result.get('success') else 'FAILED'
    record['success'] = bool(result.get('success')) and not stopped
    if result.get('final_world_state'):
        app.state.world_state = result['final_world_state']
    return {'status': 'ok'}


@app.get('/api/tasks/{task_id}')
def get_task(task_id: str):
    return _record(task_id)


@app.post('/api/tasks/{task_id}/stop')
def stop_task(task_id: str):
    record = _record(task_id)
    if record['status'] in TERMINAL:
        return {'status': record['status']}
    if record['status'] == 'QUEUED_FOR_GENESIS_VIEWER':
        genesis_command_queue[:] = [c for c in genesis_command_queue if c['task_id'] != task_id]
        record['status'] = 'STOPPED'
    else:
        record['status'] = 'STOP_REQUESTED'
    return {'status': record['status']}


@app.get('/api/executions')
def executions():
    return app.state.executions


@app.get('/api/memory')
def memory():
    store = app.state.memory
    return {'available': store.available, 'last_error': store.last_error, 'history': store.history()}


@app.post('/api/memory/reconnect')
def reconnect_memory():
    app.state.memory._connect()
    return {'available': app.state.memory.available, 'last_error': app.state.memory.last_error}


@app.get('/api/config/status')
def config_status():
    return {'simulation_engine': 'genesis', 'ai_provider': health()['provider'],
            'ai_model': app.state.viewer.get('model'), 'viewer_online': _online(),
            'neo4j': 'Connected' if app.state.memory.available else 'Fallback'}


@app.websocket('/ws')
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    await websocket.send_json({'type': 'info', 'message': 'The native Genesis viewer owns the simulation. Poll /api/scene and /api/executions for measured state.'})
    await websocket.close()


frontend_dist = Path(__file__).resolve().parents[2] / 'frontend' / 'dist'
app.add_api_route('/', simple_commands, include_in_schema=False)
if frontend_dist.exists():
    app.mount('/', StaticFiles(directory=frontend_dist, html=True), name='frontend')
