from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
import os

from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.app.memory.graph_memory import MemoryStore
from backend.app.models.schemas import TaskRequest, TaskResponse
from backend.app.robots.registry import available_robots, get_profile, import_robot, selected_robot
from pydantic import BaseModel
from fastapi import Query
from backend.app.simulation.randomization import resolve_seed
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


app = FastAPI(title='Sim-Agent — Language tasks on Genesis', version='2.1.0', lifespan=lifespan)
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
    robot = (app.state.world_state or {}).get('robot', {})
    return {'status': 'ok', 'service': 'sim-agent', 'robot': robot.get('name', 'OpenArm v2'),
            'robot_id': robot.get('type', 'openarm_v2'), 'simulation': 'Genesis viewer process',
            'simulation_engine': 'genesis', 'simulation_description': ENGINE_DESCRIPTION if robot.get('type', 'openarm_v2') == 'openarm_v2'
            else 'Genesis World, imported URDF: ' + robot.get('name', 'robot'),
            'grasp_mode': (app.state.world_state or {}).get('grasp_mode', 'physical-friction-pinch'), 'provider': app.state.viewer.get('provider', AgentPlanner().configured_provider),
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
    if command.get('randomize'):
        record['randomization_request']={k:command.get(k) for k in ('seed','shuffle_sides')}
    app.state.executions.append(record)
    return record


@app.post('/api/diagnostics/pinch')
def diagnose_pinch(object: str = 'red_cube', arm: str = 'left', observer: bool = True):
    """Ask the viewer process to grasp and lift one object and report the contact numbers.

    The viewer owns the Genesis world; the API process cannot reach it. A diagnostic that runs inside
    that process is the only way to tell "the app's world behaves differently" from "the agent path does",
    which is the question left open by measurements that agree everywhere else.
    """
    # ``observer=False`` detaches the per-step callback for the duration of the diagnostic, so the same
    # sequence can be run with and without it in the same process - the viewer's per-step callback is the
    # one thing an in-process probe cannot reproduce.
    return _enqueue({'action': 'diagnose_pinch', 'object': object, 'arm': arm,
                     'observer': bool(observer),
                     'instruction': f'pinch diagnostic: {object}'})


@app.post('/api/scene/reset')
def reset_scene(randomize: bool = False,seed:int|None=Query(default=None,ge=0,le=2**32-1),shuffle_sides:bool=True):
    seed=resolve_seed(seed) if randomize else None
    record = _enqueue({'action': 'reset', 'randomize': randomize,'seed':seed,'shuffle_sides':shuffle_sides})
    return {'status': 'queued', 'randomized': randomize,'seed':seed,'shuffle_sides':shuffle_sides,'task_id': record['task_id']}


@app.post('/api/tasks', response_model=TaskResponse)
async def create_task(request: TaskRequest):
    body=request.model_dump()
    if request.randomize:body['seed']=resolve_seed(request.seed)
    record = _enqueue({'action': 'task', **body})
    return TaskResponse(task_id=record['task_id'], status='PLANNING', message='Task queued for Genesis viewer')


class RobotImport(BaseModel):
    folder: str


class RobotSelection(BaseModel):
    robot_id: str


@app.get('/api/robots')
def robots():
    robot_id = (app.state.world_state or {}).get('robot', {}).get('type') or selected_robot()
    return {'robots': available_robots(), 'selected': robot_id}


@app.post('/api/robots/import')
def import_urdf(request: RobotImport):
    try:
        return import_robot(request.folder)
    except (ValueError, OSError) as exc:
        raise HTTPException(400, str(exc)) from None


@app.post('/api/robots/select')
def select_robot(request: RobotSelection):
    if any(record['status'] not in TERMINAL for record in app.state.executions):
        raise HTTPException(409, 'Finish or stop the current task before switching robots')
    try:
        profile = get_profile(request.robot_id)
    except (ValueError, OSError) as exc:
        raise HTTPException(400, str(exc)) from None
    record = _enqueue({'action': 'select_robot', 'robot_id': profile['id'], 'instruction': 'Select robot: ' + profile['id']})
    return {'task_id': record['task_id'], 'status': 'queued'}


@app.get('/api/robots/{robot_id}/files/{filename}')
def robot_file(robot_id: str, filename: str):
    if filename not in {'gripper_analysis.json', 'profile.json', 'scene.json', 'ik_seeds.json', 'ik_validation.json', 'spawn_validation.json',
                        'runtime_validation.json', 'auto_grasp_validation.json', 'gripper_runtime.json', 'handover_skills.json', 'solve_ik.py', 'README.md', 'robot.urdf'}:
        raise HTTPException(404, 'Unknown generated file')
    try:
        profile = get_profile(robot_id)
        path = Path(profile['directory']) / filename
    except (ValueError, OSError, KeyError):
        raise HTTPException(404, 'Generated robot profile not found') from None
    if not path.is_file():
        raise HTTPException(404, 'Generated file not found')
    return FileResponse(path, filename=filename)


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
        for field in ('plan', 'step_index', 'step', 'result', 'detail', 'agent_trace'):
            if field in body:
                record[field] = body[field]
        if body.get('event'):
            record['events'] = (record['events'] + [{'kind': body['event'], 'time': time.time(), 'step_index': body.get('step_index')}])[-200:]
    return {'status': 'ok'}


@app.post('/api/genesis/result', include_in_schema=False)
def genesis_result(result: dict[str, Any]):
    _owns_scene(result.get('viewer_id'))
    record = _record(result.get('task_id', ''))
    for field in ('plan', 'skills', 'verification', 'detail', 'retry_count', 'total_duration', 'initial_world_state', 'final_world_state', 'agent_trace', 'agent_attempts', 'agent_recoveries', 'post_task_homing', 'diagnosis'):
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


@app.get('/api/physics')
def physics_relations():
    """The physical relations the skills evaluate, with units and assumptions.

    The planner is told what exists and asks by name ("the release speed that gives this rise"), the way
    it asks IK for a joint solution; it never does the arithmetic itself.
    """
    from backend.app.physics import GRAVITY, describe
    return {'gravity_m_s2': GRAVITY, 'relations': describe()}


@app.post('/api/physics/evaluate')
def physics_evaluate(request: dict):
    """Evaluate one named relation. Unknown names are an error, never a silent zero."""
    from backend.app.physics import evaluate
    try:
        return evaluate(request.get('relation', ''), **(request.get('arguments') or {}))
    except KeyError as error:
        raise HTTPException(status_code=400, detail=str(error)) from None


@app.get('/api/config/status')
def config_status():
    # Report the planner's own switches. Without them an operator cannot tell a request that used the
    # model from one the grammar answered: both used to report the same provider.
    planner = AgentPlanner()
    return {'simulation_engine': 'genesis', 'ai_provider': health()['provider'],
            'ai_model': app.state.viewer.get('model'), 'viewer_online': _online(),
            'neo4j': 'Connected' if app.state.memory.available else 'Fallback',
            'planner': {'provider': planner.configured_provider, 'enabled': planner.enabled,
                        'model': planner.local_model if planner.local_enabled else planner.model,
                        'capability_mode': planner.capability_mode, 'shadow_mode': planner.shadow_mode,
                        'fast_language': planner.exact_grammar_fast_path},
            # Two systems, two jobs: deliberation is slow and writes the plan, judgement is fast and
            # typed. Keeping them separate means a monitor can be swapped without touching the planner.
            'roles': {'system2_planner': planner.local_model if planner.local_enabled else planner.model,
                      'system1_monitor': (os.getenv('OPENARM_DECISION_URL', '').strip() or None),
                      'physics': '/api/physics'}}


@app.websocket('/ws')
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    await websocket.send_json({'type': 'info', 'message': 'The native Genesis viewer owns the simulation. Poll /api/scene and /api/executions for measured state.'})
    await websocket.close()


frontend_dist = Path(__file__).resolve().parents[2] / 'frontend' / 'dist'
app.add_api_route('/', simple_commands, include_in_schema=False)
if frontend_dist.exists():
    app.mount('/', StaticFiles(directory=frontend_dist, html=True), name='frontend')
