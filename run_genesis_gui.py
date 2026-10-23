#!/usr/bin/env python3
"""One native Genesis owner. The web process brokers commands and measured progress only."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import time
import uuid

from dotenv import load_dotenv
import httpx

from backend.app.agent.world_state import build_world_state
from backend.app.execution.task_executor import TaskExecutor
from backend.app.planning.agent_planner import AgentPlanner, PlannerProviderError
from backend.app.planning.scene_selection import scene_for_instruction
from backend.app.simulation import create_backend
from backend.app.robots.registry import get_profile, selected_robot, remember_selection
from backend.app.simulation.randomization import SpawnSamplingError

BACKEND_URL = 'http://127.0.0.1:8000'


def make_executor(env):
    if getattr(env, 'robot_id', 'openarm_v2') == 'openarm_v2':
        return TaskExecutor(env)
    from backend.app.robots.executor import ImportedRobotExecutor
    return ImportedRobotExecutor(env)


def main() -> None:
    from backend.app.simulation.runtime_owner import acquire_genesis_owner
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--headless', action='store_true', help='Same real physics/executor without a native window')
    parser.add_argument('--once', action='store_true', help='Exit after executing one command (integration verification)')
    parser.add_argument('--backend-url', default=BACKEND_URL)
    parser.add_argument('--scene', choices=['cubes', 'wine'], default='cubes', help='Initial Genesis scene')
    parser.add_argument('--robot', help='Imported robot ID; default is the saved selection or OpenArm')
    args = parser.parse_args()
    acquire_genesis_owner()
    load_dotenv(Path(__file__).resolve().parent / '.env')
    # Fully grounded requests need no model round trip; unfamiliar language still
    # uses the configured model. An explicit shell/.env value takes precedence.
    os.environ.setdefault('OPENARM_FAST_LANGUAGE', '1')
    planner = AgentPlanner()
    viewer_id = str(uuid.uuid4())
    client = httpx.Client(base_url=args.backend_url, timeout=2.0)
    # Register before constructing a scene: a second launch cannot open a second viewer.
    response = client.post('/api/genesis/register', json={'viewer_id': viewer_id, 'provider': planner.configured_provider})
    response.raise_for_status()
    os.environ['GENESIS_SHOW_VIEWER'] = '0' if args.headless else '1'
    robot_id = get_profile(args.robot or selected_robot())['id']
    print(f'Building {robot_id} on Genesis...', flush=True)
    env = create_backend(args.scene if robot_id == 'openarm_v2' else 'cubes', robot_id=robot_id)
    env.step(200)
    executor = make_executor(env)
    active_id = None
    last_publish = 0.0
    last_control = 0.0
    last_viewer_refresh = 0.0
    stop_cache = False

    def publish(event=None, force=False, **payload):
        nonlocal last_publish
        if not force and time.monotonic() - last_publish < .75:
            return
        body = {'viewer_id': viewer_id, 'task_id': active_id, 'provider': planner.configured_provider,
                'world_state': build_world_state(env), **payload}
        if event:
            body['event'] = event
        try:
            response = client.post('/api/genesis/state', json=body)
            response.raise_for_status()
            last_publish = time.monotonic()
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 409:
                raise RuntimeError('Another viewer took ownership; stopping this simulation') from None
        except httpx.TransportError:
            pass

    def should_stop():
        nonlocal last_control, stop_cache, last_viewer_refresh
        # Private IK/collision searches can run without advancing physics.
        # Pump the native window on the same main thread while they work.
        if time.monotonic()-last_viewer_refresh>.05:
            env.refresh_viewer()
            last_viewer_refresh=time.monotonic()
        if active_id and time.monotonic() - last_control > .25:
            publish()
            try:
                stop_cache = client.get(f'/api/genesis/control/{active_id}', timeout=.5).json().get('stop', False)
                last_control = time.monotonic()
            except httpx.TransportError:
                pass
        return stop_cache

    def emit(event, **payload):
        status = payload.pop('status', 'VERIFYING' if event == 'verifying' else 'EXECUTING')
        publish(event, force=True, status=status, **payload)
        env.refresh_viewer()

    def observe_step():
        # SPH can make a 0.4s simulated gripper wait take many wall-clock seconds.
        # Keep the lease/control responsive even inside env.step(large_count).
        if active_id and should_stop():
            raise InterruptedError('Task stopped between physics steps')

    env.step_observer = observe_step
    env.planning_pulse=should_stop

    def prepare_scene(profile):
        nonlocal env, executor, last_publish
        if profile == env.scene_profile:
            return
        state = executor.world_state()
        if any(state['robot'][side + '_arm']['held_object'] for side in ('left', 'right')):
            raise PlannerProviderError('SCENE_BUSY', 'Place the held object or reset before changing the scene')
        publish('scene_preparing', force=True, status='PREPARING_SCENE', detail='Preparing ' + profile + ' scene')
        client.post('/api/genesis/register', json={'viewer_id': viewer_id, 'provider': planner.configured_provider}).raise_for_status()
        env.close()
        try:
            env = create_backend(profile)
        except Exception:
            import genesis as gs
            gs.destroy()
            env = create_backend('cubes')
            env.step_observer = observe_step
            executor = make_executor(env)
            publish(force=True)
            raise
        executor = make_executor(env)
        env.step_observer = observe_step
        env.planning_pulse=should_stop
        env.step(200)
        last_publish = 0.
        publish('scene_ready', force=True, status='PLANNING', detail=None)

    def prepare_robot(robot_id):
        nonlocal env, executor, last_publish
        current_id = getattr(env, 'robot_id', 'openarm_v2')
        profile = get_profile(robot_id)
        robot_id = profile['id']
        if current_id == robot_id and (profile.get('builtin') or
                profile.get('configuration_sha256') == getattr(env, 'profile', {}).get('configuration_sha256')):
            return
        if any(executor.world_state()['robot'][side + '_arm']['held_object'] for side in ('left', 'right')):
            raise PlannerProviderError('SCENE_BUSY', 'Release the held object or reset before switching robots')
        publish('robot_preparing', force=True, status='PREPARING_SCENE', detail='Loading ' + robot_id)
        client.post('/api/genesis/register', json={'viewer_id': viewer_id, 'provider': planner.configured_provider}).raise_for_status()
        previous_scene = env.scene_profile
        env.close()
        try:
            env = create_backend('cubes', robot_id=robot_id)
        except Exception:
            import genesis as gs
            gs.destroy()
            env = create_backend(previous_scene, robot_id=current_id)
            executor = make_executor(env)
            env.step_observer = observe_step
            publish(force=True)
            raise
        executor = make_executor(env)
        env.step_observer = observe_step
        env.planning_pulse=should_stop
        env.step(200)
        remember_selection(robot_id)
        last_publish = 0.
        publish('robot_ready', force=True, status='EXECUTING', detail=None)

    print(f'Ready: {planner.configured_provider}; arm motion {executor.controller.motion_speed:g}x; '
          f'target pacing {env.simulation_speed:g}x '
          f'(limited by computation). Open {args.backend_url}/simple', flush=True)
    with ThreadPoolExecutor(max_workers=1) as planning_pool:
        while True:
            env.refresh_viewer()
            publish()
            try:
                response = client.get('/api/genesis/next', params={'viewer_id': viewer_id})
                response.raise_for_status()
                command = response.json()
            except httpx.HTTPError:
                time.sleep(.1)
                continue
            if not command.get('action'):
                env.step(5)
                time.sleep(.04)
                continue
            active_id = command['task_id']
            stop_cache, last_control = False, 0.0
            body = {'viewer_id': viewer_id, 'task_id': active_id, 'instruction': command.get('instruction')}
            try:
                if should_stop():
                    body.update(success=False, status='STOPPED')
                elif command['action'] == 'reset':
                    publish('scene_preparing',force=True,status='PREPARING_SCENE',detail='Sampling reachable desk positions')
                    env.reset(randomize=bool(command.get('randomize')),seed=command.get('seed'),
                              shuffle_sides=command.get('shuffle_sides',True))
                    executor.reset_context()
                    env.step(200)
                    body.update(success=True, verification={'success': True, 'goal_type': 'scene_reset',
                        'randomization':env.spawn_validation.get('randomization')}, final_world_state=build_world_state(env))
                elif command['action'] == 'select_robot':
                    prepare_robot(command['robot_id'])
                    body.update(success=True, verification={'success': True, 'goal_type': 'robot_selection', 'robot_id': command['robot_id']},
                                final_world_state=build_world_state(env))
                else:
                    if getattr(env, 'robot_id', 'openarm_v2') == 'openarm_v2':
                        prepare_scene(scene_for_instruction(command['instruction'], env.scene_profile))
                    if command.get('randomize'):
                        publish('scene_preparing',force=True,status='PREPARING_SCENE',detail='Sampling reachable desk positions')
                        env.reset(randomize=True,seed=command.get('seed'),shuffle_sides=command.get('shuffle_sides',True))
                        executor.reset_context()
                        env.step(200)
                    state = executor.world_state()
                    publish('planning', force=True, status='PLANNING')
                    # Only language parsing runs in a worker. Genesis stays on the main
                    # thread, including its native macOS viewer and every motor command.
                    future = planning_pool.submit(planner.plan, command['instruction'], state, [])
                    while not future.done() and not should_stop():
                        env.refresh_viewer()
                        publish(status='PLANNING')
                        should_stop()
                        time.sleep(.04)
                    if should_stop():
                        future.cancel()
                        body.update(success=False, status='STOPPED')
                    else:
                        plan = future.result()
                        publish('plan_generated', force=True, status='VALIDATING', plan=plan.model_dump())
                        print(f"Instruction: {command['instruction']}\nProvider: {plan.provider}; actions: {len(plan.program['actions'])}", flush=True)
                        body.update(executor.run(plan, emit=emit, should_stop=should_stop, force_failure=command.get('force_failure', False)))
                print(f"Task {body.get('status', 'SUCCEEDED' if body.get('success') else 'FAILED')}", flush=True)
            except PlannerProviderError as exc:
                body.update(success=False, detail=f'{exc.category}: {exc}', final_world_state=build_world_state(env))
            except SpawnSamplingError as exc:
                body.update(success=False,detail=f'RANDOMIZATION_UNREACHABLE: {exc}',final_world_state=build_world_state(env))
            except Exception as exc:
                # Never serialize provider HTTP exceptions or environment credentials.
                body.update(success=False, detail=f'Execution stopped: {type(exc).__name__}', final_world_state=build_world_state(env))
                print(body['detail'], flush=True)
            client.post('/api/genesis/result', json=body, timeout=5).raise_for_status()
            active_id = None
            publish(force=True)
            if args.once:
                break


if __name__ == '__main__':
    main()
