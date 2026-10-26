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
from backend.app.agent.task_agent import TaskAgent
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
    # The native window needs a working GL context, and on this host it does not get one: measured
    # 2026-10-26, the native path sat at "Building ... on Genesis" for minutes at ~26-58% CPU while the
    # identical scene built in 20 s with the viewer off (the same 14 s the test suite sees). So an
    # explicit decision in the environment is kept - GENESIS_SHOW_VIEWER=0 runs the same scene through
    # the web UI - and only an unset value follows --headless.
    if args.headless:
        os.environ['GENESIS_SHOW_VIEWER'] = '0'
    else:
        os.environ.setdefault('GENESIS_SHOW_VIEWER', '1')
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
    scene_has_tasks = False

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
        if scene_has_tasks:
            raise PlannerProviderError('SCENE_CHANGE_REQUIRED',
                '現在の作業結果を維持しています。新しい指示のためにシーンを作り直しません。別のシーンは起動時に --scene で選択してください。')
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
        nonlocal env, executor, last_publish, scene_has_tasks
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
        scene_has_tasks = False
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
        def language_call(fn, *arguments):
            # The provider works off an immutable snapshot in a worker. Every
            # Genesis observation, reset, controller and window stays here.
            future = planning_pool.submit(fn, *arguments)
            while not future.done() and not should_stop():
                env.refresh_viewer()
                publish(status='PLANNING')
                time.sleep(.04)
            if should_stop():
                future.cancel()
                raise InterruptedError('Stopped during language reasoning')
            return future.result()
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
                    scene_has_tasks = False
                    env.step(200)
                    body.update(success=True, verification={'success': True, 'goal_type': 'scene_reset',
                        'randomization':env.spawn_validation.get('randomization')}, final_world_state=build_world_state(env))
                elif command['action'] == 'diagnose_pinch':
                    # Run the same skills the task path runs, inside this process, and report the contact
                    # numbers instead of a verdict: q, the pinch verdict, the pad links and the candidate
                    # contacts. This is the measurement that separates world state from agent path.
                    arm = command.get('arm', 'left'); obj = command.get('object', 'red_cube')
                    saved_observer = env.step_observer
                    if command.get('observer') is False:
                        env.step_observer = None
                    rows = []
                    for skill in ('approach_object', 'grasp_object', 'lift_object'):
                        out = getattr(executor.skills, skill)(arm, obj)
                        evidence = env.pinch_evidence(obj, arm) if hasattr(env, 'pinch_evidence') else {}
                        rows.append({'skill': skill, 'success': out.get('success'), 'error': out.get('error'),
                                     'gripper_position': float(env.gripper_position(arm)),
                                     'pinch_established': bool(evidence.get('established')),
                                     'pads': sorted({c.get('robot_link') for c in
                                                     (evidence.get('candidate_contacts') or [])}),
                                     'object_position': [round(float(v), 5) for v in env.object_pose(obj)[0]],
                                     'raw_contacts': sorted(env.object_contacts(obj, arm))})
                        if not out.get('success'):
                            break
                    env.step_observer = saved_observer
                    body.update(success=all(row['success'] for row in rows), verification={'success': True,
                        'goal_type': 'pinch_diagnostic', 'observer': command.get('observer') is not False,
                        'rows': rows}, final_world_state=build_world_state(env))
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
                        scene_has_tasks = False
                        env.step(200)
                    publish('planning', force=True, status='PLANNING')
                    agent = TaskAgent(executor,planner,
                        plan_call=lambda *a: language_call(planner.plan,*a),
                        recovery_call=lambda *a: language_call(planner.assess_failure,*a))
                    scene_has_tasks = True
                    body.update(agent.run(command['instruction'],emit=emit,should_stop=should_stop,
                        force_failure=command.get('force_failure',False)))
                print(f"Task {body.get('status', 'SUCCEEDED' if body.get('success') else 'FAILED')}", flush=True)
            except PlannerProviderError as exc:
                body.update(success=False, detail=f'{exc.category}: {exc}', final_world_state=build_world_state(env))
            except SpawnSamplingError as exc:
                body.update(success=False,detail=f'RANDOMIZATION_UNREACHABLE: {exc}',final_world_state=build_world_state(env))
            except InterruptedError:
                body.update(success=False,status='STOPPED',detail='Stopped during observation or reasoning',final_world_state=build_world_state(env))
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
