#!/usr/bin/env python3
"""One native Genesis owner. The web process brokers commands and measured progress only."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time
import uuid

from dotenv import load_dotenv
import httpx

from backend.app.agent.world_state import build_world_state
from backend.app.execution.task_executor import TaskExecutor
from backend.app.planning.agent_planner import AgentPlanner, PlannerProviderError
from backend.app.simulation import create_backend

BACKEND_URL = 'http://127.0.0.1:8000'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--headless', action='store_true', help='Same real physics/executor without a native window')
    parser.add_argument('--once', action='store_true', help='Exit after executing one command (integration verification)')
    parser.add_argument('--backend-url', default=BACKEND_URL)
    args = parser.parse_args()
    load_dotenv(Path(__file__).resolve().parent / '.env')
    planner = AgentPlanner()
    viewer_id = str(uuid.uuid4())
    client = httpx.Client(base_url=args.backend_url, timeout=2.0)
    # Register before constructing a scene: a second launch cannot open a second viewer.
    response = client.post('/api/genesis/register', json={'viewer_id': viewer_id, 'provider': planner.configured_provider})
    response.raise_for_status()
    import os
    os.environ['GENESIS_SHOW_VIEWER'] = '0' if args.headless else '1'
    print('Building the official OpenArm v2 scene on Genesis...', flush=True)
    env = create_backend()
    env.step(200)
    executor = TaskExecutor(env)
    active_id = None
    last_publish = 0.0
    last_control = 0.0
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
        nonlocal last_control, stop_cache
        if active_id and time.monotonic() - last_control > .25:
            publish()
            try:
                stop_cache = client.get(f'/api/genesis/control/{active_id}', timeout=.5).json().get('stop', False)
                last_control = time.monotonic()
            except httpx.TransportError:
                pass
        return stop_cache

    def emit(event, **payload):
        status = 'VERIFYING' if event == 'verifying' else 'EXECUTING'
        publish(event, force=True, status=status, **payload)
        env.refresh_viewer()

    print(f'Ready: {planner.configured_provider}. Open {args.backend_url}/simple', flush=True)
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
                    env.reset(randomize=bool(command.get('randomize')))
                    executor.reset_context()
                    env.step(200)
                    body.update(success=True, verification={'success': True, 'goal_type': 'scene_reset'}, final_world_state=build_world_state(env))
                else:
                    if command.get('randomize'):
                        env.reset(randomize=True)
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
