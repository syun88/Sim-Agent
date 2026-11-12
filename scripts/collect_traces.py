#!/usr/bin/env python3
"""Collect execution traces through the running GUI's API, without taking a second Genesis owner.

The simulator has exactly one owner (the viewer process). This script is a *client*: it asks that owner to
reset the scene and to run an instruction, waits for the terminal state, and saves the record. Nothing here
touches physics directly, so no second engine is created and the ownership rule holds.

What it produces is data, not a conclusion: each task's own verification decides what happened, and a batch
of them is what a mechanism predictor or a fine-tune would need. A run that fails is a sample, not a bug in
the collector, and is saved like any other.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

TERMINAL = {'SUCCEEDED', 'FAILED', 'STOPPED'}

#: The cases worth collecting: the ones this project actually failed on, plus two ordinary ones as controls.
DEFAULT_BATTERY = (
    '右手でキューブを掴んでから、掴んでいる状態のキューブの上に赤いキューブを重ねて',
    'キューブを、もう一方の手に持たせて、そのキューブの上にもう一つのキューブを重ねて',
    '青いキューブを右手に渡して右においてから、赤いキューブを青いキューブに乗せて',
    '赤いキューブを左手で左においてから、青いキューブを右手で右において',
)


def _get(url, timeout=5):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.load(response)


def _post(url, payload=None, timeout=15):
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(url, data=data, method='POST',
                                     headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def wait_for(base: str, task_id: str, *, timeout: float, poll: float = 2.0):
    """Poll until the task reaches a terminal state, or give up and say so."""
    deadline = time.monotonic() + timeout
    record = None
    while time.monotonic() < deadline:
        try:
            record = _get(f'{base}/api/tasks/{task_id}')
        except (urllib.error.URLError, TimeoutError) as error:
            print(f'    poll error: {type(error).__name__}', file=sys.stderr)
            time.sleep(poll)
            continue
        if record.get('status') in TERMINAL:
            return record
        time.sleep(poll)
    return {'task_id': task_id, 'status': 'COLLECTOR_TIMEOUT', 'detail': 'collector stopped waiting',
            'partial': record}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--base-url', default='http://127.0.0.1:8000')
    parser.add_argument('--out', required=True, help='directory for the saved execution records')
    parser.add_argument('--trials', type=int, default=1, help='repeats per instruction, each with a fresh scene')
    parser.add_argument('--randomize', action='store_true', help='let the owner sample a reachable layout')
    parser.add_argument('--seed', type=int, default=None, help='first layout seed (incremented per trial)')
    parser.add_argument('--task-timeout', type=float, default=240.)
    parser.add_argument('--budget-seconds', type=float, default=900.)
    parser.add_argument('--instruction', action='append', default=None,
                        help='collect only these instructions (repeatable); defaults to the battery')
    args = parser.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    instructions = args.instruction or list(DEFAULT_BATTERY)
    started = time.monotonic()
    saved = failures = 0

    for index in range(args.trials):
        for instruction in instructions:
            if time.monotonic() - started > args.budget_seconds:
                print('budget exhausted; stopping', file=sys.stderr)
                return 0
            seed = None if args.seed is None else args.seed + index
            query = f'?randomize={str(args.randomize).lower()}&shuffle_sides=true'
            if seed is not None:
                query += f'&seed={seed}'
            reset = _post(f'{args.base_url}/api/scene/reset{query}')
            wait_for(args.base_url, reset['task_id'], timeout=args.task_timeout)
            queued = _post(f'{args.base_url}/api/tasks', {'instruction': instruction})
            record = wait_for(args.base_url, queued['task_id'], timeout=args.task_timeout)
            record['collector'] = {'seed': seed, 'randomize': args.randomize, 'trial': index + 1}
            path = out / f"task-{queued['task_id']}.json"
            path.write_text(json.dumps(record, ensure_ascii=False), encoding='utf-8')
            saved += 1
            failures += int(record.get('status') != 'SUCCEEDED')
            detail = str(record.get('detail') or '')[:110]
            print(f"[{index + 1}/{args.trials}] {record.get('status'):<9} {instruction[:34]:<34} {detail}")
    print(f'saved {saved} records to {out} ({failures} not SUCCEEDED)', file=sys.stderr)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
