#!/usr/bin/env python3
"""Try to build each registered scene, one at a time, with a time budget; report what actually works.

A scene in the registry is a claim ("this can be built"). Measured 2026-11-10: the `mixed` scene ran for more
than ten minutes without finishing, so the claim has to be checked rather than assumed. Each build runs in its
own process so a hang can be killed and does not take the check with it.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.simulation import scenes  # noqa: E402

PROBE = '''
import json, os, sys
sys.path.insert(0, {root!r})
from backend.app.simulation import create_backend
try:
    env = create_backend({scene!r})
    env.step(5)
    print(json.dumps({{'ok': True, 'objects': list(env.frame().objects), 'profile': env.scene_profile}}))
except Exception as error:
    print(json.dumps({{'ok': False, 'error': f'{{type(error).__name__}}: {{error}}'}}))
'''


def probe(scene: str, timeout: float) -> dict:
    code = PROBE.format(root=str(ROOT), scene=scene)
    try:
        completed = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True,
                                   timeout=timeout, cwd=str(ROOT))
    except subprocess.TimeoutExpired:
        return {'ok': False, 'error': f'did not finish within {timeout:.0f}s'}
    line = (completed.stdout or '').strip().splitlines()[-1:] or ['{}']
    try:
        return json.loads(line[0])
    except json.JSONDecodeError:
        return {'ok': False, 'error': (completed.stderr or 'no output').strip().splitlines()[-1:][0][:200]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--timeout', type=float, default=180.)
    parser.add_argument('--only', default=None, help='a single scene id')
    args = parser.parse_args(argv)
    results = {}
    for scene in ([args.only] if args.only else scenes.ids()):
        result = probe(scene, args.timeout)
        results[scene] = result
        state = 'OK  ' if result.get('ok') else 'FAIL'
        print(f'{state} {scene:<7} {result.get("error") or ", ".join(result.get("objects") or [])}')
    return 0 if all(item.get('ok') for item in results.values()) else 1


if __name__ == '__main__':
    raise SystemExit(main())
