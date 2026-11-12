#!/usr/bin/env python3
"""Compare local planners on a fixed instruction battery, and report what each actually produced.

The question this answers is "should we change the model?", and the answer has to be a measurement, not an
impression. Every model is given the same instructions, the same scene and the same validation; the report
records the failure category, whether the plan passed `validate_plan`, the actions it produced, and the
latency. Adoption is decided by the numbers printed here - the script adopts nothing by itself.

Planner-only: no simulator, no motion. Physics is not measured here and must not be inferred from it.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

#: A scene both cubes exist in, the same one the failure reports used.
SCENE = {'objects': [{'id': 'red_cube', 'type': 'cube', 'position': [.44, .15, 1.031]},
                     {'id': 'blue_cube', 'type': 'cube', 'position': [.50, -.22, 1.031]},
                     {'id': 'wine_bottle', 'type': 'bottle', 'position': [.52, .02, 1.10]},
                     {'id': 'wine_glass', 'type': 'glass', 'position': [.40, -.05, 1.03]}],
         'targets': [{'id': 'left_side', 'position': [.46, .30, 1.008]},
                     {'id': 'right_side', 'position': [.46, -.30, 1.008]},
                     {'id': 'center', 'position': [.50, 0.0, 1.008]},
                     {'id': 'blue_zone', 'position': [.48, .18, 1.008]},
                     {'id': 'green_zone', 'position': [.48, -.20, 1.008]}],
         'robot': {'available_arms': ['left', 'right'],
                   'left_arm': {'held_object': None}, 'right_arm': {'held_object': None}},
         'workspace': {'table_height_m': 1.008}}

#: The instructions this project actually failed on, plus ordinary ones and one unsupported request.
BATTERY = (
    ('single_pick', '右手で青いキューブを掴んで'),
    ('stack_on_table', '青いキューブを赤いキューブに乗せて'),
    ('stack_on_held', '右手でキューブを掴んでから、掴んでいる状態のキューブの上に赤いキューブを重ねて'),
    ('contrastive_pair', 'キューブを、もう一方の手に持たせて、そのキューブの上にもう一つのキューブを重ねて'),
    ('handover_then_stack', '青いキューブを右手に渡して右においてから、赤いキューブを青いキューブに乗せて'),
    ('two_placements', '赤いキューブを左手で左においてから、青いキューブを右手で右において'),
    ('hand_reference', '右手にあるキューブを取って左手に渡して'),
    ('relative_move', '赤いキューブを右に5cm動かしてください'),
    ('throw', '青いキューブを上に投げて'),
    ('pour', 'ワインをグラスに20ml注いで'),
    ('english_stack', 'Put the red cube on top of the blue cube.'),
    ('unsupported', '机を拭いて'),
)


def provenance():
    """Which device actually served the models, so a latency number is not attributed to the wrong one."""
    info = {}
    for name, command in (('ollama_ps', ['ollama', 'ps']),
                          ('rocm', ['rocm-smi', '--showmeminfo', 'vram'])):
        try:
            info[name] = subprocess.run(command, capture_output=True, text=True, timeout=10).stdout.strip()
        except Exception as error:
            info[name] = f'unavailable: {type(error).__name__}'
    return info


def run_model(model: str, battery, *, capability_mode: bool, timeout_note=None):
    os.environ['OPENARM_LOCAL_PLANNER'] = '1'
    os.environ['OPENARM_OLLAMA_MODEL'] = model
    os.environ['OPENARM_CAPABILITY_MODE'] = '1' if capability_mode else '0'
    for key in ('AI_AND_API_KEY', 'AI_AND_BASE_URL', 'AI_AND_MODEL'):
        os.environ.pop(key, None)
    from backend.app.planning.agent_planner import AgentPlanner, PlannerProviderError, validate_plan

    rows = []
    for name, instruction in battery:
        started = time.perf_counter()
        row = {'case': name, 'instruction': instruction}
        try:
            plan = AgentPlanner().plan(instruction, SCENE)
        except PlannerProviderError as error:
            row.update(outcome='refused', category=error.category, detail=str(error)[:200])
        except Exception as error:  # a provider failure must be reported, not crash the comparison
            row.update(outcome='error', category=type(error).__name__, detail=str(error)[:200])
        else:
            valid, errors = validate_plan(plan, SCENE)
            row.update(outcome='planned', valid=bool(valid), errors=errors[:3],
                       actions=[(a.get('op'), a.get('object') or a.get('target')) for a in
                                (plan.program or {}).get('actions', [])],
                       provider=plan.provider, model_called=bool(plan.model_called),
                       grounded=bool((plan.planning_evidence or {}).get('scene_bindings')))
        row['seconds'] = round(time.perf_counter() - started, 2)
        rows.append(row)
    planned = [row for row in rows if row['outcome'] == 'planned']
    return {'model': model, 'capability_mode': capability_mode,
            'planned': len(planned), 'valid': sum(1 for row in planned if row.get('valid')),
            'refused': sum(1 for row in rows if row['outcome'] == 'refused'),
            'errors': sum(1 for row in rows if row['outcome'] == 'error'),
            'median_seconds': statistics.median([row['seconds'] for row in rows]) if rows else None,
            'cases': len(rows), 'rows': rows}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--models', nargs='+', required=True)
    parser.add_argument('--out', default='-')
    parser.add_argument('--no-capability-mode', action='store_true',
                        help='let the grammar answer first instead of forcing the model')
    args = parser.parse_args(argv)

    report = {'battery': [name for name, _ in BATTERY], 'provenance': provenance(), 'models': []}
    for model in args.models:
        print(f'--- {model}', file=sys.stderr)
        report['models'].append(run_model(model, BATTERY, capability_mode=not args.no_capability_mode))
    body = json.dumps(report, ensure_ascii=False, indent=1)
    if args.out == '-':
        print(body)
    else:
        Path(args.out).write_text(body, encoding='utf-8')
    print(f"{'model':<22}{'planned':>8}{'valid':>7}{'refused':>9}{'error':>7}{'median s':>10}", file=sys.stderr)
    for entry in report['models']:
        print(f"{entry['model']:<22}{entry['planned']:>8}{entry['valid']:>7}{entry['refused']:>9}"
              f"{entry['errors']:>7}{entry['median_seconds']:>10}", file=sys.stderr)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
