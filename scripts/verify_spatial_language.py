"""Exercise the real local model on varied requests, with independent goal checks."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.app.planning.agent_planner import AgentPlanner

WORLD = {'objects': [{'id': 'red_cube', 'type': 'cube', 'position': [.44, .15, 1.031]},
                     {'id': 'blue_cube', 'type': 'cube', 'position': [.5, -.22, 1.031]}],
         'targets': [{'id': 'blue_zone'}, {'id': 'green_zone'}]}
CASES = [
    ('右手で青いキューブを取って左手に渡して左側の赤いキューブの上に乗せて', 'blue_cube', 'red_cube', 'on'),
    ('青いやつを右で拾って、空いてる左にパスしてさ、赤い箱を土台にしてその上にちょこんと乗っけて', 'blue_cube', 'red_cube', 'on'),
    ('Use your right hand to pick up blue, pass it to your left, and stack it on the red cube. Leave red where it is.', 'blue_cube', 'red_cube', 'on'),
    ('赤には触らず、青を赤の上に重ねてください', 'blue_cube', 'red_cube', 'on'),
    ('Put the red cube next to the blue one, on its left.', 'red_cube', 'blue_cube', 'left_of'),
    ('青いブロックを赤いブロックの手前にそっと置いて', 'blue_cube', 'red_cube', 'front_of'),
    ('Pick the red cube with the left hand and keep holding it. Do not pass it.', 'red_cube', None, None),
    ('青を右手で掴んで、上に5cm動かして、そのまま持っていて', 'blue_cube', None, None),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--provider', choices=['local', 'fallback'], default='local')
    parser.add_argument('--json', type=Path, default=ROOT / 'outputs/spatial_language_verified.json')
    args = parser.parse_args()
    os.environ['OPENARM_LOCAL_PLANNER'] = '1' if args.provider == 'local' else '0'
    os.environ['OPENARM_FAST_LANGUAGE'] = '0'
    planner = AgentPlanner()
    planner.api_key = planner.base_url = planner.model = ''
    records = []
    for instruction, obj, target, relation in CASES:
        record = {'instruction': instruction, 'success': False}
        try:
            plan = planner.plan(instruction, WORLD)
            record['plan'] = plan.model_dump()
            checks = plan.goal['checks']
            if target:
                matched = any(c['type'] == 'relative_placement' and c['object'] == obj and c['target'] == target
                              and c['relation'] == relation for c in checks)
                only_moved_object = all(a['object'] in {None, obj} for a in plan.program['actions'])
                record['success'] = matched and only_moved_object
            else:
                record['success'] = (any(c['type'] == 'holding' and c['object'] == obj for c in checks)
                                     and not any(s.skill == 'handover_object' for s in plan.steps))
        except Exception as exc:
            record['error'] = str(exc)
        records.append(record)
        args.json.write_text(json.dumps(records, ensure_ascii=False, indent=2))
        print(record['success'], instruction, record.get('plan', {}).get('provider'), record.get('error', ''), flush=True)
    return 0 if all(r['success'] for r in records) else 1


if __name__ == '__main__':
    raise SystemExit(main())
