#!/usr/bin/env python3
"""How much pose error would this system survive? Computed from the margins it already recorded.

Every safeguard in this repo is a margin (approach clearance, placement accuracy, home clearance). On the
simulator the object poses are ground truth, so none of those margins has ever had to absorb a perception
error - which is exactly what a real arm will hand it. This reads executed records and reports, per task, the
smallest margin left over: that number is the pose error the run would still have tolerated.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _floats(value):
    return [float(item) for item in value] if isinstance(value, (list, tuple)) else []


def headroom(record: dict) -> list[dict]:
    """One row per margin the run recorded, with the leftover that a pose error would have to fit into."""
    rows = []
    for attempt in record.get('agent_attempts') or []:
        for step in ((attempt.get('result') or {}).get('skills') or []):
            planning = step.get('planning') or {}
            evidence = step.get('evidence') or {}
            if planning.get('measured_clearance_m') is not None and planning.get('screen_clearance_m') is not None:
                rows.append({'kind': 'approach_screen', 'skill': step.get('skill'),
                             'margin_m': float(planning['measured_clearance_m']),
                             'required_m': float(planning['screen_clearance_m'])})
            search = step.get('destination_search') or evidence.get('destination_search') or {}
            feasibility = search.get('feasibility') or {}
            if feasibility.get('required_m') is not None and feasibility.get('estimated_landing_error_m') is not None:
                rows.append({'kind': 'placement_accuracy', 'skill': step.get('skill'),
                             'margin_m': float(feasibility['required_m']),
                             'required_m': float(feasibility['estimated_landing_error_m'])})
            tightest = evidence.get('tightest') or {}
            if tightest.get('gap_m') is not None:
                rows.append({'kind': 'home_route', 'skill': step.get('skill'),
                             'margin_m': float(tightest['gap_m']), 'required_m': 0.0})
    for row in rows:
        row['headroom_m'] = round(row['margin_m'] - row['required_m'], 5)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--input', required=True, help='execution record file or directory')
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args(argv)
    source = Path(args.input)
    files = sorted(source.rglob('*.json')) if source.is_dir() else [source]
    rows, skipped = [], 0
    for path in files:
        try:
            record = json.loads(path.read_text())
        except Exception:
            skipped += 1
            continue
        for row in headroom(record):
            rows.append({'task': record.get('task_id'), 'instruction': record.get('instruction'), **row})
    if args.json:
        print(json.dumps({'rows': rows, 'skipped': skipped}, ensure_ascii=False, indent=2))
        return 0
    if not rows:
        print(f'no margins recorded in {len(files)} record(s) ({skipped} unreadable)')
        return 0
    by_kind = {}
    for row in rows:
        by_kind.setdefault(row['kind'], []).append(row['headroom_m'])
    print(f'{len(rows)} margin(s) from {len(files)} records')
    for kind, values in sorted(by_kind.items()):
        values.sort()
        worst = values[0]
        print(f'  {kind:<20} n={len(values):<4} tightest={worst*1000:8.1f} mm   median={values[len(values)//2]*1000:8.1f} mm')
    tightest = min(rows, key=lambda row: row['headroom_m'])
    print(f'  pose error this run would have survived: {tightest["headroom_m"]*1000:.1f} mm '
          f'({tightest["kind"]}, {tightest.get("instruction") or tightest.get("task")})')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
