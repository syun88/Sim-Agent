#!/usr/bin/env python3
"""Run one fixed battery against the running cell and print the counts.

The point is to answer "did that safeguard help?" with the *same* layouts and the *same* instructions. The
flags are read by the GUI process at run time, so a configuration is chosen by launching the cell with the
environment variables set; this script only runs the battery and reports.

Usage:
    OPENARM_MEASURE_DESTINATION_SEARCH=0 python scripts/ab_measurements.py --label all-off --out DIR
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

BATTERY = ('青いキューブを右手で青いゾーンに置いて',
           '赤いキューブを拾って青いゾーンに置いて',
           '青いキューブを赤いキューブの上に置いて',
           '赤いキューブを左手で左において')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--out', required=True)
    parser.add_argument('--label', default='configuration')
    parser.add_argument('--seeds', default='11,12,13', help='layout seeds: identical across configurations')
    parser.add_argument('--base-url', default='http://127.0.0.1:8000')
    args = parser.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    flags = {name: os.environ.get(f'OPENARM_MEASURE_{name.upper()}', '1')
             for name in ('approach_margin', 'destination_search', 'arm_preflight', 'home_margin')}
    print(f'[{args.label}] flags={flags}', flush=True)
    for seed in [value.strip() for value in args.seeds.split(',') if value.strip()]:
        for instruction in BATTERY:
            target = out / f'{args.label}-seed{seed}-{abs(hash(instruction)) % 10000}'
            result = subprocess.run(
                [sys.executable, str(Path(__file__).with_name('collect_traces.py')),
                 '--base-url', args.base_url, '--out', str(target), '--trials', '1',
                 '--randomize', '--seed', seed, '--instruction', instruction,
                 '--budget-seconds', '300'],
                capture_output=True, text=True)
            line = (result.stdout or result.stderr).strip().splitlines()[-1:] or ['no output']
            print(f'  seed {seed} {instruction[:26]:<26} {line[0][:90]}', flush=True)
    records = []
    for path in out.glob('*/*.json'):
        try:
            records.append(json.loads(path.read_text()))
        except Exception:
            continue
    succeeded = sum(1 for record in records if record.get('status') == 'SUCCEEDED')
    print(f'[{args.label}] {succeeded}/{len(records)} SUCCEEDED', flush=True)
    (out / f'{args.label}-summary.json').write_text(json.dumps(
        {'label': args.label, 'flags': flags, 'succeeded': succeeded, 'total': len(records)}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
