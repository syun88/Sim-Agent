#!/usr/bin/env python3
"""Check a body against the operations the skills require, and print what it does not answer.

Run against the simulated cell today; run it against a real arm's bridge before trusting a hardware port.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app.simulation import conformance  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--describe', action='store_true', help='print the contract instead of checking a body')
    args = parser.parse_args(argv)
    if args.describe:
        print(json.dumps(conformance.describe(), ensure_ascii=False, indent=2))
        return 0
    from backend.app.simulation.openarm_v2.spawn import private_spawn_environment
    problems = conformance.check(private_spawn_environment())
    if problems:
        print(json.dumps(problems, ensure_ascii=False, indent=2))
        print(f'{len(problems)} problem(s): this body would not be driven blindly')
        return 1
    print('conforming: every operation the skills require answered as documented')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
