"""Deselect the tests that need a live Genesis world, for a CPU-only run.

This plugin used to live in a scratch directory next to the traces, and that directory was removed from disk
outside this repository (measured 2026-11-09: `/home/yuu/dojo/lime-verify-20261025/` disappeared, taking the
plugin with it, and the suite could no longer even collect). A test-suite dependency belongs in the repository.

Usage:  PYTHONPATH=scripts python -m pytest -q backend/tests -p s1_cpu_only
"""

from __future__ import annotations

import pytest

#: Tests that build or step a Genesis world. They are deselected, never skipped silently: the count is printed
#: by pytest's summary line ("N deselected") so a shrinking suite stays visible.
GENESIS_TESTS = (
    'test_genesis_smoke.py',
    'test_generic_env_kinematics.py',
    'test_official_model_fidelity.py',
    'test_physics_relations.py',
)


def pytest_collection_modifyitems(config, items):
    if config.getoption('-k') and 'genesis' in str(config.getoption('-k')):
        return
    keep, dropped = [], []
    for item in items:
        if any(name in str(item.fspath) for name in GENESIS_TESTS):
            dropped.append(item)
        else:
            keep.append(item)
    if dropped:
        config.hook.pytest_deselected(items=dropped)
        items[:] = keep
