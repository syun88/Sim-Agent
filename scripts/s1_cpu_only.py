"""Deselect the tests that need a live Genesis world, for a CPU-only run.

This plugin used to live in a scratch directory next to the traces, and that directory was removed from disk
outside this repository (measured 2026-11-09: `/home/yuu/dojo/lime-verify-20261025/` disappeared, taking the
plugin with it, and the suite could no longer even collect). A test-suite dependency belongs in the repository.

Usage:  PYTHONPATH=scripts python -m pytest -q backend/tests -p s1_cpu_only
"""

from __future__ import annotations

import pytest

#: Fixtures that build or step a live Genesis world. A test that requests one of these is deselected, which is
#: the honest rule: the dependency is the fixture, not the file name. Measured 2026-11-09: an earlier list of
#: file names deselected 25 where the real answer was 29, and the four extras failed at setup instead.
GENESIS_FIXTURES = ('genesis_env', 'fresh_engine_state', 'env', 'controller')


def _needs_genesis(item) -> bool:
    requested = getattr(item, 'fixturenames', ()) or ()
    return any(name in requested for name in GENESIS_FIXTURES)


def pytest_collection_modifyitems(config, items):
    keep, dropped = [], []
    for item in items:
        (dropped if _needs_genesis(item) else keep).append(item)
    if dropped:
        config.hook.pytest_deselected(items=dropped)
        items[:] = keep
