"""Shared fixtures.

The MuJoCo backend is the default and runs everywhere. The Genesis backend is optional: its
tests run only with `SIM_AGENT_TEST_GENESIS=1` because starting it compiles Genesis kernels and
builds the scene (tens of seconds on CPU).
"""

from __future__ import annotations

import os

import pytest

from backend.app.simulation.genesis_backend.environment import genesis_available

GENESIS_FLAG = "SIM_AGENT_TEST_GENESIS"
ENGINES = ("mujoco", "genesis")


def genesis_testing_enabled() -> bool:
    return os.getenv(GENESIS_FLAG) == "1"


def require_genesis() -> None:
    if not genesis_testing_enabled():
        pytest.skip(f"set {GENESIS_FLAG}=1 to test the optional Genesis World engine")
    if not genesis_available():
        pytest.skip("genesis-world is not installed in this environment")


@pytest.fixture(scope="session")
def genesis_env():
    """One Genesis scene for the whole session: building it is the expensive part."""
    require_genesis()
    from backend.app.simulation.genesis_backend.environment import GenesisOpenArmV2Environment

    env = GenesisOpenArmV2Environment()
    yield env


@pytest.fixture
def mujoco_env():
    from backend.app.simulation.openarm_v2.environment import OpenArmV2Environment

    env = OpenArmV2Environment()
    yield env


@pytest.fixture(params=ENGINES)
def env(request, mujoco_env, genesis_env):
    """Every engine, reset to the same initial scene before each test."""
    if request.param == "genesis":
        genesis_env.reset()
        return genesis_env
    mujoco_env.reset()
    return mujoco_env
