"""Shared fixtures.

Genesis World is the engine this project runs, so the suite requires it: a machine without
`genesis-world` gets a collection error telling it how to install it rather than a silently green
run. Building the scene is the expensive part (tens of seconds on CPU, it compiles kernels), so one
scene is built for the session and every test resets it to the documented initial state.
"""

from __future__ import annotations

from typing import Any

import pytest

from backend.app.simulation import create_backend
from backend.app.simulation.genesis_backend.environment import genesis_available

INSTALL_HINT = (
    "genesis-world is not installed. Genesis World is this project's simulation engine; "
    "install it with `uv sync` (or `pip install genesis-world`) in a Python 3.12 environment."
)


#: State observed on the engine the moment it was constructed, before any test could touch it.
#: Read through the `fresh_engine_state` fixture, not imported: pytest loads this file as its own
#: module, so a direct import from a test would see a second, empty copy of this dict.
_FRESH_STATE: dict[str, Any] = {}


@pytest.fixture
def upward_flight():
    """Synthetic recorded ballistic measurements, not a live simulation success claim."""
    return [{'t':.24+i*.002, 'position':[0.,0.,1.2+i*.002-4.905*(i*.002)**2],
             'velocity':[0.,0.,1.-9.81*i*.002], 'contacts':[]} for i in range(65)]


@pytest.fixture(scope="session")
def genesis_env():
    if not genesis_available():
        pytest.exit(INSTALL_HINT, returncode=1)
    env = create_backend()
    # Capture now: the only moment the engine is untouched. The server builds one backend at startup
    # and never resets it, so a fresh engine has to already be in the documented scene.
    _FRESH_STATE.update(
        arm_left=env.arm_qpos("left").copy(),
        arm_right=env.arm_qpos("right").copy(),
        objects={name: env.object_pose(name)[0].copy() for name in ("red_cube", "blue_cube")},
        zones={name: env.zone_position(name).copy() for name in ("blue_zone", "green_zone")},
    )
    yield env


@pytest.fixture(scope="session")
def fresh_engine_state(genesis_env) -> dict[str, Any]:
    """The engine's state the moment it was constructed, before any test touched it."""
    return _FRESH_STATE


@pytest.fixture
def env(genesis_env):
    """The engine, reset to the documented initial scene before each test."""
    genesis_env.reset()
    return genesis_env


@pytest.fixture
def controller(env):
    from backend.app.control.motion import OpenArmV2Controller

    return OpenArmV2Controller(env)


@pytest.fixture(autouse=True)
def _clean_body_measurements():
    """The body's recorded landing errors are process-global: reset them around every test.

    Without this, one test's blocked carry raises the expected error for the next test and gates its
    placements off - a real hazard in the live process too, which is why the estimate needs several samples.
    """
    from backend.app.skills import motion_accuracy
    motion_accuracy.reset()
    yield
    motion_accuracy.reset()
