"""The simulation engine of this project: Genesis World.

`create_backend()` is the only place an engine is constructed. The agent layers reach it through the
`SimulationBackend` contract, and the model they simulate is the official `openarm-mujoco` 2.3.0
OpenArm v2 Cell, whose description is read from the same MJCF by MuJoCo's compiler
(`backend/app/simulation/openarm_v2/scene.py`) so names and geometry come from one source.
"""

from __future__ import annotations

import os

from backend.app.simulation.base import Frame, SimulationBackend
from backend.app.simulation.genesis_backend.environment import GenesisOpenArmV2Environment

__all__ = ["Frame", "SimulationBackend", "create_backend"]

#: Human-readable description of the running engine, for the API and the dashboard.
ENGINE_DESCRIPTION = "Genesis World, official openarm-mujoco 2.3.0 OpenArm v2 Cell"


#: Set to "1" to ask Genesis World for its GPU backend. The CPU backend is what this project is
#: verified on, and Genesis initialises its device once per process.
GPU_ENVIRONMENT_VARIABLE = "GENESIS_USE_GPU"


def prefer_gpu_from_environment() -> bool:
    """Whether the environment asks for the GPU backend."""
    return os.getenv(GPU_ENVIRONMENT_VARIABLE, "0").strip() == "1"


def create_backend() -> SimulationBackend:
    """Create the simulation backend."""
    return GenesisOpenArmV2Environment(
        prefer_gpu=prefer_gpu_from_environment(),
        show_viewer=os.getenv("GENESIS_SHOW_VIEWER", "") == "1",
    )
