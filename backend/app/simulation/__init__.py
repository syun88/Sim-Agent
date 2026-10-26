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


def create_backend(scene_profile: str | None = None, *, robot_id: str = "openarm_v2") -> SimulationBackend:
    """Create the simulation backend."""
    profile = scene_profile or os.getenv("OPENARM_SCENE", "cubes")
    gpu = prefer_gpu_from_environment()
    if robot_id != "openarm_v2":
        if profile == "wine":
            # Pouring used to be refused outright for imported robots. The wine world is engine-generic; what it needs
            # is a gripper that can physically hold the bottle, which the profile records after a verified pinch.
            from backend.app.robots import registry
            bottle_profile = registry.get_profile(robot_id)
            grippers = bottle_profile.get('grippers') or {}
            aperture = min((g.get('maximum_aperture') or 0.) for g in grippers.values()) if grippers else 0.
            if not grippers or not bottle_profile.get('grasp_calibrated'):
                raise ValueError("Wine manipulation needs a calibrated physical gripper profile: this robot has "
                                 f"grippers={bool(grippers)} grasp_calibrated="
                                 f"{bool(bottle_profile.get('grasp_calibrated'))}")
            if aperture < .05:
                raise ValueError(f"Wine manipulation needs an aperture that can hold a bottle; this gripper opens "
                                 f"to {aperture * 1000:.1f} mm")
        from backend.app.simulation.genesis_backend.generic_environment import GenericURDFEnvironment
        return GenericURDFEnvironment(robot_id, show_viewer=os.getenv("GENESIS_SHOW_VIEWER", "") == "1",
                                      prefer_gpu=gpu, scene_profile=profile)
    if profile == "wine" and GPU_ENVIRONMENT_VARIABLE not in os.environ:
        import torch
        gpu = torch.backends.mps.is_available() or torch.cuda.is_available()
    return GenesisOpenArmV2Environment(
        prefer_gpu=gpu,
        show_viewer=os.getenv("GENESIS_SHOW_VIEWER", "") == "1",
        scene_profile=profile,
    )
