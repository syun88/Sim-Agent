"""Engine selection for the OpenArm v2 Cell simulation.

MuJoCo is the default and stays authoritative. Genesis World is opt-in, either with
`SIM_BACKEND=genesis` or through `create_backend("genesis")`.
"""

from __future__ import annotations

import os

from backend.app.simulation.base import Frame, SimulationBackend

__all__ = ["Frame", "SimulationBackend", "available_backends", "create_backend"]

DEFAULT_BACKEND = "mujoco"


def available_backends() -> list[dict]:
    """Describe every engine this installation can start."""
    from backend.app.simulation.genesis_backend.environment import genesis_available

    return [
        {
            "name": "mujoco",
            "available": True,
            "authoritative": True,
            "grasp_mode": "contact-assisted-weld",
            "description": "Official openarm-mujoco 2.3.0 Cell model, MuJoCo physics and rendering.",
        },
        {
            "name": "genesis",
            "available": genesis_available(),
            "authoritative": False,
            "grasp_mode": "kinematic-grasp-hold",
            "description": (
                "Same official Cell model on Genesis World physics; grasp hold is a labelled "
                "kinematic fallback. Requires the 'genesis' extra."
            ),
        },
    ]


def create_backend(name: str | None = None) -> SimulationBackend:
    """Create the selected simulation backend."""
    selected = (name or os.getenv("SIM_BACKEND") or DEFAULT_BACKEND).strip().lower()
    if selected == "mujoco":
        from backend.app.simulation.openarm_v2.environment import OpenArmV2Environment

        return OpenArmV2Environment()
    if selected == "genesis":
        from backend.app.simulation.genesis_backend.environment import (
            GenesisOpenArmV2Environment,
        )

        return GenesisOpenArmV2Environment(
            prefer_gpu=os.getenv("GENESIS_USE_GPU", "") == "1",
            show_viewer=os.getenv("GENESIS_SHOW_VIEWER", "") == "1",
        )
    raise ValueError(f"Unknown SIM_BACKEND {selected!r}; use 'mujoco' or 'genesis'")
