"""Resolve the demo scene's references to the official OpenArm v2 assets.

`assets/scenes/openarm_v2_demo.xml` is authored with path tokens instead of
machine-specific absolute paths, so one XML string can be compiled on any machine
and fed to more than one engine. The official model itself is never copied or
substituted: every referenced file comes from the installed `openarm-mujoco`
package.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from openarm_mujoco.v2 import asset_path

ROOT = Path(__file__).resolve().parents[4]
DEMO_SCENE_PATH = ROOT / "assets" / "scenes" / "openarm_v2_demo.xml"

CELL_XML_TOKEN = "@OPENARM_MUJOCO_V2_CELL_XML@"
ASSETS_DIR_TOKEN = "@OPENARM_MUJOCO_V2_ASSETS_DIR@"

#: Cube placement of the demonstration scene.
INITIAL_OBJECT_POSES = {
    "red_cube": (0.44, 0.15, 1.032),
    "blue_cube": (0.50, -0.22, 1.032),
}
def initial_object_positions(randomize: bool = False, seed: int | None = None,
                             shuffle_sides: bool = True) -> dict[str, np.ndarray]:
    """Default setup or a broad IK-admitted layout; explicit seeds reproduce it."""
    positions = {name: np.array(pose, dtype=float) for name, pose in INITIAL_OBJECT_POSES.items()}
    if not randomize:
        return positions
    from backend.app.simulation.openarm_v2.spawn import random_spawn_layout,private_spawn_environment
    return random_spawn_layout(private_spawn_environment(),seed=seed,shuffle_sides=shuffle_sides)[0]


def official_cell_xml() -> str:
    """Absolute path of the official OpenArm v2 Cell model."""
    return asset_path("cell/cell.xml")


def official_assets_dir() -> str:
    """Absolute path of the official OpenArm v2 asset directory."""
    return asset_path("assets")


def resolved_scene_xml(path: Path | str = DEMO_SCENE_PATH) -> str:
    """Return the demo scene XML with the official asset references resolved."""
    text = Path(path).read_text()
    return text.replace(CELL_XML_TOKEN, official_cell_xml()).replace(
        ASSETS_DIR_TOKEN, official_assets_dir()
    )
