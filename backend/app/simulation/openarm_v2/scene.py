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
#: Seed of the documented randomization, shared so every engine randomizes identically.
RANDOM_SEED = 20261017


def initial_object_positions(randomize: bool = False, seed: int = RANDOM_SEED) -> dict[str, np.ndarray]:
    """Cube placement for a scene reset, identical on every engine.

    `randomize` is a documented scene reset/setup operation: it moves the cubes before a task
    starts and never during an execution. One generator serves every draw, so the two red-cube
    axes are independent samples rather than the same sample repeated.
    """
    positions = {name: np.array(pose, dtype=float) for name, pose in INITIAL_OBJECT_POSES.items()}
    if not randomize:
        return positions
    rng = np.random.default_rng(seed)
    positions["red_cube"] = np.array(
        [rng.uniform(0.37, 0.42), rng.uniform(0.11, 0.18), 1.032]
    )
    positions["blue_cube"] = positions["blue_cube"] + np.array(
        [rng.uniform(-0.012, 0.012), rng.uniform(-0.012, 0.012), 0.0]
    )
    return positions


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
