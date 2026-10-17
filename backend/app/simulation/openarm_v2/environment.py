from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
SCENE_PATH = ROOT / "assets" / "scenes" / "openarm_v2_demo.xml"

ARM_JOINTS = {
    "left": [f"openarm_left_joint{i}" for i in range(1, 8)],
    "right": [f"openarm_right_joint{i}" for i in range(1, 8)],
}
ARM_ACTUATORS = {
    "left": [f"left_joint{i}_ctrl" for i in range(1, 8)],
    "right": [f"right_joint{i}_ctrl" for i in range(1, 8)],
}
GRIPPER_ACTUATORS = {"left": "left_finger1_ctrl", "right": "right_finger1_ctrl"}
GRIPPER_OPEN = {"left": 0.0, "right": -0.7854}
GRIPPER_CLOSED = {"left": 0.02, "right": -0.02}
EE_SITES = {"left": "left_ee_control_point", "right": "right_ee_control_point"}


@dataclass
class Frame:
    sim_time: float
    qpos: list[float]
    objects: dict[str, list[float]]
    zones: dict[str, list[float]]
    ee: dict[str, list[float]]
    grippers: dict[str, float]
    contacts: list[str]


class OpenArmV2Environment:
    """Authoritative OpenArm v2 Cell simulation."""

    def __init__(self, model_path: Path = SCENE_PATH):
        self.model_path = Path(model_path)
        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.data = mujoco.MjData(self.model)
        self._object_bodies = {
            name: self._body_id(name) for name in ("red_cube", "blue_cube")
        }
        self._object_joints = {
            name: int(self.model.body_jntadr[body_id])
            for name, body_id in self._object_bodies.items()
        }
        self._zone_sites = {
            name: self._site_id(f"{name}_site") for name in ("blue_zone", "green_zone")
        }
        self._arm_ids = {
            side: [self._joint_id(name) for name in names]
            for side, names in ARM_JOINTS.items()
        }
        self._actuator_ids = {
            side: [self._actuator_id(name) for name in names]
            for side, names in ARM_ACTUATORS.items()
        }
        self._gripper_ids = {
            side: self._actuator_id(name) for side, name in GRIPPER_ACTUATORS.items()
        }
        self._site_ids = {side: self._site_id(name) for side, name in EE_SITES.items()}
        self.renderer: mujoco.Renderer | None = None
        self.reset()

    def _require(self, value: int, kind: str, name: str) -> int:
        if value < 0:
            raise ValueError(f"OpenArm v2 scene is missing {kind} {name!r}")
        return value

    def _body_id(self, name: str) -> int:
        return self._require(
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name),
            "body",
            name,
        )

    def _site_id(self, name: str) -> int:
        return self._require(
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name),
            "site",
            name,
        )

    def _joint_id(self, name: str) -> int:
        return self._require(
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name),
            "joint",
            name,
        )

    def _actuator_id(self, name: str) -> int:
        return self._require(
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name),
            "actuator",
            name,
        )

    def reset(self, randomize: bool = False) -> None:
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        base_positions = {
            "red_cube": np.array([0.44, 0.15, 1.032]),
            "blue_cube": np.array([0.50, -0.22, 1.032]),
        }
        if randomize:
            base_positions["red_cube"] = np.array(
                [self.rng.uniform(0.37, 0.42), self.rng.uniform(0.11, 0.18), 1.032]
            )
            base_positions["blue_cube"] += np.array(
                [self.rng.uniform(-0.012, 0.012), self.rng.uniform(-0.012, 0.012), 0]
            )
        for name, pos in base_positions.items():
            joint = self._object_joints[name]
            adr = self.model.jnt_qposadr[joint]
            self.data.qpos[adr : adr + 7] = [*pos.tolist(), 1, 0, 0, 0]
            self.data.qvel[self.model.jnt_dofadr[joint] : self.model.jnt_dofadr[joint] + 6] = 0
        mujoco.mj_forward(self.model, self.data)

    @property
    def rng(self) -> np.random.Generator:
        return np.random.default_rng(20261017)

    def step(self, steps: int = 1) -> None:
        for _ in range(steps):
            mujoco.mj_step(self.model, self.data)

    def set_arm(self, side: str, targets: list[float] | np.ndarray) -> None:
        ids = self._actuator_ids[side]
        limits = [self.model.actuator_ctrlrange[a] for a in ids]
        values = np.clip(np.asarray(targets, dtype=float), [x[0] for x in limits], [x[1] for x in limits])
        self.data.ctrl[ids] = values

    def set_gripper(self, side: str, value: float) -> None:
        low, high = self.model.actuator_ctrlrange[self._gripper_ids[side]]
        self.data.ctrl[self._gripper_ids[side]] = np.clip(value, low, high)

    def ee_pose(self, side: str) -> tuple[np.ndarray, np.ndarray]:
        site = self._site_ids[side]
        return self.data.site_xpos[site].copy(), self.data.site_xmat[site].reshape(3, 3).copy()

    def object_pose(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        body = self._object_bodies[name]
        return self.data.xpos[body].copy(), self.data.xmat[body].reshape(3, 3).copy()

    def zone_position(self, name: str) -> np.ndarray:
        return self.data.site_xpos[self._zone_sites[name]].copy()

    def contact_names(self) -> list[str]:
        out = []
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, int(c.geom1)) or "geom"
            g2 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, int(c.geom2)) or "geom"
            out.append(f"{g1}|{g2}")
        return out

    def object_contacts(self, object_name: str, side: str | None = None) -> list[str]:
        object_geom = f"{object_name}_geom"
        finger_marker = f"{side}_" if side else "_"
        return [
            pair
            for pair in self.contact_names()
            if object_geom in pair
            and any(part.startswith("finger") and finger_marker in part for part in pair.split("|"))
        ]

    def frame(self) -> Frame:
        objects = {name: self.object_pose(name)[0].tolist() for name in self._object_bodies}
        zones = {name: self.zone_position(name).tolist() for name in self._zone_sites}
        ee = {side: self.ee_pose(side)[0].tolist() for side in EE_SITES}
        grippers = {
            side: float(self.data.ctrl[self._gripper_ids[side]]) for side in EE_SITES
        }
        return Frame(
            sim_time=float(self.data.time),
            qpos=self.data.qpos.tolist(),
            objects=objects,
            zones=zones,
            ee=ee,
            grippers=grippers,
            contacts=self.contact_names(),
        )

    def render(self, width: int = 720, height: int = 480) -> np.ndarray | None:
        try:
            if self.renderer is None or self.renderer._width != width or self.renderer._height != height:
                self.renderer = mujoco.Renderer(self.model, height=height, width=width)
            camera = mujoco.MjvCamera()
            camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            camera.lookat[:] = [0.48, 0.0, 1.08]
            camera.distance = 1.15
            camera.azimuth = 135
            camera.elevation = -18
            self.renderer.update_scene(self.data, camera=camera)
            return self.renderer.render()
        except Exception:
            return None

    def run_controlled(
        self, command_fn: Callable[[], None], duration: float, realtime: bool = True
    ) -> None:
        steps = max(1, int(duration / self.model.opt.timestep))
        for _ in range(steps):
            command_fn()
            self.step(1)
