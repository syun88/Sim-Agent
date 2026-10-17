from __future__ import annotations

from pathlib import Path
from typing import Callable

import mujoco
import numpy as np

from backend.app.simulation.base import Frame, SimulationBackend
from backend.app.simulation.openarm_v2.definition import (
    ARM_ACTUATORS,
    ARM_JOINTS,
    EE_SITES,
    GRASP_OFFSET,
    GRIPPER_ACTUATORS,
    GRIPPER_CLOSED,
    GRIPPER_OPEN,
)
from backend.app.simulation.openarm_v2.scene import (
    DEMO_SCENE_PATH,
    initial_object_positions,
    resolved_scene_xml,
)

SCENE_PATH = DEMO_SCENE_PATH

__all__ = [
    "ARM_ACTUATORS",
    "ARM_JOINTS",
    "EE_SITES",
    "GRIPPER_ACTUATORS",
    "GRIPPER_CLOSED",
    "GRIPPER_OPEN",
    "Frame",
    "OpenArmV2Environment",
    "SCENE_PATH",
]


class OpenArmV2Environment(SimulationBackend):
    """Authoritative OpenArm v2 Cell simulation, run by MuJoCo."""

    name = "mujoco"
    #: MuJoCo holds the object with the documented equality constraint after gripper closure.
    grasp_mode = "contact-assisted-weld"
    grasp_is_physical_contact = False

    def __init__(self, model_path: Path = SCENE_PATH):
        self.model_path = Path(model_path)
        self.model = mujoco.MjModel.from_xml_string(resolved_scene_xml(self.model_path))
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
        self._arm_joint_ids = {
            side: np.array([self._joint_id(name) for name in ARM_JOINTS[side]])
            for side in ARM_JOINTS
        }
        self._arm_qadr = {
            side: np.array([int(self.model.jnt_qposadr[j]) for j in ids])
            for side, ids in self._arm_joint_ids.items()
        }
        self._arm_dofadr = {
            side: np.array([int(self.model.jnt_dofadr[j]) for j in ids])
            for side, ids in self._arm_joint_ids.items()
        }
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
        for name, pos in initial_object_positions(randomize).items():
            joint = self._object_joints[name]
            adr = self.model.jnt_qposadr[joint]
            self.data.qpos[adr : adr + 7] = [*pos.tolist(), 1, 0, 0, 0]
            self.data.qvel[self.model.jnt_dofadr[joint] : self.model.jnt_dofadr[joint] + 6] = 0
        mujoco.mj_forward(self.model, self.data)

    @property
    def timestep(self) -> float:
        return float(self.model.opt.timestep)

    def step(self, steps: int = 1) -> None:
        for _ in range(steps):
            mujoco.mj_step(self.model, self.data)

    # -- engine-neutral contract -------------------------------------------------

    def arm_qpos(self, side: str) -> np.ndarray:
        return self.data.qpos[self._arm_qadr[side]].copy()

    def set_arm_qpos(self, side: str, targets: np.ndarray) -> None:
        limits = self.arm_limits(side)
        self.data.qpos[self._arm_qadr[side]] = np.clip(np.asarray(targets, float), limits[:, 0], limits[:, 1])

    def arm_limits(self, side: str) -> np.ndarray:
        return self.model.jnt_range[self._arm_joint_ids[side]].copy()

    def kinematics(self) -> None:
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)

    def refresh(self) -> None:
        mujoco.mj_forward(self.model, self.data)

    def ee_jacobian(self, side: str) -> tuple[np.ndarray, np.ndarray]:
        site = self._site_ids[side]
        dof = self._arm_dofadr[side]
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jac(self.model, self.data, jacp, jacr, self.data.site_xpos[site], self.model.site_bodyid[site])
        return jacp[:, dof], jacr[:, dof]

    def apply_gravity_compensation(self, side: str) -> None:
        dof = self._arm_dofadr[side]
        self.data.qfrc_applied[dof] = self.data.qfrc_bias[dof]

    def object_linear_speed(self, name: str) -> float:
        dof = int(self.model.jnt_dofadr[self._object_joints[name]])
        return float(np.linalg.norm(self.data.qvel[dof : dof + 3]))

    def set_grasp_hold(self, arm: str, object_name: str, active: bool) -> None:
        """Activate or release the documented MuJoCo equality weld used as the grasp fallback."""
        eid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_EQUALITY, f"grasp_{arm}_{object_name}")
        if eid < 0:
            return
        if active:
            arm_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"openarm_{arm}_ee_base_link")
            obj_body = self._object_bodies[object_name]
            relative_rotation = np.zeros(4)
            inverse_arm_rotation = np.zeros(4)
            mujoco.mju_negQuat(inverse_arm_rotation, self.data.xquat[arm_body])
            mujoco.mju_mulQuat(relative_rotation, inverse_arm_rotation, self.data.xquat[obj_body])
            desired_local = self.data.xmat[arm_body].reshape(3, 3).T @ GRASP_OFFSET.copy()
            self.model.eq_data[eid, 3:6] = desired_local
            self.model.eq_data[eid, 6:10] = relative_rotation
            self.model.eq_data[eid, 10] = 1.0
        self.data.eq_active[eid] = bool(active)
        mujoco.mj_forward(self.model, self.data)

    def grasp_hold_active(self, arm: str, object_name: str) -> bool:
        eid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_EQUALITY, f"grasp_{arm}_{object_name}")
        return eid >= 0 and bool(self.data.eq_active[eid])

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
