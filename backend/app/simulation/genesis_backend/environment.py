"""Genesis World backend for the same official OpenArm v2 Cell scene.

Genesis World is a second, independent physics engine for the identical task: it loads the
same resolved MJCF string (official `openarm-mujoco` v2 Cell plus the demo objects), addresses
the same joint and link names, uses the same grasp geometry and the same controller, and
exposes its state through `SimulationBackend`, so `build_world_state()` and the verification
step do not know which engine produced a sample.

MuJoCo remains the authoritative engine. What this backend deliberately does differently is
listed here so nothing is silently implied:

- **Grasp hold.** Genesis takes the MJCF equality constraints at build time and exposes no
  runtime enable/disable for them, and `RigidEntity.attach()` is a one-way structural change
  with no detach. A grasp is therefore a *kinematic grasp hold*: while the hold is active the
  object's free-joint pose is prescribed from the gripper base link on every step. Real finger
  contacts are still monitored and reported, but this backend does not claim contact-only
  manipulation either.
- **Gravity compensation.** The MuJoCo controller adds the arm bias forces (`qfrc_bias`)
  during a joint path. Genesis holds the arm with the model's own PD gains, so
  `apply_gravity_compensation()` is a no-op here.
- **Sites.** Genesis does not model MJCF sites, so the gripper control point and the zone
  origins are resolved as a parent link plus a local offset, read from the same XML string with
  MuJoCo's compiler (`_OfficialModelMetadata`). The model description has one source; Genesis
  supplies only the physics.

Install it with `pip install "sim-agent[genesis]"`. The MuJoCo path never imports it.
"""

from __future__ import annotations

import importlib.util
import threading

import numpy as np

from backend.app.simulation.base import Frame, SimulationBackend
from backend.app.simulation.openarm_v2.definition import (
    ARM_JOINTS,
    EE_LINKS,
    EE_SITES,
    GRIPPER_JOINTS,
    OBJECTS,
    ZONES,
)
from backend.app.simulation.openarm_v2.scene import initial_object_positions, resolved_scene_xml
from backend.app.simulation.transform import (
    compose_pose,
    mat_to_quat,
    quat_conjugate,
    quat_mul,
    quat_to_mat,
)

#: Camera of the dashboard preview, mirroring the MuJoCo free camera.
CAMERA_LOOKAT = (0.48, 0.0, 1.08)
CAMERA_DISTANCE = 1.15
CAMERA_AZIMUTH = 135.0
CAMERA_ELEVATION = 18.0
CAMERA_FOV = 45.0
CAMERA_RESOLUTION = (720, 480)

_INIT_LOCK = threading.Lock()
_INITIALISED = False


def genesis_available() -> bool:
    """Whether the optional Genesis World dependency is importable."""
    return importlib.util.find_spec("genesis") is not None


def _ensure_genesis_init(prefer_gpu: bool):
    """Initialise Genesis exactly once per process and return the module."""
    import genesis as gs

    global _INITIALISED
    with _INIT_LOCK:
        if not _INITIALISED:
            gs.init(backend=gs.gpu if prefer_gpu else gs.cpu, precision="32", logging_level="error")
            _INITIALISED = True
    return gs


def free_camera_position(
    lookat: tuple[float, float, float] = CAMERA_LOOKAT,
    distance: float = CAMERA_DISTANCE,
    azimuth_deg: float = CAMERA_AZIMUTH,
    elevation_deg: float = CAMERA_ELEVATION,
) -> tuple[float, float, float]:
    """Camera position from MuJoCo's distance/azimuth/elevation free-camera convention."""
    azimuth = np.deg2rad(azimuth_deg)
    elevation = np.deg2rad(elevation_deg)
    return (
        lookat[0] + distance * np.cos(elevation) * np.cos(azimuth),
        lookat[1] + distance * np.cos(elevation) * np.sin(azimuth),
        lookat[2] + distance * np.sin(elevation),
    )


class _OfficialModelMetadata:
    """Link names, site frames, keyframe and actuator mapping of the same XML, read by MuJoCo.

    MuJoCo's compiler is the reader of record for the model description, so the alternative
    engine cannot drift from the official asset naming or geometry.
    """

    def __init__(self, xml: str):
        import mujoco

        self._mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_string(xml)

    def _obj_id(self, kind, name: str) -> int:
        return int(self._mujoco.mj_name2id(self.model, kind, name))

    def site_frame(self, site_name: str) -> tuple[str, np.ndarray, np.ndarray]:
        """Parent link name and the site's pose inside that link."""
        mujoco = self._mujoco
        site = self._obj_id(mujoco.mjtObj.mjOBJ_SITE, site_name)
        if site < 0:
            raise ValueError(f"OpenArm v2 model is missing site {site_name!r}")
        body = int(self.model.site_bodyid[site])
        body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body)
        if body_name is None:
            raise ValueError(f"OpenArm v2 model site {site_name!r} has no named parent body")
        return (
            body_name,
            self.model.site_pos[site].astype(float).copy(),
            self.model.site_quat[site].astype(float).copy(),
        )

    def joint_of_actuator(self, index: int) -> str:
        mujoco = self._mujoco
        joint = int(self.model.actuator_trnid[index, 0])
        name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint)
        if name is None:
            raise ValueError(f"OpenArm v2 model actuator {index} drives no named joint")
        return name

    @property
    def keyframe_qpos(self) -> np.ndarray:
        """Joint positions of the model's first keyframe (the canonical 'home' pose)."""
        return self.model.key_qpos[0].astype(float).copy()

    @property
    def keyframe_ctrl(self) -> dict[str, float]:
        """Keyframe actuator targets, keyed by the joint each actuator drives."""
        return {
            self.joint_of_actuator(index): float(self.model.key_ctrl[0, index])
            for index in range(self.model.nu)
        }


class GenesisOpenArmV2Environment(SimulationBackend):
    """OpenArm v2 Cell simulation run by Genesis World."""

    name = "genesis"
    #: Genesis prescribes the held object pose instead of constraining it; labelled as such.
    grasp_mode = "kinematic-grasp-hold"
    grasp_is_physical_contact = False

    def __init__(
        self,
        prefer_gpu: bool = False,
        camera_resolution: tuple[int, int] = CAMERA_RESOLUTION,
    ):
        if not genesis_available():
            raise RuntimeError(
                "genesis-world is not installed; install the 'genesis' extra to use this engine"
            )
        self._gs = _ensure_genesis_init(prefer_gpu)
        self._xml = resolved_scene_xml()
        self._meta = _OfficialModelMetadata(self._xml)
        self._holds: dict[str, tuple[str, np.ndarray, np.ndarray]] = {}
        self._scene = self._gs.Scene(
            sim_options=self._gs.options.SimOptions(dt=0.001),
            # Honor what the official MJCF declares: its friction cone and its torsional and
            # rolling friction (the model uses condim 4 and 6 geoms).
            rigid_options=self._gs.options.RigidOptions(
                friction_cone=self._gs.friction_cone.elliptic,
                enable_torsional_friction=True,
                enable_rolling_friction=True,
            ),
            show_viewer=False,
        )
        self._entity = self._scene.add_entity(self._gs.morphs.MJCF(file=self._xml))
        self._camera = self._scene.add_camera(
            res=camera_resolution,
            pos=free_camera_position(),
            lookat=list(CAMERA_LOOKAT),
            fov=CAMERA_FOV,
            GUI=False,
        )
        self._scene.build()
        self._index()

    # -- model indexing ----------------------------------------------------------

    def _index(self) -> None:
        entity = self._entity
        self._links = {
            name: entity.get_link(name)
            for name in (*OBJECTS, *EE_LINKS.values(), *self._site_parents())
        }
        self._dofs = {
            side: np.array([entity.get_joint(name).dofs_idx_local[0] for name in ARM_JOINTS[side]])
            for side in ARM_JOINTS
        }
        self._gripper_dofs = {
            side: int(entity.get_joint(name).dofs_idx_local[0])
            for side, name in GRIPPER_JOINTS.items()
        }
        self._object_qs = {name: int(entity.get_joint(name).qs_idx_local[0]) for name in OBJECTS}
        self._link_names_by_index = {link.idx: link.name for link in entity.links}

    def _site_parents(self) -> tuple[str, ...]:
        names = []
        for site in (*EE_SITES.values(), *(f"{zone}_site" for zone in ZONES)):
            parent = self._meta.site_frame(site)[0]
            if parent not in names:
                names.append(parent)
        return tuple(names)

    def _link_pose(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        link = self._links[name]
        return link.get_pos().numpy().astype(float), quat_to_mat(link.get_quat().numpy().astype(float))

    def _site_pose(self, site_name: str) -> tuple[np.ndarray, np.ndarray]:
        parent, local_position, local_quat = self._meta.site_frame(site_name)
        position, rotation = self._link_pose(parent)
        return compose_pose(position, rotation, local_position, local_quat)

    # -- lifecycle ---------------------------------------------------------------

    @property
    def timestep(self) -> float:
        return float(self._scene.dt)

    def reset(self, randomize: bool = False) -> None:
        self._holds.clear()
        entity = self._entity
        entity.set_qpos(self._meta.keyframe_qpos[: entity.n_qs].astype(np.float32), zero_velocity=True)
        for joint_name, value in self._meta.keyframe_ctrl.items():
            dof = int(entity.get_joint(joint_name).dofs_idx_local[0])
            entity.control_dofs_position(np.array([value], dtype=np.float32), np.array([dof]))
        for name, position in initial_object_positions(randomize).items():
            qpos = np.array([*position, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            entity.set_qpos(qpos, self._object_qs[name] + np.arange(7), zero_velocity=True)
        self.refresh()

    def step(self, steps: int = 1) -> None:
        for _ in range(steps):
            self._apply_holds()
            self._scene.step()

    def _apply_holds(self) -> None:
        for object_name, (arm, local_offset, local_quat) in self._holds.items():
            position, rotation = self._link_pose(EE_LINKS[arm])
            world_position, world_rotation = compose_pose(position, rotation, local_offset, local_quat)
            qpos = np.array([*world_position, *mat_to_quat(world_rotation)], dtype=np.float32)
            self._entity.set_qpos(
                qpos, self._object_qs[object_name] + np.arange(7), zero_velocity=True
            )

    # -- observation -------------------------------------------------------------

    def frame(self) -> Frame:
        entity = self._entity
        return Frame(
            sim_time=float(self._scene.sim.cur_t),
            qpos=entity.get_qpos().numpy().astype(float).tolist(),
            objects={name: self._link_pose(name)[0].tolist() for name in OBJECTS},
            zones={name: self.zone_position(name).tolist() for name in ZONES},
            ee={side: self.ee_pose(side)[0].tolist() for side in EE_LINKS},
            grippers={
                side: float(entity.get_dofs_position(np.array([dof])).numpy()[0])
                for side, dof in self._gripper_dofs.items()
            },
            contacts=self.contact_names(),
        )

    def render(self, width: int = 720, height: int = 480) -> np.ndarray | None:
        try:
            image = np.asarray(self._camera.render(rgb=True)[0])
        except Exception:
            return None
        if image.shape[:2] != (height, width):
            # Genesis fixes a camera's resolution at build time, so other sizes are resized.
            from PIL import Image

            image = np.asarray(Image.fromarray(image).resize((width, height), Image.BILINEAR))
        return image

    def object_pose(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        return self._link_pose(name)

    def zone_position(self, name: str) -> np.ndarray:
        return self._site_pose(f"{name}_site")[0]

    def object_linear_speed(self, name: str) -> float:
        return float(np.linalg.norm(self._links[name].get_vel().numpy()))

    def contact_names(self) -> list[str]:
        return [
            "|".join(self._link_names_by_index.get(index, "link") for index in pair)
            for pair in self._contact_link_pairs()
        ]

    def object_contacts(self, object_name: str, side: str | None = None) -> list[str]:
        prefix = f"{side}_"
        contacts = []
        for left, right in self._contact_link_pairs():
            left_name = self._link_names_by_index.get(left, "")
            right_name = self._link_names_by_index.get(right, "")
            if object_name not in (left_name, right_name):
                continue
            other = right_name if left_name == object_name else left_name
            if "finger" in other and (side is None or prefix in other):
                contacts.append(f"{other}|{object_name}")
        return contacts

    def _contact_link_pairs(self) -> list[tuple[int, int]]:
        contacts = self._entity.get_contacts()
        link_a, link_b = contacts.get("link_a"), contacts.get("link_b")
        if link_a is None or link_b is None or len(link_a) == 0:
            return []
        return list(zip(link_a.numpy().astype(int).tolist(), link_b.numpy().astype(int).tolist()))

    # -- actuation ---------------------------------------------------------------

    def set_arm(self, side: str, targets: list[float] | np.ndarray) -> None:
        values = np.clip(np.asarray(targets, dtype=float), *self.arm_limits(side).T)
        self._entity.control_dofs_position(values.astype(np.float32), self._dofs[side])

    def arm_qpos(self, side: str) -> np.ndarray:
        return self._entity.get_dofs_position(self._dofs[side]).numpy().astype(float)

    def set_arm_qpos(self, side: str, targets: np.ndarray) -> None:
        limits = self.arm_limits(side)
        values = np.clip(np.asarray(targets, dtype=float), limits[:, 0], limits[:, 1])
        self._entity.set_dofs_position(
            values.astype(np.float32), self._dofs[side], zero_velocity=False
        )

    def arm_limits(self, side: str) -> np.ndarray:
        lower, upper = self._entity.get_dofs_limit(self._dofs[side])
        return np.stack([lower.numpy(), upper.numpy()], axis=1).astype(float)

    def kinematics(self) -> None:
        """No-op: Genesis refreshes forward kinematics inside `set_arm_qpos`."""

    def refresh(self) -> None:
        self._scene.rigid_solver.update_forward_pos()

    def ee_pose(self, side: str) -> tuple[np.ndarray, np.ndarray]:
        return self._site_pose(EE_SITES[side])

    def ee_jacobian(self, side: str) -> tuple[np.ndarray, np.ndarray]:
        parent, local_position, _ = self._meta.site_frame(EE_SITES[side])
        jacobian = self._entity.get_jacobian(
            self._links[parent], local_point=local_position.astype(np.float32)
        )
        jacobian = jacobian.numpy().astype(float)
        dofs = self._dofs[side]
        return jacobian[:3][:, dofs], jacobian[3:][:, dofs]

    def set_gripper(self, side: str, value: float) -> None:
        dof = self._gripper_dofs[side]
        lower, upper = self._entity.get_dofs_limit(np.array([dof]))
        self._entity.control_dofs_position(
            np.array([np.clip(value, float(lower[0]), float(upper[0]))], dtype=np.float32),
            np.array([dof]),
        )

    def apply_gravity_compensation(self, side: str) -> None:
        """No-op: Genesis holds the arm with the model's PD gains instead of a bias feed-forward."""

    # -- grasp hold --------------------------------------------------------------

    def set_grasp_hold(self, arm: str, object_name: str, active: bool) -> None:
        if not active:
            self._holds.pop(object_name, None)
            return
        position, rotation = self._link_pose(EE_LINKS[arm])
        object_position, object_rotation = self.object_pose(object_name)
        local_offset = rotation.T @ (object_position - position)
        local_quat = quat_mul(
            quat_conjugate(mat_to_quat(rotation)), mat_to_quat(object_rotation)
        )
        self._holds[object_name] = (arm, local_offset, local_quat)

    def grasp_hold_active(self, arm: str, object_name: str) -> bool:
        hold = self._holds.get(object_name)
        return hold is not None and hold[0] == arm
