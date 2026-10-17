"""Genesis World running the official OpenArm v2 Cell scene.

The engine loads the resolved MJCF string (official `openarm-mujoco` v2 Cell plus the demo
objects), addresses the same joint and link names, uses the same grasp geometry and the same
controller, and exposes its state through `SimulationBackend`, so `build_world_state()` and the
verification step never see engine specifics.

Three things Genesis does differently from the model description, each handled explicitly here so
that nothing is silently implied:

- **Grasp hold.** Genesis takes MJCF equality constraints at build time and exposes no runtime
  enable/disable for them, and `RigidEntity.attach()` is a one-way structural change with no
  detach. A grasp is therefore a *kinematic grasp hold* (`grasp_mode`, reported in every grasp
  result, in `/api/health` and on the dashboard): while the hold is active the object's free-joint
  pose is prescribed from the gripper base link on every step. Finger contacts are still monitored
  and reported, but this engine does not claim contact-only manipulation.
- **Gravity compensation.** `apply_gravity_compensation()` feeds the arm's gravity torque into the
  constant term of the actuator model, the analogue of the bias force the model description
  provides, so the arm holds a commanded pose instead of sagging by `gravity_torque / kp`.
- **Sites.** Genesis does not implement MJCF sites, so the gripper control point and the zone
  origins are resolved as a parent link plus a local offset, read from the same XML string with
  MuJoCo's compiler (`_OfficialModelMetadata`). The model description has one source.
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
    GRASP_OFFSET,
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

#: Camera of the dashboard preview and of the recorded demo, aimed at the middle of the table from
#: above so the cubes, both zones and the working arm are all in frame.
CAMERA_LOOKAT = (0.44, 0.0, 1.04)
CAMERA_DISTANCE = 1.05
CAMERA_AZIMUTH = 140.0
CAMERA_ELEVATION = 38.0
CAMERA_FOV = 45.0
CAMERA_RESOLUTION = (720, 480)
#: Steps between two gravity-torque refreshes of the feed-forward (see `apply_gravity_compensation`).
GRAVITY_TORQUE_REFRESH_STEPS = 5

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
    def inactive_equality_names(self) -> set[str]:
        """Equalities the model declares as inactive, read from MuJoCo's compiled `eq_active0`."""
        names = set()
        for index in range(self.model.neq):
            if float(self.model.eq_active0[index]) != 0.0:
                continue
            name = self._mujoco.mj_id2name(
                self.model, self._mujoco.mjtObj.mjOBJ_EQUALITY, index
            )
            if name:
                names.add(name)
        return names

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


def genesis_scene_xml(xml: str, metadata: _OfficialModelMetadata) -> str:
    """Remove the model's inactive equalities from the XML handed to Genesis.

    Genesis builds every equality it parses and reads no `active` flag (see
    `genesis/utils/mjcf.py: parse_equalities`), and it offers no runtime enable/disable either. The
    demonstration scene declares its four grasp welds as `active="false"` so MuJoCo keeps them
    inert until `set_grasp_hold` turns one on; handed to Genesis unchanged they are permanent welds,
    which snap the cube to the gripper at reset. Inactive equalities are therefore dropped here, and
    the grasp is the backend's labelled kinematic hold instead.
    """
    import xml.etree.ElementTree as ET

    inactive = metadata.inactive_equality_names
    if not inactive:
        return xml
    root = ET.fromstring(xml)
    for block in root.findall("equality"):
        for element in list(block):
            if element.attrib.get("name") in inactive:
                block.remove(element)
    return ET.tostring(root, encoding="unicode")


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
        show_viewer: bool = False,
    ):
        if not genesis_available():
            raise RuntimeError(
                "genesis-world is not installed; install the 'genesis' extra to use this engine"
            )
        self._gs = _ensure_genesis_init(prefer_gpu)
        self._xml = resolved_scene_xml()
        self._meta = _OfficialModelMetadata(self._xml)
        self._scene_xml = genesis_scene_xml(self._xml, self._meta)
        self._holds: dict[str, tuple[str, np.ndarray, np.ndarray]] = {}
        self._scene = self._gs.Scene(
            sim_options=self._gs.options.SimOptions(dt=0.001),
            rigid_options=self._gs.options.RigidOptions(
                friction_cone=self._gs.friction_cone.elliptic,
                enable_torsional_friction=True,
                enable_rolling_friction=True,
                # Enable self-collision and adjacent collision so the free cubes collide
                # with the arm links, the cell table and walls. Without these, objects
                # pass through geometry that shares the same entity.
                enable_self_collision=True,
                enable_adjacent_collision=True,
            ),
            show_viewer=show_viewer,
        )
        self._entity = self._scene.add_entity(self._gs.morphs.MJCF(file=self._scene_xml))
        self._camera = self._scene.add_camera(
            res=camera_resolution,
            pos=free_camera_position(),
            lookat=list(CAMERA_LOOKAT),
            fov=CAMERA_FOV,
            GUI=False,
        )
        self._scene.build()
        self._index()
        # A constructed engine is already in the documented initial scene. The server builds exactly
        # one backend at startup, so without this the first task would begin from the model's zero
        # configuration instead of its keyframe, and the arm would sweep across the table.
        self.reset()

    # -- model indexing ----------------------------------------------------------

    def model_summary(self) -> dict:
        """What Genesis World built from the official MJCF.

        Genesis does not implement MJCF sites and reads no `active` flag, so it is worth being able
        to assert what it actually constructed instead of trusting the file that was handed to it.
        """
        entity = self._entity
        return {
            "engine": self.name,
            "links": len(entity.links),
            "joints": len(entity.joints),
            "degrees_of_freedom": int(entity.n_dofs),
            "positions": int(entity.n_qs),
            "equalities": int(self._scene.rigid_solver.n_equalities),
            "objects": sorted(self._object_qs),
        }

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
        self._object_dofs = {
            name: np.asarray(entity.get_joint(name).dofs_idx_local) for name in OBJECTS
        }
        self._link_names_by_index = {link.idx: link.name for link in entity.links}
        self._arm_links = {side: self._arm_descendants(side) for side in ARM_JOINTS}
        self._arm_link_indices = {
            side: np.array([link.idx_local for link in links])
            for side, links in self._arm_links.items()
        }
        self._act_biases = {
            side: tuple(
                tensor.numpy().astype(float) for tensor in entity.get_dofs_act_bias(self._dofs[side])[1:]
            )
            for side in ARM_JOINTS
        }
        self._gravity = np.asarray(self._scene.rigid_solver.get_gravity(), dtype=float).reshape(3)
        self._sim_step = 0
        self._gravity_torque_at_step = -1
        self._gravity_torque_values = {side: np.zeros(7) for side in ARM_JOINTS}

    def _arm_descendants(self, side: str) -> list:
        """Every link carried by one arm, including the gripper, in link order."""
        entity = self._entity
        root = entity.get_joint(ARM_JOINTS[side][0]).link.parent_idx
        parents = {link.idx: link.parent_idx for link in entity.links}
        descendants = []
        for link in entity.links:
            ancestors = []
            index = link.idx
            while index in parents and index not in ancestors:
                ancestors.append(index)
                index = parents[index]
            if root in ancestors:
                descendants.append(link)
        return descendants

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
        self._sim_step = 0
        self._gravity_torque_at_step = -1
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
            self._sim_step += 1

    def _apply_holds(self) -> None:
        entity = self._entity
        for object_name, (arm, local_offset, local_quat) in self._holds.items():
            position, rotation = self._link_pose(EE_LINKS[arm])
            world_position, world_rotation = compose_pose(position, rotation, local_offset, local_quat)
            qpos = np.array([*world_position, *mat_to_quat(world_rotation)], dtype=np.float32)
            entity.set_qpos(qpos, self._object_qs[object_name] + np.arange(7), zero_velocity=False)
            # Only the held object's velocity is reset: `zero_velocity=True` would zero the whole
            # entity's velocities, including the arm's, and stop the arm mid-path.
            entity.set_dofs_velocity(
                np.zeros(len(self._object_dofs[object_name]), dtype=np.float32),
                self._object_dofs[object_name],
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
        """Feed the arm's gravity torque forward, the way the model description does.

        The constant bias term of Genesis's actuator model (`act_bias[0]`, the analogue of the
        model's `biasprm[0]`) is added to the position controller's force, so the arm holds a commanded pose
        to a few millimetres instead of `gravity_torque / kp` short of it, while the damping stays
        implicit in the solver. Only the arm's dofs are compensated: the cubes keep their gravity.
        """
        dofs = self._dofs[side]
        bias1, bias2 = self._act_biases[side]
        self._entity.set_dofs_act_bias(self._gravity_torque(side), bias1, bias2, dofs)

    def _gravity_torque(self, side: str) -> np.ndarray:
        """Torque holding one arm's links against gravity.

        Recomputed every `GRAVITY_TORQUE_REFRESH_STEPS` steps: the load changes slowly compared to
        the PD response that absorbs the residual, and the Jacobian reads are the expensive part.
        """
        if self._gravity_torque_at_step + GRAVITY_TORQUE_REFRESH_STEPS <= self._sim_step:
            self._refresh_gravity_torque()
        return self._gravity_torque_values[side]

    def _refresh_gravity_torque(self) -> None:
        entity = self._entity
        for side, dofs in self._dofs.items():
            indices = self._arm_link_indices[side]
            masses = entity.get_links_mass(indices).numpy().astype(float)
            coms = entity.get_links_COM(indices).numpy().astype(float)
            torque = np.zeros(len(dofs))
            for link, mass, com in zip(self._arm_links[side], masses, coms):
                if mass <= 0.0:
                    continue
                jacobian = entity.get_jacobian(
                    link, local_point=com.astype(np.float32)
                ).numpy().astype(float)
                # Hold the link: the actuator torque cancels the gravity torque the link applies.
                torque -= jacobian[:3][:, dofs].T @ (mass * self._gravity)
            self._gravity_torque_values[side] = torque
        self._gravity_torque_at_step = self._sim_step

    # -- grasp hold --------------------------------------------------------------

    def set_grasp_hold(self, arm: str, object_name: str, active: bool) -> None:
        """Hold the object at the documented grasp offset from the gripper base link.

        The offset is the documented grasp rule: `GRASP_OFFSET` in world axes frozen into the link frame
        at grasp time, with the object's orientation relative to the link taken from the current
        poses. Both engines therefore hold the object in the same nominal place, and the difference
        left between them is the arm's own trajectory difference rather than a second grasp rule.
        """
        if not active:
            self._holds.pop(object_name, None)
            return
        position, rotation = self._link_pose(EE_LINKS[arm])
        _, object_rotation = self.object_pose(object_name)
        local_offset = rotation.T @ GRASP_OFFSET.astype(float)
        local_quat = quat_mul(
            quat_conjugate(mat_to_quat(rotation)), mat_to_quat(object_rotation)
        )
        self._holds[object_name] = (arm, local_offset, local_quat)

    def grasp_hold_active(self, arm: str, object_name: str) -> bool:
        hold = self._holds.get(object_name)
        return hold is not None and hold[0] == arm
