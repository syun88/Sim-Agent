"""Engine-neutral simulation contract shared by the agent layers.

The agent never talks to a physics engine directly:

- the planner only reads `build_world_state()`, which serializes `frame()`;
- the skills and the verification step drive the robot through the operations below.

Genesis World is the implementation of this contract. It declares how it implements the parts the
model description expresses natively (gravity compensation), so nothing engine specific
leaks into the agent, and every engine specific behaviour stays labelled - which is what lets a
replacement engine be added, or this one be swapped, without touching the agent layers.

A grasp is physical contact in every engine that implements this contract: the skills close the
gripper's fingers, read back the finger-object contacts through `object_contacts`, and verify the
hold by moving the gripper and checking that the object followed. No operation here may write an
object's pose directly; the only pose writes are scene reset and setup.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np


@dataclass
class Frame:
    """One simulation sample, in the units of the engine that produced it."""

    sim_time: float
    qpos: list[float]
    objects: dict[str, list[float]]
    zones: dict[str, list[float]]
    ee: dict[str, list[float]]
    grippers: dict[str, float]
    contacts: list[str]
    fluid: dict | None = None
    scene_profile: str = "cubes"
    # Planar pose [x, y, yaw] of the scene's mobile base, when it has one.
    base: list[float] | None = None


class ArmKinematics(Protocol):
    """Private joint-state snapshot for IK; evaluating candidates never changes the scene."""

    def evaluate(self, joints: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return position, rotation, linear Jacobian and angular Jacobian at the EE."""
        ...


class SimulationBackend(ABC):
    """Operations the agent layers need from a physics engine running the OpenArm v2 Cell.

    Every pose is a world-frame position (metres) or rotation matrix, and every joint
    value is a joint-position command in radians or metres, matching the official
    `openarm-mujoco` model semantics.
    """

    #: Stable identifier used by the API and the dashboard.
    name: str = "unknown"
    #: How the current engine holds a grasped object, reported with every grasp result.
    grasp_mode: str = "unknown"
    #: Whether the grasp is physical contact between the fingers and the object.
    grasp_is_physical_contact: bool = False
    # Called between completed physics steps, on the simulation owner's thread.
    # The native broker uses it for heartbeats and interruption of long waits.
    step_observer: Callable[[], None] | None = None

    # -- lifecycle ---------------------------------------------------------------

    @property
    @abstractmethod
    def timestep(self) -> float:
        """Simulation timestep in seconds."""

    @abstractmethod
    def reset(self, randomize: bool = False, object_positions: dict[str, np.ndarray] | None = None,
              seed: int | None = None,shuffle_sides: bool = True) -> None:
        """Restore the initial scene.

        `object_positions` overrides where the objects start - a scene *setup* operation that runs
        before a task, exactly like `randomize`; it never runs during an execution.
        """

    @abstractmethod
    def step(self, steps: int = 1) -> None:
        """Advance the simulation."""

    def configure_motion_response(self, speed: float) -> None:
        """Select actuator response for transport, retaining model force/position limits."""

    # -- observation -------------------------------------------------------------

    @abstractmethod
    def frame(self) -> Frame:
        """Serialize the engine state for the planner."""

    @abstractmethod
    def render(self, width: int = 720, height: int = 480) -> np.ndarray | None:
        """Render the scene, or return None when the engine cannot render here."""

    @abstractmethod
    def object_pose(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """World position and rotation matrix of a manipulated object."""

    def object_bounds(self, name: str) -> np.ndarray:
        """Observed world collision bounds, [minimum, maximum], in metres."""
        raise NotImplementedError

    @abstractmethod
    def zone_position(self, name: str) -> np.ndarray:
        """World position of a target zone."""

    @abstractmethod
    def object_linear_speed(self, name: str) -> float:
        """Linear speed of an object in m/s, used by the stability check."""

    @abstractmethod
    def object_velocity(self, name: str) -> np.ndarray:
        """World linear velocity (m/s), measured by the physics engine."""

    # -- mobile base (optional; only scenes with a wheeled base implement it) ----

    def base_pose(self) -> np.ndarray:
        """Measured planar pose [x, y, yaw] of the mobile base, in metres/radians."""
        raise NotImplementedError

    def command_base_velocity(self, vx: float, vy: float, omega: float) -> None:
        """Command the base's body velocity (m/s, m/s, rad/s).

        The base moves only through its wheel actuators, exactly like an arm
        moves through its joint motors: no operation may write the base's
        pose during an execution. ``vy`` is nonzero only for holonomic
        (omnidirectional) bases.
        """
        raise NotImplementedError

    @abstractmethod
    def object_contact_links(self, name: str) -> list[str]:
        """All links touching this object, including the table, palm and fingers."""

    @abstractmethod
    def contact_names(self) -> list[str]:
        """Human-readable names of the active contacts."""

    @abstractmethod
    def object_contacts(self, object_name: str, side: str | None = None) -> list[str]:
        """Active contacts between an object and the gripper fingers of one arm (or both)."""

    # -- actuation ---------------------------------------------------------------

    @abstractmethod
    def set_arm(self, side: str, targets: list[float] | np.ndarray,
                velocities: np.ndarray | None = None) -> None:
        """Command joint position and optional desired velocity through the motors."""

    @abstractmethod
    def arm_qpos(self, side: str) -> np.ndarray:
        """Current joint positions of one arm."""

    @abstractmethod
    def arm_kinematics(self, side: str) -> ArmKinematics:
        """Snapshot the current configuration for read-only, isolated IK evaluation."""

    @abstractmethod
    def set_arm_qpos(self, side: str, targets: np.ndarray) -> None:
        """Write joint positions directly for scene setup/model-fidelity checks, never IK."""

    @abstractmethod
    def arm_limits(self, side: str) -> np.ndarray:
        """Joint limits of one arm as an (N, 2) array."""

    def home_arm_qpos(self, side: str) -> np.ndarray:
        from backend.app.simulation.openarm_v2.definition import SPAWN_ARM_QPOS
        return SPAWN_ARM_QPOS.copy()

    def arm_velocity_limits(self, side: str) -> np.ndarray | None:
        """Optional source velocity caps for a generated URDF motion profile."""
        return None

    def plan_arm_pose(self, side: str, position: np.ndarray, rotation: np.ndarray | None = None,
                      *, q_init: np.ndarray | None = None) -> dict | None:
        """Plan a joint vector that reaches this pose, or ``None`` when this robot cannot plan one.

        A skill that plans a pose - a throw's wind-up search and its launch stroke are the two - asks
        here instead of assuming a shape of its own, because the two kinds of robot answer it with
        different machinery: an imported package through the planner that reads its URDF profile, the
        builtin official cell through its own kinematics. The result carries ``success``, ``q_rad``
        (continuous with ``q_init`` where the joints can wrap), and the errors that justify the verdict.
        """
        return None

    @abstractmethod
    def kinematics(self) -> None:
        """Refresh forward kinematics after `set_arm_qpos`, without stepping physics."""

    @abstractmethod
    def refresh(self) -> None:
        """Recompute derived state (contacts, bias) after a direct state write."""

    @abstractmethod
    def ee_pose(self, side: str) -> tuple[np.ndarray, np.ndarray]:
        """World position and rotation matrix of the gripper control point."""

    @abstractmethod
    def ee_jacobian(self, side: str) -> tuple[np.ndarray, np.ndarray]:
        """Linear and angular Jacobian (3x7 each) at the gripper control point."""

    @abstractmethod
    def set_gripper(self, side: str, value: float) -> None:
        """Command one gripper position."""

    def gripper_position(self, side: str) -> float:
        """Measured driven finger joint, for checking the motor response."""
        return float(self.frame().grippers[side])

    @abstractmethod
    def apply_gravity_compensation(self, side: str) -> None:
        """Compensate the arm's own bias forces, where the engine needs it."""

    # -- grasp -------------------------------------------------------------------

    # `object_contacts` (see the observation section) is the physical evidence a grasp is
    # verified against: the skills require the object to be touched by both fingers after
    # closing and to follow the gripper through a verification lift, and the goal verification
    # requires the object to be untouched by any finger before declaring it released.

    def gripper_target(self, side: str, closed: bool) -> float:
        from backend.app.simulation.openarm_v2.definition import GRIPPER_CLOSED, GRIPPER_OPEN
        return (GRIPPER_CLOSED if closed else GRIPPER_OPEN)[side]

    def pinch_established(self, name: str, side: str) -> bool:
        contacts = self.object_contacts(name, side)
        return any('inner' in c for c in contacts) and any('outer' in c for c in contacts)

    table_height: float = 1.006
