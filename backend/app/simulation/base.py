"""Engine-neutral simulation contract shared by the agent layers.

The agent never talks to a physics engine directly:

- the planner only reads `build_world_state()`, which serializes `frame()`;
- the skills and the verification step drive the robot through the operations below.

Genesis World is the implementation of this contract. It declares how it implements the parts the
model description expresses natively (grasp hold, gravity compensation), so nothing engine specific
leaks into the agent, and every engine specific behaviour stays labelled - which is what lets a
replacement engine be added, or this one be swapped, without touching the agent layers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

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
    #: Whether the grasp hold is a physical constraint or a documented fallback.
    grasp_is_physical_contact: bool = False

    # -- lifecycle ---------------------------------------------------------------

    @property
    @abstractmethod
    def timestep(self) -> float:
        """Simulation timestep in seconds."""

    @abstractmethod
    def reset(self, randomize: bool = False) -> None:
        """Restore the initial scene, optionally randomizing the cube placement."""

    @abstractmethod
    def step(self, steps: int = 1) -> None:
        """Advance the simulation and keep any active grasp hold applied."""

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

    @abstractmethod
    def zone_position(self, name: str) -> np.ndarray:
        """World position of a target zone."""

    @abstractmethod
    def object_linear_speed(self, name: str) -> float:
        """Linear speed of an object in m/s, used by the stability check."""

    @abstractmethod
    def contact_names(self) -> list[str]:
        """Human-readable names of the active contacts."""

    @abstractmethod
    def object_contacts(self, object_name: str, side: str | None = None) -> list[str]:
        """Active contacts between an object and the gripper of one arm."""

    # -- actuation ---------------------------------------------------------------

    @abstractmethod
    def set_arm(self, side: str, targets: list[float] | np.ndarray) -> None:
        """Command the seven joint positions of one arm."""

    @abstractmethod
    def arm_qpos(self, side: str) -> np.ndarray:
        """Current joint positions of one arm."""

    @abstractmethod
    def set_arm_qpos(self, side: str, targets: np.ndarray) -> None:
        """Write joint positions directly, for kinematics evaluation during IK."""

    @abstractmethod
    def arm_limits(self, side: str) -> np.ndarray:
        """Joint limits of one arm as a (7, 2) array."""

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

    @abstractmethod
    def apply_gravity_compensation(self, side: str) -> None:
        """Compensate the arm's own bias forces, where the engine needs it."""

    # -- grasp hold --------------------------------------------------------------

    @abstractmethod
    def set_grasp_hold(self, arm: str, object_name: str, active: bool) -> None:
        """Start or release the engine specific grasp hold for one arm/object pair."""

    @abstractmethod
    def grasp_hold_active(self, arm: str, object_name: str) -> bool:
        """Whether the grasp hold for one arm/object pair is currently active."""
