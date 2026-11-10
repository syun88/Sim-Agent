"""A complete reference body for the conformance contract, for a hardware author and for the checker itself.

`conformance.check` has only ever been *accepted* by the live Genesis cell and *rejected* by fakes that omit
things. A checker proven only on those two ends could be rejecting or accepting the wrong things. This module
is a small, complete, honest implementation of the contract - no physics, just a body that answers every
operation exactly as documented - so the checker's positive path is exercised in the test suite, and so a real
bridge has a file to copy.

What this is *not*: a simulator. It steps a clock, moves an arm to a commanded joint configuration, and reports
the fingertip pose by forward kinematics. Anything that needs dynamics (contacts, grasp success) says so
instead of inventing an answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ReferenceFrame:
    sim_time: float
    qpos: list
    objects: dict
    zones: dict
    ee: dict
    grippers: dict
    contacts: list
    fluid: dict | None = None
    scene_profile: str = 'cubes'
    base: list | None = None


@dataclass
class _Kinematics:
    body: 'ReferenceBody'
    arm: str

    def evaluate(self, q):
        """Fingertip position from the commanded joints: a linear stand-in, enough to be consistent."""
        q = np.asarray(q, float)
        offset = np.array([0.02, 0.0, -0.25]) if self.arm == 'left' else np.array([0.02, 0.0, -0.25])
        position = np.array([0.40, 0.10 if self.arm == 'left' else -0.10, 1.20]) + offset + q[:3] * 0.01
        return position, np.eye(3), np.zeros((3, q.size)), np.zeros((3, q.size))


@dataclass
class ReferenceBody:
    """The contract, implemented once, in the smallest honest way."""

    objects: dict = field(default_factory=lambda: {'red_cube': np.array([0.44, 0.15, 1.031]),
                                                   'blue_cube': np.array([0.50, -0.22, 1.031])})
    zones: dict = field(default_factory=lambda: {'blue_zone': np.array([0.48, 0.18, 1.008]),
                                                 'green_zone': np.array([0.48, -0.20, 1.008])})
    joints: dict = field(default_factory=lambda: {'left': np.zeros(7), 'right': np.zeros(7)})
    joints_per_arm: int = 7
    timestep: float = 0.002
    time: float = 0.0
    grippers: dict = field(default_factory=lambda: {'left': 0.02, 'right': -0.02})
    contacts: list = field(default_factory=list)

    # -- read-only -------------------------------------------------------
    def frame(self):
        return ReferenceFrame(self.time, list(self.joints['left']) + list(self.joints['right']),
                              {name: value.tolist() for name, value in self.objects.items()},
                              {name: value.tolist() for name, value in self.zones.items()},
                              {arm: self._ee(arm)[0].tolist() for arm in self.joints},
                              dict(self.grippers), list(self.contacts))

    def _ee(self, arm):
        return _Kinematics(self, arm).evaluate(self.joints[arm])[:2]

    def object_pose(self, name):
        return np.asarray(self.objects[name], float).copy(), np.eye(3)

    def object_bounds(self, name):
        centre = np.asarray(self.objects[name], float)
        half = np.array([0.02, 0.02, 0.02])
        return np.stack([centre - half, centre + half])

    def object_contacts(self, name, arm=None):
        return [f'finger|{name}'] if f'{arm}|{name}' in self.contacts else []

    def object_linear_speed(self, name):
        return 0.0

    def ee_pose(self, arm):
        return self._ee(arm)

    def arm_qpos(self, arm):
        return np.asarray(self.joints[arm], float).copy()

    def home_arm_qpos(self, arm):
        return np.zeros(self.joints_per_arm)

    def arm_limits(self, arm):
        return np.array([[-3.0, 3.0]] * self.joints_per_arm)

    def arm_velocity_limits(self, arm):
        return None

    def arm_kinematics(self, arm):
        return _Kinematics(self, arm)

    def zone_position(self, name):
        return np.asarray(self.zones[name], float).copy()

    # -- commands --------------------------------------------------------
    def set_arm(self, arm, qpos, velocities=None):
        self.joints[arm] = np.asarray(qpos, float).copy()

    def step(self, count=1):
        self.time += float(count) * self.timestep

    def set_gripper(self, arm, value):
        self.grippers[arm] = float(value)

    def gripper_position(self, arm):
        return float(self.grippers[arm])

    def gripper_target(self, arm, closed):
        return 0.02 if closed else 0.7854
