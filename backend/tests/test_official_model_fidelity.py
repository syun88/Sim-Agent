"""The engine must run the official OpenArm v2 Cell, not an approximation of it.

The reference is the model description itself: the resolved MJCF is compiled by MuJoCo's compiler
and evaluated with its own kinematics routines (`mj_kinematics`, `mj_comPos`, `mj_jac`), while the
physics under test runs in Genesis World. Those routines are pure kinematics - no solver, no
stepping - so this compares the two engines' reading of the same robot rather than two physics
implementations.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from backend.app.control.motion import OpenArmV2Controller
from backend.app.simulation.openarm_v2.definition import ARM_JOINTS, EE_SITES, ZONES
from backend.app.simulation.openarm_v2.scene import (
    initial_object_positions,
    resolved_scene_xml,
)


class OfficialKinematics:
    """The resolved MJCF compiled once, questioned through MuJoCo's forward kinematics."""

    def __init__(self, xml: str):
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self.data.qpos[:] = self.model.key_qpos[0]
        self._forward()

    def _forward(self) -> None:
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)

    def joint_qpos(self, name: str) -> int:
        identifier = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        assert identifier >= 0, f"the official model has no joint {name}"
        return int(self.model.jnt_qposadr[identifier])

    def joint_dof(self, name: str) -> int:
        identifier = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        return int(self.model.jnt_dofadr[identifier])

    def arm_qpos(self, side: str) -> np.ndarray:
        return np.array([self.data.qpos[self.joint_qpos(name)] for name in ARM_JOINTS[side]])

    def set_arm_qpos(self, side: str, values: np.ndarray) -> None:
        address = np.array([self.joint_qpos(name) for name in ARM_JOINTS[side]])
        self.data.qpos[address] = np.asarray(values, dtype=float)
        self._forward()

    def site(self, name: str) -> int:
        identifier = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
        assert identifier >= 0, f"the official model has no site {name}"
        return identifier

    def site_pose(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        identifier = self.site(name)
        return (
            self.data.site_xpos[identifier].copy(),
            self.data.site_xmat[identifier].reshape(3, 3).copy(),
        )

    def site_jacobian(self, name: str, side: str) -> tuple[np.ndarray, np.ndarray]:
        """The site's world-frame Jacobian, restricted to one arm's seven joints."""
        identifier = self.site(name)
        linear = np.zeros((3, self.model.nv))
        angular = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, linear, angular, identifier)
        dofs = [self.joint_dof(name) for name in ARM_JOINTS[side]]
        return linear[:, dofs], angular[:, dofs]


@pytest.fixture(scope="module")
def official() -> OfficialKinematics:
    return OfficialKinematics(resolved_scene_xml())


def test_reset_matches_the_official_model(env, official):
    """The initial scene is the model's own keyframe plus the shared object placement."""
    assert env.arm_qpos("left") == pytest.approx(official.arm_qpos("left"), abs=1e-5)
    assert env.arm_qpos("right") == pytest.approx(official.arm_qpos("right"), abs=1e-5)
    for name, position in initial_object_positions().items():
        assert env.object_pose(name)[0] == pytest.approx(position, abs=1e-5)
    for zone in ZONES:
        assert env.zone_position(zone) == pytest.approx(official.site_pose(f"{zone}_site")[0], abs=1e-5)


def test_forward_kinematics_matches_the_official_model(env, official):
    for side in ("left", "right"):
        position, rotation = env.ee_pose(side)
        reference_position, reference_rotation = official.site_pose(EE_SITES[side])
        assert position == pytest.approx(reference_position, abs=1e-5)
        assert rotation == pytest.approx(reference_rotation, abs=1e-5)


def test_jacobian_matches_the_official_model(env, official):
    for side in ("left", "right"):
        linear, angular = env.ee_jacobian(side)
        reference_linear, reference_angular = official.site_jacobian(EE_SITES[side], side)
        assert linear == pytest.approx(reference_linear, abs=1e-6)
        assert angular == pytest.approx(reference_angular, abs=1e-6)


def test_jacobian_and_forward_kinematics_agree_away_from_the_home_pose(env, official):
    """Both engines must track the same arm through an arbitrary configuration."""
    config = np.array([0.4, -0.3, 0.5, 0.8, -0.2, 0.3, 0.1])
    env.set_arm_qpos("left", config)
    env.kinematics()
    official.set_arm_qpos("left", config)
    reference_position, _ = official.site_pose(EE_SITES["left"])
    assert env.ee_pose("left")[0] == pytest.approx(reference_position, abs=1e-5)
    linear, _ = env.ee_jacobian("left")
    reference_linear, _ = official.site_jacobian(EE_SITES["left"], "left")
    assert linear == pytest.approx(reference_linear, abs=1e-6)


def test_ik_reaches_the_target_through_the_official_model(env, controller, official):
    """The controller's solution is checked by the official model's kinematics, not by the engine."""
    target = np.array([0.36, 0.15, 1.06])
    solution, error, _ = controller.solve_ik("left", target)
    assert solution is not None and error < 1e-3
    official.set_arm_qpos("left", solution)
    reached, _ = official.site_pose(EE_SITES["left"])
    # The engine reports its own residual; the official model's forward kinematics is what decides
    # whether the gripper is there, so it is measured against the target directly.
    assert np.linalg.norm(reached - target) < 5e-4
