"""Cross-engine checks: two engines, one model description.

These are the checks that make the Genesis backend trustworthy as a *second implementation*
rather than a second robot: loaded from the same resolved MJCF string, both engines must agree
on the initial state, on forward kinematics and on the Jacobian the controller consumes, so a
difference between them is a physics difference and not a modelling difference.
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.app.control.motion import OpenArmV2Controller
from backend.app.simulation import available_backends, create_backend
from backend.app.simulation.genesis_backend.environment import genesis_available
from backend.app.simulation.openarm_v2.definition import (
    ARM_JOINTS,
    ARM_LINKS,
    GRASP_OFFSET,
    OBJECTS,
    ZONES,
)

from .conftest import require_genesis


@pytest.fixture
def engines(mujoco_env, genesis_env):
    require_genesis()
    genesis_env.reset()
    mujoco_env.reset()
    return mujoco_env, genesis_env


def test_backend_selection_and_availability():
    names = {item["name"] for item in available_backends()}
    assert {"mujoco", "genesis"} <= names
    assert create_backend("mujoco").name == "mujoco"
    with pytest.raises(ValueError):
        create_backend("not-an-engine")


def test_genesis_is_reported_as_the_non_authoritative_engine():
    if not genesis_available():
        pytest.skip("genesis-world is not installed")
    from backend.app.simulation.genesis_backend.environment import GenesisOpenArmV2Environment

    assert GenesisOpenArmV2Environment.name == "genesis"
    assert GenesisOpenArmV2Environment.grasp_is_physical_contact is False
    # A labelled fallback, never presented as contact-only manipulation.
    assert GenesisOpenArmV2Environment.grasp_mode == "kinematic-grasp-hold"


def test_both_engines_address_the_official_openarm_v2_model(genesis_env):
    require_genesis()
    assert genesis_env.grasp_hold_active("left", "red_cube") is False
    for side in ("left", "right"):
        for joint in ARM_JOINTS[side]:
            genesis_env._entity.get_joint(joint)
        for link in ARM_LINKS[side]:
            genesis_env._entity.get_link(link)
        assert genesis_env.arm_limits(side).shape == (7, 2)
    for name in OBJECTS:
        genesis_env._entity.get_link(name)
    assert genesis_env.timestep == pytest.approx(0.001)


def test_reset_state_matches_mujoco(engines):
    mujoco, genesis = engines
    mujoco.reset()
    genesis.reset()
    for name in OBJECTS:
        assert genesis.object_pose(name)[0] == pytest.approx(mujoco.object_pose(name)[0], abs=1e-5)
    for name in ZONES:
        assert genesis.zone_position(name) == pytest.approx(mujoco.zone_position(name), abs=1e-5)
    for side in ("left", "right"):
        assert genesis.ee_pose(side)[0] == pytest.approx(mujoco.ee_pose(side)[0], abs=1e-5)
        assert genesis.arm_qpos(side) == pytest.approx(mujoco.arm_qpos(side), abs=1e-5)


def test_randomized_reset_matches_mujoco(engines):
    mujoco, genesis = engines
    mujoco.reset(randomize=True)
    genesis.reset(randomize=True)
    for name in OBJECTS:
        assert genesis.object_pose(name)[0] == pytest.approx(mujoco.object_pose(name)[0], abs=1e-5)


def test_forward_kinematics_and_jacobian_match_mujoco(engines):
    mujoco, genesis = engines
    mujoco.reset()
    genesis.reset()
    for side in ("left", "right"):
        mujoco_position, mujoco_rotation = mujoco.ee_pose(side)
        genesis_position, genesis_rotation = genesis.ee_pose(side)
        assert genesis_position == pytest.approx(mujoco_position, abs=1e-5)
        assert genesis_rotation == pytest.approx(mujoco_rotation, abs=1e-5)
        for engine_jacobian, engine in ((mujoco.ee_jacobian(side), mujoco), (genesis.ee_jacobian(side), genesis)):
            assert engine_jacobian[0].shape == (3, 7), engine.name
            assert engine_jacobian[1].shape == (3, 7), engine.name
        mujoco_linear, mujoco_angular = mujoco.ee_jacobian(side)
        genesis_linear, genesis_angular = genesis.ee_jacobian(side)
        assert genesis_linear == pytest.approx(mujoco_linear, abs=1e-6)
        assert genesis_angular == pytest.approx(mujoco_angular, abs=1e-6)


def test_inverse_kinematics_solves_the_same_solution(engines):
    """Same target, same algorithm, same joint solution: the engines differ only in physics."""
    mujoco, genesis = engines
    mujoco.reset()
    genesis.reset()
    target = np.array([0.357, 0.15, 1.051])
    mujoco_solution, mujoco_error, _ = OpenArmV2Controller(mujoco).solve_ik("left", target, q_init=mujoco.arm_qpos("left"))
    genesis_solution, genesis_error, _ = OpenArmV2Controller(genesis).solve_ik("left", target, q_init=genesis.arm_qpos("left"))
    assert mujoco_solution is not None and genesis_solution is not None
    assert genesis_error == pytest.approx(mujoco_error, abs=1e-6)
    assert genesis_solution == pytest.approx(mujoco_solution, abs=1e-5)


def test_unreachable_target_is_rejected_by_both_engines(engines):
    mujoco, genesis = engines
    mujoco.reset()
    genesis.reset()
    # The documented fault offset of the recoverable-fault demo, applied as the skill applies it:
    # an approach point far beyond the arm's reach on the other side of the cube.
    fault_offset = -0.12
    unreachable = mujoco.object_pose("red_cube")[0] - np.array([fault_offset, 0.0, -GRASP_OFFSET[2]])
    for engine in (mujoco, genesis):
        solution, error, _ = OpenArmV2Controller(engine).solve_ik("left", unreachable)
        assert solution is None, engine.name
        assert error > 0.02, engine.name
