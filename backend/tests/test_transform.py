"""Pin the engine-neutral rotation helpers against MuJoCo's own C functions.

The controller used to call `mju_mat2Quat` / `mju_mulQuat` / `mju_negQuat` / `mju_quat2Vel`
directly. They are now NumPy (`backend/app/simulation/transform.py`) so the same controller can
drive any engine, and this test is the guard on that replacement: if the formulas drift, the
inverse kinematics of every engine drifts with them.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from backend.app.simulation.transform import (
    mat_to_quat,
    orientation_error,
    quat_conjugate,
    quat_mul,
    quat_to_mat,
    quat_to_rotvec,
)


def random_rotation(rng: np.random.Generator) -> np.ndarray:
    quat = rng.normal(size=4)
    quat /= np.linalg.norm(quat)
    return quat_to_mat(quat)


def test_mat_to_quat_matches_mujoco():
    rng = np.random.default_rng(7)
    for _ in range(200):
        rotation = random_rotation(rng)
        expected = np.zeros(4)
        mujoco.mju_mat2Quat(expected, rotation.flatten())
        actual = mat_to_quat(rotation)
        # A quaternion and its negation are the same rotation.
        error = min(np.linalg.norm(actual - expected), np.linalg.norm(actual + expected))
        assert error < 1e-12, (actual, expected)


def test_quat_helpers_match_mujoco():
    rng = np.random.default_rng(11)
    for _ in range(200):
        first = rng.normal(size=4)
        second = rng.normal(size=4)
        first /= np.linalg.norm(first)
        second /= np.linalg.norm(second)

        expected = np.zeros(4)
        mujoco.mju_mulQuat(expected, first, second)
        assert quat_mul(first, second) == pytest.approx(expected, abs=1e-12)

        expected = np.zeros(4)
        mujoco.mju_negQuat(expected, first)
        assert quat_conjugate(first) == pytest.approx(expected, abs=1e-12)

        expected = np.zeros(3)
        mujoco.mju_quat2Vel(expected, second, 1.0)
        assert quat_to_rotvec(second) == pytest.approx(expected, abs=1e-12)


def test_orientation_error_matches_mujoco():
    """The rotation vector the IK drives to zero must be MuJoCo's."""
    rng = np.random.default_rng(13)
    for _ in range(200):
        current = random_rotation(rng)
        desired = random_rotation(rng)

        current_quat = np.zeros(4)
        desired_quat = np.zeros(4)
        inverse_quat = np.zeros(4)
        error_quat = np.zeros(4)
        expected = np.zeros(3)
        mujoco.mju_mat2Quat(current_quat, current.flatten())
        mujoco.mju_mat2Quat(desired_quat, desired.flatten())
        mujoco.mju_negQuat(inverse_quat, current_quat)
        mujoco.mju_mulQuat(error_quat, desired_quat, inverse_quat)
        mujoco.mju_quat2Vel(expected, error_quat, 1.0)

        assert orientation_error(current, desired) == pytest.approx(expected, abs=1e-12)


def test_quat_to_mat_round_trip():
    rng = np.random.default_rng(17)
    for _ in range(200):
        rotation = random_rotation(rng)
        assert quat_to_mat(mat_to_quat(rotation)) == pytest.approx(rotation, abs=1e-12)
