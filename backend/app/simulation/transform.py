"""Engine-neutral rotation and pose helpers.

The damped least-squares controller and the alternative physics backend both need
quaternion algebra, so it lives here in NumPy instead of behind one engine's C helpers.
The formulas follow MuJoCo's `mju_mat2Quat`, `mju_mulQuat`, `mju_negQuat` and
`mju_quat2Vel`, which the original MuJoCo-only controller used, and `tests/test_transform.py`
pins this implementation against those functions.
"""

from __future__ import annotations

import numpy as np


def quat_to_mat(quat: np.ndarray) -> np.ndarray:
    """Rotation matrix of a (w, x, y, z) quaternion."""
    w, x, y, z = np.asarray(quat, dtype=float)
    norm = np.sqrt(w * w + x * x + y * y + z * z)
    if norm == 0.0:
        return np.eye(3)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def mat_to_quat(matrix: np.ndarray) -> np.ndarray:
    """(w, x, y, z) quaternion of a rotation matrix, in MuJoCo's convention."""
    m = np.asarray(matrix, dtype=float).reshape(3, 3)
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        quat = np.array(
            [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
        )
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        quat = np.array(
            [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
        )
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        quat = np.array(
            [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
        )
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        quat = np.array(
            [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
        )
    return quat / np.linalg.norm(quat)


def quat_conjugate(quat: np.ndarray) -> np.ndarray:
    """Inverse rotation of a unit quaternion."""
    w, x, y, z = np.asarray(quat, dtype=float)
    return np.array([w, -x, -y, -z])


def quat_mul(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Hamilton product, matching MuJoCo's `mju_mulQuat(res, left, right)`."""
    lw, lx, ly, lz = np.asarray(left, dtype=float)
    rw, rx, ry, rz = np.asarray(right, dtype=float)
    return np.array(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ]
    )


def quat_to_rotvec(quat: np.ndarray, dt: float = 1.0) -> np.ndarray:
    """Rotation vector of a quaternion, matching MuJoCo's `mju_quat2Vel`.

    MuJoCo returns the rotation vector of the quaternion's *canonical* representative: with a
    negative scalar part it reports the negated quaternion's vector, i.e. the shorter rotation
    in the opposite direction. `tests/test_transform.py` pins that behaviour against MuJoCo.
    """
    w, x, y, z = np.asarray(quat, dtype=float)
    if w < 0.0:
        w, x, y, z = -w, -x, -y, -z
    axis = np.array([x, y, z])
    norm = float(np.linalg.norm(axis))
    if norm > 0.0:
        axis = axis / norm
    return axis * (2.0 * np.arctan2(norm, w)) / dt


def orientation_error(current: np.ndarray, desired: np.ndarray) -> np.ndarray:
    """Rotation vector taking `current` to `desired`, both rotation matrices."""
    quat_current = mat_to_quat(current)
    quat_desired = mat_to_quat(desired)
    return quat_to_rotvec(quat_mul(quat_desired, quat_conjugate(quat_current)))


def compose_pose(
    frame_position: np.ndarray, frame_rotation: np.ndarray, local_position: np.ndarray, local_quat: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a child frame given in a parent frame, returning world position and rotation."""
    rotation = np.asarray(frame_rotation, dtype=float).reshape(3, 3)
    position = np.asarray(frame_position, dtype=float)
    offset = rotation @ np.asarray(local_position, dtype=float)
    child_rotation = rotation @ quat_to_mat(np.asarray(local_quat, dtype=float))
    return position + offset, child_rotation
