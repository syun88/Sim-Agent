from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from backend.app.simulation.base import SimulationBackend
from backend.app.simulation.openarm_v2.definition import (
    GRASP_ROTATION,
    GRIPPER_CLOSED,
    GRIPPER_OPEN,
    SPAWN_ARM_QPOS,
)
from backend.app.simulation.transform import orientation_error

__all__ = [
    "MotionResult",
    "OpenArmV2Controller",
]


@dataclass
class MotionResult:
    success: bool
    error: str | None = None
    final_position: list[float] | None = None
    position_error: float | None = None
    rotation_error: float | None = None


class OpenArmV2Controller:
    """Damped least-squares joint-space controller, independent of the physics engine.

    IK evaluates a private kinematic snapshot through `SimulationBackend`. Candidate
    poses never enter the live scene or its viewer; only motor commands execute there.
    Endpoint verification always reads the actual physics state.
    """

    def __init__(self, env: SimulationBackend):
        self.env = env
        self.cancelled = False
        self.should_stop: Callable[[], bool] | None = None
        self.target_rotations: dict[str, np.ndarray] = {}

    def stop_requested(self) -> bool:
        if self.should_stop and self.should_stop():
            self.cancelled = True
        return self.cancelled

    @property
    def timestep(self) -> float:
        return float(self.env.timestep)

    def cancel(self) -> None:
        self.cancelled = True

    def solve_ik(
        self,
        side: str,
        target_position: np.ndarray,
        target_rotation: np.ndarray | None = None,
        q_init: np.ndarray | None = None,
        max_iterations: int = 500,
    ) -> tuple[np.ndarray | None, float, float]:
        limits = self.env.arm_limits(side)
        if q_init is None:
            q = self.env.arm_qpos(side)
        else:
            q = np.clip(np.asarray(q_init, float), limits[:, 0], limits[:, 1])
        if target_rotation is None:
            target_rotation = self.env.ee_pose(side)[1]
        kinematics = self.env.arm_kinematics(side)
        best_q = q.copy()
        best_error = np.inf
        for _ in range(max_iterations):
            position, rotation, linear_jacobian, angular_jacobian = kinematics.evaluate(q)
            perr = np.asarray(target_position, float) - position
            oerr = orientation_error(rotation, target_rotation)
            error = float(np.linalg.norm(perr) + np.linalg.norm(oerr) * 0.2)
            if error < best_error:
                best_error = error
                best_q = q.copy()
            if np.linalg.norm(perr) < 0.002 and np.linalg.norm(oerr) < 0.015:
                break
            J = np.vstack([linear_jacobian, angular_jacobian])
            # Weighted task vector matching the official openarm_control cost ratio
            # (position_cost=12.0, orientation_cost=1.5 → weight 1.5/12.0 = 0.125).
            task = np.concatenate([perr, oerr * 0.125])
            # Damping tuned between the official lm_damping (0.01) and the previous
            # value (0.0007): large enough to avoid oscillation near singularities,
            # small enough to converge to < 1e-3.
            damping = 0.002
            dq = J.T @ np.linalg.solve(J @ J.T + damping * np.eye(6), task)
            step = np.linalg.norm(dq)
            if step > 0.12:
                dq *= 0.12 / step
            # Joint limit avoidance: pull away from limits when close (official joint_limit_gain=0.95).
            for j in range(len(q)):
                margin = limits[j, 1] - limits[j, 0]
                lower_dist = q[j] - limits[j, 0]
                upper_dist = limits[j, 1] - q[j]
                braking_distance = min(0.2, margin)
                if lower_dist < braking_distance and dq[j] < 0.0:
                    scale = max(0.0, (lower_dist / braking_distance) ** 2)
                    dq[j] *= scale
                elif upper_dist < braking_distance and dq[j] > 0.0:
                    scale = max(0.0, (upper_dist / braking_distance) ** 2)
                    dq[j] *= scale
            q = np.clip(q + dq, limits[:, 0], limits[:, 1])
        if best_error > 0.02:
            if q_init is None:
                # A successful handover leaves a very different redundant-arm posture.
                # Retry the official spawn posture before calling a reachable goal impossible.
                return self.solve_ik(side, target_position, target_rotation, q_init=SPAWN_ARM_QPOS, max_iterations=max_iterations)
            return None, best_error, float(best_error)
        return best_q, best_error, float(best_error)

    def execute_joint_path(
        self,
        side: str,
        path: list[np.ndarray],
        seconds_per_segment: float = 0.06,
        progress: Callable[[str], None] | None = None,
    ) -> MotionResult:
        steps_per = max(1, int(seconds_per_segment / self.timestep))
        for i, target in enumerate(path):
            if self.stop_requested():
                return MotionResult(False, "STOPPED")
            if progress and i % 8 == 0:
                progress(f"joint_path_{i}/{len(path)}")
            q_start = self.env.arm_qpos(side)
            for alpha in np.linspace(0, 1, max(2, int(np.linalg.norm(target - q_start) / 0.005))):
                qcmd = (1 - alpha) * q_start + alpha * target
                self.env.set_arm(side, qcmd)
                self.env.apply_gravity_compensation(side)
                self.env.step(1)
                if self.stop_requested():
                    return MotionResult(False, "STOPPED")
            self.env.set_arm(side, target)
            self.env.apply_gravity_compensation(side)
            self.env.step(steps_per)
        pos, _ = self.env.ee_pose(side)
        return MotionResult(True, final_position=pos.tolist(), position_error=0.0)

    def move_to_site(
        self,
        side: str,
        position: np.ndarray,
        rotation: np.ndarray | None = None,
        progress: Callable[[str], None] | None = None,
        q_init: np.ndarray | None = None,
    ) -> MotionResult:
        if rotation is None:
            rotation = GRASP_ROTATION.copy()
        q, pos_err, _ = self.solve_ik(side, position, rotation, q_init=q_init)
        if q is None:
            return MotionResult(False, "IK_FAILURE", position.tolist(), pos_err)
        self.target_rotations[side] = rotation.copy()
        start = self.env.arm_qpos(side)
        path = [(1 - a) * start + a * q for a in np.linspace(0, 1, 8)]
        result = self.execute_joint_path(side, path, 0.08, progress)
        if not result.success:
            return result
        actual, actual_rotation = self.env.ee_pose(side)
        residual = float(np.linalg.norm(actual - position))
        rotation_residual = float(np.linalg.norm(orientation_error(actual_rotation, rotation)))
        # Read a settled endpoint. The official wrist's damping can leave a short
        # orientation transient after a large handover rotation or lateral carry.
        for _ in range(15):
            if residual <= .004 and rotation_residual <= .06:
                break
            if self.stop_requested():
                return MotionResult(False, "STOPPED")
            self.env.step(20)
            actual, actual_rotation = self.env.ee_pose(side)
            residual = float(np.linalg.norm(actual - position))
            rotation_residual = float(np.linalg.norm(orientation_error(actual_rotation, rotation)))
        # Solving IK is not evidence that the motors reached it: a collision can block them.
        return MotionResult(residual <= 0.012 and rotation_residual <= 0.15,
                            None if residual <= 0.012 and rotation_residual <= 0.15 else "MOTION_BLOCKED",
                            actual.tolist(), residual, rotation_residual)

    def set_gripper(self, side: str, closed: bool) -> None:
        self.env.set_gripper(side, GRIPPER_CLOSED[side] if closed else GRIPPER_OPEN[side])

    def wait_gripper(self, side: str, closed: bool, seconds: float = 0.45) -> MotionResult:
        self.set_gripper(side, closed)
        self.env.step(max(1, int(seconds / self.timestep)))
        return MotionResult(True)
