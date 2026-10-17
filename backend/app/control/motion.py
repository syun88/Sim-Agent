from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from backend.app.simulation.base import SimulationBackend
from backend.app.simulation.openarm_v2.definition import (
    GRASP_ROTATION,
    GRIPPER_CLOSED,
    GRIPPER_OPEN,
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


class OpenArmV2Controller:
    """Damped least-squares joint-space controller, independent of the physics engine.

    The controller reads poses and Jacobians through `SimulationBackend`, so the same
    inverse kinematics and the same joint path execute on every backend, and an engine
    difference shows up as a physical difference rather than as a different controller.
    """

    def __init__(self, env: SimulationBackend):
        self.env = env
        self.cancelled = False

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
        qpos_backup = self.env.arm_qpos(side)
        best_q = q.copy()
        best_error = np.inf
        for _ in range(max_iterations):
            self.env.set_arm_qpos(side, q)
            self.env.kinematics()
            position, rotation = self.env.ee_pose(side)
            linear_jacobian, angular_jacobian = self.env.ee_jacobian(side)
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
        self.env.set_arm_qpos(side, qpos_backup)
        self.env.refresh()
        if best_error > 0.02:
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
            if self.cancelled:
                return MotionResult(False, "STOPPED")
            if progress and i % 8 == 0:
                progress(f"joint_path_{i}/{len(path)}")
            q_start = self.env.arm_qpos(side)
            for alpha in np.linspace(0, 1, max(2, int(np.linalg.norm(target - q_start) / 0.005))):
                qcmd = (1 - alpha) * q_start + alpha * target
                self.env.set_arm(side, qcmd)
                self.env.apply_gravity_compensation(side)
                self.env.step(1)
                if self.cancelled:
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
    ) -> MotionResult:
        if rotation is None:
            rotation = GRASP_ROTATION.copy()
        q, pos_err, _ = self.solve_ik(side, position, rotation)
        if q is None:
            return MotionResult(False, "IK_FAILURE", position.tolist(), pos_err)
        start = self.env.arm_qpos(side)
        path = [(1 - a) * start + a * q for a in np.linspace(0, 1, 8)]
        return self.execute_joint_path(side, path, 0.08, progress)

    def set_gripper(self, side: str, closed: bool) -> None:
        self.env.set_gripper(side, GRIPPER_CLOSED[side] if closed else GRIPPER_OPEN[side])

    def wait_gripper(self, side: str, closed: bool, seconds: float = 0.45) -> MotionResult:
        self.set_gripper(side, closed)
        self.env.step(max(1, int(seconds / self.timestep)))
        return MotionResult(True)
