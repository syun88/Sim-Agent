from __future__ import annotations

from dataclasses import dataclass
import os
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

# Motor targets need not cross the Python/engine boundary at the 1 kHz physics
# rate. The solver still integrates every step; a target is held for at most 4 ms.
MOTOR_COMMAND_PERIOD = .004

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
    nominal_duration: float | None = None
    trajectory_duration: float | None = None
    settling_duration: float | None = None


class OpenArmV2Controller:
    """Damped least-squares joint-space controller, independent of the physics engine.

    IK evaluates a private kinematic snapshot through `SimulationBackend`. Candidate
    poses never enter the live scene or its viewer; only motor commands execute there.
    Endpoint verification always reads the actual physics state.
    """

    def __init__(self, env: SimulationBackend, *, motion_speed: float | None = None):
        self.env = env
        self.motion_speed = float(os.getenv("OPENARM_MOTION_SPEED", "6")) if motion_speed is None else float(motion_speed)
        if not np.isfinite(self.motion_speed) or self.motion_speed <= 0:
            raise ValueError("OPENARM_MOTION_SPEED must be a finite positive number")
        self.env.motion_speed = self.motion_speed
        self.env.configure_motion_response(self.motion_speed)
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
        command_steps = max(1, int(round(MOTOR_COMMAND_PERIOD / self.timestep)))
        for i, target in enumerate(path):
            if self.stop_requested():
                return MotionResult(False, "STOPPED")
            if progress and i % 8 == 0:
                progress(f"joint_path_{i}/{len(path)}")
            q_start = self.env.arm_qpos(side)
            distance = float(np.linalg.norm(target - q_start))
            if distance < 1e-9:
                # The first old waypoint was exactly the observed start pose,
                # but still added an 80 ms dwell to every move, even a 1 cm one.
                continue
            interpolation_steps = max(2, int(distance / .005))
            for index in range(0, interpolation_steps, command_steps):
                alpha = index / (interpolation_steps - 1)
                qcmd = (1 - alpha) * q_start + alpha * target
                self.env.set_arm(side, qcmd)
                self.env.apply_gravity_compensation(side)
                self.env.step(min(command_steps, interpolation_steps - index))
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
        duration: float | None = None,
        scale_speed: bool = True,
        via: list[np.ndarray] | None = None,
        via_rotations: list[np.ndarray] | None = None,
        on_sample: Callable[[], None] | None = None,
    ) -> MotionResult:
        if rotation is None:
            rotation = GRASP_ROTATION.copy()
        self.env.configure_motion_response(self.motion_speed if scale_speed else 1.)
        joint_waypoints = [self.env.arm_qpos(side)]
        q, pos_err = None, float("inf")
        rotations = via_rotations if via_rotations is not None else [rotation] * len(via or [])
        if len(rotations) != len(via or []):
            raise ValueError("Each via position needs one via rotation")
        for waypoint, attitude in zip([*(via or []), position], [*rotations, rotation]):
            seed = joint_waypoints[-1] if via else q_init
            q, pos_err, _ = self.solve_ik(side, waypoint, attitude, q_init=seed)
            if q is None:
                break
            joint_waypoints.append(q)
        if q is None:
            return MotionResult(False, "IK_FAILURE", position.tolist(), pos_err)
        self.target_rotations[side] = rotation.copy()
        start = self.env.arm_qpos(side)
        nominal_duration = duration
        trajectory_duration = None
        if duration is None and self.motion_speed == 1. and not via:
            path = [(1 - a) * start + a * q for a in np.linspace(0, 1, 8)]
            result = self.execute_joint_path(side, path, .08, progress)
        else:
            if nominal_duration is None:
                # The previous path spent 80ms at each of seven waypoints plus
                # its interpolation time. Schedule one continuous trajectory.
                nominal_duration = sum(.56 + max(14, int(np.linalg.norm(b - a) / .005)) * self.timestep
                                       for a, b in zip(joint_waypoints, joint_waypoints[1:]))
            trajectory_duration = nominal_duration / (self.motion_speed if scale_speed else 1.)
            result = MotionResult(True)
            steps = max(2, int(np.ceil(trajectory_duration / self.timestep)))
            trajectory_duration = steps * self.timestep
            curve = derivative = None
            if via:
                from scipy.interpolate import PchipInterpolator
                knots = np.linspace(0., 1., len(joint_waypoints))
                # Shape preservation prevents a high-clearance approach from
                # overshooting the last insertion segment into the workpiece.
                curve = PchipInterpolator(knots, joint_waypoints, axis=0)
                derivative = curve.derivative()
            command_steps = max(1, int(round(MOTOR_COMMAND_PERIOD / self.timestep))) if scale_speed else 1
            indices = range(0, steps, command_steps) if scale_speed else range(1, steps + 1)
            completed = False
            try:
                for index in indices:
                    if self.stop_requested():
                        return MotionResult(False, "STOPPED")
                    fraction = index / steps
                    alpha = fraction ** 3 * (10. - 15. * fraction + 6. * fraction ** 2)
                    alpha_dot = 30. * fraction ** 2 * (1. - fraction) ** 2 / trajectory_duration
                    reference = start + (q - start) * alpha if curve is None else curve(alpha)
                    velocity = ((q - start) if curve is None else derivative(alpha)) * alpha_dot if scale_speed else None
                    self.env.set_arm(side, reference, velocities=velocity)
                    self.env.step(min(command_steps, steps - index) if scale_speed else 1)
                    if on_sample is not None and index % max(command_steps, round(.02 / self.timestep)) == 0:
                        on_sample()
                completed = True
            finally:
                # A stop/error must also clear the velocity reference; otherwise
                # the still-running scene would continue driving the stopped arm.
                self.env.set_arm(side, q if completed else self.env.arm_qpos(side))
        if not result.success:
            return result
        actual, actual_rotation = self.env.ee_pose(side)
        residual = float(np.linalg.norm(actual - position))
        rotation_residual = float(np.linalg.norm(orientation_error(actual_rotation, rotation)))
        # Read a settled endpoint. The official wrist's damping can leave a short
        # orientation transient after a large handover rotation or lateral carry.
        settling_steps = 0
        previous_position, previous_rotation = actual.copy(), actual_rotation.copy()
        stable_steps = 0
        settle_quantum = 10 if scale_speed else 20
        require_stable = scale_speed and self.motion_speed != 1.
        for _ in range(300 // settle_quantum):
            if not require_stable and residual <= .004 and rotation_residual <= .06:
                break
            if self.stop_requested():
                return MotionResult(False, "STOPPED")
            self.env.step(settle_quantum)
            settling_steps += settle_quantum
            if on_sample is not None:
                on_sample()
            actual, actual_rotation = self.env.ee_pose(side)
            residual = float(np.linalg.norm(actual - position))
            rotation_residual = float(np.linalg.norm(orientation_error(actual_rotation, rotation)))
            if (residual <= .012 and rotation_residual <= .15
                    and np.linalg.norm(actual - previous_position) < .0001
                    and np.linalg.norm(orientation_error(previous_rotation, actual_rotation)) < .001):
                stable_steps += settle_quantum
            else:
                stable_steps = 0
            previous_position, previous_rotation = actual.copy(), actual_rotation.copy()
            if require_stable and stable_steps * self.timestep >= .03:
                break
        # Solving IK is not evidence that the motors reached it: a collision can block them.
        return MotionResult(residual <= 0.012 and rotation_residual <= 0.15,
                            None if residual <= 0.012 and rotation_residual <= 0.15 else "MOTION_BLOCKED",
                            actual.tolist(), residual, rotation_residual,
                            nominal_duration, trajectory_duration, settling_steps * self.timestep)

    def set_gripper(self, side: str, closed: bool) -> None:
        self.env.set_gripper(side, GRIPPER_CLOSED[side] if closed else GRIPPER_OPEN[side])

    def wait_gripper(self, side: str, closed: bool, seconds: float = 0.45) -> MotionResult:
        self.set_gripper(side, closed)
        budget = max(1, int(seconds / self.timestep))
        if self.motion_speed == 1.:
            self.env.step(budget)
            return MotionResult(True)
        # Closure speed is produced by the official finger motors. Finish once
        # their measured response settles, rather than always idling for 650ms.
        quantum = max(1, int(round(.01 / self.timestep)))
        previous = self.env.gripper_position(side)
        stable_steps = elapsed = 0
        completed = False
        try:
            while elapsed < budget:
                if self.stop_requested():
                    return MotionResult(False, "STOPPED")
                count = min(quantum, budget - elapsed)
                self.env.step(count)
                elapsed += count
                measured = self.env.gripper_position(side)
                near_goal = closed or abs(measured - GRIPPER_OPEN[side]) < .03
                stable_steps = stable_steps + count if near_goal and abs(measured - previous) < .005 else 0
                previous = measured
                if elapsed * self.timestep >= (.18 if closed else .04) and stable_steps * self.timestep >= .03:
                    break
            completed = True
        finally:
            if not completed:
                self.env.set_gripper(side, self.env.gripper_position(side))
        return MotionResult(True)
