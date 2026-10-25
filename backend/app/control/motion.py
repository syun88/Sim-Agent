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
# rate. The target cadence is 4ms, or one physics step when its dt is larger.
MOTOR_COMMAND_PERIOD = .004


def maximum_spline_slope(curve) -> np.ndarray:
    """Exact per-joint derivative bound for a piecewise cubic spline."""
    # q'(t) = 3*a*t**2 + 2*b*t + c on each local interval.
    a, b, c = curve.c[:3]
    widths = np.diff(curve.x)[:, None]
    at_end = 3. * a * widths**2 + 2. * b * widths + c
    vertex = np.divide(-b, 3. * a, out=np.zeros_like(a), where=abs(a) > 1e-15)
    inside = (vertex > 0.) & (vertex < widths)
    at_vertex = np.where(inside, 3.*a*vertex**2 + 2.*b*vertex + c, 0.)
    return np.max(np.maximum(np.maximum(abs(c), abs(at_end)), abs(at_vertex)), axis=0)


def joint_spline(waypoints, velocity_limits=None):
    from scipy.interpolate import PchipInterpolator
    points = np.asarray(waypoints,float)
    if velocity_limits is None:
        knots = np.linspace(0.,1.,len(points))
    else:
        # Allocate progress by joint travel time. Uniform knots make one large
        # posture correction slow down every tiny Cartesian sample in the path.
        weights = np.maximum(np.max(abs(np.diff(points,axis=0))/velocity_limits,axis=1),1e-6)
        knots = np.r_[0.,np.cumsum(weights)]
        knots /= knots[-1]
    return PchipInterpolator(knots,points,axis=0)



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

    def __init__(self, env: SimulationBackend, *, motion_speed: float | None = None, configure_motors: bool = True):
        self.env = env
        self.motion_speed = float(os.getenv("OPENARM_MOTION_SPEED", "6")) if motion_speed is None else float(motion_speed)
        if not np.isfinite(self.motion_speed) or self.motion_speed <= 0:
            raise ValueError("OPENARM_MOTION_SPEED must be a finite positive number")
        if configure_motors:
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
        # A robot with fewer than six joints cannot hold an arbitrary attitude, so its attitude is not
        # asked for. Solving *with* one makes the least-squares task spend its freedom on a term it can
        # never satisfy, and the iterate the mixed error keeps can be tens of millimetres off in
        # position: measured 2026-10-25 on the four-joint OpenManipulator-X, every carry was refused
        # with a 40-90 mm residual - including a carry to the very pose the arm was already holding -
        # while the planner's IK, which asks for the position alone below six joints, solved the same
        # poses with zero error. Six-joint arms still hold the attitude they are given.
        under_actuated = len(limits) < 6
        if q_init is None:
            q = self.env.arm_qpos(side)
        else:
            q = np.clip(np.asarray(q_init, float), limits[:, 0], limits[:, 1])
        if under_actuated:
            # Not merely "do not substitute one": an arm that cannot hold an attitude must not be
            # asked for one even when the caller passes it, because the least-squares task then spends
            # its freedom on a term it can never satisfy and the iterate the mixed error keeps is tens
            # of millimetres off in position. Measured 2026-10-25 on the four-joint
            # OpenManipulator-X: carries were refused with 40-90 mm residuals for targets the
            # planner's position-only IK solved with zero error, including the pose the arm was
            # already holding. Calls that want an attitude on such an arm still get the position.
            target_rotation = None
        elif target_rotation is None:
            target_rotation = self.env.ee_pose(side)[1]
        kinematics = self.env.arm_kinematics(side)
        best_q = q.copy()
        best_error = np.inf
        for _ in range(max_iterations):
            position, rotation, linear_jacobian, angular_jacobian = kinematics.evaluate(q)
            perr = np.asarray(target_position, float) - position
            oerr = np.zeros(3) if target_rotation is None else orientation_error(rotation, target_rotation)
            error = float(np.linalg.norm(perr) + np.linalg.norm(oerr) * 0.2)
            if error < best_error:
                best_error = error
                best_q = q.copy()
            if np.linalg.norm(perr) < getattr(self.env, "ik_position_tolerance", .002) and np.linalg.norm(oerr) < getattr(self.env, "ik_rotation_tolerance", .015):
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
        # Judge the solution by the task that was actually asked for. The error above mixes position
        # with a weighted orientation term, and an arm with fewer than six joints cannot drive that
        # orientation term to zero at all: measured 2026-10-25, every carry of the four-joint
        # OpenManipulator-X failed here with the same ~90 mm residual because a full six-degree-of-
        # freedom pose was demanded, while the planner's IK - which already falls back to the position
        # below six joints - solved the very same pose with zero error. Six-joint arms are unchanged.
        best_position, best_rotation, _, _ = kinematics.evaluate(best_q)
        position_error = float(np.linalg.norm(np.asarray(target_position, float) - best_position))
        attitude_error = 0. if target_rotation is None else float(np.linalg.norm(orientation_error(best_rotation, target_rotation)))
        position_tolerance = getattr(self.env, "ik_position_tolerance", .002)
        attitude_tolerance = getattr(self.env, "ik_rotation_tolerance", .015)
        if best_error > 0.02 and not (under_actuated and position_error <= position_tolerance):
            if q_init is None:
                # A successful handover leaves a very different redundant-arm posture.
                # Retry the official spawn posture before calling a reachable goal impossible.
                return self.solve_ik(side, target_position, target_rotation, q_init=self.env.home_arm_qpos(side), max_iterations=max_iterations)
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
        planned_joints: list[np.ndarray] | None = None,
    ) -> MotionResult:
        if rotation is None:
            # The official cell's grasp frame. An imported robot's own attitude is passed in explicitly by
            # the caller that knows the robot (see ``RobotSkills._carry_aim``) rather than substituted
            # here: measured 2026-10-25, changing this default broke the official handover, whose right
            # arm is mirrored and depends on exactly this constant.
            rotation = GRASP_ROTATION.copy()
        self.env.configure_motion_response(self.motion_speed if scale_speed else 1.)
        joint_waypoints = [self.env.arm_qpos(side)]
        q, pos_err = None, float("inf")
        rotations = via_rotations if via_rotations is not None else [rotation] * len(via or [])
        if len(rotations) != len(via or []):
            raise ValueError("Each via position needs one via rotation")
        if planned_joints is not None:
            path = np.asarray(planned_joints, float)
            limits = self.env.arm_limits(side)
            if (path.ndim != 2 or len(path)<2 or path.shape[1]!=len(joint_waypoints[0])
                    or not np.isfinite(path).all() or np.max(abs(path[0]-joint_waypoints[0]))>.02
                    or np.any(path<limits[:,0]) or np.any(path>limits[:,1])):
                return MotionResult(False,'STALE_MOTION_PLAN')
            q = path[-1].copy()
            point, attitude, _, _ = self.env.arm_kinematics(side).evaluate(q)
            pos_err = float(np.linalg.norm(point-position))
            if pos_err>.002 or np.linalg.norm(orientation_error(attitude,rotation))>.03:
                return MotionResult(False,'INVALID_MOTION_PLAN',position.tolist(),pos_err)
            joint_waypoints.extend(path[1:])
        else:
            for waypoint, attitude in zip([*(via or []), position], [*rotations, rotation]):
                seed = joint_waypoints[-1] if via else q_init
                q, pos_err, _ = self.solve_ik(side, waypoint, attitude, q_init=seed)
                if q is None:
                    break
                joint_waypoints.append(q)
        if q is None:
            return MotionResult(False, "IK_FAILURE", position.tolist(), pos_err)
        planner = getattr(self.env, 'plan_joint_waypoints', None)
        if planned_joints is not None:
            check = getattr(self.env,'joint_waypoints_are_clear',None)
            if check is not None and not check(side,joint_waypoints):
                return MotionResult(False,'STALE_MOTION_PLAN',position.tolist(),pos_err)
        elif planner is not None:
            joint_waypoints = planner(side, joint_waypoints, self.stop_requested)
            if joint_waypoints is None:
                return MotionResult(False, "COLLISION_PATH", position.tolist(), pos_err)
        curved_path = bool(via) or len(joint_waypoints)>2
        self.target_rotations[side] = rotation.copy()
        start = self.env.arm_qpos(side)
        nominal_duration = duration
        trajectory_duration = None
        velocity_limits = self.env.arm_velocity_limits(side)
        if duration is None and self.motion_speed == 1. and not curved_path and velocity_limits is None:
            path = [(1 - a) * start + a * q for a in np.linspace(0, 1, 8)]
            result = self.execute_joint_path(side, path, .08, progress)
        else:
            if nominal_duration is None:
                # The previous path spent 80ms at each of seven waypoints plus
                # its interpolation time. Schedule one continuous trajectory.
                if velocity_limits is not None:
                    # Timing belongs to the whole curve. Adding 560ms per IK
                    # sample made a densely sampled, smooth path needlessly slow.
                    nominal_duration=max(.12,1.875*float(np.sum(np.max(
                        abs(np.diff(joint_waypoints,axis=0))/velocity_limits,axis=1))))
                else:
                    nominal_duration = sum(.56 + max(14, int(np.linalg.norm(b - a) / .005)) * self.timestep
                                           for a, b in zip(joint_waypoints, joint_waypoints[1:]))
            trajectory_duration = nominal_duration / (self.motion_speed if scale_speed else 1.)
            curve = derivative = None
            if curved_path:
                curve = joint_spline(joint_waypoints,velocity_limits)
                derivative = curve.derivative()
                check_curve = getattr(self.env, 'joint_curve_is_clear', None)
                if check_curve is not None and not check_curve(side, curve, joint_waypoints):
                    return MotionResult(False, 'COLLISION_PATH', position.tolist(), pos_err)
            if velocity_limits is not None:
                # Bound the actual spline derivative. Multiplying the largest
                # segment by 3*N needlessly slowed finely sampled approaches.
                slopes = maximum_spline_slope(curve) if curve is not None else abs(q-start)
                trajectory_duration = max(trajectory_duration, float(np.max(1.875 * slopes / velocity_limits)))
            timing=None
            tool_limits=getattr(self.env,'cartesian_motion_limits',None)
            if tool_limits:
                from backend.app.control.path_timing import PathTiming
                timing=PathTiming(self.env.arm_kinematics(side),curve,start,q,velocity_limits,tool_limits)
                # The locally timed profile enforces motor limits too. Retain
                # the caller's nominal pacing; remove the global worst-corner
                # bound which would slow every other part of the motion.
                trajectory_duration=max(nominal_duration/(self.motion_speed if scale_speed else 1.),timing.duration)
            result = MotionResult(True)
            steps = max(2, int(np.ceil(trajectory_duration / self.timestep)))
            trajectory_duration = steps * self.timestep
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
                    if timing is not None:
                        stretch=trajectory_duration/timing.duration
                        alpha,alpha_dot=timing.phase(index*self.timestep/stretch)
                        alpha_dot/=stretch
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
        position_tolerance = getattr(self.env, "motion_position_tolerance", .012)
        rotation_tolerance = getattr(self.env, "motion_rotation_tolerance", .15)
        settling_steps = 0
        previous_position, previous_rotation = actual.copy(), actual_rotation.copy()
        stable_steps = 0
        settle_quantum = 10 if scale_speed else 20
        settling_iterations=300//settle_quantum
        if getattr(self.env,'cartesian_motion_limits',None):
            settle_quantum=max(1,round(.01/self.timestep))
            settling_iterations=int(np.ceil(.6/(settle_quantum*self.timestep)))
        require_stable = scale_speed and self.motion_speed != 1.
        for _ in range(settling_iterations):
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
            if (residual <= position_tolerance and rotation_residual <= rotation_tolerance
                    and np.linalg.norm(actual - previous_position) < .0001
                    and np.linalg.norm(orientation_error(previous_rotation, actual_rotation)) < .001):
                stable_steps += settle_quantum
            else:
                stable_steps = 0
            previous_position, previous_rotation = actual.copy(), actual_rotation.copy()
            if require_stable and stable_steps * self.timestep >= .03:
                break
        # Solving IK is not evidence that the motors reached it: a collision can block them.
        return MotionResult(residual <= position_tolerance and rotation_residual <= rotation_tolerance,
                            None if residual <= position_tolerance and rotation_residual <= rotation_tolerance else "MOTION_BLOCKED",
                            actual.tolist(), residual, rotation_residual,
                            nominal_duration, trajectory_duration, settling_steps * self.timestep)

    def set_gripper(self, side: str, closed: bool) -> None:
        self.env.set_gripper(side, self.env.gripper_target(side, closed))

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
                near_goal = closed or abs(measured - self.env.gripper_target(side, False)) < .03
                stable_steps = stable_steps + count if near_goal and abs(measured - previous) < .005 else 0
                previous = measured
                if elapsed * self.timestep >= (.18 if closed else .04) and stable_steps * self.timestep >= .03:
                    break
            completed = True
        finally:
            if not completed:
                self.env.set_gripper(side, self.env.gripper_position(side))
        return MotionResult(True)
