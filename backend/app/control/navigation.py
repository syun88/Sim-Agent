"""Engine-neutral navigation core for a wheeled base (draft for Sim-Agent).

This module is the *planning/control* half of scene navigation for a mobile
base or cart. It deliberately knows nothing about Genesis or MJCF:

- it consumes a measured base pose (from the backend's ``base_pose()``),
- it produces body velocity commands ``(v, omega)`` that the backend turns
  into wheel motor targets, exactly like arm joint targets today,
- arrival is decided from the *measured* pose, never from the commanded one,
  mirroring the project rule that a grasp is verified from real contacts and
  never faked by writing a pose.

Nothing here writes a pose into a scene. The only pose integration in this
file is the closed-form differential-drive update used by the unit tests to
simulate a base offline; the real base moves by physics, through its wheels.

Integration sketch (not in this draft): ``SimulationBackend`` gains optional
``base_pose()`` / ``command_base_velocity(v, omega)`` operations (default
``NotImplementedError``, like ``object_bounds``), ``Frame`` gains an optional
``base`` field, and a Genesis scene adds a wheeled cart entity whose wheel
actuators are driven at the existing motor-command cadence.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

_TWO_PI = 2.0 * math.pi


def wrap_angle(angle: float) -> float:
    """Wrap an angle to [-pi, pi)."""
    return (angle + math.pi) % _TWO_PI - math.pi


@dataclass
class BasePose:
    """Planar pose of a mobile base in the world frame (metres, radians)."""

    x: float
    y: float
    yaw: float

    def position(self) -> np.ndarray:
        return np.array([self.x, self.y])


@dataclass
class BaseCommand:
    """One control tick: body forward velocity and yaw rate, plus status."""

    v: float
    omega: float
    status: str  # "running" | "arrived" | "blocked"


class DiffDrive:
    """Differential-drive kinematics for a two-wheel base with casters.

    ``wheel_radius`` and ``track_width`` (distance between the driven wheel
    contact patches) are in metres. Wheel speeds are rad/s.
    """

    def __init__(self, wheel_radius: float, track_width: float):
        if wheel_radius <= 0 or track_width <= 0:
            raise ValueError("wheel_radius and track_width must be positive")
        self.wheel_radius = float(wheel_radius)
        self.track_width = float(track_width)

    def wheel_speeds(self, v: float, omega: float) -> tuple[float, float]:
        """Body twist -> (left, right) wheel angular speeds."""
        half = omega * self.track_width / 2.0
        return ((v - half) / self.wheel_radius, (v + half) / self.wheel_radius)

    def twist(self, left: float, right: float) -> tuple[float, float]:
        """(left, right) wheel angular speeds -> body twist (v, omega)."""
        vl = left * self.wheel_radius
        vr = right * self.wheel_radius
        return ((vl + vr) / 2.0, (vr - vl) / self.track_width)

    def integrate(self, pose: BasePose, v: float, omega: float, dt: float) -> BasePose:
        """Exact constant-curvature pose update over one tick.

        Offline simulation helper for tests and planning previews; the real
        base pose always comes from the physics engine.
        """
        if abs(omega) < 1e-12:
            return BasePose(
                pose.x + v * dt * math.cos(pose.yaw),
                pose.y + v * dt * math.sin(pose.yaw),
                pose.yaw,
            )
        radius = v / omega
        yaw1 = pose.yaw + omega * dt
        return BasePose(
            pose.x + radius * (math.sin(yaw1) - math.sin(pose.yaw)),
            pose.y - radius * (math.cos(yaw1) - math.cos(pose.yaw)),
            wrap_angle(yaw1),
        )


class KiwiDrive:
    """Three-wheel omnidirectional (kiwi) drive kinematics, as in LeKiwi.

    Each wheel sits at ``wheel_angles[i]`` (its position angle around the
    base centre, radians) and rolls tangentially. Wheel speeds here are the
    *linear* rolling speeds (m/s); divide by the wheel radius for rad/s.

    Draft note: the default angles follow the LeKiwi layout (wheels at
    30/150/270 degrees). Sign conventions must be re-checked against
    LeRobot's own ``lekiwi`` kinematics when this is integrated, the same
    way grasp geometry is checked against the official model description.
    """

    def __init__(
        self,
        base_radius: float,
        wheel_angles: tuple[float, float, float] = (
            math.radians(30.0),
            math.radians(150.0),
            math.radians(270.0),
        ),
    ):
        if base_radius <= 0:
            raise ValueError("base_radius must be positive")
        self.base_radius = float(base_radius)
        self.wheel_angles = tuple(float(a) for a in wheel_angles)
        # Row i maps body velocity (vx, vy, omega) to wheel i's rolling
        # speed: the tangential component of the contact-point velocity.
        self._matrix = np.array(
            [
                [-math.sin(a), math.cos(a), self.base_radius]
                for a in self.wheel_angles
            ]
        )
        self._inverse = np.linalg.inv(self._matrix)

    def wheel_speeds(self, vx: float, vy: float, omega: float) -> np.ndarray:
        """Body velocity -> the three wheels' linear rolling speeds."""
        return self._matrix @ np.array([vx, vy, omega])

    def body_velocity(self, wheel_speeds: np.ndarray) -> np.ndarray:
        """The three wheels' rolling speeds -> body velocity (vx, vy, omega)."""
        return self._inverse @ np.asarray(wheel_speeds, dtype=float)

    def integrate(self, pose: BasePose, vx: float, vy: float, omega: float, dt: float) -> BasePose:
        """Exact SE(2) pose update for constant body velocity over one tick.

        Offline simulation helper, like ``DiffDrive.integrate``.
        """
        if abs(omega) < 1e-12:
            dx_b, dy_b = vx * dt, vy * dt
        else:
            theta = omega * dt
            dx_b = (math.sin(theta) * vx - (1.0 - math.cos(theta)) * vy) / omega
            dy_b = ((1.0 - math.cos(theta)) * vx + math.sin(theta) * vy) / omega
        cos_y, sin_y = math.cos(pose.yaw), math.sin(pose.yaw)
        return BasePose(
            pose.x + cos_y * dx_b - sin_y * dy_b,
            pose.y + sin_y * dx_b + cos_y * dy_b,
            wrap_angle(pose.yaw + omega * dt),
        )


def clearance_to_aabbs(pose: BasePose, radius: float, aabbs: np.ndarray) -> float:
    """Minimum distance from a circular footprint to axis-aligned boxes.

    ``aabbs`` is an (N, 2, 2) array of [minimum, maximum] corners in the
    ground plane, matching the shape ``SimulationBackend.object_bounds``
    already returns per object. Returns ``inf`` when there are no obstacles
    and 0.0 when the footprint already overlaps a box.
    """
    if aabbs is None or len(aabbs) == 0:
        return math.inf
    boxes = np.asarray(aabbs, dtype=float).reshape(-1, 2, 2)
    point = np.array([pose.x, pose.y])
    clamped = np.clip(point, boxes[:, 0], boxes[:, 1])
    distances = np.linalg.norm(clamped - point, axis=1)
    return float(max(0.0, distances.min() - radius))


class WaypointController:
    """Proportional go-to-pose control for one waypoint at a time.

    The controller turns in place when the heading error is large, drives
    with a heading correction while approaching, slows down near the goal,
    and (when ``target_yaw`` is given) finishes by aligning the heading.
    Speeds are hard-clamped to ``max_v`` / ``max_omega``; the obstacle guard
    can only ever *reduce* the forward speed, never increase it.
    """

    def __init__(
        self,
        max_v: float = 0.5,
        max_omega: float = 1.2,
        arrive_tolerance: float = 0.05,
        yaw_tolerance: float = 0.05,
        turn_in_place_angle: float = 0.6,
        slow_radius: float = 0.6,
    ):
        self.max_v = float(max_v)
        self.max_omega = float(max_omega)
        self.arrive_tolerance = float(arrive_tolerance)
        self.yaw_tolerance = float(yaw_tolerance)
        self.turn_in_place_angle = float(turn_in_place_angle)
        self.slow_radius = float(slow_radius)

    def update(
        self,
        pose: BasePose,
        target: np.ndarray,
        target_yaw: float | None = None,
        obstacle_clearance: float = math.inf,
        stop_distance: float = 0.05,
        slow_distance: float = 0.4,
    ) -> BaseCommand:
        target = np.asarray(target, dtype=float)
        offset = target - pose.position()
        distance = float(np.linalg.norm(offset))

        if distance <= self.arrive_tolerance:
            if target_yaw is None:
                return BaseCommand(0.0, 0.0, "arrived")
            yaw_error = wrap_angle(target_yaw - pose.yaw)
            if abs(yaw_error) <= self.yaw_tolerance:
                return BaseCommand(0.0, 0.0, "arrived")
            omega = float(np.clip(2.0 * yaw_error, -self.max_omega, self.max_omega))
            return BaseCommand(0.0, omega, "running")

        # Obstacle guard: stop before contact; report blocked instead of
        # pushing through, so the caller (skill layer) can replan or fail
        # honestly rather than claim an arrival that never happened.
        if obstacle_clearance <= stop_distance:
            return BaseCommand(0.0, 0.0, "blocked")

        heading_error = wrap_angle(math.atan2(offset[1], offset[0]) - pose.yaw)
        if abs(heading_error) > self.turn_in_place_angle:
            omega = float(np.clip(2.0 * heading_error, -self.max_omega, self.max_omega))
            return BaseCommand(0.0, omega, "running")

        v = min(self.max_v, max(0.15 * self.max_v, 1.2 * distance))
        if distance < self.slow_radius:
            v = min(v, self.max_v * distance / self.slow_radius + 0.05)
        if obstacle_clearance < slow_distance:
            scale = max(0.0, (obstacle_clearance - stop_distance) / (slow_distance - stop_distance))
            v *= scale
            if v < 0.01:
                # The guard has effectively stopped the base short of the
                # goal: report blocked instead of creeping forever.
                return BaseCommand(0.0, 0.0, "blocked")
        omega = float(np.clip(2.0 * heading_error, -self.max_omega, self.max_omega))
        return BaseCommand(float(v), omega, "running")
