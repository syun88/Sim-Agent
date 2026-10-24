"""Tests for the engine-neutral navigation core (draft for Sim-Agent).

Run with: uv run pytest -q tests/test_navigation.py
The tests simulate a base offline with DiffDrive.integrate; the real base
pose always comes from the physics engine.
"""

import math

import numpy as np

from backend.app.control.navigation import (
    BasePose,
    DiffDrive,
    KiwiDrive,
    WaypointController,
    clearance_to_aabbs,
    wrap_angle,
)


def test_wrap_angle():
    assert wrap_angle(0.0) == 0.0
    assert abs(wrap_angle(3 * math.pi) - (-math.pi)) < 1e-12
    assert abs(wrap_angle(-3 * math.pi / 2) - math.pi / 2) < 1e-12


def test_diff_drive_round_trip():
    drive = DiffDrive(wheel_radius=0.05, track_width=0.4)
    left, right = drive.wheel_speeds(0.3, 0.5)
    v, omega = drive.twist(left, right)
    assert abs(v - 0.3) < 1e-12 and abs(omega - 0.5) < 1e-12
    # Pure rotation: equal and opposite wheel speeds, no forward motion.
    left, right = drive.wheel_speeds(0.0, 1.0)
    assert abs(left + right) < 1e-12


def test_kiwi_drive_round_trip():
    drive = KiwiDrive(base_radius=0.12)
    wheels = drive.wheel_speeds(0.2, -0.1, 0.7)
    vx, vy, omega = drive.body_velocity(wheels)
    assert abs(vx - 0.2) < 1e-12 and abs(vy + 0.1) < 1e-12 and abs(omega - 0.7) < 1e-12
    # Pure rotation: all three wheels roll at the same speed.
    wheels = drive.wheel_speeds(0.0, 0.0, 1.0)
    assert np.allclose(wheels, 0.12)


def test_kiwi_drive_integrate_lateral():
    drive = KiwiDrive(base_radius=0.12)
    # Facing +y (yaw = pi/2), body-frame +x points along world +y.
    pose = drive.integrate(BasePose(0.0, 0.0, math.pi / 2), 0.3, 0.0, 0.0, 1.0)
    assert abs(pose.x) < 1e-12 and abs(pose.y - 0.3) < 1e-12
    # Pure lateral body velocity at yaw 0 moves the base sideways.
    pose = drive.integrate(BasePose(0.0, 0.0, 0.0), 0.0, 0.4, 0.0, 0.5)
    assert abs(pose.x) < 1e-12 and abs(pose.y - 0.2) < 1e-12


def test_clearance_to_aabbs():
    box = np.array([[[1.0, -0.5], [2.0, 0.5]]])
    pose = BasePose(0.0, 0.0, 0.0)
    assert abs(clearance_to_aabbs(pose, 0.3, box) - 0.7) < 1e-9
    inside = BasePose(1.5, 0.0, 0.0)
    assert clearance_to_aabbs(inside, 0.3, box) == 0.0
    assert clearance_to_aabbs(pose, 0.3, np.empty((0, 2, 2))) == math.inf


def _run(controller, drive, start, target, target_yaw=None, aabbs=None,
         radius=0.25, dt=0.02, max_steps=4000):
    pose = start
    for _ in range(max_steps):
        clearance = clearance_to_aabbs(pose, radius, aabbs) if aabbs is not None else math.inf
        cmd = controller.update(pose, np.asarray(target, dtype=float),
                                target_yaw=target_yaw, obstacle_clearance=clearance)
        if cmd.status != "running":
            return pose, cmd.status
        pose = drive.integrate(pose, cmd.v, cmd.omega, dt)
    return pose, "timeout"


def test_controller_reaches_target():
    controller = WaypointController()
    drive = DiffDrive(wheel_radius=0.05, track_width=0.4)
    pose, status = _run(controller, drive, BasePose(0.0, 0.0, 0.0), (1.5, 0.8))
    assert status == "arrived"
    assert np.linalg.norm(np.array([pose.x, pose.y]) - np.array([1.5, 0.8])) <= 0.05


def test_controller_aligns_final_yaw():
    controller = WaypointController()
    drive = DiffDrive(wheel_radius=0.05, track_width=0.4)
    pose, status = _run(controller, drive, BasePose(0.0, 0.0, 0.0), (1.0, 0.0),
                         target_yaw=math.pi / 2)
    assert status == "arrived"
    assert abs(wrap_angle(pose.yaw - math.pi / 2)) <= 0.05


def test_controller_stops_for_obstacle():
    controller = WaypointController()
    drive = DiffDrive(wheel_radius=0.05, track_width=0.4)
    # Wall segment directly on the path to the goal.
    aabbs = np.array([[[0.9, -1.0], [1.1, 1.0]]])
    pose, status = _run(controller, drive, BasePose(0.0, 0.0, 0.0), (2.0, 0.0),
                        aabbs=aabbs)
    assert status == "blocked"
    # The footprint never penetrated the wall.
    assert clearance_to_aabbs(pose, 0.25, aabbs) >= 0.05 - 1e-9
