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
    correct_twist,
    limit_wheel_speeds,
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


def test_kiwi_drive_per_wheel_levers():
    drive = KiwiDrive(base_radius=[0.10, 0.20, 0.30])
    # Pure rotation: each wheel rolls at omega * its own lever arm.
    assert np.allclose(drive.wheel_speeds(0.0, 0.0, 1.0), [0.10, 0.20, 0.30])
    vx, vy, omega = drive.body_velocity(np.array([0.10, 0.20, 0.30]))
    assert abs(vx) < 1e-12 and abs(vy) < 1e-12 and abs(omega - 1.0) < 1e-12


def test_kiwi_drive_integrate_lateral():
    drive = KiwiDrive(base_radius=0.12)
    # Facing +y (yaw = pi/2), body-frame +x points along world +y.
    pose = drive.integrate(BasePose(0.0, 0.0, math.pi / 2), 0.3, 0.0, 0.0, 1.0)
    assert abs(pose.x) < 1e-12 and abs(pose.y - 0.3) < 1e-12
    # Pure lateral body velocity at yaw 0 moves the base sideways.
    pose = drive.integrate(BasePose(0.0, 0.0, 0.0), 0.0, 0.4, 0.0, 0.5)
    assert abs(pose.x) < 1e-12 and abs(pose.y - 0.2) < 1e-12


def test_correct_twist_inverts_measured_model():
    # A leaky drive model like LeKiwi's: forward leaks into yaw.
    a = np.array([[0.15, 0.04, 0.0], [0.05, 0.24, 0.0], [-2.29, -0.42, 0.87]])
    cmd = correct_twist(a, [0.1, 0.0, 0.0])
    assert np.allclose(a @ cmd, [0.1, 0.0, 0.0])
    assert np.allclose(correct_twist(None, [0.1, 0.2, 0.3]), [0.1, 0.2, 0.3])


def test_limit_wheel_speeds_scales_uniformly():
    out = limit_wheel_speeds(np.array([4.0, -2.0, 1.0]), np.array([2.0, 2.0, 2.0]))
    assert np.allclose(out, [2.0, -1.0, 0.5])
    assert np.allclose(limit_wheel_speeds(np.array([1.0, 1.0]), np.array([2.0, 2.0])), [1.0, 1.0])


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


def test_detect_mobile_base_kiwi(tmp_path):
    from backend.app.robots.registry import _detect_mobile_base
    from backend.app.robots.urdf import URDFModel

    wheel = ('<link name="{name}_link"><collision><geometry>'
             '<cylinder radius="0.05" length="0.03"/></geometry></collision></link>')
    joint = ('<joint name="base_{name}_wheel" type="continuous">'
             '<parent link="base_link"/><child link="{name}_link"/>'
             '<origin xyz="{xyz}" rpy="0 0 0"/><axis xyz="0 1 0"/>'
             '<limit velocity="6" effort="1"/></joint>')
    urdf = ('<robot name="tiny_kiwi">'
            '<link name="base_link"/>'
            + wheel.format(name="left") + wheel.format(name="back") + wheel.format(name="right")
            + joint.format(name="left", xyz="0.10 0.12 0")
            + joint.format(name="back", xyz="-0.14 0 0")
            + joint.format(name="right", xyz="0.10 -0.12 0")
            + '</robot>')
    path = tmp_path / "tiny.urdf"
    path.write_text(urdf)
    spec = _detect_mobile_base(URDFModel(path), {})
    assert spec["drive"] == "kiwi"
    assert spec["wheel_joints"] == ["base_left_wheel", "base_back_wheel", "base_right_wheel"]
    assert abs(spec["wheel_radius"] - 0.05) < 1e-9
    assert abs(spec["base_radius"] - 0.14) < 0.02
    assert len(spec["wheel_angles"]) == 3
    # Per-wheel lever arms come from the wheel bodies' own positions.
    assert len(spec["wheel_levers"]) == 3
    assert abs(spec["wheel_levers"][1] - 0.14) < 1e-9
    # Axles point along +y in this fixture: the left wheel's axle points
    # radially outward (sign -1), the other two do not.
    assert spec["wheel_signs"] == [-1, 1, 1]


class _FakeNavEnv:
    """Duck-typed backend: a diff-drive base integrated exactly, no engine."""

    timestep = 0.002

    def __init__(self, blocked=False):
        self._drive = DiffDrive(wheel_radius=0.05, track_width=0.4)
        self._pose = BasePose(0.0, 0.0, 0.0)
        self._command = (0.0, 0.0)
        self._blocked = blocked

    def base_pose(self):
        return np.array([self._pose.x, self._pose.y, self._pose.yaw])

    def command_base_velocity(self, vx, vy, omega):
        self._command = (vx, omega)

    def step(self, steps):
        for _ in range(steps):
            self._pose = self._drive.integrate(self._pose, *self._command, self.timestep)

    def frame(self):
        from types import SimpleNamespace
        return SimpleNamespace(objects={"crate": [0.0, 0.0, 0.0]} if self._blocked else {})

    def object_bounds(self, name):
        return np.array([[0.9, -1.0, 0.0], [1.1, 1.0, 0.3]])


def test_navigate_to_skill_arrives():
    from backend.app.skills.robot_skills import SkillRunner
    runner = SkillRunner(_FakeNavEnv(), None)
    result = runner.navigate_to((0.5, 0.3), timeout=30.)
    assert result["success"] is True
    assert result["final_distance_m"] <= 0.05


def test_navigate_to_skill_blocked():
    from backend.app.skills.robot_skills import SkillRunner
    runner = SkillRunner(_FakeNavEnv(blocked=True), None)
    result = runner.navigate_to((2.0, 0.0), timeout=30.)
    assert result["success"] is False
    assert result["error"] == "NAVIGATION_BLOCKED"


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
