"""Tests for the engine-neutral navigation core (draft for Sim-Agent).

Run with: uv run pytest -q tests/test_navigation.py
The tests simulate a base offline with DiffDrive.integrate; the real base
pose always comes from the physics engine.
"""

import math

import numpy as np
import pytest

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


def _agent_state(navigation=True):
    return {
        "objects": [{"id": "red_cube", "position": [0.6, 0.0, 0.014]}],
        "targets": [],
        "robot": {"available_arms": ["left"], "base_pose": [0.0, 0.0, 0.0],
                  "left_arm": {"held_object": None}, "right_arm": {"held_object": None}},
        "capabilities": {"skills": ["navigate_to", "approach_object", "grasp_object"],
                         "navigation": navigation},
        "discourse": {},
    }


def test_compile_navigate_action_to_standoff_step():
    from backend.app.planning.task_program import TaskAction, TaskProgram, compile_program

    plan = compile_program(TaskProgram(actions=[TaskAction(op="navigate", object="red_cube")]),
                           _agent_state())
    assert len(plan.steps) == 1
    step = plan.steps[0]
    assert step.skill == "navigate_to"
    # Standoff 0.32 m from the cube, on the base's side of it, facing it.
    assert np.allclose(step.target_xy, [0.28, 0.0])
    assert abs(step.target_yaw) < 1e-9


def test_compile_navigate_requires_navigation_capability():
    from backend.app.planning.task_program import GroundingError, TaskAction, TaskProgram, compile_program

    with pytest.raises(GroundingError):
        compile_program(TaskProgram(actions=[TaskAction(op="navigate", object="red_cube")]),
                        _agent_state(navigation=False))


def test_execute_plan_dispatches_navigate_to():
    from backend.app.skills.robot_skills import SkillRunner

    env = _FakeNavEnv()
    runner = SkillRunner(env, None)
    results = runner.execute_plan([{"skill": "navigate_to", "arm": "left",
                                    "target_xy": [0.3, 0.0], "target_yaw": None}])
    assert results[0]["success"]


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
             '<origin xyz="{xyz}" rpy="0 0 0"/><axis xyz="{axis}"/>'
             '<limit velocity="6" effort="1"/></joint>')
    urdf = ('<robot name="tiny_kiwi">'
            '<link name="base_link"/>'
            + wheel.format(name="left") + wheel.format(name="back") + wheel.format(name="right")
            + joint.format(name="left", xyz="0 0.14 0", axis="0 1 0")
            + joint.format(name="back", xyz="-0.14 0 0", axis="-1 0 0")
            + joint.format(name="right", xyz="0 -0.14 0", axis="0 -1 0")
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
    # Every axle points radially outward in this fixture, so every sign is
    # -1: the no-slip sign t . (a x z) with an outward axle is always -1.
    assert spec["wheel_signs"] == [-1, -1, -1]


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


# -- base placement search (reverse-computed standoff) ----------------------

def _placement_key(placement):
    return (placement.distance_m, placement.bearing_offset_deg, placement.yaw_offset_deg)


def test_base_placement_default_candidate_matches_legacy_standoff():
    from backend.app.skills.base_placement import candidate_base_placements

    candidates = candidate_base_placements((0.6, 0.0), (0.0, 0.0))
    first = candidates[0]
    # The legacy navigate standoff: 0.32 m from the cube, on the base's side
    # of it, facing it - the search must prefer exactly this placement.
    assert _placement_key(first) == (0.32, 0.0, 0.0)
    assert np.allclose(first.target_xy, [0.28, 0.0])
    assert abs(first.target_yaw) < 1e-9
    assert len(candidates) == 6 * 8 * 7
    assert len({_placement_key(c) for c in candidates}) == len(candidates)


def test_base_placement_bearing_and_yaw_offsets():
    from backend.app.skills.base_placement import candidate_base_placements

    candidates = candidate_base_placements((0.6, 0.0), (0.0, 0.0))
    by_key = {_placement_key(c): c for c in candidates}
    # A +90 deg bearing offset places the base perpendicular to the default
    # side (default bearing from the target toward the reference is +x here
    # seen from the target: the reference sits at angle pi), still facing it.
    side = by_key[(0.32, 90.0, 0.0)]
    assert np.allclose(side.target_xy, [0.6, -0.32])
    assert abs(side.target_yaw - math.pi / 2) < 1e-9
    # A yaw offset leaves the position unchanged and rotates the final yaw.
    turned = by_key[(0.32, 0.0, -40.0)]
    assert np.allclose(turned.target_xy, [0.28, 0.0])
    assert abs(turned.target_yaw - math.radians(-40.0)) < 1e-9


def test_base_placement_preference_expands_from_default():
    from backend.app.skills.base_placement import candidate_base_placements

    candidates = candidate_base_placements((0.6, 0.0), (0.0, 0.0))
    order = {_placement_key(c): i for i, c in enumerate(candidates)}
    # One step in any single dimension comes before two steps in one
    # dimension or a 180 deg flip.
    assert order[(0.32, 0.0, -20.0)] < order[(0.32, 0.0, -40.0)]
    assert order[(0.32, 0.0, -20.0)] < order[(0.32, 180.0, 0.0)]
    assert order[(0.28, 0.0, 0.0)] < order[(0.32, 180.0, 0.0)]
    assert order[(0.32, -30.0, 0.0)] < order[(0.32, -60.0, 0.0)]


def test_search_base_placement_returns_first_accepted_and_stops():
    from backend.app.skills.base_placement import candidate_base_placements, search_base_placement

    expected = candidate_base_placements((0.6, 0.0), (0.0, 0.0))
    calls = []

    def evaluate(placement):
        calls.append(placement)
        return {"arm_q": [0.0]} if len(calls) == 3 else None

    placement, result = search_base_placement((0.6, 0.0), (0.0, 0.0), evaluate)
    assert placement == expected[2]
    assert result == {"arm_q": [0.0]}
    assert calls == expected[:3]


def test_search_base_placement_none_when_all_rejected():
    from backend.app.skills.base_placement import candidate_base_placements, search_base_placement

    calls = []
    outcome = search_base_placement((0.6, 0.0), (0.0, 0.0),
                                    lambda placement: calls.append(placement) or None)
    assert outcome is None
    assert len(calls) == len(candidate_base_placements((0.6, 0.0), (0.0, 0.0)))


def _stub_imported_skills(mobile, planned, retry_result=None):
    from types import SimpleNamespace

    from backend.app.robots.executor import ImportedRobotSkills

    skills = ImportedRobotSkills.__new__(ImportedRobotSkills)
    skills.env = SimpleNamespace(profile={"mobile_base": {"drive": "kiwi"} if mobile else None})
    skills._approach_planned = planned
    calls = {"retry": 0}

    def retry(*args):
        calls["retry"] += 1
        skills._placement_search_evidence = {"evaluated_candidates": 5}
        return retry_result

    skills._retry_approach_from_searched_placements = retry
    return skills, calls


def test_approach_wrapper_passes_through_success():
    from backend.app.skills.robot_skills import SkillError  # noqa: F401

    skills, calls = _stub_imported_skills(True, lambda *a: {"target": [0.0]})
    assert skills._approach("left", "red_cube") == {"target": [0.0]}
    assert calls["retry"] == 0


def test_approach_wrapper_relocates_only_mobile_collision_path():
    from backend.app.skills.robot_skills import SkillError

    def collide(*a):
        raise SkillError("COLLISION_PATH", "no corridor")

    # Desk robot: the original failure propagates, no search is attempted.
    skills, calls = _stub_imported_skills(False, collide)
    with pytest.raises(SkillError):
        skills._approach("left", "red_cube")
    assert calls["retry"] == 0

    # Mobile base, searched retry succeeds: its result is the approach result.
    skills, calls = _stub_imported_skills(True, collide, retry_result={"base_placement": {"distance_m": 0.32}})
    assert skills._approach("left", "red_cube") == {"base_placement": {"distance_m": 0.32}}
    assert calls["retry"] == 1

    # Mobile base, search finds nothing: the original error is re-raised
    # with the search evidence attached.
    skills, calls = _stub_imported_skills(True, collide, retry_result=None)
    with pytest.raises(SkillError) as caught:
        skills._approach("left", "red_cube")
    assert caught.value.category == "COLLISION_PATH"
    assert caught.value.evidence["base_placement_search"] == {"evaluated_candidates": 5}


def test_approach_wrapper_ignores_other_errors():
    from backend.app.skills.robot_skills import SkillError

    def fail(*a):
        raise SkillError("IK_FAILURE", "unreachable")

    skills, calls = _stub_imported_skills(True, fail)
    with pytest.raises(SkillError):
        skills._approach("left", "red_cube")
    assert calls["retry"] == 0


def test_controller_arrive_tolerance_is_configurable():
    # The grasp-placement retry evaluates a base pose exactly and drives to
    # it with tight tolerances: 45 mm short of the target counts as arrived
    # at the default 50 mm tolerance but must keep driving at 10 mm
    # (measured on LeKiwi: the base "relocated" without translating).
    pose = BasePose(0.0, 0.0, 0.0)
    target = np.array([0.045, 0.0])
    assert WaypointController().update(pose, target).status == "arrived"
    tight = WaypointController(arrive_tolerance=0.01, yaw_tolerance=0.02)
    assert tight.update(pose, target).status == "running"
    assert tight.update(pose, np.array([0.005, 0.0])).status == "arrived"


# -- base hold during arm motion ---------------------------------------------


def _hold_env(drive="kiwi"):
    from backend.app.simulation.genesis_backend.generic_environment import (
        GenericURDFEnvironment,
    )

    env = GenericURDFEnvironment.__new__(GenericURDFEnvironment)
    env._mobile_base = {"drive": drive}
    env._base_hold_pose = None
    env._base_hold_suspended = False
    env._base_hold_depth = 0
    env._pose = np.array([0.1, 0.2, 0.3])
    env.commands = []
    env.base_pose = lambda: env._pose.copy()
    env.base_speed_limits = lambda: (0.3, 2.0)
    env.command_base_velocity = lambda vx, vy, omega: env.commands.append(
        (float(vx), float(vy), float(omega)))
    return env


def test_base_hold_engage_release_nesting():
    env = _hold_env()
    env.engage_base_hold()
    assert env.base_hold_active
    assert np.allclose(env._base_hold_pose, [0.1, 0.2, 0.3])
    env._pose = np.array([0.4, 0.5, 0.6])
    env.engage_base_hold()  # nested: the first captured pose is kept
    assert np.allclose(env._base_hold_pose, [0.1, 0.2, 0.3])
    env.release_base_hold()
    assert env.base_hold_active  # still held by the outer engagement
    env.release_base_hold()
    assert env._base_hold_pose is None
    assert env.commands[-1] == (0.0, 0.0, 0.0)  # wheels stopped on release


def test_base_hold_servo_corrects_pose_error():
    env = _hold_env()
    env._base_hold_pose = np.array([0.0, 0.0, 0.0])
    env._pose = np.array([0.05, 0.0, 0.1])  # drifted +x and +yaw
    env._command_base_hold()
    vx, vy, omega = env.commands[-1]
    # The command opposes the measured error on every axis.
    assert vx < 0 and omega < 0
    assert abs(vx - 2.5 * -0.0498) < 0.01
    assert abs(omega - 3.5 * -0.1) < 1e-9
    # Within the deadband the hold commands a stop instead of jittering.
    env._pose = np.array([0.001, 0.0, 0.001])
    env._command_base_hold()
    assert env.commands[-1] == (0.0, 0.0, 0.0)


def test_base_hold_diff_drive_never_commands_strafe():
    env = _hold_env(drive="diff")
    env._base_hold_pose = np.array([0.0, 0.0, 0.0])
    env._pose = np.array([0.0, 0.08, 0.0])  # pure lateral drift
    env._command_base_hold()
    _, vy, _ = env.commands[-1]
    assert vy == 0.0


def test_base_hold_suspend_and_resume_reanchors():
    env = _hold_env()
    env.engage_base_hold()
    env.suspend_base_hold()
    assert not env.base_hold_active
    env._pose = np.array([0.9, 0.9, 1.0])  # deliberately driven elsewhere
    env.resume_base_hold()
    assert env.base_hold_active
    assert np.allclose(env._base_hold_pose, [0.9, 0.9, 1.0])


def test_navigate_suspends_and_resumes_the_hold():
    from backend.app.skills.robot_skills import SkillRunner

    class HoldNavEnv(_FakeNavEnv):
        def __init__(self):
            super().__init__()
            self.events = []

        def suspend_base_hold(self):
            self.events.append("suspend")

        def resume_base_hold(self):
            self.events.append("resume")

        def command_base_velocity(self, vx, vy, omega):
            self.events.append("cmd")
            super().command_base_velocity(vx, vy, omega)

    env = HoldNavEnv()
    result = SkillRunner(env, None).navigate_to((0.5, 0.3), timeout=30.)
    assert result["success"] is True
    assert env.events[0] == "suspend"
    assert env.events[-1] == "resume"
    assert "cmd" in env.events


# -- gravity-ordered placement drives -----------------------------------------


def _retry_skills(seed_outcomes, approach_outcomes):
    """ImportedRobotSkills stubbed at the planning/driving seams.

    seed_outcomes: gravity ratios (or None) handed out in candidate order.
    approach_outcomes: booleans per drive (True = the retried approach
    succeeds there).
    """
    from types import SimpleNamespace

    from backend.app.robots.executor import ImportedRobotSkills

    skills = ImportedRobotSkills.__new__(ImportedRobotSkills)
    skills.env = SimpleNamespace(
        profile={"mobile_base": {"drive": "kiwi"},
                 "root_transform": np.eye(4).tolist()},
        object_pose=lambda obj: (np.array([0.55, 0.30, 0.014]), None),
        base_pose=lambda: np.array([0.0, 0.0, 0.0]),
        joint_snapshot=lambda: {},
    )
    calls = {"seeds": [], "drives": [], "approaches": 0}
    outcomes = list(seed_outcomes)

    def seed_at(arm, target, snapshot, placement, original_root):
        ratio = outcomes.pop(0) if outcomes else None
        calls["seeds"].append((placement, ratio))
        if ratio is None:
            return None
        return {"arm_q": [0.0] * 5, "closure_planned": True,
                "gravity_worst_ratio": ratio}

    def navigate(target_xy, target_yaw, **kwargs):
        calls["drives"].append(tuple(target_xy))
        return {"success": True}

    def planned(arm, obj, *args):
        from backend.app.skills.robot_skills import SkillError

        index = calls["approaches"]
        calls["approaches"] += 1
        if approach_outcomes[index]:
            return {"target": [0.55, 0.30, 0.014]}
        raise SkillError("COLLISION_PATH", "still blocked")

    skills._grasp_seed_at_placement = seed_at
    skills.navigate_to = navigate
    skills._approach_planned = planned
    return skills, calls


def _placement_for_ratio(calls, ratio):
    return next(p for p, r in calls["seeds"] if r == ratio)


def test_retry_drives_to_in_margin_placement_first():
    # Candidate order finds an over-margin seed (3.0) before an in-margin
    # one (0.5); the drive must go to the in-margin placement, and the
    # over-margin one must not be driven to at all once it succeeds.
    skills, calls = _retry_skills([3.0, 0.5], [True])
    result = skills._retry_approach_from_searched_placements("left", "red_cube", None, None, 0.)
    assert result is not None
    assert result["base_placement"]["gravity_worst_ratio"] == 0.5
    assert result["base_placement"]["gravity_over_limit"] is False
    good = _placement_for_ratio(calls, 0.5)
    assert calls["drives"] == [tuple(good.target_xy)]
    # The evidence keeps candidate (preference) order even though the
    # drive order differed.
    evidence = skills._placement_search_evidence
    assert [row["gravity_worst_ratio"] for row in evidence["clear_seed_placements"]] == [3.0, 0.5]


def test_retry_orders_over_margin_drives_by_margin():
    # Every seed is over the margin: drives go best-margin first, the
    # excess is surfaced on the result, and a failed approach falls
    # through to the next placement.
    skills, calls = _retry_skills([3.0, 2.0], [False, True])
    result = skills._retry_approach_from_searched_placements("left", "red_cube", None, None, 0.)
    assert result is not None
    assert result["base_placement"]["gravity_worst_ratio"] == 3.0
    assert result["base_placement"]["gravity_over_limit"] is True
    best = _placement_for_ratio(calls, 2.0)
    worst = _placement_for_ratio(calls, 3.0)
    assert calls["drives"] == [tuple(best.target_xy), tuple(worst.target_xy)]


def test_retry_bounded_evaluation_when_nothing_is_clear():
    from backend.app.robots.executor import _PLACEMENT_EVALUATION_BUDGET

    skills, calls = _retry_skills([], [])
    assert skills._retry_approach_from_searched_placements(
        "left", "red_cube", None, None, 0.) is None
    evidence = skills._placement_search_evidence
    assert evidence["evaluated_candidates"] == _PLACEMENT_EVALUATION_BUDGET
    assert evidence["drives"] == []
