"""Guards for the multi-seed IK search on arms with fewer joints than the task needs.

The controller is the single owner every skill reaches the arm through, so this is where a local solve
must stop being the only attempt. The case that motivated it: on the four-joint OpenManipulator-X a
motion to an arm's-length point failed with a 2 cm residual that the planner's IK solved exactly.

The synthetic arm below has the property that makes a single solve fail and nothing to do with Genesis:
its reachable set is cos(q0), cos(q1), so at the zero posture the Jacobian is exactly flat and the damped
solve cannot start. Only a different seed can find the pose, which is precisely what the sweep adds.
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.app.control.motion import OpenArmV2Controller


class _FlatKinematics:
    """Two links whose positions are cos(q): gradient is zero at q = 0, so a local solve stalls."""

    def __init__(self, joints=2, start=0.):
        self.joints = joints
        self.start = float(start)

    def evaluate(self, q):
        position = np.array([np.cos(q[0]), np.cos(q[1]), 0.])
        linear = np.zeros((3, self.joints))
        linear[0, 0] = -np.sin(q[0])
        linear[1, 1] = -np.sin(q[1])
        return position, np.eye(3), linear, np.zeros((3, self.joints))


class _FlatArm:
    """An env exposing only what solve_ik touches, with a two-joint arm."""

    ik_position_tolerance = .002
    ik_rotation_tolerance = .015

    def __init__(self, joints=2, start=0.):
        self.joints = joints
        self.start = float(start)
        self._kinematics = _FlatKinematics(joints)
        self.calls = 0

    def arm_limits(self, side):
        return np.array([[0., np.pi]] * self.joints)

    def arm_qpos(self, side):
        return np.full(self.joints, self.start)

    def home_arm_qpos(self, side):
        return np.full(self.joints, self.start)

    def arm_kinematics(self, side):
        self.calls += 1
        return self._kinematics

    def ee_pose(self, side):
        return np.zeros(3), np.eye(3)

    def configure_motion_response(self, speed):
        return None


def _controller(env):
    return OpenArmV2Controller(env, configure_motors=False)


def test_a_stalled_local_solve_finds_the_pose_from_another_seed():
    env = _FlatArm()
    target = np.array([np.cos(2.5), np.cos(2.5), 0.])
    solution, error, _ = _controller(env).solve_ik("left", target)
    assert solution is not None, "the sweep must find a pose the first seed cannot"
    reached, _, _, _ = _FlatKinematics().evaluate(solution)
    assert np.linalg.norm(reached - target) <= env.ik_position_tolerance
    assert min(abs(solution[0] - 2.5), abs(solution[0] + 2.5 - 2 * np.pi)) < .05


def test_the_sweep_is_deterministic():
    target = np.array([np.cos(2.5), np.cos(2.5), 0.])
    first = _controller(_FlatArm()).solve_ik("left", target)[0]
    second = _controller(_FlatArm()).solve_ik("left", target)[0]
    assert np.allclose(first, second), "a failure must be reproducible"


def test_the_first_seed_is_used_when_it_works(monkeypatch):
    # Started away from q = 0, where this arm's Jacobian is flat: the first solve can converge.
    env = _FlatArm(start=1.0)
    controller = _controller(env)
    solved = {"count": 0}
    original = controller._solve_from_seed

    def counting(*args, **kwargs):
        solved["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(controller, "_solve_from_seed", counting)
    controller.solve_ik("left", np.array([np.cos(.3), np.cos(.3), 0.]))
    assert solved["count"] == 1, "a target next to the current posture must not pay for a sweep"


def test_an_arm_with_six_joints_is_unchanged():
    """Six joints have one attempt plus the documented home retry, and no sweep."""
    env = _FlatArm(joints=6)
    controller = _controller(env)
    seen = []
    original = controller._solve_from_seed

    def counting(*args, **kwargs):
        seen.append(args[4].copy())
        return original(*args, **kwargs)

    controller._solve_from_seed = counting
    target = np.array([np.cos(2.5), np.cos(2.5), 0.])
    controller.solve_ik("left", target)
    assert len(seen) == 2, "six-joint arms keep exactly the behaviour they had"
    assert controller._ik_seeds("left", env.arm_limits("left"))[0].tolist() == [0.] * 6
