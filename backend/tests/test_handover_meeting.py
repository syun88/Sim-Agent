"""Where the hands meet is measured from both arms, not read from a fixed point in the cell."""
from types import SimpleNamespace

import numpy as np
import pytest

from backend.app.simulation.openarm_v2.definition import FINGERTIP_MIDPOINT, GRASP_ROTATION
from backend.app.simulation.openarm_v2.handover import measured_meeting_point

CUBE = np.array([0.45, 0.10, 1.10])


def _env():
    return SimpleNamespace(
        object_pose=lambda name: (CUBE.copy(), np.eye(3)),
        ee_pose=lambda arm: (np.array([0.45, 0.10, 1.12]), np.eye(3)))


def _controller(preferred_y, *, reaches=(-.10, .10)):
    """IK whose error is smallest at the meeting offset both arms prefer; outside `reaches` it fails."""
    asked = []

    def solve(arm, target, rotation, **kwargs):
        asked.append((arm, np.asarray(target, float).copy()))
        # The pinch target encodes the meeting point through the fingertip offset.
        pinch = np.asarray(target, float) + np.asarray(rotation, float) @ FINGERTIP_MIDPOINT[arm]
        offset = float(pinch[1] - CUBE[1])
        if not (reaches[0] <= offset <= reaches[1]):
            return None, None, 0.
        return np.zeros(7), 0.001 + abs(offset - preferred_y) * .02, 0.
    return SimpleNamespace(solve_ik=solve), asked


def test_the_meeting_offset_is_the_one_both_arms_can_hold():
    controller, asked = _controller(preferred_y=.06)
    answer = measured_meeting_point(_env(), controller, 'left', 'right', 'blue_cube', .015)
    assert answer['measured'] is True
    assert answer['offset_y_m'] == pytest.approx(.06, abs=.031)
    assert answer['point'][1] == pytest.approx(CUBE[1] + .06, abs=.031)
    # Both arms were asked about every candidate: the choice is a measurement of the pair.
    assert {arm for arm, _ in asked} == {'left', 'right'}


def test_a_meeting_point_only_one_arm_can_hold_is_not_chosen():
    """The donor is comfortable off-centre; the receiver is not. The pair decides, not the donor."""
    def solve(arm, target, rotation, **kwargs):
        pinch = np.asarray(target, float) + np.asarray(rotation, float) @ FINGERTIP_MIDPOINT[arm]
        offset = float(pinch[1] - CUBE[1])
        if arm == 'left':
            return np.zeros(7), .001, 0.               # indifferent along y
        return np.zeros(7), .001 + abs(offset) * .5, 0.  # the receiver wants y = 0
    answer = measured_meeting_point(_env(), SimpleNamespace(solve_ik=solve), 'left', 'right', 'blue_cube', .015)
    assert answer['measured'] is True
    assert answer['offset_y_m'] == pytest.approx(0., abs=.031)


def test_an_unmeasurable_body_reports_the_grasp_point_and_says_it_was_not_measured():
    def solve(arm, target, rotation, **kwargs):
        return None, None, 0.
    answer = measured_meeting_point(_env(), SimpleNamespace(solve_ik=solve), 'left', 'right', 'blue_cube', .015)
    assert answer['measured'] is False and answer['reason'] == 'UNMEASURED'
    assert answer['point'] == pytest.approx(CUBE.tolist())
