"""Guards for the physical-relation substrate.

The point of the substrate is that a number has one owner and can be checked: the planner names a
relation, the skill evaluates it, and the engine still measures the outcome. These tests pin the
relations' own arithmetic and the failure modes that would make the substrate useless - an unknown name
returning zero, or a missing argument being filled with a default.
"""

from __future__ import annotations

import pytest

from backend.app.physics import (GRAVITY, closing_reserve, contact_time_for_speed, describe, evaluate,
                                 friction_speed_bound, pour_time_for_volume, ramp_for_release_speed,
                                 release_speed_for_rise, rise_for_release_speed)


def test_release_speed_and_rise_are_inverses():
    """3 cm of free rise needs 0.767 m/s; that speed gives 3 cm back."""
    speed = release_speed_for_rise(.03)
    assert speed == pytest.approx((2 * GRAVITY * .03) ** .5)
    assert speed == pytest.approx(.767, abs=.002)
    assert rise_for_release_speed(speed) == pytest.approx(.03)


def test_the_ramp_relation_reproduces_the_measured_design():
    """20 cm stroke, 1.04 m/s at phase 0.9 -> the 0.346 s ramp measured on the robot."""
    ramp = ramp_for_release_speed(.20, 1.042, .9)
    assert ramp == pytest.approx(.345, abs=.005)
    assert 2. * .9 * .20 / ramp == pytest.approx(1.042, abs=.01)


def test_friction_is_the_reason_a_short_stroke_cannot_throw():
    """At mu*g ~ 9.8 m/s^2, 0.77 m/s needs ~0.079 s of contact - longer than a 0.06 s stroke."""
    assert contact_time_for_speed(.767, 1.) == pytest.approx(.0782, abs=.001)
    assert friction_speed_bound(1., .06) == pytest.approx(.5886, abs=.001)
    assert friction_speed_bound(1., .06) < .767, "a 0.06 s contact cannot deliver a 3 cm throw"


def test_pour_volume_and_time_are_inverses():
    assert pour_time_for_volume(20., 4.) == pytest.approx(5.)
    assert closing_reserve(4., 5., 100.) == pytest.approx(20.)


def test_the_closing_reserve_is_capped_by_the_caller():
    assert closing_reserve(1000., 1., 8.) == pytest.approx(8.)


def test_an_unknown_relation_is_an_error_not_a_zero():
    with pytest.raises(KeyError):
        evaluate("looks_about_right", rise_m=.03)


def test_a_missing_argument_is_an_error_not_a_default():
    with pytest.raises(KeyError):
        evaluate("release_speed_for_rise")


def test_every_relation_reports_its_units_and_what_it_assumes():
    described = {item["name"]: item for item in describe()}
    assert described, "the substrate must be discoverable"
    for item in described.values():
        assert item["unit"], f"{item['name']} has no unit"
        assert item["assumes"], f"{item['name']} does not say what it assumes"
        assert item["args"], f"{item['name']} has no declared arguments"


def test_evaluate_returns_the_relation_that_produced_the_number():
    result = evaluate("release_speed_for_rise", rise_m=.03)
    assert result["relation"] == "release_speed_for_rise"
    assert result["unit"] == "m/s"
    assert result["inputs"] == {"rise_m": .03}


def test_an_aimed_launch_points_at_the_target():
    """10 cm away, 0.30 s of flight: the velocity must land on the target, not merely go up."""
    from backend.app.physics import launch_for_target
    aim = launch_for_target((.10, 0., 0.), .30)
    vz, vx = aim["velocity"][2], aim["velocity"][0]
    # Reconstruct the flight and check it passes through the requested point.
    travel = [vx * t for t in (0., .15, .30)]
    height = [vz * t - .5 * 9.81 * t * t for t in (0., .15, .30)]
    assert travel[-1] == pytest.approx(.10, abs=1e-6)
    assert height[-1] == pytest.approx(0., abs=1e-6)
    assert aim["direction"][2] > .9, "a 10 cm throw at 0.3 s is mostly upward"


def test_closest_approach_measures_the_trajectory_not_the_plan():
    from backend.app.skills.dynamic_skills import closest_approach
    trace = [{"position": [0., 0., 0.]}, {"position": [.05, 0., .02]}, {"position": [.10, 0., 0.]}]
    assert closest_approach(trace, [.04, 0., 0.]) == pytest.approx(.0005 ** .5, abs=1e-6)
    assert closest_approach(trace, [.05, 0., .021]) == pytest.approx(.001, abs=.001)
    assert closest_approach([], [0., 0., 0.]) is None


def test_a_flat_aimed_throw_is_recognised_as_released():
    """Measured 2026-10-27: a hit 57 mm from centre was read as "no release" because the arc was flat.

    With upward_min=0 an aimed throw counts contact loss with any upward speed; without it, the 0.1 m/s rule
    still applies to a pure upward throw.
    """
    from backend.app.skills.dynamic_skills import evaluate_throw

    flat = [
        {"t": 0.30, "position": [0., 0., 1.0], "velocity": [1.0, 0., 0.04], "contacts": ["g|obj"]},
        {"t": 0.31, "position": [0.01, 0., 1.0001], "velocity": [1.0, 0., 0.02], "contacts": []},
        {"t": 0.32, "position": [0.02, 0., 0.9999], "velocity": [1.0, 0., 0.0], "contacts": []},
    ]
    aimed = evaluate_throw(flat, 0., release_after=0.30, upward_min=0.)
    upward = evaluate_throw(flat, 0., release_after=0.30)
    assert aimed.get("release_velocity") is not None, "a flat arc still has a release"
    assert upward.get("release_velocity") is None, "the upward rule must still reject a flat arc"
    assert upward["failure_category"] == "NO_UPWARD_FREE_FLIGHT"


def test_an_aimed_throw_counts_only_a_measured_hit():
    from backend.app.skills.dynamic_skills import HIT_TOLERANCE_M, closest_approach

    trace = [{"t": 0.30, "position": [0., 0., 1.0], "velocity": [1.0, 0., 0.04], "contacts": ["g|obj"]},
             {"t": 0.31, "position": [0.05, 0., 1.0], "velocity": [1.0, 0., 0.], "contacts": []},
             {"t": 0.32, "position": [0.12, 0., 0.99], "velocity": [1.0, 0., 0.], "contacts": []}]
    target = [0.06, 0., 1.0]
    assert closest_approach(trace, target) <= HIT_TOLERANCE_M     # this one lands
    assert closest_approach(trace, [0.30, 0., 1.0]) > HIT_TOLERANCE_M   # this one does not
