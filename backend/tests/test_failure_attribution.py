"""A failed skill must name its mechanism from the measurements, not from its error code."""
from backend.app.execution.failure_evidence import failure_observation
from backend.app.planning.task_contract import TaskContract, requested_program
from backend.app.planning.task_program import TaskProgram, compile_program
from backend.app.skills.robot_skills import SkillError
from backend.app.verification.failure_attribution import POSITION_TOLERANCE_M, attribute

WORLD = {'objects': [{'id': 'red_cube', 'type': 'cube', 'position': [.44, .15, 1.031]},
                     {'id': 'blue_cube', 'type': 'cube', 'position': [.50, -.22, 1.031]}],
         'targets': [], 'robot': {'left_arm': {'held_object': None}, 'right_arm': {'held_object': None}}}


def test_tracking_is_blamed_on_the_servo_not_the_plan():
    """A blocked motion whose residual exceeded its tolerance is servo lag, not an unreachable target."""
    mechanism, _ = attribute("MOTION_BLOCKED", {"position_error": 0.021, "rotation_error": 0.05})
    assert mechanism == "TRACKING"


def test_residuals_inside_the_reported_tolerances_are_not_called_tracking():
    """The classifier honours the tolerances the motion was judged with, when the record reports them."""
    inside = {"position_error": 0.02, "rotation_error": 0.02,
              "position_tolerance_m": 0.03, "rotation_tolerance_m": 0.15}
    mechanism, reason = attribute("MOTION_BLOCKED", inside)
    assert mechanism == "UNKNOWN" and "inside the tolerances" in reason
    over = {"position_error": 0.04, "rotation_error": 0.02,
            "position_tolerance_m": 0.03, "rotation_tolerance_m": 0.15}
    assert attribute("MOTION_BLOCKED", over)[0] == "TRACKING"


def test_grasp_without_contact_is_a_frame_fault():
    evidence = {"close_trace": [{"object_contacts": [], "aperture_center_error_m": 5.0e-05}]}
    assert attribute("GRASP_LOST", evidence)[0] == "GRASP_FRAME"


def test_grasp_with_contacts_is_not_a_frame_fault():
    """The guard: the frame did reach the object, so force or surface is the mechanism."""
    evidence = {"close_trace": [{"object_contacts": ["left_link_7|red_cube"],
                                "aperture_center_error_m": 8.0e-04}]}
    assert attribute("GRASP_LOST", evidence)[0] == "GRASP_CONTACT"


def test_self_collision_outranks_every_other_signal():
    evidence = {"rotation_error": 0.2,
                "collisions": [{"links": ["a", "b"], "penetration_m": 0.001}]}
    assert attribute("MOTION_BLOCKED", evidence)[0] == "GEOMETRY"


def test_planning_and_language_stay_distinct():
    assert attribute("IK_FAILURE", {})[0] == "PLANNING"
    assert attribute("UNSUPPORTED_INSTRUCTION", {})[0] == "LANGUAGE"


def test_missing_evidence_is_unknown_rather_than_guessed():
    assert attribute("GRASP_LOST", None)[0] == "UNKNOWN"
    assert attribute("APPROACH_COLLISION", {})[0] == "UNKNOWN"


def test_a_large_rotation_residual_carries_a_reachability_caveat():
    """Rotation far over tolerance is not proven servo lag; the reason must say so."""
    _, small = attribute("MOTION_BLOCKED", {"position_error": 0.0031, "rotation_error": 0.03})
    _, large = attribute("MOTION_BLOCKED", {"position_error": 0.0067, "rotation_error": 0.2709})
    assert "unreachable" not in small
    assert "unreachable" in large


def test_an_object_moved_during_the_approach_is_its_own_mechanism():
    mechanism, reason = attribute("APPROACH_COLLISION", {"displacement_m": 0.012, "tolerance_m": 0.005})
    assert mechanism == "APPROACH_CONTACT"
    assert "12.0 mm" in reason and "5.0 mm" in reason


def test_a_carried_object_shoving_another_is_a_carry_contact():
    """Measured 2026-11-07: carrying a cube past a resting one moved it 7.4 mm and the task stopped."""
    mechanism, reason = attribute("CARRY_COLLISION",
                                  {"object": "red_cube", "carrying": "blue_cube",
                                   "displacement_m": 0.0074, "tolerance_m": 0.005, "phase": "carry"})
    assert mechanism == "CARRY_CONTACT"
    assert "blue_cube" in reason and "7.4 mm" in reason


def test_a_reported_contact_names_the_pair_and_the_movement():
    mechanism, reason = attribute("APPROACH_CONTACT",
                                  {"contact_pairs": ["openarm_left_ee_outer_finger|blue_cube"],
                                   "displacement_m": 0.00021})
    assert mechanism == "APPROACH_CONTACT"
    assert "openarm_left_ee_outer_finger|blue_cube" in reason and "0.2 mm" in reason


def test_a_zero_displacement_is_not_reported_as_a_contact():
    """A record can carry the field with no movement; zero is not 'the object was moved'."""
    mechanism, _ = attribute("APPROACH_CONTACT", {"contact_pairs": [], "displacement_m": 0.0})
    assert mechanism == "UNKNOWN", "the error code alone must not produce a contact verdict"


def test_a_launch_that_never_reached_the_speed_is_tracking_not_a_mystery():
    mechanism, reason = attribute("THROW_SPEED_NOT_REACHED",
                                  {"peak_object_speed_m_s": 0.459, "needed_release_speed_m_s": 0.767})
    assert mechanism == "TRACKING"
    assert "0.459" in reason and "0.767" in reason


def test_a_launch_that_did_reach_the_speed_is_not_blamed_on_the_servo():
    mechanism, _ = attribute("THROW_SPEED_NOT_REACHED",
                             {"peak_object_speed_m_s": 0.9, "needed_release_speed_m_s": 0.767})
    assert mechanism != "TRACKING"


def test_the_failure_receipt_carries_the_mechanism_beside_the_code():
    """The receipt a caller receives must name the mechanism, not only repeat the code."""
    program = TaskProgram(actions=[{'op': 'pick', 'object': 'red_cube', 'arm': 'left'}])
    plan = compile_program(program, WORLD)
    contract = TaskContract.freeze('赤を左手で取って', plan, requested_program(plan))
    before = dict(WORLD, sim_time=1.0)
    result = {'success': False, 'status': 'FAILED',
              'skills': [{'skill': 'move_to_object', 'arm': 'left', 'object': 'red_cube', 'success': False,
                          'error': 'CARRY_COLLISION', 'action_index': 0, 'step_index': 0,
                          'evidence': {'object': 'blue_cube', 'carrying': 'red_cube',
                                       'displacement_m': 0.0074, 'tolerance_m': 0.005, 'phase': 'carry'},
                          'detail': 'Carrying red_cube disturbed resting blue_cube by 7.4 mm'}],
              'verification': {'success': False, 'failure_category': 'CARRY_COLLISION'}}
    receipt = failure_observation(contract, 0, 1, result, before, before)
    assert receipt['code'] == 'CARRY_COLLISION'
    assert receipt['mechanism'] == 'CARRY_CONTACT'
    assert receipt['mechanism_reason']
    assert receipt['classification'] == 'PATH_COLLISION', 'a contact made while moving is a path collision'
