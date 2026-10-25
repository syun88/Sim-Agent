"""Closed-loop pouring with the real held vessel and Genesis SPH particles."""
import time
import logging
import numpy as np
from scipy.spatial.transform import Rotation


def move_held_vessel(runner, arm, obj, target, origin, rotation, phase, duration=.40, *, transport=False, path=None):
    """Move the physically pinched vessel; also used to verify upright A-to-B transport."""
    from backend.app.skills.robot_skills import SkillError
    from backend.app.simulation.genesis_backend.wine import BOTTLE_MOUTH, vessel_clearance

    env, control = runner.env, runner.controller
    started = env.frame().sim_time
    runner._require_hold(arm, obj)
    p, r = env.object_pose(obj)
    ee, er = env.ee_pose(arm)
    relative_position = r.T @ (ee - p)
    relative_rotation = r.T @ er
    if transport:
        # A partially filled bottle can slip during a 0.1s, 18cm return even
        # when its upright outward carry succeeded. Bound physical acceleration
        # and peak speed rather than scaling every explicit transit by six.
        distance = float(np.linalg.norm(np.asarray(origin) - p))
        minimum_time = max(np.sqrt(5.774 * distance / 4.), 1.875 * distance / .65)
        duration = max(duration, minimum_time * control.motion_speed)
    waypoints = path or []
    minimum_clearance = float("inf")
    samples = []

    def observe():
        nonlocal minimum_clearance
        runner._require_hold(arm, obj)
        bp, br = env.object_pose(obj)
        gp, gr = env.object_pose(target)
        clearance = vessel_clearance(bp, br, gp, gr)
        minimum_clearance = min(minimum_clearance, clearance)
        samples.append({"sim_time": env._sim_step * env.timestep, "position": bp.tolist(),
                        "rotation": br.tolist()})
        if clearance < 0.:
            raise SkillError("VESSEL_COLLISION", f"Bottle/glass contact during {phase}")

    moved = control.move_to_site(arm, np.asarray(origin) + rotation @ relative_position,
        rotation @ relative_rotation, duration=duration, scale_speed=transport,
        via=[np.asarray(point) + attitude @ relative_position for point, attitude in waypoints] or None,
        via_rotations=[attitude @ relative_rotation for _, attitude in waypoints] or None,
        on_sample=observe if waypoints else None)
    if not moved.success:
        raise SkillError(moved.error or "POUR_MOTION_FAILED", f"Bottle motion blocked during {phase} "
                         f"(position error={moved.position_error}, rotation error={moved.rotation_error})")
    runner._require_hold(arm, obj)
    desired_mouth = np.asarray(origin) + rotation @ BOTTLE_MOUTH
    corrections = 0
    for _ in range(2):
        bp, br = env.object_pose(obj)
        mouth_error = desired_mouth - (bp + br @ BOTTLE_MOUTH)
        if np.linalg.norm(mouth_error) <= .003:
            break
        ep, er = env.ee_pose(arm)
        corrected = control.move_to_site(arm, ep + mouth_error, rotation @ br.T @ er,
                                         duration=.12, scale_speed=False)
        if not corrected.success:
            raise SkillError(corrected.error or "POUR_MOTION_FAILED", f"Mouth correction blocked during {phase}")
        corrections += 1
        runner._require_hold(arm, obj)
    bp, br = env.object_pose(obj)
    gp, gr = env.object_pose(target)
    finished = env.frame().sim_time
    measured = {"phase": phase, "sim_time": finished,
                "motion_sim_duration": finished - started,
                "mouth_corrections": corrections,
                "vessel_clearance_mm": vessel_clearance(bp, br, gp, gr) * 1000., **env.fluid_state()}
    if waypoints:
        measured.update(continuous_waypoints=len(waypoints) + 1, motion_samples=samples,
                        minimum_path_clearance_mm=minimum_clearance * 1000.)
    if measured["vessel_clearance_mm"] < 0.:
        raise SkillError("VESSEL_COLLISION", f"Measured bottle/glass interference during {phase}")
    return measured


def pour_wine(runner, arm: str, obj: str, target: str, volume_ml: float) -> dict:
    from backend.app.skills.robot_skills import SkillError
    context = {}
    try:
        return _pour_wine(runner, arm, obj, target, volume_ml, context)
    except SkillError as exc:
        exc.evidence.update(context)
        exc.evidence["final_liquid"] = runner.env.fluid_state()
        raise


def _pour_wine(runner, arm: str, obj: str, target: str, volume_ml: float, context: dict) -> dict:
    from backend.app.skills.robot_skills import SkillError
    from backend.app.simulation.genesis_backend.wine import BOTTLE_MOUTH, BOTTLE_START, MAX_SPILL_ML, vessel_clearance

    env, control = runner.env, runner.controller
    if obj != "wine_bottle" or target != "wine_glass" or env.scene_profile != "wine":
        raise SkillError("UNSUPPORTED_POUR", "Pour requires the wine bottle, glass, and wine scene")
    runner._require_hold(arm, obj)
    initial = env.fluid_state()
    if initial["spilled_ml"] > MAX_SPILL_ML:
        raise SkillError("SCENE_HAS_SPILL", "The scene already exceeds the spill limit; reset it before another pour")
    if initial["bottle_ml"] < volume_ml + 2.:
        raise SkillError("INSUFFICIENT_LIQUID", "The bottle does not contain the requested amount plus flow margin")
    initial_grip = env.object_contacts(obj, arm)
    original_rotation = env.object_pose(obj)[1]
    trace = [{"phase": "held", "sim_time": env.frame().sim_time, **initial}]
    context.update(initial_liquid=initial, fluid_trace=trace)
    glass_rim = env.zone_position(target)
    direction = 1. if arm == "left" else -1.

    def move_bottle(origin, rotation, phase, duration=.40, *, transport=False, path=None):
        context['active_phase'] = phase
        trace.append(move_held_vessel(runner, arm, obj, target, origin, rotation, phase, duration, transport=transport, path=path))
        logging.getLogger(__name__).info('%s: glass=%.4fmL spill=%.4fmL sim=%.3fs', phase,
            trace[-1]['glass_ml'], trace[-1]['spilled_ml'], trace[-1]['sim_time'])

    def angle_pose(angle, *, closing=False, side_clearance=0.):
        radians = np.deg2rad(angle)
        rotation = Rotation.from_rotvec(np.array([direction * radians, 0., 0.])).as_matrix() @ original_rotation
        # Approach the glass from the holder's side. The mouth arrives above it
        # once the bottle is tilted, keeping the upright bottle outside the bowl.
        # The stream retains horizontal velocity after the lip. Aim slightly
        # upstream so its ballistic path enters the bowl instead of its far side.
        offset = (.015 if closing and angle < 60. else .005) + side_clearance
        offset += 0. if closing else .18 * max(0., 1. - angle / 80.)
        lip_height = .020 if closing else .018
        mouth = glass_rim + [0., direction * offset, lip_height]
        mouth[2] = max(mouth[2], 1.021 + .210 * max(0., np.cos(radians)) + .0263 * abs(np.sin(radians)))
        origin = mouth - rotation @ BOTTLE_MOUTH
        gp, gr = env.object_pose(target)
        clearance = vessel_clearance(origin, rotation, gp, gr)
        if clearance < .001:
            raise SkillError("VESSEL_COLLISION", f"Bottle/glass clearance is only {clearance * 1000:.2f}mm")
        return origin, rotation

    def at_angle(angle, *, closing=False, side_clearance=0., duration=.40):
        origin, rotation = angle_pose(angle, closing=closing, side_clearance=side_clearance)
        move_bottle(origin, rotation, f"{'close' if closing else 'tilt'}_{angle:g}", duration)

    # First move forward while remaining on the bottle's side of the table,
    # keeping the upright bottle clear of the receiving glass.
    current = env.object_pose(obj)[0]
    front = np.array([glass_rim[0], current[1], max(current[2], 1.12)])
    move_bottle(front, original_rotation, "clear_table", 1.2, transport=True)
    baseline = initial["glass_ml"]
    previous_flow = [env.frame().sim_time, initial["bottle_ml"], 0.]
    def arriving_volume(measured):
        return measured["glass_ml"] - baseline + measured["in_flight_ml"]
    def enough_with_closing_flow(measured):
        now = env.frame().sim_time
        elapsed = now - previous_flow[0]
        if elapsed > 1e-6:
            previous_flow[2] = max(0., (previous_flow[1] - measured["bottle_ml"]) / elapsed)
            previous_flow[0:2] = [now, measured["bottle_ml"]]
        from backend.app.physics import closing_reserve
        # The stream keeps arriving while the bottle is brought upright. The cap is the task's tolerance
        # (40% of the request, floored) and the rate is the flow measured from this pour, so the reserve
        # is a physical relation rather than a private constant.
        reserve = max(.5, closing_reserve(previous_flow[2], .18, volume_ml * .4))
        return arriving_volume(measured) >= volume_ml - reserve
    target_reached = False
    final_angle = 0.
    # Approach the flow angle along one continuously timed path. The former
    # 20-degree moves each stopped, settled and corrected the mouth separately.
    # Retain the fine volume feedback once liquid can reach the lip.
    at_angle(0., duration=.4)
    poses = [angle_pose(angle) for angle in np.arange(2., 107., 2.)]
    move_bottle(*poses[-1], "tilt_continuous_106", 3.4, path=poses[:-1])
    for angle in np.arange(108., 119., 2.):
        at_angle(angle, duration=.4 if angle <= 75 else .10)
        final_angle = angle
        measured = env.fluid_state()
        arriving = arriving_volume(measured)
        if measured["spilled_ml"] > MAX_SPILL_ML:
            raise SkillError("POUR_SPILL", "The measured spill exceeded the pouring tolerance")
        if enough_with_closing_flow(measured):
            target_reached = True
            break
        if arriving > 1.:
            # Once flow starts, hold this angle and meter it before tilting
            # further. A whole extra tilt/correction can discharge several mL.
            for _ in range(max(1, int(2. / env.timestep / 10))):
                if control.stop_requested():
                    raise SkillError("STOPPED", "Pour stopped")
                env.step(10)
                measured = env.fluid_state()
                if measured["spilled_ml"] > MAX_SPILL_ML:
                    raise SkillError("POUR_SPILL", "The measured spill exceeded the pouring tolerance")
                if enough_with_closing_flow(measured):
                    target_reached = True
                    break
            trace.append({"phase": "meter_flow", **env.fluid_state()})
            if target_reached:
                break
    deadline = time.monotonic() + 180.
    for _ in range(int(5. / env.timestep / 10)):
        if target_reached:
            break
        if control.stop_requested():
            raise SkillError("STOPPED", "Pour stopped")
        env.step(10)
        measured = env.fluid_state()
        if measured["spilled_ml"] > MAX_SPILL_ML:
            raise SkillError("POUR_SPILL", "The measured spill exceeded the pouring tolerance")
        if enough_with_closing_flow(measured):
            target_reached = True
            break
        if time.monotonic() > deadline:
            raise SkillError("POUR_TIMEOUT", "Liquid flow exceeded the wall-clock limit")
    trace.append({"phase": "stop_flow", **env.fluid_state()})
    # Close the stream by bringing the opening up; all remaining motion is motor
    # control of the bottle while particles continue moving under fluid physics.
    # First raise the lip just beyond horizontal. Turning all the way from the
    # flow angle to 65 degrees in one short motion flung the last drops sideways.
    at_angle(min(85., final_angle), closing=True, duration=.20)
    at_angle(min(65., final_angle), closing=True, duration=.25)
    at_angle(45., closing=True, duration=.2)
    # Keep the mouth above the bowl while closing the stream. Only after the
    # opening points upward is it safe to move away and stand the bottle upright.
    at_angle(45., closing=True, side_clearance=.08, duration=.3)
    at_angle(0., closing=True, side_clearance=.08, duration=.3)
    current = env.object_pose(obj)[0]
    move_bottle([glass_rim[0], BOTTLE_START[1], max(current[2], 1.12)], original_rotation, "return_side", .6, transport=True)
    move_bottle(BOTTLE_START + [0., 0., .018], original_rotation, "return_bottle", .6, transport=True)
    move_bottle(BOTTLE_START, original_rotation, "lower_bottle", .4, transport=True)
    control.wait_gripper(arm, False, .4)
    retreat = runner.retreat(arm)
    if not retreat["success"]:
        raise SkillError(retreat.get("error", "RETREAT_FAILED"), "Could not retreat after pouring")
    env.step(max(1, int(.8 / env.timestep)))
    final = env.fluid_state()
    poured = final["glass_ml"] - baseline
    spilled = final["spilled_ml"]
    tolerance = max(3., volume_ml * .25)
    if (not final["finite"] or abs(poured - volume_ml) > tolerance
            or final["spilled_ml"] > MAX_SPILL_ML or final["in_flight_ml"] != 0.):
        raise SkillError("POUR_NOT_REACHED", f"Measured {poured:.4f} mL in glass, {spilled:.4f} mL spilled, "
                         f"and {final['in_flight_ml']:.4f} mL still outside the vessels; requested {volume_ml:g} mL")
    if env.object_contacts(obj):
        raise SkillError("BOTTLE_NOT_RELEASED", "The bottle is still touching a finger")
    bp, br = env.object_pose(obj)
    upright_error = float(np.arccos(np.clip(br[2, 2], -1., 1.)))
    return_error = float(np.linalg.norm(bp - BOTTLE_START))
    stable = env.object_linear_speed(obj) < .02
    if upright_error > .12 or return_error > .012 or not stable:
        raise SkillError("BOTTLE_NOT_RETURNED", "The released bottle is not stable and upright at its starting place")
    trace.append({"phase": "settled", **final})
    return {"object": obj, "target": target, "requested_ml": volume_ml, "poured_ml": poured,
            "spilled_ml": spilled, "additional_spilled_ml": spilled - initial["spilled_ml"],
            "tolerance_ml": tolerance, "initial_liquid": initial, "final_liquid": final,
            "pinch_contacts_before_pour": initial_grip, "fluid_trace": trace,
            "bottle_released": True, "bottle_upright_error": upright_error,
            "bottle_return_error": return_error, "bottle_stable": stable, "grasp_mode": env.grasp_mode}
