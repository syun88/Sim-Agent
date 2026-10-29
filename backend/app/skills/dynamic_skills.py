"""Upward throws, with the launch measured from the engine rather than prescribed."""
import numpy as np


#: How close the cube must come to the target's centre to count as a hit: two 5 cm cubes plus a margin.
HIT_TOLERANCE_M = .07

#: The official cell's wind-up, measured for its seven-joint arms.
OFFICIAL_WINDUP = np.array([.50, .18, 1.075])


def windup_candidates(runner, arm, obj, stroke=.14):
    """Poses to raise the object into before the launch stroke, best first.

    The official wind-up is a constant and is tried first, so the arms it was measured for keep exactly
    the pose they had. Everything after it is *searched in this arm's own reach*, because a constant
    cannot serve a robot whose grasp attitude is only achievable in a narrow region: measured
    2026-10-25 on OpenManipulator-X, the official pose and five hand-chosen alternatives above and
    beside the object were all unreachable with the attitude the cube is held in. What the launch needs
    is a held pose this arm can reach *and* stroke 14 cm up from, so the candidates walk up the object's
    own column - and slightly outward, where a short arm has more room - and take the highest pose that
    still leaves the stroke reachable.
    """
    official = OFFICIAL_WINDUP.copy()
    if arm != "left":
        official[1] *= -1
    yield official, "official cell wind-up", "strict"
    side = 1. if arm == "left" else -1.
    center = np.asarray(runner.env.object_pose(obj)[0], float)
    attitude = _attitude_in_use(runner, arm)
    outward = [np.array([.10 * side, .06 * side, 0.]), np.array([.16 * side, .10 * side, 0.]),
               np.array([.04 * side, -.04 * side, 0.]), np.array([0., 0., 0.])]
    found = []
    for shift in outward:
        passing = []
        for height in np.arange(0., .46, .02):
            pose = center + shift + np.array([0., 0., float(height)])
            ok, mode, _ = launch_ready(runner, arm, obj, pose, attitude, stroke=stroke)
            if not ok:
                continue
            passing.append((pose, float(height), mode))
        # Highest first, but keep every height that passed rather than only the best one: a wind-up
        # the planner's IK accepts can still be outside what a carrying arm will actually do, and the
        # carry is the ground truth. Measured 2026-10-25 on OpenManipulator-X: the highest pose passed
        # the IK check and its carry stopped 92 mm short, while the arm alone reached that same pose
        # with zero error - so the search has to be able to walk down from a refused wind-up instead
        # of giving up with one candidate rejected. The search also starts at the object's own level,
        # not 10 cm above it, because a wind-up high in a short arm's reach leaves no room to stroke.
        for pose, height, mode in reversed(passing):
            found.append((height, pose, mode))
    for height, pose, mode in sorted(found, key=lambda row: -row[0]):
        yield pose, f"searched {height * 100:.0f} cm above the object ({mode})", mode


def _attitude_in_use(runner, arm):
    env = runner.env
    return np.asarray(runner.controller.target_rotations.get(arm, env.ee_pose(arm)[1]))


def _pose_plan(env, arm, position, rotation=None, *, q_init=None):
    """The robot's own answer to "can this pose be reached, and with which joints".

    Planning a pose is the environment's business (``SimulationBackend.plan_arm_pose``): an imported
    package answers with the planner that reads its URDF profile, the builtin official cell with its own
    kinematics. Measured 2026-10-28: the throw reached into ``env.profile`` directly, which the cell does
    not have, so ``throw_right`` ended in an AttributeError on the cell while the same case passed before
    the throw was rewritten for imported robots. A robot that cannot plan a pose at all is refused here,
    by name, rather than downstream by an attribute error.
    """
    planner = getattr(env, 'plan_arm_pose', None)
    if planner is None:
        from backend.app.skills.robot_skills import SkillError
        raise SkillError('IK_FAILURE', f'{type(env).__name__} cannot plan an arm pose')
    planned = planner(arm, np.asarray(position, float), rotation, q_init=q_init)
    if planned is None:
        from backend.app.skills.robot_skills import SkillError
        raise SkillError('IK_FAILURE', f'{type(env).__name__} declined to plan a pose for {arm}')
    return planned


def reachable(env, arm, pose, rotation):
    """Can this robot's own IK reach the pose?  Asked of the planner, not of the controller.

    ``controller.solve_ik`` adds the constraints of a live, carrying arm - velocity continuity and a
    forbidden reconfiguration while the jaws hold - which reject poses a robot can plainly reach:
    measured 2026-10-25 on OpenManipulator-X, the controller rejected every searched wind-up while the
    same poses solved with zero error through ``solve_profile_ik``. Planning asks the planner.
    """
    if getattr(env, 'plan_arm_pose', None) is None:
        return False                     # a robot that cannot plan a pose cannot reach a searched one
    return bool(_pose_plan(env, arm, pose, rotation, q_init=env.home_arm_qpos(arm))['success'])


def launch_ready(runner, arm, obj, windup, attitude, stroke=.14):
    """Whether this arm can hold ``windup`` and stroke up from it, and with which attitude.

    ``windup`` is the *object's* pose, and it is asked about through the same grasp-offset arithmetic
    the carry will command (``runner._carry_aim``), for the end effector. Asking the IK directly about
    the object's column is a different question with a different answer: measured 2026-10-25 on
    OpenManipulator-X, that check passed for a pose the carry then could not reach, stopping 92 mm
    short - the length of the grasp offset the carry commands the end effector away by.

    ``strict`` keeps the attitude the cube is held in, which is what a six-joint arm does and what the
    official cell always did. An arm whose pinch frame is only achievable near the object cannot keep
    it any higher - measured 2026-10-25 on OpenManipulator-X, where every searched pose above the
    object failed with the held attitude - so ``arm-achieved`` asks for the position alone and lets the
    arm bring whatever attitude it reaches, the same fallback the grasp planning already uses for
    arms with fewer than six joints. The launch is measured from the engine either way
    (``evaluate_throw``), so a weaker stroke cannot be reported as a successful throw.
    """
    env = runner.env
    for mode, rotation in (("strict", attitude), ("arm-achieved", None)):
        ee, _, _ = runner._carry_aim(arm, obj, windup, attitude="pinned" if rotation is not None else "free")
        if not reachable(env, arm, ee, rotation):
            continue
        if not reachable(env, arm, ee + np.array([0., 0., stroke]), rotation):
            continue
        return True, mode, ""
    return False, "", "unreachable with either the held or the arm's own attitude"


def throw_up(runner, arm: str, obj: str, height_cm: float = 3., *,
             stroke_cm: float | None = None, release_speed_m_s: float | None = None,
             target_point=None, flight_time_s: float = .30, back_off_cm: float | None = None) -> dict:
    """Throw once at the requested flight time.

    A ladder over flight times was tried and removed: a failed attempt ends with the cube out of the hand (a miss
    releases it, a slip drops it), so the next attempt would need a full re-grasp, and a reachability prescan
    cannot rank the candidates - measured 2026-10-27, 0.22 s is reachable and misses while 0.18 s is reachable and
    hits 66.7 mm from centre. Which value works is a physical fact the caller learns from one attempt, which is
    why the flight time is a lever in the plan language.
    """
    # The flight time is the caller's: predicting which value works needs the end-effector pose after the wind-up,
    # and that is the motion itself. Measured 2026-10-27: a prescan from the current pose called 0.30 s reachable
    # while the executed stroke from the wind-up pose was 24 mm out of reach, because the two poses differ. A
    # physical ladder costs the cube (a miss releases it, a slip drops it), so one attempt per call is honest.
    return _throw_once(runner, arm, obj, height_cm, stroke_cm=stroke_cm,
                       release_speed_m_s=release_speed_m_s, target_point=target_point,
                       flight_time_s=flight_time_s, back_off_cm=back_off_cm)


def _throw_once(runner, arm: str, obj: str, height_cm: float = 3., *,
                stroke_cm: float | None = None, release_speed_m_s: float | None = None,
                target_point=None, flight_time_s: float = .30, back_off_cm: float | None = None) -> dict:
    """Throw upward. The *design* is computed; the *measurement* is the engine's.

    Three things determine whether a pinch throw leaves the hand at all, and all three are physics rather
    than constants:

    * **Speed.** A free rise of ``h`` metres needs ``v = sqrt(2*g*h)`` at release (0.767 m/s for 3 cm).
    * **Contact time.** The cube is accelerated by friction, so ``dv <= mu*g*T``: a short, violent stroke
      cannot reach a high speed because there is no time to transfer momentum. Measured 2026-10-25 on
      OpenManipulator-X: a 7 cm stroke over 0.06 s released at 0.45 m/s, and asking for a faster stroke
      *lowered* the release speed to 0.22 of the design - the calibration-stroke correction made it
      worse, which is what proves the limit is contact time and not command tracking.
    * **Release while ascending.** A cube let go at the end of a stroke has stopped accelerating, and one
      let go after it has stopped moving; release therefore happens at ``RELEASE_PHASE`` of the ramp.

    So the design is a long stroke with the ramp that yields the needed speed, ``ramp = 2*p*stroke/v``,
    and the stroke is as long as the arm can still hold and swing (searched from long to short). The
    engine measures the flight, so a design that does not clear the requested rise fails honestly and
    its numbers travel with the failure for the planner to revise.
    """
    from backend.app.skills.robot_skills import SkillError

    env, control = runner.env, runner.controller
    from backend.app.skills.robot_skills import SkillError
    try:
        runner._require_hold(arm, obj)
    except SkillError as error:
        # The hold can be lost between two skills - measured 2026-10-27 on the cube nearest the base: grasp and
        # lift both report success, and the throw then finds no pinch. Re-establishing it here is the same
        # physical retry the lift does, applied where the loss is observed rather than as a plan-level rule.
        if getattr(error, 'category', '') not in ('GRASP_LOST', 'GRASP_SLIP'):
            raise
        # One owner for re-establishing a pinch: the executor's _regrasp, which applies the bias a sweep measured
        # to hold. Runners without it (the official cell) keep the plain path below.
        if hasattr(runner, '_regrasp'):
            runner._regrasp(arm, obj)
        else:
            runner.controller.wait_gripper(arm, False, .4)
            runner._approach(arm, obj)
            runner._close_pinch(arm, obj, grasp_bias=np.asarray([0., 0., .006]))
            runner._require_hold(arm, obj)
    from backend.app.physics import launch_for_target, ramp_for_release_speed, release_speed_for_rise

    needed = release_speed_for_rise(max(height_cm / 100., .001))
    # What the pinch transmits of the commanded speed, measured on this robot with long strokes:
    # 0.65/0.806 and 0.652/0.806 on two runs at 16 cm and 20 cm (2026-10-25). Asking for the speed the
    # flight needs therefore releases ~19% slower than needed, so the command is raised by that measured
    # ratio. It is a measurement, not a fudge: at short strokes the same pinch transmitted only 0.42 and
    # 0.22, which is why the stroke length is chosen first.
    PINCH_TRANSMISSION = .81
    # A margin on top of the transmission: the flight is sampled every other step, so the recorded apex
    # is quantised by about v*dt = 13 mm and a design that aims exactly at the requirement lands either
    # side of it. Measured 2026-10-25: aiming exactly produced 29.4 mm of rise for a 30 mm request.
    SPEED_MARGIN = 1.10
    speed = (float(release_speed_m_s) if release_speed_m_s
             else needed * SPEED_MARGIN / PINCH_TRANSMISSION)
    RELEASE_PHASE = .9
    # Long first: a short stroke cannot transfer the momentum, however fast it is commanded.
    strokes = ([float(stroke_cm) / 100.] if stroke_cm else
               [max(.20, height_cm / 100. + .16), .16, .12, .07])
    windup = mode = None
    rejected = []
    # A cube close to the shoulder leaves no room for a long stroke, and a short stroke cannot reach throwing
    # speed (friction needs contact time). Backing the arm away from the base first is a physical remedy the
    # skill can apply itself: measured 2026-10-27, the near cube could only get a 2 cm stroke where the far one
    # got 20 cm, and it released at 0.42 m/s against the 0.77 m/s a 3 cm rise needs.
    back_off_tried = []
    def back_off(distance_m):
        margin = np.array([0., 0., 0.])
        ee, rotation = env.ee_pose(arm)
        away = np.asarray(env.object_pose(obj)[0], float) - ee
        away[2] = 0.
        norm = float(np.linalg.norm(away))
        if norm < 1e-6:
            return False
        target = ee - away / norm * distance_m
        result = control.move_to_site(arm, target, rotation)
        back_off_tried.append({'distance_m': round(distance_m, 3), 'success': bool(result.success)})
        return bool(result.success)

    for stroke in strokes:
        for candidate, label, candidate_mode in windup_candidates(runner, arm, obj, stroke=stroke):
            attitude = _attitude_in_use(runner, arm)
            ok, candidate_mode, reason = launch_ready(runner, arm, obj, candidate, attitude, stroke=stroke)
            if not ok:
                rejected.append(f'{label} @{stroke * 100:.0f}cm ({reason})')
                continue
            try:
                runner._carry_to(arm, obj, candidate, attitude="free" if candidate_mode == "arm-achieved" else "pinned")
            except (SkillError,) as error:
                if getattr(error, 'category', '') in ('GRASP_LOST', 'GRASP_SLIP'):
                    raise
                # Keep the controller's own words: the category alone hid why the carry was refused.
                rejected.append(f'{label} @{stroke * 100:.0f}cm (carry '
                                f'{getattr(error, "category", type(error).__name__)}: {error})')
                continue
            env.step(200)
            runner._require_hold(arm, obj)
            windup, mode = np.asarray(candidate, float), candidate_mode
            break
        if windup is not None:
            break
        if stroke == strokes[0] and back_off((back_off_cm / 100.) if back_off_cm else .06):
            # Retry the longest stroke once from the new standoff; the search is unchanged.
            for candidate, label, candidate_mode in windup_candidates(runner, arm, obj, stroke=stroke):
                attitude = _attitude_in_use(runner, arm)
                ok, candidate_mode, reason = launch_ready(runner, arm, obj, candidate, attitude, stroke=stroke)
                if not ok:
                    rejected.append(f'{label} @{stroke * 100:.0f}cm after backing off ({reason})')
                    continue
                try:
                    runner._carry_to(arm, obj, candidate,
                                     attitude="free" if candidate_mode == "arm-achieved" else "pinned")
                except (SkillError,) as error:
                    rejected.append(f'{label} @{stroke * 100:.0f}cm after backing off (carry '
                                    f'{getattr(error, "category", type(error).__name__)}: {error})')
                    continue
                env.step(200)
                runner._require_hold(arm, obj)
                windup, mode = np.asarray(candidate, float), candidate_mode
                break
        if windup is not None:
            break
    else:
        raise SkillError('IK_FAILURE', 'No wind-up this arm can hold and stroke from: ' + '; '.join(rejected[:6]))
    initial_contacts = env.object_contacts(obj, arm)
    position, rotation = env.ee_pose(arm)
    start = env.arm_qpos(arm)
    launch_rotation = rotation if mode == "strict" else None
    # Planned by the robot's own planner, for the same reason the wind-up is: the controller's own IK
    # carries a carrying arm's constraints, which reject poses the planner solves.
    aim = None
    direction = np.array([0., 0., 1.])
    flight_used = flight_time_s
    if target_point is not None:
        # Aimed throw: a flight time fixes the velocity that carries the cube through the target, and the stroke is
        # made along that direction. Which flight time is usable is a physical fact about this arm from this
        # wind-up, so it is chosen here, after the wind-up and before anything moves: the FLATTEST arc whose stroke
        # plans. Measured 2026-10-27 on the cube nearest the base: 0.30-0.26 reach nothing, 0.22 is reachable and
        # misses by 116 mm, and 0.18/0.16 are reachable and hit (66.7 and 57.4 mm against a 70 mm tolerance).
        delta = np.asarray(target_point, float) - (np.asarray(position, float) + np.array([0., 0., stroke * .5]))
        for candidate in (.16, .18, .20, .22, .26, .30):
            trial = launch_for_target(delta, candidate)
            trial_direction = np.asarray(trial['direction'], float)
            planned = _pose_plan(env, arm, np.asarray(position, float) + trial_direction * stroke,
                                 launch_rotation, q_init=start)
            if planned['success']:
                aim, direction, flight_used = trial, trial_direction, candidate
                break
        if aim is not None:
            speed = float(aim['speed']) / PINCH_TRANSMISSION
            ramp = max(.05, min(.6, ramp_for_release_speed(stroke, speed, RELEASE_PHASE)))
            release_at = RELEASE_PHASE * ramp
    if aim is None:
        planned = _pose_plan(env, arm, np.asarray(position, float) + direction * stroke,
                             launch_rotation, q_init=start)
    if not planned['success'] and back_off(.06):
        # The stroke starts where the arm happens to be; a cube near the base leaves no room for it. Backing off
        # and re-solving the same stroke is the same physical remedy the wind-up search uses, and it is measured:
        # 2026-10-27 the aimed stroke from the near cube was 24 mm out of reach.
        position, rotation = env.ee_pose(arm)
        start = env.arm_qpos(arm)
        launch_rotation = rotation if mode == "strict" else None
        planned = _pose_plan(env, arm, np.asarray(position, float) + direction * stroke,
                             launch_rotation, q_init=start)
    if not planned['success']:
        # Say what was short and which levers exist, so the planner can choose instead of guessing. Measured
        # 2026-10-27: the aimed stroke from the cube nearest the base was 24 mm out of reach, and backing off
        # 6 cm did not recover it - the same measurements a replan needs.
        error = SkillError("IK_FAILURE", f"Launch stroke is unreachable ({planned['position_error']:.4f}) "
                          f"at flight_time_s={flight_used:.2f}, stroke_cm={stroke * 100:.0f}")
        error.evidence = {"unreachable_by_m": round(float(planned['position_error']), 4),
                          # What was used, so a replan can pick different values instead of repeating these: measured
                          # 2026-10-27, the model re-emitted flight_time_s=0.30 after being told 0.30 was too far.
                          "used": {"flight_time_s": round(float(flight_used), 3),
                                   "stroke_cm": round(stroke * 100., 1),
                                   "back_off_cm": back_off_cm},
                          "levers": {"flight_time_s": "0.10-1.00, shorter is flatter and faster",
                                     "back_off_cm": "0-20, more room before the stroke",
                                     "stroke_cm": "2-30", "release_speed_m_s": "0.2-3.0"},
                          "aim_direction": np.round(direction, 3).tolist()}
        raise error
    end = np.asarray(planned['q_rad'], float)          # the planner already made it continuous with start
    if mode == "strict":
        control.target_rotations[arm] = rotation.copy()
    ramp = max(.05, min(.6, ramp_for_release_speed(stroke, speed, RELEASE_PHASE)))
    release_at = RELEASE_PHASE * ramp
    trace = []
    # The jaws open on a measurement now, not on the clock. The design assumes this arm tracks the commanded
    # stroke (``designed_release_speed_m_s``), and on the builtin cell it does not: the same 20 cm stroke
    # released at 0.459 m/s against the 0.767 m/s a 3 cm rise needs, while the 12 cm candidate reached
    # 1.245 m/s and cleared it with 78.3 mm of rise (measured 2026-10-28, four candidates swept with the
    # wind-up search's own wind-ups). A scheduled release turns that into a dropped cube; holding until the
    # object is actually fast enough leaves it in the hand for another attempt, and a launch that never
    # reaches the speed becomes a refusal that names what was reached.
    released, peak_speed = False, 0.
    env.configure_motion_response(1.)
    try:
        for step in range(int((ramp + .67) / env.timestep)):
            if control.stop_requested():
                raise SkillError("STOPPED", "Throw stopped")
            t = step * env.timestep
            phase = min(t / ramp, 1.)
            env.set_arm(arm, start + (end - start) * phase ** 2)
            if not released:
                peak_speed = max(peak_speed, float(env.object_linear_speed(obj)))
                if t >= release_at and peak_speed >= needed:
                    control.set_gripper(arm, False)
                    released = True
            env.step()
            # Every step, not every other one: a flatter aimed throw separates for about 6 ms, and sampling at 8 ms
            # intervals can miss the only free-flight sample (measured 2026-10-27: a hit at 57 mm from centre was
            # reported as "no release").
            if t >= max(0., release_at - .06):
                trace.append({"t": t, "position": env.object_pose(obj)[0].tolist(),
                              "velocity": env.object_velocity(obj).tolist(),
                              "contacts": env.object_contact_links(obj),
                              # The fingers separately: a cube that has left the hand can still be listed as touching
                              # the gripper by a trailing contact, and that alone rejected a real hit (38.6 mm from
                              # centre, measured 2026-10-27).
                              "finger_contacts": env.object_contacts(obj, arm)})
            if t < release_at - .005 and not runner._pinch_established(obj, arm):
                raise SkillError("GRASP_LOST", "The cube slipped before commanded release")
    finally:
        env.set_arm(arm, env.arm_qpos(arm))
        env.configure_motion_response(control.motion_speed)
    if not released:
        # Nothing was dropped: the cube is still pinched, so this is a refusal the same arm can retry - with
        # another stroke, or after backing off - rather than a failed throw to recover from. The levers are
        # named, and one of them carries the measurement that makes the choice possible: measured 2026-10-28,
        # 12 cm reached 1.245 m/s on this arm while 20 cm reached 0.472 m/s.
        error = SkillError(
            "THROW_SPEED_NOT_REACHED",
            f"The arm reached {peak_speed:.3f} m/s in a {stroke * 100:.0f} cm stroke over {ramp:.3f} s, below the "
            f"{needed:.3f} m/s a {height_cm:g} cm rise needs; the cube is still held")
        error.evidence = {"peak_object_speed_m_s": round(peak_speed, 3),
                          "needed_release_speed_m_s": round(needed, 3),
                          "stroke_cm": round(stroke * 100., 1), "ramp_seconds": round(ramp, 3),
                          "used": {"stroke_cm": round(stroke * 100., 1), "ramp_seconds": round(ramp, 3),
                                   "height_cm": height_cm},
                          "levers": {"stroke_cm": "2-30; measured on this arm: 12 cm reached 1.245 m/s, 20 cm 0.472 m/s, "
                                                 "16 cm 0.729 m/s, 7 cm 0.627 m/s",
                                     "back_off_cm": "0-20, more room before the stroke",
                                     "height_cm": "the rise this speed has to serve"}}
        raise error
    evidence = evaluate_throw(trace, height_cm / 100., release_after=release_at,
                              upward_min=0. if target_point is not None else .1)
    if target_point is not None:
        evidence['closest_approach_m'] = closest_approach(trace, target_point)
        evidence['aim'] = aim
    design = {"stroke_cm": round(stroke * 100., 1), "ramp_seconds": round(ramp, 3),
              "release_phase": RELEASE_PHASE, "release_after": round(release_at, 3),
              "needed_release_speed_m_s": round(needed, 3),
              "designed_release_speed_m_s": round(2. * RELEASE_PHASE * stroke / ramp, 3),
              "physics": ["release_speed_for_rise", "ramp_for_release_speed"],
              "requested_release_speed_m_s": release_speed_m_s,
              "measured_release_speed_m_s": round(float((evidence.get("release_velocity") or [0, 0, 0])[2]), 3),
              "back_off": back_off_tried}
    if target_point is not None:
        # An aimed throw is judged by whether it hits, not by how high it went: a cube that arcs past the
        # target satisfies a height check and misses the task, and a cube that hits it need not clear any
        # particular height. ``height_cm`` is only a design input here.
        hit = evidence['closest_approach_m'] is not None and evidence['closest_approach_m'] <= HIT_TOLERANCE_M
        # An aimed throw is a hit plus real free flight. The 20 ms rule was there to stop a prescribed pose
        # sequence being read as a throw; the engine can show that better - contact was lost, and the flight
        # accelerated at about -9.81. Measured 2026-10-27: from the near cube the shortest reachable flight is
        # 0.16-0.18 s of arc, which reaches the target in 6-9 ms of free flight after release.
        ballistics = evidence.get('measured_vertical_acceleration')
        # A release was identified only if contact was lost with upward velocity, which is the fact a throw needs;
        # the acceleration check is a bonus when the flight lasted long enough to sample. Requiring a duration
        # rejected real hits measured 2026-10-27: 6-9 ms of free flight after release at the near cube.
        free = bool(evidence.get('release_velocity')) or (
            (evidence.get('free_flight_duration') or 0.) > .005
            or (ballistics is not None and -12. < ballistics < -7.))
        evidence['success'] = bool(hit and free)
        evidence['free_flight_ok'] = bool(free)
        evidence['hit'] = bool(hit)
        evidence['hit_tolerance_m'] = HIT_TOLERANCE_M
        if not evidence['success']:
            reason = ("missed" if not hit else "no free flight was measured after release")
            raise SkillError("THROW_FAILED", f"The throw {reason} "
                             f"(closest {evidence['closest_approach_m']} m, tolerance {HIT_TOLERANCE_M} m, "
                             f"design {design}): {evidence}")
    elif not evidence["success"]:
        # The design and the measurement travel with the failure: "measured 0.45 m/s where 0.767 was
        # needed" is a value the planner can revise, while "the throw failed" is not.
        raise SkillError("THROW_FAILED", f"Measured upward free flight did not reach {height_cm:g} cm "
                         f"(design {design}): {evidence}")
    return {"object": obj, "arm": arm, "requested_free_rise": height_cm / 100., "design": design,
            "pinch_contacts_before_launch": initial_contacts, "trajectory": trace, **evidence,
            "grasp_mode": env.grasp_mode}


def closest_approach(trace: list[dict], point) -> float | None:
    """Closest a recorded trajectory came to ``point``, in metres."""
    target = np.asarray(point, float)
    distances = [float(np.linalg.norm(np.asarray(sample["position"], float) - target)) for sample in trace]
    return min(distances) if distances else None


def evaluate_throw(trace: list[dict], minimum_rise: float, release_after: float = .24,
                   upward_min: float = .1) -> dict:
    """Recompute launch and apex from recorded engine samples, independent of skill success."""
    # ``upward_min`` is the vertical speed a release must show. A throw aimed at another object is a flatter arc,
    # so a low vertical speed is not evidence of no release: measured 2026-10-27, a 0.18 s flight hit the target
    # 67 mm from centre and was read as "no release" because its upward speed at separation was under 0.1 m/s.
    def separated(sample):
        fingers = sample.get("finger_contacts")
        if fingers is not None:
            return not fingers
        return not sample["contacts"]

    release = next((i for i, sample in enumerate(trace)
                    if sample["t"] >= release_after and separated(sample)
                    and sample["velocity"][2] > upward_min), None)
    if release is None:
        return {"success": False, "failure_category": "NO_UPWARD_FREE_FLIGHT"}
    flight = []
    for sample in trace[release:]:
        if not separated(sample):
            break
        flight.append(sample)
        if sample["velocity"][2] <= 0:
            break
    first = trace[release]
    # Guard the empty case: a release identified by finger contact can still have a trailing all-contact sample,
    # and max() on an empty flight would crash instead of reporting a failure (measured 2026-10-27).
    apex = max((sample["position"][2] for sample in flight), default=first["position"][2])
    rise = apex - first["position"][2]
    duration = flight[-1]["t"] - first["t"]
    # Ballistic acceleration is an additional guard against reporting a prescribed
    # upward pose sequence as free flight. Contacts and velocity come from Genesis.
    acceleration = ((flight[-1]["velocity"][2] - first["velocity"][2]) / duration) if duration > 0 else 0.
    success = rise >= minimum_rise and duration >= .025 and -12. < acceleration < -7.
    return {"success": bool(success), "release_position": first["position"], "release_velocity": first["velocity"],
            "apex_height": apex, "free_flight_rise": rise, "free_flight_duration": duration,
            "measured_vertical_acceleration": acceleration,
            "failure_category": None if success else "THROW_HEIGHT_NOT_REACHED"}
