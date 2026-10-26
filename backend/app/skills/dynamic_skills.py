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


def reachable(env, arm, pose, rotation):
    """Can this robot's own IK reach the pose?  Asked of the planner, not of the controller.

    ``controller.solve_ik`` adds the constraints of a live, carrying arm - velocity continuity and a
    forbidden reconfiguration while the jaws hold - which reject poses a robot can plainly reach:
    measured 2026-10-25 on OpenManipulator-X, the controller rejected every searched wind-up while the
    same poses solved with zero error through ``solve_profile_ik``. Planning asks the planner.
    """
    from backend.app.robots.urdf import solve_profile_ik
    result = solve_profile_ik(env.profile, arm, np.asarray(pose, float), rotation, model=env.model,
                              q_init=env.home_arm_qpos(arm), snapshot=env.joint_snapshot())
    return bool(result['success'])


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
             target_point=None, flight_time_s: float = .30) -> dict:
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
        if stroke == strokes[0] and back_off(.06):
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
    # Solved by the planner's IK, for the same reason the wind-up is: the controller's own IK is a
    # local, damped solve that a four-joint arm cannot satisfy a full six-degree-of-freedom pose with.
    from backend.app.robots.urdf import solve_profile_ik
    aim = None
    direction = np.array([0., 0., 1.])
    if target_point is not None:
        # Aimed throw: the physics substrate says which velocity carries the cube through the target
        # after a chosen flight time, and the stroke is made along that direction so the release
        # velocity points at it. The engine still measures whether it actually got there.
        delta = np.asarray(target_point, float) - (np.asarray(position, float) + np.array([0., 0., stroke * .5]))
        aim = launch_for_target(delta, flight_time_s)
        direction = np.asarray(aim['direction'], float)
        speed = float(aim['speed']) / PINCH_TRANSMISSION
        ramp = max(.05, min(.6, ramp_for_release_speed(stroke, speed, RELEASE_PHASE)))
        release_at = RELEASE_PHASE * ramp
    planned = solve_profile_ik(env.profile, arm, np.asarray(position, float) + direction * stroke,
                               launch_rotation, model=env.model, q_init=start, snapshot=env.joint_snapshot())
    if not planned['success']:
        raise SkillError("IK_FAILURE", f"Launch stroke is unreachable ({planned['position_error']:.4f})")
    end = env.model.nearest_continuous(env.profile['chains'][arm]['joints'], planned['q_rad'], start)
    if mode == "strict":
        control.target_rotations[arm] = rotation.copy()
    ramp = max(.05, min(.6, ramp_for_release_speed(stroke, speed, RELEASE_PHASE)))
    release_at = RELEASE_PHASE * ramp
    trace = []
    env.configure_motion_response(1.)
    try:
        for step in range(int((ramp + .67) / env.timestep)):
            if control.stop_requested():
                raise SkillError("STOPPED", "Throw stopped")
            t = step * env.timestep
            phase = min(t / ramp, 1.)
            env.set_arm(arm, start + (end - start) * phase ** 2)
            if t >= release_at:
                control.set_gripper(arm, False)
            env.step()
            if t >= max(0., release_at - .06) and step % 2 == 0:
                trace.append({"t": t, "position": env.object_pose(obj)[0].tolist(),
                              "velocity": env.object_velocity(obj).tolist(),
                              "contacts": env.object_contact_links(obj)})
            if t < release_at - .005 and not runner._pinch_established(obj, arm):
                raise SkillError("GRASP_LOST", "The cube slipped before commanded release")
    finally:
        env.set_arm(arm, env.arm_qpos(arm))
        env.configure_motion_response(control.motion_speed)
    evidence = evaluate_throw(trace, height_cm / 100., release_after=release_at)
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
        evidence['success'] = bool(hit and (evidence.get('free_flight_duration') or 0.) > .02)
        evidence['hit'] = bool(hit)
        evidence['hit_tolerance_m'] = HIT_TOLERANCE_M
        if not evidence['success']:
            raise SkillError("THROW_FAILED", f"The throw did not hit the target "
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


def evaluate_throw(trace: list[dict], minimum_rise: float, release_after: float = .24) -> dict:
    """Recompute launch and apex from recorded engine samples, independent of skill success."""
    release = next((i for i, sample in enumerate(trace)
                    if sample["t"] >= release_after and not sample["contacts"] and sample["velocity"][2] > .1), None)
    if release is None:
        return {"success": False, "failure_category": "NO_UPWARD_FREE_FLIGHT"}
    flight = []
    for sample in trace[release:]:
        if sample["contacts"]:
            break
        flight.append(sample)
        if sample["velocity"][2] <= 0:
            break
    first = trace[release]
    apex = max(sample["position"][2] for sample in flight)
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
