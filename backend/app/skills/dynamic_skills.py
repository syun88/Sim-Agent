"""Upward throws, with the launch measured from the engine rather than prescribed."""
import numpy as np


#: The official cell's wind-up, measured for its seven-joint arms.
OFFICIAL_WINDUP = np.array([.50, .18, 1.075])


def windup_candidates(runner, arm, obj):
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
            ok, mode, _ = launch_ready(runner, arm, obj, pose, attitude)
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


def throw_up(runner, arm: str, obj: str, height_cm: float = 3.) -> dict:
    from backend.app.skills.robot_skills import SkillError

    env, control = runner.env, runner.controller
    runner._require_hold(arm, obj)
    # Raise the object into a pose this arm can hold and stroke up from. The official wind-up is tried
    # first and is what the official arms keep; a robot that cannot reach it gets one derived from the
    # object's own position, because the launch only needs headroom above a held pose, not a constant.
    rejected = []
    for windup, label, mode in windup_candidates(runner, arm, obj):
        attitude = _attitude_in_use(runner, arm)
        ok, mode, reason = launch_ready(runner, arm, obj, windup, attitude)
        if not ok:
            rejected.append(f'{label} ({reason})')
            continue
        try:
            runner._carry_to(arm, obj, windup, attitude="free" if mode == "arm-achieved" else "pinned")
        except (SkillError,) as error:
            if getattr(error, 'category', '') in ('GRASP_LOST', 'GRASP_SLIP'):
                raise
            # Keep the controller's own words: the category alone hid the reason the carry was refused.
            rejected.append(f'{label} (carry {getattr(error, "category", type(error).__name__)}: {error})')
            continue
        env.step(200)
        runner._require_hold(arm, obj)
        break
    else:
        raise SkillError('IK_FAILURE', 'No wind-up this arm can hold and stroke from: ' + '; '.join(rejected))
    launch_from = np.asarray(windup, float)
    initial_contacts = env.object_contacts(obj, arm)
    position, rotation = env.ee_pose(arm)
    start = env.arm_qpos(arm)
    launch_rotation = rotation if mode == "strict" else None
    end, error, _ = control.solve_ik(arm, position + [0, 0, .14], launch_rotation)
    if end is None:
        raise SkillError("IK_FAILURE", f"Upward launch stroke is unreachable ({error:.4f})")
    if mode == "strict":
        control.target_rotations[arm] = rotation.copy()
    trace = []
    # Wind-up/transport use the fast trajectory. Preserve the calibrated launch
    # response and release timing that define the requested free-flight height.
    env.configure_motion_response(1.)
    try:
        for step in range(int(.95 / env.timestep)):
            if control.stop_requested():
                raise SkillError("STOPPED", "Throw stopped")
            t = step * env.timestep
            phase = min(t / .28, 1.)
            env.set_arm(arm, start + (end - start) * phase ** 2)
            if t >= .24:
                control.set_gripper(arm, False)
            env.step()
            if t >= .20:
                contacts = env.object_contact_links(obj)
                if step % 2 == 0:
                    trace.append({"t": t, "position": env.object_pose(obj)[0].tolist(),
                                  "velocity": env.object_velocity(obj).tolist(), "contacts": contacts})
            if t < .235 and not runner._pinch_established(obj, arm):
                raise SkillError("GRASP_LOST", "The cube slipped before commanded release")
    finally:
        env.set_arm(arm, env.arm_qpos(arm))
        env.configure_motion_response(control.motion_speed)
    evidence = evaluate_throw(trace, height_cm / 100.)
    if not evidence["success"]:
        raise SkillError("THROW_FAILED", f"Measured upward free flight did not reach {height_cm:g} cm: {evidence}")
    return {"object": obj, "arm": arm, "requested_free_rise": height_cm / 100.,
            "pinch_contacts_before_launch": initial_contacts, "trajectory": trace, **evidence,
            "grasp_mode": env.grasp_mode}


def evaluate_throw(trace: list[dict], minimum_rise: float) -> dict:
    """Recompute launch and apex from recorded engine samples, independent of skill success."""
    release = next((i for i, sample in enumerate(trace)
                    if sample["t"] >= .24 and not sample["contacts"] and sample["velocity"][2] > .1), None)
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
