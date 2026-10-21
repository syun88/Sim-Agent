"""Dynamic manipulation driven by motors, with measured event trajectories."""
import numpy as np


def throw_up(runner, arm: str, obj: str, height_cm: float = 3.) -> dict:
    from backend.app.skills.robot_skills import SkillError

    env, control = runner.env, runner.controller
    runner._require_hold(arm, obj)
    # A common wind-up pose gives both official seven-joint arms an upward stroke.
    runner._carry_to(arm, obj, np.array([.50, .18 if arm == "left" else -.18, 1.075]))
    env.step(200)
    runner._require_hold(arm, obj)
    initial_contacts = env.object_contacts(obj, arm)
    position, rotation = env.ee_pose(arm)
    start = env.arm_qpos(arm)
    end, error, _ = control.solve_ik(arm, position + [0, 0, .14], rotation)
    if end is None:
        raise SkillError("IK_FAILURE", f"Upward launch stroke is unreachable ({error:.4f})")
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
