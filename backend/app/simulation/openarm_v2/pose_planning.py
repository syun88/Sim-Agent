"""Planning an arm pose: the cell's solve, its seed policy, and the one place both live.

A skill that needs to know "can this pose be reached, and with which joint vector" - the throw's wind-up
search and the launch stroke are the two that do - must not care which robot it is driving. The two kinds
of robot answer it with different machinery, and this module is the cell's half:

* An imported package is planned by the planner that reads its URDF profile (``robots.urdf``), which is
  what ``GenericURDFEnvironment.plan_arm_pose`` asks for.
* The builtin official cell is planned by the damped least-squares solve below, over its own kinematics.

The two are not interchangeable, and the difference is measured rather than stylistic: a wind-up search
that asked the *controller* for a pose was rejected on every candidate it tried on OpenManipulator-X,
because a controller's solve carries the constraints of a live, carrying arm, while the same poses solved
with zero error through the planner's IK (measured 2026-10-25). Planning asks a planner; moving asks a
controller. ``OpenArmV2Controller.solve_ik`` uses this same solve, so the cell has one implementation of
"reach this pose" rather than two that can drift apart.
"""
from __future__ import annotations

import numpy as np

from backend.app.simulation.transform import orientation_error


def solve_from_seed(env, kinematics, limits, target_position, target_rotation, seed, max_iterations) -> dict:
    """One damped least-squares solve from one seed, with joint-limit avoidance."""
    rotation_tolerance_ok = getattr(env, "ik_rotation_tolerance", .015)
    position_tolerance_ok = getattr(env, "ik_position_tolerance", .002)
    q = np.clip(np.asarray(seed, float), limits[:, 0], limits[:, 1])
    best_q, best_error = q.copy(), np.inf
    for _ in range(max_iterations):
        position, rotation, linear_jacobian, angular_jacobian = kinematics.evaluate(q)
        perr = np.asarray(target_position, float) - position
        oerr = np.zeros(3) if target_rotation is None else orientation_error(rotation, target_rotation)
        error = float(np.linalg.norm(perr) + np.linalg.norm(oerr) * 0.2)
        if error < best_error:
            best_error = error
            best_q = q.copy()
        if np.linalg.norm(perr) < position_tolerance_ok and np.linalg.norm(oerr) < rotation_tolerance_ok:
            break
        J = np.vstack([linear_jacobian, angular_jacobian])
        # Weighted task vector matching the official openarm_control cost ratio
        # (position_cost=12.0, orientation_cost=1.5 → weight 1.5/12.0 = 0.125).
        task = np.concatenate([perr, oerr * 0.125])
        # Damping tuned between the official lm_damping (0.01) and the previous
        # value (0.0007): large enough to avoid oscillation near singularities,
        # small enough to converge to < 1e-3.
        damping = 0.002
        dq = J.T @ np.linalg.solve(J @ J.T + damping * np.eye(6), task)
        step = np.linalg.norm(dq)
        if step > 0.12:
            dq *= 0.12 / step
        # Joint limit avoidance: pull away from limits when close (official joint_limit_gain=0.95).
        for j in range(len(q)):
            margin = limits[j, 1] - limits[j, 0]
            lower_dist = q[j] - limits[j, 0]
            upper_dist = limits[j, 1] - q[j]
            braking_distance = min(0.2, margin)
            if lower_dist < braking_distance and dq[j] < 0.0:
                scale = max(0.0, (lower_dist / braking_distance) ** 2)
                dq[j] *= scale
            elif upper_dist < braking_distance and dq[j] > 0.0:
                scale = max(0.0, (upper_dist / braking_distance) ** 2)
                dq[j] *= scale
        q = np.clip(q + dq, limits[:, 0], limits[:, 1])
    best_position, best_rotation, _, _ = kinematics.evaluate(best_q)
    return {"q": best_q, "error": float(best_error),
            "position_error": float(np.linalg.norm(np.asarray(target_position, float) - best_position)),
            "attitude_error": (0. if target_rotation is None else
                               float(np.linalg.norm(orientation_error(best_rotation, target_rotation))))}


def ik_seeds(env, side, limits, count=12) -> list:
    """Deterministic alternative seeds for an arm that has fewer joints than the task needs.

    A damped local solve stalls wherever its Jacobian is flat for the requested task - measured
    2026-10-25 on the four-joint OpenManipulator-X, where a 2 cm residual persisted for targets the
    planner's IK solved exactly. The home posture alone does not cover the joint space, so the sweep
    adds the middle of the range plus a fixed pseudo-random sample: fixed so a failure is
    reproducible, sampled so the sweep does not depend on which posture the arm happens to be in.
    """
    seeds = [np.asarray(env.home_arm_qpos(side), float),
             (limits[:, 0] + limits[:, 1]) / 2.]
    rng = np.random.default_rng(0)
    for _ in range(count):
        unit = rng.random(len(limits))
        seeds.append(limits[:, 0] + unit * (limits[:, 1] - limits[:, 0]))
    return seeds


def plan_pose(env, side: str, position, rotation=None, *, q_init=None, max_iterations: int = 500,
              solve=None, seeds=None, planner_grade: bool = False) -> dict:
    """Plan a joint vector that reaches ``position`` (and ``rotation`` where the arm can hold it).

    ``solve`` and ``seeds`` are the two collaborators, injected so that a caller which already owns a
    tested seam (``OpenArmV2Controller`` patches its own ``_solve_from_seed`` in tests) keeps calling
    through it while the policy below stays in one place.

    ``planner_grade`` chooses which of the two questions is being asked. A *mover* wants the arm driven to
    a pose and accepts a basin that is good enough to move to (and, below six joints, is not failed for
    an attitude it cannot hold). A *planner* - the throw's wind-up search and its launch stroke - is asking
    whether a pose is reachable at all, so it is held to the same 2 mm and 0.03 rad the imported planner
    reports success with, and it searches the alternative seeds whatever the arm's joint count. Enabling
    the planner grade alone did not change this branch's throw either way (measured 2026-10-28: the same
    release position to nine decimals with and without it); it is here because the two questions differ,
    not because it was measured to rescue a case.

    Notable policy, each measured: a robot with fewer than six joints is not asked for an attitude, and
    its position alone is the task; a failed first solve retries the home posture before a reachable goal
    is called impossible; and an under-actuated arm then searches other basins, because one local solve
    is one basin among many.
    """
    solve = solve or (lambda kinematics, limits, target_position, target_rotation, seed, iterations:
                      solve_from_seed(env, kinematics, limits, target_position, target_rotation, seed, iterations))
    seeds = seeds or (lambda limits: ik_seeds(env, side, limits))
    limits = env.arm_limits(side)
    position = np.asarray(position, float)
    # A robot with fewer than six joints cannot hold an arbitrary attitude, so its attitude is not
    # asked for. Solving *with* one makes the least-squares task spend its freedom on a term it can
    # never satisfy, and the iterate the mixed error keeps can be tens of millimetres off in
    # position: measured 2026-10-25 on the four-joint OpenManipulator-X, every carry was refused
    # with a 40-90 mm residual - including a carry to the very pose the arm was already holding -
    # while the planner's IK, which asks for the position alone below six joints, solved the same
    # poses with zero error. Six-joint arms still hold the attitude they are given.
    under_actuated = len(limits) < 6
    if under_actuated:
        rotation = None
    elif rotation is None:
        rotation = env.ee_pose(side)[1]
    kinematics = env.arm_kinematics(side)
    position_tolerance = getattr(env, "ik_position_tolerance", .002)
    # The imported planner's own success thresholds: a pose planned for a launch is not allowed the slack
    # a move to a pose is, because the stroke is commanded to those joints and the cube leaves with
    # whatever speed the arm actually reached.
    planning_position_tolerance, planning_rotation_tolerance = .002, .03

    def accepted(result):
        if planner_grade:
            return bool(result["position_error"] <= planning_position_tolerance
                        and result["attitude_error"] <= planning_rotation_tolerance)
        # The mixed error still decides, except that an arm which cannot hold the attitude is not
        # failed for it: its position is the task.
        if result["error"] <= .02:
            return True
        return bool(under_actuated and result["position_error"] <= position_tolerance)

    def outcome(result):
        return {"success": True, "q_rad": np.asarray(result["q"], float).tolist(),
                "q": np.asarray(result["q"], float), "position_error": result["position_error"],
                "rotation_error": result["attitude_error"], "mixed_error": result["error"]}

    attempts = []
    seed = env.arm_qpos(side) if q_init is None else q_init
    first = solve(kinematics, limits, position, rotation, seed, max_iterations)
    if accepted(first):
        return outcome(first)
    attempts.append(first)
    if q_init is None:
        # A successful handover leaves a very different redundant-arm posture.
        # Retry the official spawn posture before calling a reachable goal impossible.
        home = solve(kinematics, limits, position, rotation, env.home_arm_qpos(side), max_iterations)
        if accepted(home):
            return outcome(home)
        attempts.append(home)
        if not under_actuated and not planner_grade:
            return _refusal(attempts, under_actuated)
    elif not under_actuated and not planner_grade:
        return _refusal(attempts, under_actuated)
    # Fewer joints than the task asks for, or a planner being asked whether the pose is reachable at all:
    # a local solve is one basin among many, so search the others.
    for alternative in seeds(limits):
        attempt = solve(kinematics, limits, position, rotation, alternative, max_iterations)
        if accepted(attempt):
            return outcome(attempt)
        attempts.append(attempt)
    return _refusal(attempts, under_actuated)


def _refusal(attempts, under_actuated) -> dict:
    """The best attempt's numbers, marked unsuccessful: a failure reports what was short, not just "no"."""
    best = min(attempts, key=lambda attempt: attempt["position_error"])
    return {"success": False, "q_rad": np.asarray(best["q"], float).tolist(),
            "q": np.asarray(best["q"], float), "position_error": best["position_error"],
            "rotation_error": best["attitude_error"], "mixed_error": best["error"],
            "unreachable_by_m": best["position_error"]}
