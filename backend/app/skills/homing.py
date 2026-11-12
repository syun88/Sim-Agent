"""Home routes from the measured posture and current objects, without physics writes.

Getting the arm home is a *search*, not a fixed lift. Measured 2026-11-09: after placing a cube, the only
candidate that solved was a vertical lift to 1.24 m, and it brushed the cube it had just released by 1.1 mm
(`PARK_COLLISION`, `displacement_m: 0.0011`) - the arm never moved away from the object first. The candidates
below therefore include a retreat *away* from the nearest object before any lift, and every candidate is
screened and then ranked by the measured clearance of its whole path, so the return is the roomiest route the
body can actually reach rather than the first one that happens to solve.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from backend.app.simulation.transform import orientation_error


def plan_empty_arm_home(env, controller, arm):
    kin = env.arm_kinematics(arm)
    start_q = env.arm_qpos(arm)
    start, rotation = env.ee_pose(arm)
    home = env.home_arm_qpos(arm)
    target, target_rotation, _, _ = kin.evaluate(home)
    objects = {name: env.object_pose(name) for name in env.frame().objects}
    sizes = [float(np.max(env.object_bounds(name)[1]-env.object_bounds(name)[0]))
             for name in objects]
    clearance = max(.08, 2*max(sizes, default=.04))
    heights = list(dict.fromkeys([target[2]+clearance,
                                 max(start[2], target[2])+clearance, target[2]+clearance/2]))
    check = getattr(kin, 'empty_arm_path_is_clear', None)
    measure = getattr(kin, 'path_minimum_clearance', None)
    # How much room the home posture needs is a property of this body: the clearance a target must have is
    # derived from how far the arm actually lands from where it was told to go (measured), not from a
    # constant. Measured 2026-11-09: a 30 mm margin lost to a ~35 mm positioning difference.
    from backend.app.skills import motion_accuracy
    from backend.app.planning.measurement_flags import enabled as _measured
    home_clearance_required = (motion_accuracy.required_clearance(arm) if _measured('home_margin') else .03)
    nominal_clear = (check is None
                     or check([home, home], objects, minimum_object_clearance=home_clearance_required))

    def parked_postures():
        """Where this arm could actually be left: the model home first, then the canonical set.

        Measured 2026-11-09/10: the task reported "could not return the empty arm home" because the only target
        ever considered was the model home. Where the arm may rest is a question the body can answer - it is a
        posture that is reachable (it is a joint configuration) and clear (the scene says so) - so it is chosen
        the same way the staging point and the grasp attitude are chosen.
        """
        options = [('model_home', np.asarray(home, float))]
        try:
            from backend.app.planning.capability_queries import canonical_postures
            for index, posture in enumerate(canonical_postures(env, arm)):
                if np.allclose(posture, home):
                    continue
                options.append((f'canonical_{index}', np.asarray(posture, float)))
        except Exception:
            pass
        return options

    def build(poses, target_q=None):
        """Solve a joint route through these (position, attitude) poses, ending at `target_q`; None if IK cannot."""
        route = [start_q]
        for point, attitude in poses:
            solved = None
            if attitude is None:
                # An empty arm may rotate while clearing vertically. Fixing its original wrist attitude can
                # make a perfectly reachable lift impossible near the base of a redundant arm.
                limits = env.arm_limits(arm)
                def residual(q):
                    return np.r_[kin.evaluate(q)[0]-point, (q-route[-1])*.0002]
                lifted = least_squares(residual,np.clip(route[-1],*limits.T),bounds=limits.T,max_nfev=100)
                if np.linalg.norm(kin.evaluate(lifted.x)[0]-point)<=.003:
                    solved=lifted.x
            for seed in () if attitude is None else (route[-1], home):
                q, _, _ = controller.solve_ik(arm, point, attitude, q_init=seed)
                if q is None:
                    continue
                actual, actual_rotation, _, _ = kin.evaluate(q)
                if (np.linalg.norm(actual-point)<=.004
                        and np.linalg.norm(orientation_error(actual_rotation,attitude))<=.04):
                    solved = q
                    break
            if solved is None:
                return None
            route.append(solved)
        # The last waypoint is the posture the arm is *left in*: the model home when it works, otherwise a
        # clear posture the arm can actually hold.
        if target_q is not None:
            route.append(np.asarray(target_q, float))
        return route

    # How to get away from the object before climbing: the direction that leaves the object behind. The
    # straight-up lift is kept as a candidate, but it is no longer the assumption.
    def retreat_directions():
        directions = []
        if objects:
            nearest = min(objects.items(), key=lambda item: float(np.linalg.norm(item[1][0]-start)))
            away = start - np.asarray(nearest[1][0], float)
            away[2] = 0.
            if np.linalg.norm(away) > 1e-6:
                directions.append(('away_from_nearest', away / np.linalg.norm(away)))
            centroid = np.mean([np.asarray(pose[0], float) for pose in objects.values()], axis=0)
            outward = start - centroid
            outward[2] = 0.
            if np.linalg.norm(outward) > 1e-6:
                directions.append(('away_from_objects', outward / np.linalg.norm(outward)))
        directions.extend([('lateral_left', np.array([0., 1., 0.])),
                           ('lateral_right', np.array([0., -1., 0.]))])
        return directions

    candidates = []
    targets = []
    for target_name, target_q in parked_postures():
        target_pose, target_attitude, _, _ = kin.evaluate(target_q)
        clear = check is None or check([target_q, target_q], objects,
                                       minimum_object_clearance=home_clearance_required)
        targets.append({'target': target_name, 'clearance_ok': bool(clear), 'posture': target_q.tolist()})
        if not clear:
            continue
        for height in heights:
            high_target = np.asarray(target_pose, float).copy()
            high_target[2] = height
            # Ending at the posture, and stopping at the high point above it: the second is what the project
            # did before ("park above that frame rather than pushing the result"), and it is still the right
            # answer when no posture is clear. Both are candidates; clearance decides.
            # The transit only needs *an* attitude the arm can hold up there; the final posture is commanded as
            # joints. Measured 2026-11-10: demanding the target posture's attitude at the high point rejected
            # every candidate for a canonical park target (57 IK_FAILUREs), which is the same over-specification
            # the earlier "free attitude lift" existed to avoid.
            enders = [(f'lift_then_{target_name}', target_q, None),
                      (f'lift_then_park_above_{target_name}', None, None),
                      (f'lift_fixed_{target_name}', target_q, target_attitude),
                      (f'high_{target_name}', target_q, None)]
            for label, end_q, transit_attitude in enders:
                candidates.append((label,
                                   [(start.copy(), rotation), (high_target, transit_attitude)], height, end_q))
            for name, direction in retreat_directions():
                for distance in (.03, .06):
                    retreat = start + direction * distance
                    candidates.append((f'retreat_{name}_{int(distance*1000)}mm_then_{target_name}',
                                       [(retreat, rotation), (high_target, None)], height, target_q))
                    candidates.append((f'retreat_{name}_{int(distance*1000)}mm_then_park_above_{target_name}',
                                       [(retreat, rotation), (high_target, None)], height, None))
    # Parking above a frame needs no clear posture - it ends in the air above it - so it is always available,
    # including when no posture could be left in. Measured 2026-11-10: generating targets only from *clear*
    # postures removed the previous fallback and made "no candidate at all" the answer.
    for height in heights:
        high_home = np.asarray(target, float).copy()
        high_home[2] = height
        candidates.append((f'park_above_home_{int(height*1000)}mm', [(start.copy(), rotation), (high_home, None)],
                           height, None))
        for name, direction in retreat_directions():
            retreat = start + direction * .05
            candidates.append((f'retreat_{name}_50mm_park_above_home',
                               [(retreat, rotation), (high_home, None)], height, None))
    if nominal_clear:
        candidates.append(('direct_home', [], float(target[2]), np.asarray(home, float)))
        candidates.append(('direct_park_above_home', [], float(target[2]), None))

    def clip_first(value):
        # An unmeasured clearance must not outrank a measured one, and must not be read as zero either.
        return float('-inf') if value is None else float(value)

    rejected, considered = [], []
    best = None
    for name, poses, height, target_q in candidates:
        if controller.stop_requested():
            return None, {'error':'STOPPED', 'rejected_routes':rejected, 'considered':considered}
        route = build(poses, target_q)
        if route is None:
            rejected.append({'route':name, 'height_m':height, 'error':'IK_FAILURE'})
            continue
        # The travelled route keeps the ordinary clearance; it is the *home target* that needs the bigger
        # margin, because that is the pose the arm has to land on.
        if check is not None and not check(route, objects, minimum_object_clearance=.03,
                                            allow_clearance_exit=True):
            rejected.append({'route':name, 'height_m':height, 'error':'COLLISION_PATH'})
            continue
        # The whole-path minimum is dominated by the final home posture, which every candidate shares, so it
        # cannot tell a retreat from a straight lift. The segment that actually brushes the object just
        # released is the first one, so that is measured separately - and it is what gets ranked first.
        detail = (getattr(kin, 'path_clearance_detail', None) or (lambda *a, **k: None))(route, objects, ceiling=.05)
        clearance_all = (measure(route, objects, ceiling=.05) if measure is not None else None)
        clearance_first = (measure(route[:2], objects, ceiling=.05)
                           if measure is not None and len(route) >= 2 else None)
        considered.append({'route':name, 'height_m':height, 'clearance_m':clearance_all,
                           'clearance_first_segment_m':clearance_first,
                           'tightest': detail})
        score = (clip_first(clearance_first), clip_first(clearance_all))
        if best is None or score > best[3]:
            best = (name, height, route, score, clearance_all, clearance_first)
    if best is None:
        return None, {'error':'HOME_PATH_UNAVAILABLE','rejected_routes':rejected,'considered':considered}
    name, height, route, _, clearance_all, clearance_first = best
    planner = getattr(env, 'plan_joint_waypoints', None)
    if planner is not None:
        planned = planner(arm, route, controller.stop_requested)
        if planned is None:
            rejected.append({'route':name, 'height_m':height, 'error':'COLLISION_PATH'})
            return None, {'error':'HOME_PATH_UNAVAILABLE','rejected_routes':rejected,'considered':considered}
        route = planned
    chosen_target = next((item for item in targets if name.endswith(item['target'])), None)
    return route, {'route':name,'clearance_height_m':height,'measured_clearance_m':clearance_first,
                   'park_target':None if chosen_target is None else chosen_target['target'],
                   'park_targets_considered':targets,
                   'measured_clearance_whole_path_m':clearance_all,
                   'tightest': next((item['tightest'] for item in considered
                                     if item['route'] == name and item['height_m'] == height), None),
                   'rejected_routes':rejected,'considered':considered,
                   'target_joint_positions':np.asarray(route[-1]).tolist(),
                   'home_mode':'model_home' if nominal_clear else 'scene_clearance_home',
                   'home_clearance_required_m':home_clearance_required,
                   'observed_positioning_error_m':motion_accuracy.observed_error(arm),
                   'home_adjustment_reason':None if nominal_clear
                   else f'Home needs {home_clearance_required*1000:.0f} mm of clearance for this body'}
