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
    nominal_clear = check is None or check([home, home], objects, minimum_object_clearance=.03)

    def build(poses):
        """Solve a joint route through these (position, attitude) poses; None if IK cannot."""
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
        # Keep the canonical home when it is clear; otherwise park above that frame rather than erasing or
        # pushing the result to force the original configuration.
        if nominal_clear:
            route.append(home)
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
    for height in heights:
        high_target = target.copy()
        high_target[2] = height
        candidates.extend([
            ('lift_then_home', [(start.copy(), rotation), (high_target, target_rotation)], height),
            ('free_attitude_lift_then_home', [(start.copy(), None), (high_target, target_rotation)], height),
            ('high_home', [(high_target, target_rotation)], height),
        ])
        for name, direction in retreat_directions():
            for distance in (.03, .06):
                retreat = start + direction * distance
                candidates.append((f'retreat_{name}_{int(distance*1000)}mm_then_home',
                                   [(retreat, rotation), (high_target, target_rotation)], height))
    if nominal_clear:
        candidates.append(('direct_home', [], float(target[2])))

    def clip_first(value):
        # An unmeasured clearance must not outrank a measured one, and must not be read as zero either.
        return float('-inf') if value is None else float(value)

    rejected, considered = [], []
    best = None
    for name, poses, height in candidates:
        if controller.stop_requested():
            return None, {'error':'STOPPED', 'rejected_routes':rejected, 'considered':considered}
        route = build(poses)
        if route is None:
            rejected.append({'route':name, 'height_m':height, 'error':'IK_FAILURE'})
            continue
        if check is not None and not check(route, objects, minimum_object_clearance=.03,allow_clearance_exit=True):
            rejected.append({'route':name, 'height_m':height, 'error':'COLLISION_PATH'})
            continue
        # The whole-path minimum is dominated by the final home posture, which every candidate shares, so it
        # cannot tell a retreat from a straight lift. The segment that actually brushes the object just
        # released is the first one, so that is measured separately - and it is what gets ranked first.
        clearance_all = (measure(route, objects, ceiling=.05) if measure is not None else None)
        clearance_first = (measure(route[:2], objects, ceiling=.05)
                           if measure is not None and len(route) >= 2 else None)
        considered.append({'route':name, 'height_m':height, 'clearance_m':clearance_all,
                           'clearance_first_segment_m':clearance_first})
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
    return route, {'route':name,'clearance_height_m':height,'measured_clearance_m':clearance_first,
                   'measured_clearance_whole_path_m':clearance_all,
                   'rejected_routes':rejected,'considered':considered,
                   'target_joint_positions':np.asarray(route[-1]).tolist(),
                   'home_mode':'model_home' if nominal_clear else 'scene_clearance_home',
                   'home_adjustment_reason':None if nominal_clear else 'Current objects occupy the model home clearance'}
