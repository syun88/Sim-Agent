"""Home routes from the measured posture and current objects, without physics writes."""
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
    # A retreat can leave the wrist close to the base: a vertical lift with its
    # current attitude may be unreachable even though a high home route exists.
    # Compile and screen complete alternatives BEFORE moving the live robot.
    heights = list(dict.fromkeys([target[2]+clearance,
                                 max(start[2], target[2])+clearance, target[2]+clearance/2]))
    check = getattr(kin, 'empty_arm_path_is_clear', None)
    nominal_clear = check is None or check([home, home], objects, minimum_object_clearance=.03)
    candidates = []
    for height in heights:
        high_start, high_target = start.copy(), target.copy()
        high_start[2] = high_target[2] = height
        candidates.extend([
            ('lift_then_home', [(high_start, rotation), (high_target, target_rotation)], height),
            ('free_attitude_lift_then_home', [(high_start, None), (high_target, target_rotation)], height),
            ('high_home', [(high_target, target_rotation)], height),
        ])
    if nominal_clear:
        candidates.append(('direct_home', [], float(target[2])))
    rejected = []
    for name, poses, height in candidates:
        if controller.stop_requested():
            return None, {'error':'STOPPED', 'rejected_routes':rejected}
        route = [start_q]
        for point, attitude in poses:
            solved = None
            if attitude is None:
                # An empty arm may rotate while clearing vertically. Fixing its
                # original wrist attitude can make a perfectly reachable lift
                # impossible near the base of a redundant arm.
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
                break
            route.append(solved)
        else:
            if nominal_clear:
                route.append(home)
            # Keep the canonical home when possible. If the task's current
            # objects occupy it, park above that frame instead of erasing or
            # pushing the result to force the original configuration.
            # The official model query includes both arms, the cell, and the
            # latest object poses. It never integrates a second simulation.
            if check is not None and not check(route, objects, minimum_object_clearance=.03,allow_clearance_exit=True):
                rejected.append({'route':name, 'height_m':height, 'error':'COLLISION_PATH'})
                continue
            planner = getattr(env, 'plan_joint_waypoints', None)
            if planner is not None:
                route = planner(arm, route, controller.stop_requested)
                if route is None:
                    rejected.append({'route':name, 'height_m':height, 'error':'COLLISION_PATH'})
                    continue
            return route, {'route':name,'clearance_height_m':height,'rejected_routes':rejected,
                           'target_joint_positions':np.asarray(route[-1]).tolist(),
                           'home_mode':'model_home' if nominal_clear else 'scene_clearance_home',
                           'home_adjustment_reason':None if nominal_clear else 'Current objects occupy the model home clearance'}
        rejected.append({'route':name,'height_m':height,'error':'IK_FAILURE'})
    return None, {'error':'HOME_PATH_UNAVAILABLE','rejected_routes':rejected}
