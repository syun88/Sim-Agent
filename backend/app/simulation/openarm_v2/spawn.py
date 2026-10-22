"""Check the real fingertip geometry, attitude and approach before cube setup."""
import numpy as np

from backend.app.control.motion import OpenArmV2Controller
from backend.app.simulation.openarm_v2.definition import FINGERTIP_MIDPOINT, GRASP_ROTATION
from backend.app.simulation.transform import orientation_error


def validate_spawn_positions(env, placements):
    controller = OpenArmV2Controller(env, configure_motors=False)
    cache = getattr(env, '_spawn_ik_cache', {})
    report = {}
    for name, point in placements.items():
        point = np.asarray(point, float)
        if point.shape != (3,) or not np.isfinite(point).all():
            raise ValueError(f'{name}: spawn needs three finite coordinates')
        side = 'left' if name == 'wine_bottle' or point[1] >= 0. else 'right'
        key = (side, *point)
        if key not in cache:
            pinch = point - GRASP_ROTATION @ FINGERTIP_MIDPOINT[side]
            standoff = pinch - .06 * (GRASP_ROTATION @ np.array([0., 0., -1.]))
            goals = [('pinch', pinch), ('pregrasp', standoff)]
            if name.endswith('_cube'):
                goals.append(('pregrasp_high', standoff + [0., 0., .10]))
            checks, seed = [], env.home_arm_qpos(side)
            kin = env.arm_kinematics(side)
            for label, target in goals:
                q, _, _ = controller.solve_ik(side, target, GRASP_ROTATION, q_init=seed)
                if q is None:
                    raise ValueError(f'{name}: {side} {label} is outside IK reach')
                actual, rotation, _, _ = kin.evaluate(q)
                error = float(np.linalg.norm(actual - target))
                angular_error = float(np.linalg.norm(orientation_error(rotation, GRASP_ROTATION)))
                if error > .003 or angular_error > .03:
                    raise ValueError(f'{name}: {side} {label} fails precise spawn IK ({error:.4f}m)')
                checks.append({'pose': label, 'target_position': target.tolist(), 'q_rad': q.tolist(),
                               'position_error': error, 'rotation_error': angular_error})
                seed = q
            cache[key] = {'position': point.tolist(), 'arm': side, 'checks': checks,
                          'scope': 'official fingertip pinch and approach IK; physical grasp verified during task'}
        report[name] = cache[key]
    env._spawn_ik_cache = cache
    return {'objects': report, 'method': 'private official-model IK before scene setup'}
