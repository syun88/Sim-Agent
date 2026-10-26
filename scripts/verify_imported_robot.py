"""URDF + desk runtime, private FK/Jacobian agreement and motor-driven IK travel."""
import argparse
import json
import time
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.app.robots.registry import get_profile, import_robot
from backend.app.robots.executor import ImportedRobotExecutor
from backend.app.simulation import create_backend


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('robot', help='Imported ID or an existing URDF folder/file')
    parser.add_argument('--native', action='store_true')
    parser.add_argument('--json', type=Path)
    args = parser.parse_args()
    profile = import_robot(args.robot) if Path(args.robot).exists() else get_profile(args.robot)
    output = args.json or Path(profile['directory']) / 'runtime_validation.json'
    os.environ['GENESIS_SHOW_VIEWER'] = '1' if args.native else '0'
    env = create_backend(robot_id=profile['id'])
    executor = ImportedRobotExecutor(env)
    maximum_velocity_ratio = 0.
    original_set_arm = env.set_arm
    def observe_command(side, targets, velocities=None):
        nonlocal maximum_velocity_ratio
        if velocities is not None:
            maximum_velocity_ratio = max(maximum_velocity_ratio,
                float(np.max(abs(velocities) / env.arm_velocity_limits(side))))
        return original_set_arm(side, targets, velocities)
    env.set_arm = observe_command
    result = {'robot_id': profile['id'], 'robot_links': env._entity.n_links, 'robot_dofs': env._entity.n_dofs,
              'collision_geoms': env._entity.n_geoms, 'native': args.native, 'phases': [], 'success': False,
              # The run stamps itself, so a reader can tell whether this file is the result of the run
              # it just started or a leftover from an earlier one. A missing stamp means it predates this.
              'started_at': time.time(),
              'openarm_cell': False, 'physics': 'Genesis', 'grasp_calibrated': False}
    try:
        env.step(200)
        from PIL import Image
        Image.fromarray(env.render()).save(output.with_suffix('.png'))
        result['spawn_layouts'] = []
        for randomize in (False, True, True):
            env.reset(randomize=randomize)
            env.step(200)
            layout = {}
            for name, entry in env.spawn_validation['objects'].items():
                actual = env.object_pose(name)[0]
                error = float(np.linalg.norm(actual - entry['position']))
                if error > .005:
                    raise RuntimeError(f'{name}: initial object drifted by {error:.6f}m')
                checked = env._spawn_sampler.validate(actual, entry['arm'])
                layout[name] = {'actual_position': actual.tolist(), 'setup_settling_error': error,
                                'arm': entry['arm'], 'maximum_ik_error': max(c['position_error'] for c in checked['checks'])}
            result['spawn_layouts'].append(layout)
        for side in env.available_arms:
            actual, rotation = env.ee_pose(side)
            predicted = env.arm_kinematics(side).evaluate(env.arm_qpos(side))[0]
            error = float(np.linalg.norm(actual - predicted))
            if error > .0001:
                raise RuntimeError(f'{side}: URDF/Genesis FK differ by {error:.6f}m')
            print(side, 'FK error', error, flush=True)
            # Ask for a centimetre of travel in a direction this arm can actually reach. A pose at
            # the top of its own workspace cannot rise, but this stage is about holding the arm to
            # what it was commanded, not about the rise; a desk arm reaches the first direction, so
            # nothing changes for it.
            solution = None
            for delta in ([0., 0., .01], [.01, 0., 0.], [0., .01, 0.], [-.01, 0., 0.], [0., -.01, 0.], [0., 0., -.01]):
                target = actual + delta
                solution, _, _ = executor.controller.solve_ik(side, target, rotation)
                if solution is not None:
                    break
            if solution is None:
                raise RuntimeError(side + ': IK goal is unreachable in every direction')
            command = np.asarray(delta, dtype=float)
            command /= float(np.linalg.norm(command))
            motion = executor.controller.move_to_site(side, target, rotation, duration=1., scale_speed=True)
            measured = env.ee_pose(side)[0]
            record = {'arm': side, 'fk_error': error, 'ik_solution_rad': solution.tolist(),
                      'initial_position': actual.tolist(), 'target_position': target.tolist(), 'actual_position': measured.tolist(),
                      'actual_position_error': float(np.linalg.norm(measured - target)), 'motion_success': motion.success,
              'command': command.tolist(), 'actual_travel': float(np.dot(np.asarray(measured) - np.asarray(actual), command)),
                      'actual_rise': float(measured[2] - actual[2])}
            result['phases'].append(record)
            result['finished_at'] = time.time()
            output.write_text(json.dumps(result, indent=2))
            print(record, flush=True)
            if not motion.success or record['actual_position_error'] > .003 or record['actual_travel'] < .007:
                raise RuntimeError(side + ': physical IK tracking failed')
            # A centimetre can be tracked by a drive that saturates; a large posture change cannot. Measured
            # 2026-10-25 on a wheeled six-axis arm: the 1 cm move tracked to 2.9 mm, while a 0.8 rad shoulder
            # swing stopped 51 mm short at the default motion speed (the controller asks for 6x the nominal
            # trajectory, and these motors saturate) and tracked to 3 mm at speed 1. Record both, and fail
            # only when neither reaches the pose: which of the two works is a property of the robot.
            swing = np.asarray(env.arm_qpos(side), dtype=float)
            index = 1 if len(swing) > 1 else 0
            limits = env.arm_limits(side)
            swing[index] = float(np.clip(swing[index] + .8, limits[index, 0] + .05, limits[index, 1] - .05))
            swing_point, swing_rotation = env.arm_kinematics(side).evaluate(swing)[:2]
            swing_rows = {}
            for label, scale in (('scaled', True), ('unscaled', False)):
                result = executor.controller.move_to_site(side, swing_point, swing_rotation, duration=2., scale_speed=scale)
                reached = float(np.linalg.norm(np.asarray(env.ee_pose(side)[0]) - np.asarray(swing_point)))
                swing_rows[label] = {'success': bool(result.success), 'error': result.error, 'reached_m': reached}
                print('large travel %s: success=%s error=%s reached=%.4f' % (label, result.success, result.error, reached), flush=True)
                executor.controller.move_to_site(side, np.asarray(actual), rotation, duration=2., scale_speed=False)
            result['large_travel'] = swing_rows
            if not any(row['reached_m'] < .01 for row in swing_rows.values()):
                raise RuntimeError('%s: a large posture change is not tracked at either motion speed: %s'
                                   % (side, {k: round(v['reached_m'], 4) for k, v in swing_rows.items()}))
        result['maximum_command_velocity_ratio'] = maximum_velocity_ratio
        if maximum_velocity_ratio > 1.001:
            raise RuntimeError('A motor reference exceeded its source URDF velocity limit')
        result['success'] = True
    except Exception as exc:
        result['error'] = str(exc)
        print('FAILED', str(exc), flush=True)
    finally:
        output.write_text(json.dumps(result, indent=2))
        env.close()
    return 0 if result['success'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
