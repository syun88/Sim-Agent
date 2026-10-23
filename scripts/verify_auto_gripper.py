"""One Genesis world: inferred cube dimensions, physical opposed contact and lift."""
import argparse
import json
import os
from pathlib import Path
import sys
import traceback
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ['GENESIS_SHOW_VIEWER']='0'
from backend.app.simulation.genesis_backend.generic_environment import GenericURDFEnvironment
from backend.app.robots.executor import ImportedRobotExecutor

def main(env=None, ex=None, argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('robot',nargs='?',default='scorpion-77bd0313b2')
    parser.add_argument('--arm',choices=['left','right','both'],default='both')
    parser.add_argument('--layouts',type=int,choices=range(1,11),default=3)
    parser.add_argument('--seed',type=int,default=20261022)
    parser.add_argument('--start-layout',type=int,default=0)
    args=parser.parse_args(argv)
    robot=args.robot
    env=GenericURDFEnvironment(robot) if env is None else env
    ex=ImportedRobotExecutor(env) if ex is None else ex
    output=Path(env.profile['directory'])/'auto_grasp_validation.json'
    result={'robot':env.robot_id,'requested_robot':robot,'configuration_sha256':env.profile['configuration_sha256'],
            'model':env.model_summary(),'cube_size':env.cube_size,'cube_mass':env.cube_mass,'phases':[],'success':False}
    obj = next(iter(env._objects), None)
    peak = {'maximum_penetration_m': 0., 'sample_count': 0}
    def observe_collisions():
        report = env.self_collision_report()
        peak['sample_count'] += 1
        if report['maximum_penetration_m'] > peak['maximum_penetration_m']:
            peak.update(maximum_penetration_m=report['maximum_penetration_m'], pairs=report['pairs'])
    env.step_observer = observe_collisions
    try:
        env.reset()
        env.step(200)
        result['initial_self_collision'] = env.self_collision_report()
        print('Initial collision report:', result['initial_self_collision'], flush=True)
        assignments = [(entry['arm'], name) for name, entry in env.spawn_validation['objects'].items()
                       if args.arm=='both' or entry['arm']==args.arm]
        expected_arms = set(env.grippers) if args.arm=='both' else {args.arm}
        if not assignments or expected_arms-{side for side,_ in assignments}:
            raise RuntimeError('Every requested gripper needs a reachable cube for validation')
        result.update(requested_case_count=args.layouts*len(assignments),requested_arm=args.arm,
                      start_layout=args.start_layout,layout_count=args.layouts,random_seed=args.seed)
        for arm,obj,layout in [(s,o,i) for i in range(args.start_layout,args.start_layout+args.layouts) for s,o in assignments]:
            env.reset(randomize=layout>0,seed=args.seed+layout);env.step(200)
            if env.spawn_validation['objects'][obj]['arm']!=arm:
                obj=next(name for name,entry in env.spawn_validation['objects'].items() if entry['arm']==arm)
            result.setdefault('tested_layouts',{})[str(layout)]=env.spawn_validation
            expected=env.cube_size
            bounds=env.object_bounds(obj)
            result['measured_cube_size']=(bounds[1]-bounds[0]).tolist()
            if max(abs(bounds[1]-bounds[0]-expected))>.0005:
                raise RuntimeError('Collision cube size does not match automatic scale')
            for fn,call_args in [(ex.skills.approach_object,(arm,obj)),(ex.skills.grasp_object,(arm,obj)),
                            (ex.skills.lift_object,(arm,obj)),
                            (ex.skills.move_to_target,(arm,'right_side' if arm=='right' else 'left_side')),
                            (ex.skills.release_object,(arm,obj)),
                            (ex.skills.retreat,(arm,))]:
                phase=fn(*call_args);phase['arm']=arm;phase['object']=obj
                phase['layout_seed'] = args.seed+layout
                phase['self_collision'] = env.self_collision_report()
                phase['actual_arm_q'] = env.arm_qpos(arm).tolist()
                phase['object_position'] = env.object_pose(obj)[0].tolist()
                phase['robot_table_contacts'] = [pair for pair in env.contact_names()
                    if env._table.links[0].name in pair.split('|')
                    and any(link.name in pair.split('|') for link in env._entity.links)]
                phase['motor_effort'] = env._entity.get_dofs_control_force(env._dofs[arm]).cpu().numpy().tolist()
                phase['collision_peak'] = dict(peak)
                result['phases'].append(phase)
                print({k:phase.get(k) for k in ('skill','arm','success','duration','error','detail','object_rose','robot_table_contacts')},flush=True)
                output.write_text(json.dumps(result,indent=2))
                if not phase['success']:raise RuntimeError(str(phase))
                if peak['maximum_penetration_m'] > .0005:
                    raise RuntimeError('Robot self-collision penetration exceeded 0.5mm during the trajectory')
                if phase['skill']=='grasp_object':
                    from PIL import Image
                    center=env.object_pose(obj)[0]
                    env._camera.set_pose(pos=center+[.13,-.17,.10],lookat=center)
                    Image.fromarray(env.render()).save(output.with_name(f'grasp_{arm}_{len(result["phases"])}.png'))
            env.step(200)
            if env.object_contacts(obj):raise RuntimeError('Cube was not released')
            if env.object_linear_speed(obj)>.02:raise RuntimeError('Released cube is not stable')
            target=env.zone_position('right_side' if arm=='right' else 'left_side')
            actual=env.object_pose(obj)[0]
            error=float(np.linalg.norm(actual[:2]-target[:2]))
            contacts=env.object_contact_links(obj)
            if error>.008:raise RuntimeError(f'Released cube missed its planned placement by {error:.6f}m')
            if env._table.links[0].name not in contacts or abs(env.object_bounds(obj)[0,2]-env.table_height)>.002:
                raise RuntimeError('Released cube is not supported by the desk')
            result.setdefault('placements', []).append({'arm':arm,'object':obj,'layout_seed':args.seed+layout,
                'position':actual.tolist(),'target_position':target.tolist(),'horizontal_error_m':error,'contacts':contacts})
        if len(result.get('placements',[]))!=result['requested_case_count']:
            raise RuntimeError('Not every requested case completed')
        result['success']=True
    except Exception as exc:
        result['error']=str(exc);traceback.print_exc()
    finally:
        from PIL import Image
        Image.fromarray(env.render()).save(output.with_suffix('.png'))
        if obj is not None:
            center=env.object_pose(obj)[0]
            env._camera.set_pose(pos=center+[.20,-.20,.15],lookat=center)
            Image.fromarray(env.render()).save(output.with_name('auto_grasp_closeup.png'))
        result['collision_peak'] = peak
        output.write_text(json.dumps(result,indent=2))
        if os.getenv('ROBOT_VALIDATION_KEEP_OPEN') == '1':
            globals()['LIVE_ENV'], globals()['LIVE_EXECUTOR'] = env, ex
        else:
            env.close()
    return 0 if result['success'] else 1
if __name__=='__main__':
    status = main()
    if os.getenv('ROBOT_VALIDATION_KEEP_OPEN') != '1':
        raise SystemExit(status)
