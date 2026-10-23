"""Measure layout diversity and IK admission, optionally with real Genesis grasps.

Each layout is an independent test case. A failed grasp remains a failed case;
resets between cases never convert an unsuccessful task into success.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--robot',default='openarm_v2')
    parser.add_argument('--layouts',type=int,default=16)
    parser.add_argument('--seed',type=int,default=20261023)
    parser.add_argument('--physics',action='store_true')
    parser.add_argument('--grasp',action='store_true',help='Execute one contact-verified pick per physics layout')
    parser.add_argument('--json',type=Path,required=True)
    args=parser.parse_args()
    if not 1<=args.layouts<=100:parser.error('--layouts must be 1–100')
    if args.seed<0 or args.seed+args.layouts-1>2**32-1:parser.error('Every layout seed must fit in 0–4294967295')
    if args.grasp and not args.physics:parser.error('--grasp requires --physics')
    os.environ['GENESIS_SHOW_VIEWER']='0'
    report={'robot':args.robot,'mode':'genesis_physics' if args.physics else 'private_kinematics',
            'layouts':[],'success':False,'seed_start':args.seed,'requested_layouts':args.layouts}
    args.json.parent.mkdir(parents=True,exist_ok=True)
    def save():args.json.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    env=None;save()
    try:
        if args.physics:
            from backend.app.simulation import create_backend
            from run_genesis_gui import make_executor
            env=create_backend(robot_id=args.robot);executor=make_executor(env)
        elif args.robot=='openarm_v2':
            from backend.app.simulation.openarm_v2.spawn import private_spawn_environment
            private=private_spawn_environment()
        else:
            from backend.app.robots.registry import get_profile
            from backend.app.robots.spawn import ReachableDeskSpawns
            profile=get_profile(args.robot);scene=json.loads((Path(profile['directory'])/'scene.json').read_text())
            sampler=ReachableDeskSpawns(profile,scene);base=sampler.layout()['objects']
            report['configuration_sha256']=profile['configuration_sha256']
        for index in range(args.layouts):
            seed=args.seed+index;started=time.perf_counter();row={'seed':seed,'success':False}
            try:
                if env is not None:
                    env.reset(randomize=True,seed=seed);env.step(200)
                    layout=env.spawn_validation
                    row['layout']=layout
                    row['settled_objects']={}
                    for name,entry in layout['objects'].items():
                        bounds=env.object_bounds(name)
                        robot_links=(set(env.model.links) if hasattr(env,'model') else
                            {link.name for link in env._entity.links if link.name.startswith('openarm_')})
                        links=set(env.object_contact_links(name))
                        observed={'position':env.object_pose(name)[0].tolist(),
                            'support_error_m':abs(float(bounds[0,2]-env.table_height)),
                            'speed_m_s':env.object_linear_speed(name),'robot_contacts':sorted(links&robot_links)}
                        row['settled_objects'][name]=observed
                        if observed['support_error_m']>.003 or observed['speed_m_s']>.02 or observed['robot_contacts']:
                            raise ValueError(f'{name} did not settle on the desk clear of the robot: {observed}')
                    row['layout_success']=True
                    if args.grasp:
                        from backend.app.planning.task_program import TaskProgram,TaskAction,compile_program
                        assignments=list(layout['objects'].items());name,entry=assignments[index%len(assignments)]
                        plan=compile_program(TaskProgram(actions=[TaskAction(op='pick',object=name,arm=entry['arm'])]),executor.world_state())
                        plan.provider='validation_program'
                        deadline=time.monotonic()+180.
                        result=executor.run(plan,should_stop=lambda:time.monotonic()>deadline)
                        row['physical_pick']=result
                        if not result['success']:raise ValueError('Physical pick failed; see physical_pick evidence')
                elif args.robot=='openarm_v2':
                    from backend.app.simulation.openarm_v2.spawn import random_spawn_layout
                    _,layout=random_spawn_layout(private,seed=seed)
                else:layout=sampler.layout(randomize=True,seed=seed,base=base)
                row.update(layout=layout,layout_success=True,success=True)
            except Exception as error:
                row['error']=str(error)
                if env is not None and env.spawn_validation.get('randomization',{}).get('seed')==seed:
                    row['layout']=env.spawn_validation
            row['wall_s']=time.perf_counter()-started;report['layouts'].append(row);save()
            print('LAYOUT',seed,row['success'],round(row['wall_s'],3),
                  {n:(v['arm'],np.round(v['position'][:2],3).tolist()) for n,v in row.get('layout',{}).get('objects',{}).items()},row.get('error',''),flush=True)
        names={name for row in report['layouts'] for name in row.get('layout',{}).get('objects',{})}
        report['coverage']={}
        for name in sorted(names):
            entries=[row['layout']['objects'][name] for row in report['layouts'] if name in row.get('layout',{}).get('objects',{})]
            xy=np.asarray([e['position'][:2] for e in entries])
            report['coverage'][name]={'span_x_m':float(np.ptp(xy[:,0])),'span_y_m':float(np.ptp(xy[:,1])),
                'unique_positions':len(np.unique(xy,axis=0)),'assigned_arms':sorted({e['arm'] for e in entries})}
        report['success']=all(row['success'] for row in report['layouts']) and len(report['layouts'])==args.layouts
        report['layout_success_count']=sum(row.get('layout_success',False) for row in report['layouts'])
        if args.grasp:report['physical_pick_success_count']=sum(row.get('physical_pick',{}).get('success',False) for row in report['layouts'])
        save();print('RESULT',report['success'],report['coverage'],flush=True)
        return 0 if report['success'] else 1
    finally:
        if env is not None:env.close()
        if args.physics:
            import genesis as gs
            if getattr(gs,'_initialized',False):gs.destroy()


if __name__=='__main__':raise SystemExit(main())
