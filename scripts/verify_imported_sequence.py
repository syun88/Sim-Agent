"""Run one language task through the normal imported-robot executor and physics."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from backend.app.planning.agent_planner import AgentPlanner
from backend.app.robots.executor import ImportedRobotExecutor
from backend.app.simulation import create_backend


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('robot_id')
    parser.add_argument('--instruction',default='右手で青いキューブを取って左手に渡して左側の赤いキューブの上に乗せて')
    parser.add_argument('--json',type=Path,required=True)
    parser.add_argument('--timeout',type=float,default=600.)
    parser.add_argument('--image',type=Path,help='Render the measured final stack after execution')
    args=parser.parse_args()
    os.environ['GENESIS_SHOW_VIEWER']='0'
    os.environ.setdefault('OPENARM_FAST_LANGUAGE','1')
    env=None;report={'instruction':args.instruction,'success':False,'status':'starting','pid':os.getpid()}
    args.json.parent.mkdir(parents=True,exist_ok=True)
    def save():args.json.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    save()
    try:
        env=create_backend(robot_id=args.robot_id);env.step(200)
        executor=ImportedRobotExecutor(env);planner=AgentPlanner()
        plan=planner.plan(args.instruction,executor.world_state())
        print('PLAN',plan.provider,[(s.skill,s.arm,s.object,s.target) for s in plan.steps],flush=True)
        report.update(profile_version=env.profile['profile_version'],configuration_sha256=env.profile['configuration_sha256'],
            model=env.model_summary(),actuators=env.profile.get('actuator_configuration'),moves=[],
            observed_steps=0,peak_observed_self_penetration_m=0.,commanded_joint_velocity_peak_rad_s={})
        original_move=executor.controller.move_to_site
        original_set_arm=env.set_arm
        def set_arm(side,target,velocities=None):
            if velocities is not None:
                peak=max(abs(velocities),default=0.)
                report['commanded_joint_velocity_peak_rad_s'][side]=max(report['commanded_joint_velocity_peak_rad_s'].get(side,0.),float(peak))
            return original_set_arm(side,target,velocities)
        env.set_arm=set_arm
        def move(*a,**kw):
            started=time.perf_counter();step=env._sim_step
            result=original_move(*a,**kw)
            data={'arm':a[0],'result':vars(result),'wall_s':time.perf_counter()-started,'sim_s':(env._sim_step-step)*env.timestep}
            if kw.get('planned_joints') is not None:
                import numpy as np
                from backend.app.control.motion import joint_spline,maximum_spline_slope
                path=kw['planned_joints'];curve=joint_spline(path,env.arm_velocity_limits(a[0])) if len(path)>2 else None
                data['planned_end_joints']=np.asarray(path[-1]).tolist()
                slope=maximum_spline_slope(curve) if curve is not None else abs(np.asarray(path[-1])-path[0])
                previous=sum(.56+max(14,int(np.linalg.norm(np.asarray(b)-q)/.005))*env.timestep for q,b in zip(path,path[1:]))
                data['previous_timing_on_same_path_s']=float(max(previous/6.,max(1.875*slope/2.,default=0.)))
            report['moves'].append(data)
            print('MOVE',a[0],result.success,result.trajectory_duration,round(data['wall_s'],3),flush=True)
            return result
        executor.controller.move_to_site=move
        first_step=env._sim_step;last_checked=first_step
        observation_stride=max(1,round(.02/env.timestep))
        report['self_collision_observation_period_s']=observation_stride*env.timestep
        report['self_collision_observation_count']=0
        def observe():
            nonlocal last_checked
            # Planning also pulses the native step observer for lease/stop
            # responsiveness. Count only newly advanced Genesis physics steps.
            report['observed_steps']=env._sim_step-first_step
            if env._sim_step-last_checked>=observation_stride:
                report['peak_observed_self_penetration_m']=max(report['peak_observed_self_penetration_m'],env.self_collision_report()['maximum_penetration_m'])
                report['self_collision_observation_count']+=1;last_checked=env._sim_step
        env.step_observer=observe
        def emit(event,**payload):
            if event=='skill_finished':
                r=payload['result'];print('SKILL',r['skill'],r['success'],r.get('error'),r.get('detail'),flush=True)
        deadline=time.monotonic()+args.timeout
        result=executor.run(plan,emit=emit,should_stop=lambda:time.monotonic()>deadline)
        report.update(success=result['success'],status=result['status'],result=result)
        if args.image:
            from PIL import Image
            import numpy as np
            center=np.mean([env.object_pose(name)[0] for name in env._objects],axis=0)
            env._camera.set_pose(pos=center+[.20,-.25,.18],lookat=center)
            args.image.parent.mkdir(parents=True,exist_ok=True)
            Image.fromarray(env.render()).save(args.image)
            report['final_image']=str(args.image.resolve())
        save();print('RESULT',result['success'],result['verification'],flush=True)
        return 0 if result['success'] else 1
    except BaseException as error:
        report.update(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'error',error=repr(error));save();raise
    finally:
        if env is not None:env.close()
        import genesis as gs
        if getattr(gs,'_initialized',False):gs.destroy()


if __name__=='__main__':raise SystemExit(main())
