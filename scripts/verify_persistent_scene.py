"""Stack -> unstack/place -> restack in ONE Genesis scene, with no resets between prompts."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--json',type=Path,required=True)
    args=parser.parse_args()
    os.environ['GENESIS_SHOW_VIEWER']='0'
    os.environ['OPENARM_LOCAL_PLANNER']='0'
    os.environ['OPENARM_FAST_LANGUAGE']='1'
    from backend.app.agent.task_agent import TaskAgent
    from backend.app.planning.agent_planner import AgentPlanner
    from backend.app.simulation import create_backend
    from run_genesis_gui import make_executor
    report={'mode':'one_continuous_genesis_world','tasks':[],'reset_calls_between_tasks':0,'success':False}
    args.json.parent.mkdir(parents=True,exist_ok=True)
    def save():args.json.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    env=None
    try:
        env=create_backend(robot_id='openarm_v2')
        env.step(200)
        # Count any supervisor reset, rather than manufacturing a reset-free
        # result from unrelated standalone runs.
        reset=env.reset
        def counted_reset(*a,**kw):
            report['reset_calls_between_tasks']+=1
            return reset(*a,**kw)
        env.reset=counted_reset
        executor=make_executor(env)
        planner=AgentPlanner()
        planner.enabled=planner.local_enabled=False
        agent=TaskAgent(executor,planner)
        instructions=[
            '右手で青いキューブを取って左手に渡して左側の赤いキューブの上に乗せて',
            '左手で青いキューブを取って右手に渡して右側に置いて',
            '右手で青いキューブを取って左手に渡して左側の赤いキューブの上に乗せて',
        ]
        for text in instructions:
            deadline=time.monotonic()+180
            def emit(event,**payload):
                if event=='skill_finished':
                    r=payload['result'];print(r['skill'],r['success'],r.get('error'),r.get('detail',''),flush=True)
                elif event=='agent_decision':print(event,payload['agent_trace'][-1]['explanation'],flush=True)
            result=agent.run(text,emit=emit,should_stop=lambda:time.monotonic()>deadline)
            row={'instruction':text,'result':result,'success':result['success']}
            from numpy.linalg import norm
            import numpy as np
            homes={a['arm']:a.get('target_joint_positions',env.home_arm_qpos(a['arm']))
                   for a in result.get('post_task_homing',{}).get('arms',[])}
            row['home_joint_error_rad']={s:float(np.max(abs(env.arm_qpos(s)-homes.get(s,env.home_arm_qpos(s))))) for s in ('left','right')}
            row['home_modes']={a['arm']:a.get('home_mode','already_home') for a in result.get('post_task_homing',{}).get('arms',[])}
            row['success']=row['success'] and all(v<.025 for v in row['home_joint_error_rad'].values())
            if report['tasks']:
                previous={o['id']:o['position'] for o in report['tasks'][-1]['result']['final_world_state']['objects']}
                row['inherited_position_errors_m']={o['id']:float(norm(np.asarray(o['position'])-previous[o['id']]))
                    for o in result['initial_world_state']['objects']}
                row['success']=row['success'] and all(v<.001 for v in row['inherited_position_errors_m'].values())
            report['tasks'].append(row)
            save()
            print('TASK',len(report['tasks']),row['success'],'home',row['home_joint_error_rad'],flush=True)
            if not row['success']:break
        report['success']=(len(report['tasks'])==3 and all(r['success'] for r in report['tasks'])
                           and report['reset_calls_between_tasks']==0)
        save()
        return 0 if report['success'] else 1
    finally:
        save()
        if env is not None:env.close()
        import genesis as gs
        if getattr(gs,'_initialized',False):gs.destroy()


if __name__=='__main__':raise SystemExit(main())
