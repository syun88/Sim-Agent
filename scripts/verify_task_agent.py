"""Real Genesis checks for missing preconditions and same-scene failed-layout recovery.

No contact/hold is mocked. Layout seed 20261023 previously failed physical pickup;
the recovery case records its actual attempts whether it reproduces or succeeds.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--json',type=Path,required=True)
    parser.add_argument('--local-reasoning',action='store_true')
    parser.add_argument('--case',choices=['empty_hand','occupied_receiver','goal_receiver','failed_layout','all'],default='all')
    args=parser.parse_args()
    os.environ['GENESIS_SHOW_VIEWER']='0'
    os.environ['OPENARM_FAST_LANGUAGE']='1'
    os.environ['OPENARM_LOCAL_PLANNER']='1' if args.local_reasoning else '0'
    from backend.app.agent.task_agent import TaskAgent
    from backend.app.planning.agent_planner import AgentPlanner
    from backend.app.simulation import create_backend
    from run_genesis_gui import make_executor
    report={'mode':'actual_genesis_physics','cases':[],'success':False}
    args.json.parent.mkdir(parents=True,exist_ok=True)
    def save():
        args.json.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    env=None
    try:
        env=create_backend(robot_id='openarm_v2')
        executor=make_executor(env)
        planner=AgentPlanner()
        planner.api_key=planner.base_url=planner.model=''
        agent=TaskAgent(executor,planner)
        for name in ['empty_hand','occupied_receiver','goal_receiver','failed_layout']:
            if args.case not in ('all',name): continue
            env.reset(randomize=name=='failed_layout',seed=20261023 if name=='failed_layout' else None)
            executor.reset_context()
            env.step(200)
            row={'case':name,'success':False}
            report['cases'].append(row)
            deadline=time.monotonic()+240
            def emit(event,**payload):
                if event=='agent_decision': print(event,payload['agent_trace'][-1],flush=True)
                if event=='skill_finished':
                    r=payload['result']; print(r['skill'],r['success'],r.get('error'),r.get('detail',''),flush=True)
            if name in {'occupied_receiver','goal_receiver'}:
                prep=executor.run(planner.plan('左手で赤いキューブを掴んで',executor.world_state()),should_stop=lambda:time.monotonic()>deadline)
                row['preparation']=prep
                save()
                if not prep['success']: continue
            text='青いキューブを掴んで' if name=='failed_layout' else '右手にあるキューブを取って左手に渡して'
            active_agent=agent
            if name=='goal_receiver':
                # Isolate preparation/search physics from free-language quality.
                from backend.app.planning.preparation import compile_prepared_program
                from backend.app.planning.task_program import TaskProgram
                text='青を左手で保持する最終目標'
                active_agent=TaskAgent(executor,planner,plan_call=lambda text,state,memory:compile_prepared_program(
                    TaskProgram(goals=[{'type':'holding','object':'blue_cube','arm':'left'}]),state))
                row['interpretation_mode']='typed_goal_physics_only'
            result=active_agent.run(text,emit=emit,should_stop=lambda:time.monotonic()>deadline)
            row.update(instruction=text,result=result,success=result['success'])
            row['recovery_reproduced']=bool(result.get('agent_recoveries'))
            row['scene_resets']=result['verification'].get('scene_resets',0)
            if name=='failed_layout':
                row['success']=row['success'] and row['scene_resets']==0
            save()
            print('CASE',name,row['success'],'resets',row['scene_resets'],flush=True)
        report['success']=all(c['success'] for c in report['cases'])
        save()
        return 0 if report['success'] else 1
    finally:
        save()
        if env is not None:env.close()
        import genesis as gs
        if getattr(gs,'_initialized',False):gs.destroy()


if __name__=='__main__':raise SystemExit(main())
