"""Compare direct planning with observed goal search, using one real Genesis world.

Scene resets are only between independent cases. Every preparation is a physical
pick; the goal executor is never given a fabricated hold or object pose.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--robot', default='openarm_v2')
    parser.add_argument('--case', choices=['swap', 'stack', 'all'], default='all')
    parser.add_argument('--provider', choices=['fallback', 'local'], default='fallback')
    parser.add_argument('--json', type=Path, required=True)
    args = parser.parse_args()
    os.environ['GENESIS_SHOW_VIEWER'] = '0'
    os.environ['OPENARM_FAST_LANGUAGE'] = '1'
    os.environ['OPENARM_LOCAL_PLANNER'] = '1' if args.provider == 'local' else '0'
    from backend.app.planning.agent_planner import AgentPlanner
    from backend.app.planning.task_program import GroundingError, TaskAction, TaskGoal, TaskProgram, compile_program
    from backend.app.simulation import create_backend
    from run_genesis_gui import make_executor
    report = {'robot': args.robot, 'cases': [], 'success': False}
    args.json.parent.mkdir(parents=True, exist_ok=True)
    def save():
        args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    env = None
    try:
        env = create_backend(robot_id=args.robot)
        executor = make_executor(env)
        planner = AgentPlanner()
        planner.api_key = planner.base_url = planner.model = ''
        planner.exact_grammar_fast_path = args.provider != 'local'
        for case in (['swap', 'stack'] if args.case == 'all' else [args.case]):
            env.reset()
            executor.reset_context()
            env.step(200)
            row = {'case': case, 'success': False, 'preparation': []}
            report['cases'].append(row)
            deadline = time.monotonic()+240.
            def emit(event, **payload):
                if event == 'skill_finished':
                    result = payload['result']
                    print(event, result.get('skill'), result.get('arm'), result['success'], result.get('detail', ''), flush=True)
                elif event == 'goal_replanned':
                    print(event, payload['plan']['program']['actions'], flush=True)
            if case == 'swap':
                for side, obj in [('left', 'red_cube'), ('right', 'blue_cube')]:
                    pick = compile_program(TaskProgram(actions=[TaskAction(op='pick', object=obj, arm=side)]), executor.world_state())
                    preparation = executor.run(pick, emit=emit, should_stop=lambda: time.monotonic() > deadline)
                    row['preparation'].append(preparation)
                    save()
                    if not preparation['success']:
                        break
                if not all(result['success'] for result in row['preparation']):
                    row['detail'] = 'Physical preparation failed'
                    continue
                try:
                    compile_program(TaskProgram(actions=[TaskAction(op='handover', object='blue_cube', arm='right', receiver='left')]), executor.world_state())
                    row['direct_plan_rejected'] = False
                except GroundingError as error:
                    row['direct_plan_rejected'] = True
                    row['direct_plan_error'] = str(error)
                instruction = ('I want each of the cubes to finish in the other hand. Work out any temporary placements yourself.'
                               if args.provider == 'local' else '両手のキューブを入れ替えて')
                plan = planner.plan(instruction, executor.world_state())
            else:
                plan = compile_program(TaskProgram(goals=[TaskGoal(type='relative_placement', object='blue_cube', target='red_cube', relation='on')], keep_in_place=['red_cube']), executor.world_state())
            row['plan'] = plan.model_dump()
            save()
            result = executor.run(plan, emit=emit, should_stop=lambda: time.monotonic() > deadline)
            row['execution'] = result
            row['success'] = result['success']
            if args.provider == 'local' and case == 'swap':
                row['success'] = row['success'] and plan.provider == 'local_ollama' and bool(plan.goal_program)
            save()
            print('CASE', case, row['success'], result.get('detail'), flush=True)
        report['success'] = all(row['success'] for row in report['cases'])
        save()
        print('RESULT', report['success'], args.json, flush=True)
        return 0 if report['success'] else 1
    finally:
        save()
        if env is not None:
            env.close()
        import genesis as gs
        if getattr(gs, '_initialized', False):
            gs.destroy()


if __name__ == '__main__':
    raise SystemExit(main())
