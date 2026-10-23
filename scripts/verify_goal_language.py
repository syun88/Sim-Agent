"""Check actual Ollama interpretation against frozen, measured Genesis states.

This is language/plan evidence, not another physical execution. Generate the input
with verify_goal_planning.py first. Wrong goal bindings cannot pass by mode alone.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--physics-report', type=Path, default=Path('outputs/goal-planning-verified.json'))
    parser.add_argument('--json', type=Path, required=True)
    parser.add_argument('--case', choices=['all','exchange_en','exchange_ja','stack_goal','ordered','unsupported'], default='all')
    args = parser.parse_args()
    from backend.app.planning.agent_planner import AgentPlanner, PlannerProviderError
    source = json.loads(args.physics_report.read_text())
    swap = next(c for c in source['cases'] if c['case'] == 'swap')
    stack = next(c for c in source['cases'] if c['case'] == 'stack')
    held = swap['preparation'][-1]['final_world_state']
    table = stack['execution']['initial_world_state']
    planner = AgentPlanner()
    planner.api_key = planner.base_url = planner.model = ''
    planner.local_enabled = True
    planner.enabled = True
    planner.exact_grammar_fast_path = False
    cases = [
        ('exchange_en', 'Could you arrange things so that blue finishes in the left gripper and red in the right? Work out the intermediate steps.', held, 'holding'),
        ('exchange_ja', '両手の中身が逆になるようにして。途中の置き方は任せる。', held, 'holding'),
        ('stack_goal', '完成形は青が赤の上にある状態。赤の位置は変えないで。', table, 'stack'),
        ('ordered', '右手で青いキューブを取って左手に渡して左側の赤いキューブの上に乗せて', table, 'actions'),
        ('unsupported', '積んだ二つを同時に一つの手で掴んで渡して', table, 'error'),
    ]
    report = {'mode':'frozen_genesis_state_language_only', 'model':planner.local_model, 'cases':[], 'success':False}
    args.json.parent.mkdir(parents=True, exist_ok=True)
    def save():
        args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    for name, instruction, state, expected in cases:
        if args.case not in ('all', name):
            continue
        row = {'case':name, 'instruction':instruction, 'expected':expected, 'success':False}
        outputs = []
        extract = planner._extract_json
        def capture(text):
            value = extract(text)
            outputs.append(value)
            return value
        planner._extract_json = capture
        try:
            plan = planner.plan(instruction, state)
            row['plan'] = plan.model_dump()
            goals = (plan.goal_program or {}).get('goals', [])
            if expected == 'holding':
                correct = (len(goals)==2 and {(g['object'],g['arm']) for g in goals if g['type']=='holding'} == {('blue_cube','left'),('red_cube','right')})
            elif expected == 'stack':
                correct = (len(goals)==1 and goals[0]['type']=='relative_placement' and goals[0]['object']=='blue_cube'
                    and goals[0]['target']=='red_cube' and goals[0]['relation']=='on'
                    and 'red_cube' in plan.planning_evidence['protected_objects'])
            elif expected == 'actions':
                actions = plan.program['actions']
                correct = (not plan.goal_program and [a['op'] for a in actions]==['pick','handover','place']
                    and all(a['object']=='blue_cube' for a in actions)
                    and [(a['arm'],a.get('receiver')) for a in actions]==[('right',None),('right','left'),('left',None)]
                    and actions[-1]['target']=='red_cube' and actions[-1]['relation']=='on')
            else:
                correct = False
            row['success'] = correct and plan.provider=='local_ollama'
        except PlannerProviderError as error:
            row.update(error_category=error.category, success=expected=='error' and error.category=='UNSUPPORTED_TASK')
            row['validation_feedback'] = planner._repair_feedback
        finally:
            planner._extract_json = extract
        row['model_outputs'] = outputs
        report['cases'].append(row)
        save()
        print(name, row['success'], row.get('error_category'), row.get('plan',{}).get('provider'), flush=True)
    report['success'] = all(row['success'] for row in report['cases'])
    save()
    return 0 if report['success'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
