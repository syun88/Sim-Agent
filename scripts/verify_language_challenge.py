"""Adversarial language checks on recorded Genesis snapshots (no physics execution).

Expectations are task semantics, not the planner's own output. Report each failure,
including accepted but wrong programs. Use --report-only for a baseline comparison.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def cases():
    holding = lambda obj, arm: {'type': 'holding', 'object': obj+'_cube', 'arm': arm}
    relative = lambda obj, target, relation='on': {'type': 'relative_placement', 'object': obj+'_cube', 'target': target+'_cube', 'relation': relation}
    placement = lambda obj, target: {'type': 'placement', 'object': obj+'_cube', 'target': target}
    return [
        {'id':'pick_polite', 'text':'青いやつ、右手でちょっと持ってもらえる？', 'checks':[holding('blue','right')]},
        {'id':'source_side_not_hand', 'text':'右側にある青いキューブを左手で取って', 'checks':[holding('blue','left')]},
        {'id':'source_left_not_hand', 'text':'左側の赤いキューブを右手で取って', 'checks':[holding('red','right')]},
        {'id':'source_side_en', 'text':'Pick up the blue cube on the right with your left hand.', 'checks':[holding('blue','left')]},
        {'id':'polite_stack', 'text':'赤を土台にして、その上に青を載せてもらえる？', 'checks':[relative('blue','red')]},
        {'id':'reversed_noun_order', 'text':'赤いキューブの上に青いキューブを乗せて', 'checks':[relative('blue','red')]},
        {'id':'stack_shorthand', 'text':'赤が下、青が上になるようにして', 'checks':[relative('blue','red')]},
        {'id':'stack_goal_en', 'text':'The red cube should support the blue one. Leave red where it is.', 'checks':[relative('blue','red')], 'forbid_objects':['red_cube']},
        {'id':'preserve_and_hold', 'text':'赤には触らないで、青だけ右手に持ってて', 'checks':[holding('blue','right')], 'forbid_objects':['red_cube']},
        {'id':'no_handover', 'text':'青を右手で掴んで。左手には渡さないで。', 'checks':[holding('blue','right')], 'forbid_ops':['handover']},
        {'id':'negation_en', 'text':'Grab blue with the right hand; do not pass it to the left.', 'checks':[holding('blue','right')], 'forbid_ops':['handover']},
        {'id':'place_subject_en', 'text':'Put the blue cube on the left side using the right hand.', 'checks':[placement('blue','left_side')], 'actions':[{'op':'place','object':'blue_cube','arm':'right','target':'left_side'}]},
        {'id':'left_of', 'text':'青を赤の左隣に置いて', 'checks':[relative('blue','red','left_of')]},
        {'id':'front_of', 'text':'赤の手前に青を置いて', 'checks':[relative('blue','red','front_of')]},
        {'id':'distance', 'text':'右手を上に10cm動かして', 'actions':[{'op':'move','arm':'right','direction':'up','distance_cm':10}]},
        {'id':'distance_units', 'text':'Move the right arm upward by 100 millimeters.', 'actions':[{'op':'move','arm':'right','direction':'up','distance_cm':10}]},
        {'id':'ordered', 'text':'右手で青を取って、左手に渡して、赤の上に置いて', 'checks':[relative('blue','red')], 'actions':[{'op':'pick','object':'blue_cube','arm':'right'},{'op':'handover','object':'blue_cube','arm':'right','receiver':'left'},{'op':'place','object':'blue_cube','arm':'left','target':'red_cube','relation':'on'}]},
        {'id':'reverse_order_en', 'text':'After picking up blue with the right hand, pass it to the left hand and put it on red.', 'checks':[relative('blue','red')], 'actions':[{'op':'pick','object':'blue_cube','arm':'right'},{'op':'handover','object':'blue_cube','arm':'right','receiver':'left'},{'op':'place','object':'blue_cube','arm':'left','target':'red_cube','relation':'on'}]},
        {'id':'hold_assignments', 'text':'青は左手、赤は右手で持ってる状態にして。手順は任せる。', 'checks':[holding('blue','left'),holding('red','right')]},
        {'id':'hold_assignments_en', 'text':'I want red in the right gripper and blue in the left gripper.', 'checks':[holding('blue','left'),holding('red','right')]},
        {'id':'swap_held', 'state':'held', 'text':'両手で持っている物を交換して', 'checks':[holding('blue','left'),holding('red','right')]},
        {'id':'swap_held_en', 'state':'held', 'text':'Switch which hand is holding each cube. You can use the table.', 'checks':[holding('blue','left'),holding('red','right')]},
        {'id':'held_reference', 'state':'held', 'text':'右手に持っているキューブを右側に置いて', 'checks':[placement('blue','right_side')], 'forbid_objects':['red_cube']},
        {'id':'discourse', 'state':'discourse', 'text':'それを左手に渡して', 'checks':[holding('blue','left')], 'forbid_objects':['red_cube']},
        {'id':'spatial_reference', 'text':'右にある方を右手で持って', 'checks':[holding('blue','right')]},
        {'id':'clear_support', 'state':'stacked', 'text':'青を右側に置いてから赤を左手で取って', 'checks':[placement('blue','right_side'),holding('red','left')], 'actions':[{'op':'place','object':'blue_cube','target':'right_side'},{'op':'pick','object':'red_cube','arm':'left'}]},
        {'id':'unknown_suffix', 'text':'青を右手で取って、そのあと机を拭いて', 'error':True},
        {'id':'unknown_prefix', 'text':'机を拭いてから青いキューブを掴んで', 'error':True},
        {'id':'unknown_object', 'text':'黄色いボールを取って', 'error':True},
        {'id':'group_grasp', 'state':'stacked', 'text':'積んだ二つを片手でまとめて掴んで', 'error':True},
        {'id':'contradictory_holders', 'text':'最後は青を左手と右手でそれぞれ一個ずつ持って', 'error':True},
        {'id':'vague_reference', 'text':'あれをそこに置いて', 'error':True},
    ]


def matches(actual, expected):
    return all(actual.get(k) == v for k, v in expected.items())


def holdout_cases():
    """Additional wording, kept separate from the initial 32-case diagnosis."""
    return [
        {'id':'holdout_hold','text':'青だけを右のグリッパーで保持。赤は動かさないこと。',
         'checks':[{'type':'holding','object':'blue_cube','arm':'right'}],'forbid_objects':['red_cube']},
        {'id':'holdout_actor','text':'左側に見える赤いブロック、つかむのは右手でお願い。',
         'checks':[{'type':'holding','object':'red_cube','arm':'right'}]},
        {'id':'holdout_right_of','text':"Put blue next to red, on red's right.",
         'checks':[{'type':'relative_placement','object':'blue_cube','target':'red_cube','relation':'right_of'}]},
        {'id':'holdout_behind','text':'青を赤の後ろへ。赤には触らずに。',
         'checks':[{'type':'relative_placement','object':'blue_cube','target':'red_cube','relation':'behind'}],'forbid_objects':['red_cube']},
        {'id':'holdout_stack_en','text':'Blue on red, please.',
         'checks':[{'type':'relative_placement','object':'blue_cube','target':'red_cube','relation':'on'}]},
        {'id':'holdout_stack_ja','text':'最後は赤を下、青を上にしておいて。',
         'checks':[{'type':'relative_placement','object':'blue_cube','target':'red_cube','relation':'on'}]},
        {'id':'holdout_spatial','text':'右にあるほうを左手で持って',
         'checks':[{'type':'holding','object':'blue_cube','arm':'left'}]},
        {'id':'holdout_current_hand','state':'held','text':'左手の物を中央に置いて',
         'checks':[{'type':'placement','object':'red_cube','target':'center'}],'forbid_objects':['blue_cube']},
    ]


def evaluate(plan, case):
    errors = []
    if case.get('error'):
        return ['unsupported or ambiguous request was accepted']
    checks = plan.goal.get('checks', [])
    for check in case.get('checks', []):
        if not any(matches(c, check) for c in checks):
            errors.append('missing final condition: '+str(check))
    actions = plan.program['actions']
    if 'actions' in case:
        if len(actions) != len(case['actions']) or not all(matches(a,b) for a,b in zip(actions,case['actions'])):
            errors.append('changed requested procedure/arguments')
    if any(a['op'] in case.get('forbid_ops', []) for a in actions):
        errors.append('executed a prohibited operation')
    if any(a.get('object') in case.get('forbid_objects', []) for a in actions):
        errors.append('manipulated a protected/nonrequested object')
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--physics-report', type=Path, default=Path('outputs/goal-planning-verified.json'))
    parser.add_argument('--json', type=Path, required=True)
    parser.add_argument('--case', action='append', help='Run just these case IDs')
    parser.add_argument('--suite',choices=['challenge','holdout'],default='challenge')
    parser.add_argument('--report-only', action='store_true')
    parser.add_argument('--model-only', action='store_true', help='Disable the deployed grammar fast path')
    args = parser.parse_args()
    from backend.app.planning.agent_planner import AgentPlanner, PlannerProviderError
    source = json.loads(args.physics_report.read_text())
    swap = next(c for c in source['cases'] if c['case']=='swap')
    stack = next(c for c in source['cases'] if c['case']=='stack')
    states = {'table':stack['execution']['initial_world_state'],
              'held':swap['preparation'][-1]['final_world_state'],
              'stacked':stack['execution']['final_world_state']}
    states['discourse'] = deepcopy(states['table'])
    states['discourse']['discourse'] = {'last_object':'blue_cube','last_arm':'right'}
    planner = AgentPlanner()
    planner.api_key = planner.base_url = planner.model = ''
    planner.local_enabled = planner.enabled = True
    planner.exact_grammar_fast_path = not args.model_only
    report = {'mode':'recorded_genesis_language_only', 'model':planner.local_model,
              'grammar_fast_path':planner.exact_grammar_fast_path, 'cases':[]}
    args.json.parent.mkdir(parents=True, exist_ok=True)
    for case in (cases() if args.suite=='challenge' else holdout_cases()):
        if args.case and case['id'] not in args.case:
            continue
        row = {**case, 'success':False}
        outputs = []
        extract = planner._extract_json
        def capture(text):
            value = extract(text)
            outputs.append(value)
            return value
        planner._extract_json = capture
        try:
            plan = planner.plan(case['text'], deepcopy(states[case.get('state','table')]))
            row['plan'] = plan.model_dump()
            row['errors'] = evaluate(plan, case)
            row['success'] = not row['errors']
        except PlannerProviderError as e:
            row.update(success=bool(case.get('error')), error_category=e.category, detail=str(e),
                       validation_feedback=planner._repair_feedback)
        finally:
            planner._extract_json = extract
        row['model_outputs'] = outputs
        report['cases'].append(row)
        report['passed'] = sum(c['success'] for c in report['cases'])
        report['total'] = len(report['cases'])
        report['success'] = report['passed']==report['total']
        args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
        print(case['id'], row['success'], row.get('errors',row.get('detail','')), flush=True)
    return 0 if args.report_only or report.get('success') else 1


if __name__ == '__main__':
    raise SystemExit(main())
