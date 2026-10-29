"""Satisfy machine-reported occupancy and support preconditions with real table placements.

This is above the strict compiler: callers can still use compile_program to test
whether a procedure is executable as stated. No object or hold state is written.
"""
from copy import deepcopy
from backend.app.planning.task_program import GroundingError, TABLE_TARGETS, TaskAction, TaskGoal, TaskProgram, compile_program, observed_holds


def buffer_clear(point, item, others):
    def half(obj, axis):
        b = obj.get('bounds')
        return (b[1][axis]-b[0][axis])/2 if b else .025
    return all(other['id']==item['id'] or other.get('held_by') or
        any(abs(point[i]-other['position'][i]) > half(item,i)+half(other,i)+.015 for i in (0,1))
        for other in others)


def projected_buffer_objects(state, actions, targets):
    """Known preceding setdowns occupy buffers; unknown motion cannot justify parking.

    This projection only chooses a preparation destination, never supplies a simulated
    scene state for verification or permits a placement through an unknown motion.
    """
    objects = deepcopy(state['objects'])
    by_id = {item['id']:item for item in objects}
    for action in actions:
        obj = by_id.get(action.object) if action.object else None
        if action.op == 'place':
            if obj is None or action.relation or (action.target or f'{action.arm}_side') not in targets:
                return None
            obj['position'] = list(targets[action.target or f'{action.arm}_side'])
            obj['held_by'] = None
        elif action.op in {'pick','lift','handover'}:
            if obj is None:
                return None
            obj['held_by'] = action.receiver or action.arm
        elif action.op in {'throw','pour','move','push','navigate','turn','rotate'}:
            # A moving object's later position cannot be inferred from this static plan.
            return None
    return objects


def compile_prepared_program(program: TaskProgram, state: dict):
    candidate = program.model_copy(deep=True)
    evidence = []
    if program.goals:
        subjects={g.object for g in program.goals}
        holds=observed_holds(state)
        targets={**TABLE_TARGETS,**{t['id']:t['position'] for t in state['targets']}}
        for goal in program.goals:
            blocker=holds.get(goal.arm) if goal.type=='holding' else None
            if not blocker or blocker in subjects or blocker in program.keep_in_place:
                continue
            item=next(o for o in state['objects'] if o['id']==blocker)
            if item.get('type','cube')!='cube':continue
            target=next((name for name in [goal.arm+'_side','center',*sorted(targets)]
                         if name in targets and buffer_clear(targets[name],item,state['objects'])),None)
            if target is None:continue
            candidate.goals.append(TaskGoal(type='placement',object=blocker,target=target))
            subjects.add(blocker)
            evidence.append({'kind':'free_hand_goal','arm':goal.arm,'object':blocker,'target':target,
                'explanation':f'{goal.arm}は{blocker}を保持しているため、{target}へ置いてから要求された保持目標を達成します。'})
    for _ in range(5):
        try:
            plan = compile_program(candidate, state)
            plan.planning_evidence = {**(plan.planning_evidence or {}), 'requested_program':program.model_dump()}
            if evidence:
                plan.planning_evidence = {**(plan.planning_evidence or {}), 'preparations':evidence}
                plan.assumptions += [e['explanation'] for e in evidence]
                # Preparation home must clear objects before changing posture.
                for step in plan.steps:
                    if step.skill=='return_home' and any(e.get('action_index') is not None and step.action_index==e['action_index']+1 for e in evidence):
                        step.skill='park_empty_arm'
            return plan
        except GroundingError as error:
            requirement = error.preparation
            if not requirement or requirement['kind'] not in {'free_hand','ground_held_reference'} or len(evidence)==4:
                raise
            obj, arm, index = (requirement[k] for k in ('object','arm','action_index'))
            item = next((o for o in state['objects'] if o['id']==obj), {})
            if item.get('type','cube') != 'cube' or obj in program.keep_in_place:
                raise
            targets = {**TABLE_TARGETS, **{t['id']:t['position'] for t in state['targets']}}
            projected = projected_buffer_objects(state, candidate.actions[:index], targets)
            if projected is None:
                raise GroundingError('PREPARATION_BLOCKED: earlier object motion has unknown destination',
                                     preparation=requirement) from error
            preferred = arm+'_side'
            target = next((name for name in [preferred,'center',*sorted(targets)] if name in targets and
                buffer_clear(targets[name],item,projected)), None)
            if target is None:
                raise GroundingError('PREPARATION_BLOCKED: no clear known table buffer', preparation=requirement) from error
            # Freeze 'it'/held references before adding another object's place:
            # preparation must not become the antecedent of the user's action.
            candidate.actions[index] = TaskAction.model_validate(requirement['resolved_action'])
            candidate.actions[index:index] = [TaskAction(op='place',object=obj,arm=arm,target=target),
                                               TaskAction(op='home',arm=arm)]
            for previous in evidence:
                if previous['action_index']>=index:
                    previous['action_index'] += 2
            side = '左手' if arm=='left' else '右手'
            evidence.append({**requirement, 'target':target,
                'explanation':f'{side}は{obj}を保持しているため、{target}へ置き、空の手を準備してから要求された作業を続けます。'})
    raise GroundingError('PREPARATION_BUDGET: too many missing preconditions')
