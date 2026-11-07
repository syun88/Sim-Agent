"""Instruction relations must survive every provider and reach the success oracle."""
import pytest

from backend.app.planning.agent_planner import AgentPlanner, PlannerProviderError


WORLD = {'objects': [{'id': 'red_cube', 'position': [.44, .15, 1.031]},
                     {'id': 'blue_cube', 'position': [.5, -.22, 1.031]}], 'targets': []}


@pytest.fixture(autouse=True)
def no_provider(monkeypatch):
    for key in ('AI_AND_API_KEY', 'AI_AND_BASE_URL', 'AI_AND_MODEL',
                'OPENARM_LOCAL_PLANNER', 'OPENARM_FAST_LANGUAGE', 'OPENARM_CAPABILITY_MODE'):
        monkeypatch.delenv(key, raising=False)


def test_pronoun_support_is_grounded_without_substituting_a_table_side():
    # An intermediate acquisition may be set down, but the requested final support cannot be lost.
    plan=AgentPlanner().plan('赤いキューブを左手で持ち上げて、その上に右手で青いキューブを重ねて', WORLD)
    assert plan.planning_evidence['preparations'][0]['kind']=='ground_held_reference'
    assert any(a['op']=='place' and a['object']=='blue_cube' and a['target']=='red_cube'
               and a['relation']=='on' for a in plan.program['actions'])
    assert not any(c['type']=='holding' and c['object']=='red_cube' for c in plan.goal['checks'])


@pytest.mark.parametrize('text,support,subject,side,receiver', [
    ('青いキューブを右手に渡して右においてから、赤いキューブを青いキューブに乗せて',
     'blue_cube','red_cube','right_side','right'),
    ('赤いキューブを左手に渡して左においてから、青いキューブを赤いキューブに乗せて',
     'red_cube','blue_cube','left_side','left'),
])
def test_sequential_set_down_releases_support_before_stack(text,support,subject,side,receiver):
    # A completed handover is an intermediate step; its holder is not a final goal.
    plan=AgentPlanner().plan(text,WORLD)
    actions=plan.program['actions']
    assert [(a['op'],a['object']) for a in actions]==[
        ('handover',support),('place',support),('place',subject)]
    assert actions[0]['receiver']==receiver and actions[1]['target']==side
    assert actions[2]['target']==support and actions[2]['relation']=='on'
    assert not any(c['type']=='holding' and c['object']==support for c in plan.goal['checks'])
    assert any(c['type']=='relative_placement' and c['object']==subject
               and c['target']==support and c['relation']=='on' for c in plan.goal['checks'])
    assert any(s.skill=='release_object' and s.object==support for s in plan.steps)


@pytest.mark.parametrize('source,receiver', [('left','right'),('right','left')])
def test_unspecified_handover_hand_uses_the_other_observed_robot_arm(source,receiver):
    # Live failure c3d74b0f: grammar rejected an omitted recipient, then an invalid model
    # "move" hid the actual problem behind an irrelevant direction/distance error.
    world={**WORLD,'robot':{'available_arms':['left','right'],
        'left_arm':{'held_object':'red_cube' if source=='left' else None},
        'right_arm':{'held_object':'red_cube' if source=='right' else None}}}
    text='赤いキューブを手に渡して右においてから、青いキューブを赤いキューブに乗せて'
    plan=AgentPlanner().plan(text,world)
    assert [(a['op'],a['object']) for a in plan.program['actions']]==[
        ('handover','red_cube'),('place','red_cube'),('place','blue_cube')]
    assert plan.program['actions'][0]['arm']==source
    assert plan.program['actions'][0]['receiver']==receiver
    assert plan.program['actions'][1]['target']=='right_side'
    assert plan.program['actions'][2]['target']=='red_cube'
    assert plan.program['actions'][2]['relation']=='on'
    assert any('recipient hand unspecified' in assumption.lower()
               for assumption in plan.assumptions)
    assert not any(a['op']=='move' for a in plan.program['actions'])


def test_placed_support_updates_projected_position_and_avoids_unnecessary_handover():
    # When red_cube is placed at right_side, its projected position is on the right.
    # blue_cube is also on the right. Stacking blue on red must be performed directly
    # by the right arm without an absurd handover from right to left across the table!
    text = '赤いキューブを手に渡して右においてから、青いキューブを赤いキューブに乗せて'
    plan = AgentPlanner().plan(text, WORLD)
    actions = plan.program['actions']
    assert [(a['op'], a['object']) for a in actions] == [
        ('handover', 'red_cube'), ('place', 'red_cube'), ('place', 'blue_cube')]
    assert actions[1]['target'] == 'right_side'
    assert actions[2]['target'] == 'red_cube' and actions[2]['relation'] == 'on'
    assert actions[2]['arm'] == 'right'
    assert not any(s.skill == 'handover_object' and s.object == 'blue_cube' for s in plan.steps)


def _regrasp_fixture(pinch_survives_relief=True, puzzle_drop=False, sideways=0.0, relief_limit=1.0):
    from types import SimpleNamespace
    import numpy as np
    cube = np.array([0.45, -0.04, 1.10])
    ee = np.array([0.30, -0.04, 1.10])
    calls = {'set_gripper': [], 'moves': [], 'closed': 0}
    state = {'pinched': True}

    def set_gripper(arm, value):
        # The pinch survives only while the jaws stay closed enough for this object; closing restores it.
        calls['set_gripper'].append(value)
        fraction = (value - 0.02) / (0.7854 - 0.02)
        state['pinched'] = pinch_survives_relief and fraction <= relief_limit
        if puzzle_drop:
            cube[1] += sideways or 0.02

    def move(arm, target, rotation=None, **kwargs):
        calls['moves'].append(np.asarray(target).copy())
        if kwargs.get('on_sample'):
            kwargs['on_sample']()
        return SimpleNamespace(success=True, final_position=list(target))

    def wait_gripper(arm, closed, seconds):
        if closed:
            calls['closed'] += 1
            state['pinched'] = pinch_survives_relief
        return SimpleNamespace(success=True)

    env = SimpleNamespace(
        object_pose=lambda name: (cube.copy(), np.eye(3)),
        frame=lambda: SimpleNamespace(objects={'blue_cube': cube}, grippers={'left': 0.02, 'right': 0.7854}),
        ee_pose=lambda arm: (ee.copy(), np.eye(3)),
        pinch_established=lambda obj, arm: state['pinched'] if arm in ('left', 'right') else False,
        object_contacts=lambda obj, arm=None: (["openarm_right_ee_inner_finger|blue_cube",
                                                "openarm_right_ee_outer_finger|blue_cube"]
                                               if arm == 'right' and state['pinched'] else []),
        gripper_position=lambda arm: 0.02 if arm == 'right' else -0.02,
        gripper_target=lambda arm, closed: (0.02 if closed else 0.7854) if arm == 'right' else (-0.02 if closed else -0.7854),
        set_gripper=set_gripper,
        step=lambda n: None,
        timestep=0.002,
        grasp_mode='physical-friction-pinch',
    )
    controller = SimpleNamespace(motion_speed=6., target_rotations={}, move_to_site=move,
                                 wait_gripper=wait_gripper)
    return env, controller, calls, cube


def test_a_failed_placement_still_reports_whether_the_support_was_regrasped():
    """The preparation's measurement must survive the failure that follows it."""
    from types import SimpleNamespace
    from backend.app.skills.robot_skills import SkillRunner
    import numpy as np
    support = np.array([0.45, 0.04, 1.06])
    held = np.array([0.44, 0.15, 1.12])

    left_moves = {'count': 0}
    def move(arm, target, rotation=None, **kwargs):
        # Reorienting the grasp is allowed; the placement carry after it is blocked, as measured live.
        if arm == 'left':
            left_moves['count'] += 1
            if left_moves['count'] > 2:
                return SimpleNamespace(success=False, error='MOTION_BLOCKED',
                                       position_error=0.02, rotation_error=0.11)
        return SimpleNamespace(success=True, final_position=list(target))

    env = SimpleNamespace(
        ee_pose=lambda side: (np.array([0.40, 0.04, 1.10]) if side == 'right' else np.array([0.40, 0.15, 1.12]), np.eye(3)),
        object_pose=lambda name: (support.copy() if name == 'blue_cube' else held.copy(), np.eye(3)),
        object_bounds=lambda name: np.array([[.42, .01, 1.03], [.48, .07, 1.09]]),
        frame=lambda: SimpleNamespace(objects={'blue_cube': support, 'red_cube': held}),
        pinch_established=lambda obj, arm: (arm == 'right' and obj == 'blue_cube') or (arm == 'left' and obj == 'red_cube'),
        object_linear_speed=lambda name: 0.0, object_contacts=lambda name, arm=None: ['finger'],
        gripper_position=lambda arm: 0.02, gripper_target=lambda arm, closed: 0.02 if closed else 0.7854,
        set_gripper=lambda arm, value: None, step=lambda n: None, timestep=0.002,
        grasp_mode='physical-friction-pinch')
    controller = SimpleNamespace(motion_speed=6., target_rotations={}, move_to_site=move,
                                 wait_gripper=lambda arm, closed, seconds: SimpleNamespace(success=True),
                                 solve_ik=lambda arm, site, rot, **kw: (np.zeros(7), 0.004, 0.01))
    skills = SkillRunner(env, controller)
    skills.held_object = lambda arm: 'blue_cube' if arm == 'right' else ('red_cube' if arm == 'left' else None)
    skills._regrasp = lambda arm, obj, drop_m: {'object': obj, 'arm': arm, 'drop_m': drop_m, 'relief_fraction': 0.1}
    result = skills.move_to_object('left', 'red_cube', 'blue_cube', 'on')
    assert result['success'] is False and result['error'] == 'MOTION_BLOCKED'
    assert result['evidence']['support_regrasp']['relief_fraction'] == 0.1
    assert result['evidence']['reference_object'] == 'blue_cube'


def test_a_regrasp_slides_the_pinch_down_without_dropping_the_object():
    from backend.app.skills.robot_skills import SkillRunner, SUPPORT_REGRASP_DROP_M
    env, controller, calls, cube = _regrasp_fixture()
    skills = SkillRunner(env, controller)
    skills.held_object = lambda arm: 'blue_cube' if arm == 'right' else None
    result = skills.regrasp_object('right', 'blue_cube')
    assert result['success'], result
    assert result['drop_m'] == -SUPPORT_REGRASP_DROP_M
    # The jaws were relieved in searched steps, never opened fully.
    assert 0.02 < max(calls['set_gripper']) < 0.7854
    assert 0. < result['relief_fraction'] <= 0.25
    # The arm slid down by the drop, then lifted and returned for the hold check.
    drops = [round(float(target[2] - cube[2]), 4) for target in calls['moves']]
    assert any(abs(drop + SUPPORT_REGRASP_DROP_M) < 1e-4 for drop in drops)


def test_a_regrasp_refuses_to_slide_on_a_lost_pinch():
    from backend.app.skills.robot_skills import SkillRunner
    env, controller, calls, cube = _regrasp_fixture(pinch_survives_relief=False)
    skills = SkillRunner(env, controller)
    skills.held_object = lambda arm: 'blue_cube' if arm == 'right' else None
    result = skills.regrasp_object('right', 'blue_cube')
    assert result['success'] is False and result['error'] == 'REGRASP_LOST'
    assert calls['moves'] == [], 'the arm must not move after the pinch was lost'


def test_a_regrasp_keeps_the_largest_relief_that_still_pinches():
    """Measured 2026-11-07: 35% relief dropped the cube; the search must stop below that."""
    from backend.app.skills.robot_skills import SkillRunner
    env, controller, calls, cube = _regrasp_fixture(relief_limit=0.12)
    skills = SkillRunner(env, controller)
    skills.held_object = lambda arm: 'blue_cube' if arm == 'right' else None
    result = skills.regrasp_object('right', 'blue_cube')
    assert result['success'], result
    assert result['relief_fraction'] <= 0.12


def test_a_regrasp_that_cannot_slide_reports_that_instead_of_claiming_success():
    """If the object rides down with the hand, the grip point on it never moved."""
    from backend.app.skills.robot_skills import SkillRunner
    env, controller, calls, cube = _regrasp_fixture()
    def move(arm, target, rotation=None, **kwargs):
        calls['moves'].append(list(target))
        cube[2] = float(target[2]) - 0.11      # object follows the hand down
        if kwargs.get('on_sample'):
            kwargs['on_sample']()
        from types import SimpleNamespace as NS
        return NS(success=True, final_position=list(target))
    controller.move_to_site = move
    skills = SkillRunner(env, controller)
    skills.held_object = lambda arm: 'blue_cube' if arm == 'right' else None
    result = skills.regrasp_object('right', 'blue_cube')
    assert result['success'] is False and result['error'] == 'REGRASP_NO_SLIDE'


def test_a_regrasp_reports_an_object_that_slips_sideways():
    from backend.app.skills.robot_skills import SkillRunner
    env, controller, calls, cube = _regrasp_fixture(sideways=0.02)

    def move(arm, target, rotation=None, **kwargs):
        calls['moves'].append(list(target))
        if len(calls['moves']) == 1:
            cube[1] += 0.02          # slips only while the pinch slides
        if kwargs.get('on_sample'):
            kwargs['on_sample']()
        from types import SimpleNamespace as NS
        return NS(success=True, final_position=list(target))

    controller.move_to_site = move
    skills = SkillRunner(env, controller)
    skills.held_object = lambda arm: 'blue_cube' if arm == 'right' else None
    result = skills.regrasp_object('right', 'blue_cube')
    assert result['success'] is False and result['error'] == 'REGRASP_MOVED'


def test_in_hand_stacking_regrasps_the_held_support_before_placing():
    """The measured 7.15 mm gap is avoided by lowering the holder's pinch before the stack."""
    from types import SimpleNamespace
    from backend.app.skills.robot_skills import SkillRunner
    import numpy as np
    target_pos = np.array([0.48, -0.22, 1.12])
    placed_pos = np.array([0.44, 0.15, 1.12])
    regrasps = []

    mode = {'sliding': False}
    def mock_move(arm, target, rotation=None, **kwargs):
        # A carry takes the object with the hand. A regrasp slide is different: the hand moves down the
        # object while the object stays where it is, which is the whole point of the operation.
        if not mode['sliding']:
            if arm == 'right':
                target_pos[:] = target[:3]
            else:
                placed_pos[:] = target[:3]
        return SimpleNamespace(success=True, final_position=list(target))

    env = SimpleNamespace(
        ee_pose=lambda side: (np.array([0.40, -0.22, 1.12]) if side == 'right' else np.array([0.40, 0.15, 1.12]), np.eye(3)),
        object_pose=lambda name: (target_pos.copy() if name == 'blue_cube' else placed_pos.copy(), np.eye(3)),
        object_bounds=lambda name: np.array([[.45, -.25, 1.10], [.51, -.19, 1.15]]) if name == 'blue_cube'
        else np.array([[.41, .12, 1.10], [.47, .18, 1.15]]),
        frame=lambda: SimpleNamespace(objects={'blue_cube': target_pos, 'red_cube': placed_pos}),
        pinch_established=lambda obj, arm: (arm == 'right' and obj == 'blue_cube') or (arm == 'left' and obj == 'red_cube'),
        object_linear_speed=lambda name: 0.0,
        object_contacts=lambda name, arm=None: ['finger'],
        gripper_position=lambda arm: 0.02,
        gripper_target=lambda arm, closed: 0.02 if closed else 0.7854,
        set_gripper=lambda arm, value: None,
        step=lambda n: None, timestep=0.002, grasp_mode='physical-friction-pinch',
    )
    controller = SimpleNamespace(motion_speed=6., target_rotations={}, move_to_site=mock_move,
                                 wait_gripper=lambda arm, closed, seconds: SimpleNamespace(success=True))
    skills = SkillRunner(env, controller)
    skills.held_object = lambda arm: 'blue_cube' if arm == 'right' else ('red_cube' if arm == 'left' else None)
    original = skills._regrasp
    def spy(arm, obj, drop_m):
        regrasps.append((arm, obj, drop_m))
        mode['sliding'] = True
        try:
            return original(arm, obj, drop_m)
        finally:
            mode['sliding'] = False
    skills._regrasp = spy
    result = skills.move_to_object('left', 'red_cube', 'blue_cube', 'on')
    assert result['success'], result
    assert regrasps and regrasps[0][0] == 'right' and regrasps[0][1] == 'blue_cube'
    assert regrasps[0][2] < 0
    assert result['support_regrasp'], 'the support regrasp outcome must be recorded, achieved or not'


def test_in_hand_stacking_bimanually_stages_held_support_toward_center_workspace():
    # User insight: "今回の場合、右腕をまんなからへんによせるひつようがあった"
    # When support is held by right arm on the far right (y=-0.22), left arm cannot reach y=-0.22.
    # The bimanual skill must autonomously stage the held support toward center workspace (y ≈ 0),
    # enabling left arm to smoothly stack on it without reaching out of workspace bounds!
    from backend.app.skills.robot_skills import SkillRunner
    from types import SimpleNamespace
    import numpy as np

    target_pos = np.array([0.48, -0.22, 1.12])  # right arm holds cube on far right
    placed_pos = np.array([0.44, 0.15, 1.12])   # left arm holds cube on left

    moves = []
    rotations = []
    mode = {'sliding': False}
    def mock_move(arm, target, rotation=None, **kwargs):
        moves.append((arm, np.asarray(target).copy()))
        if rotation is not None:
            rotations.append((arm, np.asarray(rotation).copy()))
        # A carry carries; a regrasp slide moves the hand along the object and leaves it in place.
        if not mode['sliding']:
            if arm == 'right':
                target_pos[:] = target[:3]
            elif arm == 'left':
                placed_pos[:] = target[:3]
        return SimpleNamespace(success=True, final_position=list(target))

    env = SimpleNamespace(
        ee_pose=lambda side: (np.array([0.40, -0.22, 1.12]) if side == 'right' else np.array([0.40, 0.15, 1.12]), np.eye(3)),
        object_pose=lambda name: (target_pos.copy() if name == 'blue_cube' else placed_pos.copy(), np.eye(3)),
        object_bounds=lambda name: np.array([[target_pos[0]-.023, target_pos[1]-.023, target_pos[2]-.023],
                                            [target_pos[0]+.023, target_pos[1]+.023, target_pos[2]+.023]]) if name == 'blue_cube' else np.array([[placed_pos[0]-.023, placed_pos[1]-.023, placed_pos[2]-.023], [placed_pos[0]+.023, placed_pos[1]+.023, placed_pos[2]+.023]]),
        frame=lambda: SimpleNamespace(objects={'blue_cube': target_pos, 'red_cube': placed_pos}),
        pinch_established=lambda obj, arm: (arm == 'right' and obj == 'blue_cube') or (arm == 'left' and obj == 'red_cube'),
        object_linear_speed=lambda name: 0.0,
        object_contacts=lambda name, arm=None: ['finger'],
        step=lambda n: None,
        timestep=0.002,
        grasp_mode='physical-friction-pinch',
        gripper_position=lambda arm: 0.02 if arm == 'right' else -0.02,
        gripper_target=lambda arm, closed: (0.02 if closed else 0.7854) if arm == 'right' else (-0.02 if closed else -0.7854),
        set_gripper=lambda arm, value: None,
    )
    controller = SimpleNamespace(
        motion_speed=6.,
        target_rotations={},
        move_to_site=mock_move,
        solve_ik=lambda arm, site, rot, **kw: (np.zeros(7), 0.004, 0.01),
        wait_gripper=lambda arm, closed, seconds: SimpleNamespace(success=True),
    )
    skills = SkillRunner(env, controller)
    sliding = skills._regrasp
    def regrasp_spy(arm, obj, drop_m):
        mode['sliding'] = True
        try:
            return sliding(arm, obj, drop_m)
        finally:
            mode['sliding'] = False
    skills._regrasp = regrasp_spy
    # Execute move_to_object: left arm stacks red_cube on blue_cube held by right arm
    result = skills.move_to_object('left', 'red_cube', 'blue_cube', 'on')
    assert result['success']
    # Check that right arm was commanded to stage blue_cube toward center (y ≈ 0)!
    right_moves = [target for arm, target in moves if arm == 'right']
    assert len(right_moves) > 0, "Right arm must stage the held support toward center"
    staged_y = min(abs(float(target[1])) for target in right_moves)
    assert staged_y < 0.05, f"Right arm should stage close to center line, closest was {staged_y}"
    # The placing hand must not reuse the holder's approach side: it picks another reachable yaw.
    left_rotations = [r for arm, r in rotations if arm == 'left']
    assert left_rotations, 'the placing arm must be reoriented before entering the holder corridor'
    assert any(not np.allclose(r, np.eye(3)) for r in left_rotations)


def test_held_state_support_in_hand_stacking_never_places_support_on_table():
    # User instruction: "右手でキューブを掴んでから、掴んでいる状態のキューブの上に赤いキューブを重ねて"
    # Condition:
    # 1. Right arm picks blue_cube.
    # 2. Right arm NEVER places blue_cube on the table! It maintains its hold in the air!
    # 3. Left arm picks red_cube and stacks it directly on top of the held blue_cube.
    # 4. Final verification requires BOTH:
    #    - blue_cube is still held by right arm
    #    - red_cube is placed on blue_cube
    plan = AgentPlanner().plan('右手でキューブを掴んでから、掴んでいる状態のキューブの上に赤いキューブを重ねて', WORLD)
    actions = [(a['op'], a.get('object'), a.get('arm'), a.get('target'), a.get('relation'))
               for a in plan.program['actions']]
    # blue_cube must NEVER be placed on the table!
    assert not any(a[0] == 'place' and a[1] == 'blue_cube' for a in actions), f"blue_cube was placed on table: {actions}"
    # blue_cube is picked by right arm
    assert any(a[0] == 'pick' and a[1] == 'blue_cube' and a[2] == 'right' for a in actions)
    # red_cube is placed on blue_cube by left arm
    assert actions[-1] == ('place', 'red_cube', 'left', 'blue_cube', 'on')
    # Steps: right arm grasps blue_cube, left arm grasps red_cube and places on blue_cube
    skills = [(s.skill, s.arm, s.object) for s in plan.steps]
    assert ('release_object', 'right', 'blue_cube') not in skills, "blue_cube must NOT be released!"
    assert any(s[0] == 'move_to_object' and s[1] == 'left' and s[2] == 'red_cube' for s in skills)
    # Final goal checks: blue_cube is held by right arm, red_cube is on blue_cube!
    checks = plan.goal['checks']
    assert any(c['type'] == 'holding' and c['object'] == 'blue_cube' and c['arm'] == 'right' for c in checks)
    assert any(c['type'] == 'relative_placement' and c['object'] == 'red_cube' and c['target'] == 'blue_cube' and c['relation'] == 'on' for c in checks)


def test_arm_specific_uncolored_cube_and_held_state_support_reference_are_grounded():
    # User instruction: "右手でキューブを掴んでから、掴んでいる状態のキューブの上に赤いキューブを重ねて"
    # 1. "右手でキューブ": since right arm is specified, pick the cube on the right (blue_cube)
    # 2. "掴んでいる状態のキューブの上に": refers to the newly grasped support (blue_cube)
    # 3. "赤いキューブを重ねて": red_cube is placed on blue_cube
    plan = AgentPlanner().plan('右手でキューブを掴んでから、掴んでいる状態のキューブの上に赤いキューブを重ねて', WORLD)
    actions = [(a['op'], a.get('object'), a.get('arm'), a.get('target'), a.get('relation'))
               for a in plan.program['actions']]
    assert actions[0][0] == 'pick' and actions[0][1] == 'blue_cube' and actions[0][2] == 'right'
    assert actions[-1] == ('place', 'red_cube', 'left', 'blue_cube', 'on')


def test_stack_on_held_cube_maintains_hold_and_places_directly_without_table_setdown():
    # User intent: "右手で持っている状態を維持したまま、そのキューブの上に左手で赤いキューブを重ねて"
    # When right arm is holding blue_cube, the robot should NOT drop or park blue_cube on the table,
    # but directly pick red_cube with left arm and stack it directly onto held blue_cube in-hand.
    state = {
        'objects': [{'id': 'red_cube', 'type': 'cube', 'position': [0.44, 0.15, 1.031]},
                    {'id': 'blue_cube', 'type': 'cube', 'position': [0.40, -0.15, 1.12]}],
        'targets': [],
        'robot': {'available_arms': ['left', 'right'],
                  'left_arm': {'held_object': None},
                  'right_arm': {'held_object': 'blue_cube'}}
    }
    plan = AgentPlanner().plan('右手で持っている状態を維持したまま、そのキューブの上に左手で赤いキューブを重ねて', state)
    actions = [(a['op'], a.get('object'), a.get('arm'), a.get('target'), a.get('relation'))
               for a in plan.program['actions']]
    # blue_cube must NOT be placed on the table!
    assert not any(a[0] == 'place' and a[1] == 'blue_cube' for a in actions)
    # red_cube is picked by left arm and placed directly on blue_cube in-hand
    assert any(s.skill == 'grasp_object' and s.object == 'red_cube' and s.arm == 'left' for s in plan.steps)
    assert any(s.skill == 'move_to_object' and s.object == 'red_cube' and s.arm == 'left' for s in plan.steps)
    assert actions[-1] == ('place', 'red_cube', 'left', 'blue_cube', 'on')


def test_held_cube_reference_with_explicit_acting_hand_does_not_create_duplicate_actors():
    # User instruction: "左手で、右手で掴んでいるキューブの上に赤いキューブを重ねて"
    # The reference "右手で掴んでいるキューブ" modifies the support noun.
    # It must NOT inject a second conflicting actor "右手で" before the support,
    # which broke grammar and caused "Multiple acting hands in one clause".
    state = {
        'objects': [{'id': 'red_cube', 'type': 'cube', 'position': [0.44, 0.15, 1.031]},
                    {'id': 'blue_cube', 'type': 'cube', 'position': [0.40, -0.15, 1.12]}],
        'targets': [],
        'robot': {'available_arms': ['left', 'right'],
                  'left_arm': {'held_object': None},
                  'right_arm': {'held_object': 'blue_cube'}}
    }
    plan = AgentPlanner().plan('左手で、右手で掴んでいるキューブの上に赤いキューブを重ねて', state)
    actions = [(a['op'], a.get('object'), a.get('arm'), a.get('target'), a.get('relation'))
               for a in plan.program['actions']]
    # blue_cube is grounded to table by right arm first
    assert actions[0] == ('place', 'blue_cube', 'right', 'right_side', None)
    # red_cube is placed on blue_cube by left arm
    assert actions[-1] == ('place', 'red_cube', 'left', 'blue_cube', 'on')
    # Execution steps must use left arm for red_cube
    red_step = next(s for s in plan.steps if s.skill == 'move_to_object' and s.object == 'red_cube')
    assert red_step.arm == 'left'


@pytest.mark.parametrize('phrase', [
    '右手で掴んでいるキューブの上に赤いキューブを重ねて',
    '右手でつかんでいるキューブの上に赤いキューブを重ねて',
    '右手で握っているキューブの上に赤いキューブを重ねて',
    '右手で保持しているキューブの上に赤いキューブを重ねて',
    'put the red cube on top of the cube held by the right hand',
])
def test_held_cube_reference_with_tsukandeiru_and_nigitteiru_is_grounded_and_stacked(phrase):
    # User instruction: "右手で掴んでいるキューブの上に赤いキューブを重ねて"
    # When right arm is holding blue_cube:
    # 1. "右手で掴んでいるキューブ" must resolve to blue_cube
    # 2. Preparation grounds blue_cube onto table as stationary support
    # 3. red_cube is stacked on top of blue_cube
    state = {
        'objects': [{'id': 'red_cube', 'type': 'cube', 'position': [0.44, 0.15, 1.031]},
                    {'id': 'blue_cube', 'type': 'cube', 'position': [0.40, -0.15, 1.12]}],
        'targets': [],
        'robot': {'available_arms': ['left', 'right'],
                  'left_arm': {'held_object': None},
                  'right_arm': {'held_object': 'blue_cube'}}
    }
    plan = AgentPlanner().plan(phrase, state)
    actions = [(a['op'], a.get('object'), a.get('target'), a.get('relation')) for a in plan.program['actions']]
    assert actions[0] == ('place', 'blue_cube', 'right_side', None)
    assert actions[-1] == ('place', 'red_cube', 'blue_cube', 'on')
    assert any(c['type'] == 'relative_placement' and c['object'] == 'red_cube' and c['target'] == 'blue_cube'
               for c in plan.goal['checks'])


def test_carry_guard_stops_on_disturbing_stationary_cube():
    # If carrying an object bumps or displaces another resting table object,
    # the carry skill must immediately raise CARRY_COLLISION and stop.
    import numpy as np
    from backend.app.skills.robot_skills import SkillRunner, SkillError
    from types import SimpleNamespace
    positions = {'red_cube': np.array([.44, .15, 1.15]), 'blue_cube': np.array([.46, -.20, 1.031])}
    held_obj = 'red_cube'
    def move(arm, target, rotation, **kwargs):
        assert callable(kwargs.get('on_sample'))
        # Simulate physical disturbance: bumping the stationary blue_cube
        positions['blue_cube'][1] += .008
        kwargs['on_sample']()
        return SimpleNamespace(success=True, final_position=target.tolist())
    env = SimpleNamespace(
        object_pose=lambda name: (positions[name].copy(), np.eye(3)),
        frame=lambda: SimpleNamespace(objects=positions),
        ee_pose=lambda arm: (np.array([.40, .15, 1.15]), np.eye(3)),
        pinch_established=lambda *a: True,
        profile={},
    )
    controller = SimpleNamespace(
        motion_speed=6., target_rotations={}, move_to_site=move, timestep=.002
    )
    skills = SkillRunner(env, controller)
    with pytest.raises(SkillError) as err:
        skills._carry_to('left', 'red_cube', np.array([.44, -.10, 1.15]))
    assert err.value.category == 'CARRY_COLLISION'
    assert 'blue_cube' in str(err.value)


def test_sequential_different_arms_autonomously_parks_empty_arm_to_prevent_self_collision():
    # When left arm finishes placing an object and right arm begins working,
    # the planner must autonomously schedule park_empty_arm for left arm BEFORE
    # right arm enters the workspace, preventing arm-to-arm self-collision.
    plan = AgentPlanner().plan('赤いキューブを左手で左においてから、青いキューブを右手で右において', WORLD)
    skills = [s.skill for s in plan.steps]
    assert 'park_empty_arm' in skills
    left_release_idx = next(i for i, s in enumerate(plan.steps) if s.skill == 'release_object' and s.arm == 'left')
    right_approach_idx = next(i for i, s in enumerate(plan.steps) if s.skill == 'approach_object' and s.arm == 'right')
    park_idx = next(i for i, s in enumerate(plan.steps) if s.skill == 'park_empty_arm' and s.arm == 'left')
    assert left_release_idx < park_idx < right_approach_idx


def test_handover_adapts_to_irregular_single_finger_contact_with_local_micro_adjustment():
    # Irregular contact handling: when receiver initially establishes only a single-finger
    # touch (e.g. outer finger touches, inner finger misses), the skill must NOT blindly crash.
    # It must perform adaptive micro-adjustment (re-opening and adjusting depth) to secure the pinch.
    from backend.app.skills.robot_skills import SkillRunner
    from types import SimpleNamespace
    import numpy as np

    cube_pos = np.array([0.44, 0.0, 1.15])
    pinch_status = [False, True]  # 1st attempt misses inner finger, 2nd attempt succeeds
    attempts = []

    def mock_wait_gripper(arm, closed, duration):
        if arm == 'right' and closed:
            attempts.append(closed)

    def mock_move(arm, target, rotation=None, **kwargs):
        if arm == 'right' and donor_released[0]:
            # Cube follows right arm after transfer
            cube_pos[2] = target[2] - 0.03
        return SimpleNamespace(success=True, final_position=list(target))

    contacts_seq = [
        ['openarm_right_ee_outer_finger|blue_cube'],  # 1st attempt: single finger
        ['openarm_right_ee_inner_finger|blue_cube', 'openarm_right_ee_outer_finger|blue_cube'],  # 2nd attempt: two fingers
    ]

    contact_idx = [0]
    donor_released = [False]
    def mock_contacts(obj, arm=None):
        if arm == 'left':
            return [] if donor_released[0] else ['openarm_left_ee_inner_finger|blue_cube', 'openarm_left_ee_outer_finger|blue_cube']
        idx = min(contact_idx[0], len(contacts_seq) - 1)
        return contacts_seq[idx]

    def mock_pinch_established(obj, arm):
        if arm == 'left':
            return True
        val = pinch_status[min(contact_idx[0], len(pinch_status) - 1)]
        return val

    env = SimpleNamespace(
        ee_pose=lambda side: (np.array([0.40, 0.15, 1.15]) if side == 'left' else np.array([0.40, -0.15, 1.15]), np.eye(3)),
        object_pose=lambda name: (cube_pos.copy(), np.eye(3)),
        frame=lambda: SimpleNamespace(objects={'blue_cube': cube_pos}, grippers={'left': 0.02, 'right': 0.7854}),
        object_contacts=mock_contacts,
        pinch_established=mock_pinch_established,
        step=lambda n: None,
        timestep=0.002,
        grasp_mode='physical-friction-pinch',
    )
    controller = SimpleNamespace(
        motion_speed=6.,
        wait_gripper=mock_wait_gripper,
        move_to_site=mock_move,
        target_rotations={},
    )
    skills = SkillRunner(env, controller)
    # Monkeypatch contact_idx advancement on re-opening gripper
    orig_wait = mock_wait_gripper
    def adaptive_wait(arm, closed, duration):
        orig_wait(arm, closed, duration)
        if arm == 'left' and not closed:
            donor_released[0] = True
        if arm == 'right' and not closed and len(attempts) == 1:
            contact_idx[0] += 1
    controller.wait_gripper = adaptive_wait

    # Handover blue_cube from left to right with irregular initial contact
    result = skills._handover('left', 'blue_cube', 'right')
    assert result['object'] == 'blue_cube'
    assert result['receiver'] == 'right'
    assert len(attempts) >= 2, "Adaptive handover must have retried the pinch after single-finger contact"


def test_handover_geometry_adapts_to_measured_cube_location():
    # Adaptive workspace: cube at x=0.49 should not be violently dragged to x=0.44.
    # The solver adapts the coupled handover location to the measured cube position.
    import numpy as np
    from backend.app.simulation.openarm_v2.handover import plan_handover_geometry
    from types import SimpleNamespace
    cube_pos = np.array([0.48, 0.02, 1.12])
    env = SimpleNamespace(
        ee_pose=lambda side: (np.array([0.40, 0.15, 1.12]), np.eye(3)),
        object_pose=lambda name: (cube_pos.copy(), np.eye(3)),
        frame=lambda: SimpleNamespace(grippers={'left': 0.7854, 'right': -0.7854})
    )
    geom = plan_handover_geometry(env, 'left', 'red_cube')
    assert geom.residual < .003
    assert 0.40 <= geom.position[0] <= 0.52
    assert abs(geom.position[1]) < .02


def test_handover_skill_accepts_minimal_arguments_and_infers_arms():
    # Declarative / Minimal-information skill contract:
    # handover_object should only require the object name ("red_cube").
    # The skill runner must autonomously infer donor (holder) and receiver (empty arm).
    from backend.app.skills.robot_skills import SkillRunner
    from types import SimpleNamespace
    holder = 'left'
    held_state = {holder: 'red_cube', 'right': None}
    calls = []
    def fake_handover(arm, obj, receiver):
        calls.append((arm, obj, receiver))
        return {'success': True, 'giver': arm, 'receiver': receiver, 'object': obj}
    skills = SkillRunner.__new__(SkillRunner)
    skills.held_object = lambda arm: held_state[arm]
    skills._require_hold = lambda arm, obj: None
    skills._handover = fake_handover
    skills._skill = lambda name, fn, started: fn()
    # Call with minimal information: arm=None, receiver=None
    result = skills.handover_object(None, 'red_cube')
    assert result['success']
    assert calls == [('left', 'red_cube', 'right')]


def test_referring_expressions_resolve_other_hand_and_other_cube():
    # Instruction: "キューブを、もう一方の手に持たせて、そのキューブの上にもう一つのキューブを重ねて"
    # The resolver must ground:
    # 1. "キューブ": one candidate cube (e.g. red_cube)
    # 2. "もう一方の手": the opposite hand
    # 3. "そのキューブ": the first chosen cube as the support
    # 4. "もう一つのキューブ": the remaining cube in the scene
    # And execute with minimal orthogonal skills (no new ad-hoc skills)!
    plan = AgentPlanner().plan('キューブを、もう一方の手に持たせて、そのキューブの上にもう一つのキューブを重ねて', WORLD)
    actions = [(a['op'], a.get('object'), a.get('arm'), a.get('receiver'), a.get('target'), a.get('relation'))
               for a in plan.program['actions']]
    # Must involve both cubes in the scene
    manipulated = {a[1] for a in actions if a[1]}
    assert manipulated == {'red_cube', 'blue_cube'}
    # The final action must place one cube on top of the other
    last = actions[-1]
    assert last[0] == 'place' and last[5] == 'on'
    assert {last[1], last[4]} == {'red_cube', 'blue_cube'}
    assert not any(c['type'] == 'holding' for c in plan.goal['checks'])
    assert any(c['type'] == 'relative_placement' and c['relation'] == 'on' for c in plan.goal['checks'])


def test_planner_error_reporting_preserves_instruction_grammar_failure_over_model_repair_syntax_error(monkeypatch):
    # Live failure c3d74b0f: grammar rejected an omitted hand in handover, but the user was told
    # "Relative motion requires a direction and distance (1–30 cm)" because the model guessed a bad move.
    # The reported error must retain the human instruction's actual deficit, not only the model's syntax error.
    planner = AgentPlanner()
    planner.enabled = planner.local_enabled = True
    planner.capability_mode = True
    monkeypatch.setattr(planner, '_ai_plan', lambda *args: {
        'payload': {'actions': [{'op': 'move', 'object': 'red_cube'}]},
        'raw': 'unmeasured move'})
    state = {**WORLD, 'robot': {'available_arms': ['left']}} # only 1 arm, so handover fails
    with pytest.raises(PlannerProviderError) as error:
        planner.plan('赤いキューブを手に渡して', state)
    msg = str(error.value)
    assert 'handover needs two different hands' in msg
    assert 'model repair rejected' in msg


def test_invalid_model_move_cannot_hide_grounded_unspecified_handover(monkeypatch):
    planner=AgentPlanner()
    planner.enabled=planner.local_enabled=True
    planner.capability_mode=True
    monkeypatch.setattr(planner,'_ai_plan',lambda *args:{'payload':{
        'actions':[{'op':'move','object':'red_cube'}]},'raw':'unmeasured move'})
    plan=planner.plan('赤いキューブを手に渡して右においてから、青いキューブを赤いキューブに乗せて',WORLD)
    assert plan.model_called and not plan.model_verdict['ok']
    assert plan.provider=='deterministic_fallback'
    assert [(a['op'],a['object']) for a in plan.program['actions']]==[
        ('handover','red_cube'),('place','red_cube'),('place','blue_cube')]
    assert not any(a['op']=='move' for a in plan.program['actions'])


def test_explicit_human_recipient_is_not_replaced_with_another_robot_hand():
    with pytest.raises(PlannerProviderError) as error:
        AgentPlanner().plan('赤いキューブを私の手に渡して',WORLD)
    assert error.value.category=='ROBOT_CAPABILITY'


def test_unspecified_handover_requires_two_available_robot_arms():
    from backend.app.planning.task_program import GroundingError, TaskProgram, compile_program
    state={**WORLD,'robot':{'available_arms':['left']}}
    with pytest.raises(GroundingError,match='two different hands'):
        compile_program(TaskProgram(actions=[{'op':'handover','object':'red_cube'}]),state)


def test_support_parking_cannot_erase_an_earlier_throw_event():
    from backend.app.planning.task_contract import intent_checks
    from backend.app.planning.task_program import TaskProgram, compile_program
    program=TaskProgram(actions=[{'op':'throw','object':'red_cube','arm':'left'},
        {'op':'pick','object':'red_cube','arm':'left'},
        {'op':'place','object':'red_cube','arm':'left','target':'left_side'},
        {'op':'place','object':'blue_cube','arm':'right','target':'red_cube','relation':'on'}])
    world={**WORLD,'objects':[{**o,'type':'cube'} for o in WORLD['objects']]}
    plan=compile_program(program,world)
    original=TaskProgram(actions=[program.actions[0],program.actions[1],program.actions[3]])
    plan.planning_evidence={'requested_program':original.model_dump(),
        'preparations':[{'kind':'ground_held_reference','object':'red_cube',
                         'target':'left_side','action_index':2}]}
    assert any(c['type']=='throw' and c['object']=='red_cube' for c in intent_checks(plan))


def test_support_preparation_avoids_a_previously_placed_object():
    from backend.app.planning.task_program import TaskProgram
    from backend.app.planning.preparation import compile_prepared_program
    program=TaskProgram(actions=[{'op':'place','object':'blue_cube','arm':'right','target':'left_side'},
        {'op':'pick','object':'red_cube','arm':'left'},
        {'op':'place','object':'blue_cube','arm':'right','target':'red_cube','relation':'on'}])
    plan=compile_prepared_program(program,WORLD)
    support=next(p for p in plan.planning_evidence['preparations']
                 if p['kind']=='ground_held_reference')
    assert support['target']!='left_side'


def test_auxiliary_oite_is_not_misread_as_a_setdown():
    from backend.app.planning.language_fallback import parse_fallback
    source=parse_fallback('青いキューブを右手に渡しておいてから、赤いキューブを青いキューブに乗せて',WORLD)
    assert [a.op for a in source.actions]==['handover','place']
    assert source.actions[1].target=='blue_cube' and source.actions[1].relation=='on'
    second=parse_fallback('赤いキューブを左手で掴んでおいてから、青いキューブを右手で取って',WORLD)
    assert [a.op for a in second.actions]==['pick','pick']


def test_temporary_held_support_can_be_grounded_before_a_relative_placement():
    from backend.app.planning.task_program import GroundingError, TaskProgram, compile_program
    from backend.app.planning.preparation import compile_prepared_program
    program=TaskProgram(actions=[{'op':'pick','object':'red_cube','arm':'left'},
                                  {'op':'place','object':'blue_cube','arm':'right','target':'red_cube','relation':'on'}])
    with pytest.raises(GroundingError,match='REFERENCE_SUPPORT_HELD'):
        compile_program(program,WORLD)
    plan=compile_prepared_program(program,WORLD)
    actions=plan.program['actions']
    assert [(a['op'],a['object']) for a in actions]==[
        ('pick','red_cube'),('place','red_cube'),('home',None),('place','blue_cube')]
    assert actions[1]['target'] in {'left_side','right_side','center'}
    assert actions[3]['target']=='red_cube' and actions[3]['relation']=='on'
    assert plan.planning_evidence['preparations'][0]['kind']=='ground_held_reference'
    assert plan.goal['checks'][-1]['type']=='relative_placement'


def test_support_held_at_task_start_may_be_prepared_without_claiming_it_is_stationary():
    from backend.app.planning.task_contract import TaskContract, requested_program
    state={**WORLD,'robot':{'left_arm':{'held_object':'red_cube'},'right_arm':{'held_object':None}}}
    plan=AgentPlanner().plan('青いキューブを赤いキューブの上に置いて',state)
    assert plan.planning_evidence['preparations'][0]['kind']=='ground_held_reference'
    contract=TaskContract.freeze('青いキューブを赤いキューブの上に置いて',plan,requested_program(plan))
    assert contract.checks()[0]['object']=='blue_cube'
    assert not any(c['type']=='holding' or c['object']=='red_cube' for c in contract.checks())


def test_explicit_continuous_hold_is_not_silently_replaced_by_a_table_support():
    from backend.app.agent.task_agent import TaskAgent
    from tests.test_task_agent import FakeExecutor
    world={**WORLD,'robot':{'left_arm':{'held_object':None},'right_arm':{'held_object':None}}}
    executor=FakeExecutor(world,[])
    result=TaskAgent(executor,AgentPlanner()).run('赤いキューブを左手で持ったまま、その上に青いキューブを乗せて')
    assert not result['success'] and not executor.calls
    assert result['verification']['failure_category']=='CONTRACT_UNVERIFIABLE'


def test_rejected_model_cannot_replace_a_valid_support_plan_with_a_table_side(monkeypatch):
    planner = AgentPlanner()
    planner.enabled = planner.local_enabled = True
    planner.capability_mode = True
    monkeypatch.setattr(planner, '_ai_plan', lambda *args: {
        'payload': {'actions': [{'op': 'place', 'object': 'blue_cube', 'target': 'right_side'}]},
        'raw': 'lost the support'})
    plan=planner.plan('赤いキューブを左手で持ち上げて、その上に右手で青いキューブを重ねて', WORLD)
    assert plan.provider=='deterministic_fallback' and plan.model_called
    assert plan.model_verdict['ok'] is False
    assert any(c['type']=='relative_placement' and c['object']=='blue_cube'
               and c['target']=='red_cube' for c in plan.goal['checks'])


@pytest.mark.parametrize('support,subject,word,relation', [
    ('赤', '青', '上', 'on'), ('青', '赤', '左', 'left_of'),
    ('赤', '青', '右隣', 'right_of'), ('青', '赤', '手前', 'front_of'),
    ('赤', '青', '奥', 'behind'), ('青', '赤', 'そば', 'near'),
])
def test_relative_pronouns_preserve_arbitrary_support_and_relation(support,subject,word,relation):
    from backend.app.planning.spatial_language import placement_references
    text = f'{support}いキューブを持ち上げて、その{word}に{subject}いキューブを置いて'
    refs = placement_references(text, WORLD)
    expected = 'red_cube' if support == '赤' else 'blue_cube'
    assert [(r['target'], r['relation']) for r in refs] == [(expected, relation)]


def test_model_extra_setdown_does_not_bypass_canonical_preparation(monkeypatch):
    planner = AgentPlanner()
    planner.enabled = planner.local_enabled = True
    planner.capability_mode = True
    monkeypatch.setattr(planner, '_ai_plan', lambda *args: {'payload': {'actions': [
        {'op': 'pick', 'object': 'red_cube', 'arm': 'left'},
        {'op': 'place', 'object': 'red_cube', 'arm': 'left', 'target': 'left_side'},
        {'op': 'place', 'object': 'blue_cube', 'arm': 'right', 'target': 'red_cube', 'relation': 'on'}]},
        'raw': 'put the held support down first'})
    plan=planner.plan('赤いキューブを左手で持ち上げて、その上に右手で青いキューブを重ねて', WORLD)
    assert plan.provider=='deterministic_fallback' and plan.model_called
    assert plan.planning_evidence['preparations'][0]['kind']=='ground_held_reference'
    assert [(a['op'],a['object']) for a in plan.program['actions']]==[
        ('pick','red_cube'),('place','red_cube'),('home',None),('place','blue_cube')]


def test_hand_reference_is_not_an_object_relative_position():
    from backend.app.planning.spatial_language import placement_references
    assert placement_references('青を右手で取って、その左手にある赤を掴んで', WORLD) == []


def test_ambiguous_antecedent_is_not_guessed():
    from backend.app.planning.spatial_language import placement_references
    from backend.app.planning.task_program import GroundingError
    with pytest.raises(GroundingError, match='Unresolved'):
        placement_references('赤と青を持ち上げて、その上に置いて', WORLD)


@pytest.mark.parametrize('text', ['赤と青を重ねて', 'Stack red and blue'])
def test_support_free_stacking_never_means_two_table_placements(text):
    with pytest.raises(PlannerProviderError):
        AgentPlanner().plan(text, WORLD)


def test_intermediate_relations_need_not_remain_true_at_completion():
    from backend.app.planning.spatial_language import validate_instruction_relations
    from backend.app.planning.task_program import compile_program, TaskProgram
    plan = compile_program(TaskProgram.model_validate({'actions': [
        {'op': 'place', 'object': 'blue_cube', 'target': 'red_cube', 'relation': 'left_of'},
        {'op': 'place', 'object': 'blue_cube', 'target': 'red_cube', 'relation': 'right_of'}]}), WORLD)
    validate_instruction_relations(plan, 'Place blue to the left of red, then place blue to the right of red', WORLD)


def test_explicit_level_roles_are_resolved_supports():
    from backend.app.planning.spatial_language import validate_instruction_relations
    from backend.app.planning.task_program import compile_program, TaskProgram
    plan = compile_program(TaskProgram.model_validate({'goals': [
        {'type': 'relative_placement', 'object': 'blue_cube', 'target': 'red_cube', 'relation': 'on'}]}), WORLD)
    validate_instruction_relations(plan, '赤が下、青が上になるように重ねて', WORLD)


@pytest.mark.parametrize('text', [
    'Do not stack red and blue, pick blue',
    '赤と青を重ねないで、青を取って',
    '赤には触らないで、青を重ねずに右側に置いて',
])
def test_prohibiting_support_does_not_require_a_positive_support_goal(text):
    from backend.app.planning.spatial_language import validate_instruction_relations
    from backend.app.planning.task_program import TaskProgram, compile_program
    plan = compile_program(TaskProgram.model_validate({'actions': [
        {'op': 'pick', 'object': 'blue_cube', 'arm': 'right'}]}), WORLD)
    validate_instruction_relations(plan, text, WORLD)


def test_success_oracle_must_contain_the_requested_relation():
    from backend.app.planning.spatial_language import validate_instruction_relations
    from backend.app.planning.task_program import compile_program, TaskProgram, GroundingError
    plan = compile_program(TaskProgram.model_validate({'actions': [
        {'op': 'place', 'object': 'blue_cube', 'target': 'red_cube', 'relation': 'on'}]}), WORLD)
    # The path may still stack correctly; a success oracle checking a table side is not admissible.
    plan.goal['checks'] = [{'type': 'placement', 'object': 'blue_cube', 'target': 'right_side'}]
    with pytest.raises(GroundingError, match='success'):
        validate_instruction_relations(plan, '青を赤の上に置いて', WORLD)


def test_a_held_support_whose_own_fingers_cover_its_top_face_is_refused_before_moving():
    """Measured live 2026-11-07: 40.5 mm of finger above the face; no regrasp depth clears it."""
    from types import SimpleNamespace
    from backend.app.skills.robot_skills import SkillRunner
    import numpy as np
    support = np.array([0.4363, -0.0152, 1.0294])
    held = np.array([0.44, 0.15, 1.12])

    def move(arm, target, rotation=None, **kwargs):
        return SimpleNamespace(success=True, final_position=list(target))

    env = SimpleNamespace(
        ee_pose=lambda side: (np.array([0.40, -0.02, 1.10]) if side == 'right' else np.array([0.40, 0.15, 1.12]), np.eye(3)),
        object_pose=lambda name: (support.copy() if name == 'blue_cube' else held.copy(), np.eye(3)),
        object_bounds=lambda name: np.array([[.41, -.041, 1.003], [.462, .011, 1.056]]) if name == 'blue_cube'
        else np.array([[.414, .124, 1.09], [.466, .176, 1.15]]),
        frame=lambda: SimpleNamespace(objects={'blue_cube': support, 'red_cube': held}),
        pinch_established=lambda obj, arm: (arm == 'right' and obj == 'blue_cube') or (arm == 'left' and obj == 'red_cube'),
        object_linear_speed=lambda name: 0.0, object_contacts=lambda name, arm=None: ['finger'],
        gripper_position=lambda arm: 0.02, gripper_target=lambda arm, closed: 0.02 if closed else 0.7854,
        set_gripper=lambda arm, value: None, step=lambda n: None, timestep=0.002,
        grasp_mode='physical-friction-pinch')
    controller = SimpleNamespace(motion_speed=6., target_rotations={}, move_to_site=move,
                                 wait_gripper=lambda arm, closed, seconds: SimpleNamespace(success=True),
                                 solve_ik=lambda arm, site, rot, **kw: (np.zeros(7), 0.004, 0.01))
    skills = SkillRunner(env, controller)
    skills.held_object = lambda arm: 'blue_cube' if arm == 'right' else ('red_cube' if arm == 'left' else None)
    skills._regrasp = lambda arm, obj, drop_m: {'object': obj, 'arm': arm, 'drop_m': drop_m}
    # The live measurement: pads 40.5 mm above the support's top face.
    skills._support_face_clearance = lambda arm, obj: {'clearance_m': -0.0405,
                                                       'pad_over_face_m': 1.0969, 'support_top_m': 1.0563,
                                                       'pad_geoms_over_face': 12}
    result = skills.move_to_object('left', 'red_cube', 'blue_cube', 'on')
    assert result['success'] is False and result['error'] == 'SUPPORT_FACE_BLOCKED'
    assert '40.5 mm' in result['detail']
    assert result['evidence']['clearance']['clearance_m'] == -0.0405
    # The refusal happens before any motion, so no regrasp is claimed.
    assert result['evidence']['support_regrasp'] is None
    assert result['evidence']['phase'] == 'support_face'


def test_an_unmeasured_clearance_does_not_block_a_placement():
    """No geometry, no claim: an unknown clearance must not stop a placement."""
    from backend.app.skills.robot_skills import clearances_blocks
    assert clearances_blocks(-0.0405) is True
    assert clearances_blocks(0.02) is False
