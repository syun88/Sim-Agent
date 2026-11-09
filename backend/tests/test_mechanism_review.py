"""A second opinion may withhold the one irreversible recovery, and nothing else."""
import pytest

from backend.app.agent.recovery_policy import admit_reset
from backend.app.agent.task_agent import TaskAgent
from backend.app.planning.agent_planner import AgentPlanner
from backend.app.verification.failure_attribution import CRITERIA, cross_check, withholds
from tests.test_task_agent import FakeExecutor

WORLD = {'objects': [{'id': 'red_cube', 'type': 'cube', 'position': [.44, .15, 1.031]},
                     {'id': 'blue_cube', 'type': 'cube', 'position': [.50, -.22, 1.031]}],
         'targets': [], 'robot': {'left_arm': {'held_object': None}, 'right_arm': {'held_object': None}},
         'workspace': {'table_height_m': 1.008}}


def test_a_matching_reviewer_agrees_and_withholds_nothing():
    verdict = cross_check('TRACKING', {}, lambda request: {'mechanism': 'tracking', 'confidence': 0.8})
    assert verdict['verdict'] == 'agree'
    assert withholds(verdict, action='reset_and_retry') is False


def test_a_disagreeing_reviewer_withholds_only_the_reset():
    verdict = cross_check('APPROACH_CONTACT', {}, lambda request: {'mechanism': 'tracking'})
    assert verdict['verdict'] == 'disagree'
    assert withholds(verdict, action='reset_and_retry') is True
    assert withholds(verdict, action='resume') is False, 'only the irreversible step is affected'


def test_an_abstaining_or_failing_reviewer_changes_nothing():
    for reviewer in (lambda request: {'mechanism': None},
                     lambda request: 'not a dict',
                     lambda request: (_ for _ in ()).throw(RuntimeError('model down'))):
        verdict = cross_check('PLANNING', {}, reviewer)
        assert verdict['verdict'] in {'abstain', 'unavailable'}
        assert withholds(verdict, action='reset_and_retry') is False


def test_no_reviewer_is_reported_as_unavailable():
    assert cross_check('PLANNING', {}, None)['verdict'] == 'unavailable'


def test_an_undecided_mechanism_cannot_be_confirmed_by_a_named_one():
    """UNKNOWN has no criterion, so a reviewer naming anything is a disagreement, not an agreement."""
    assert CRITERIA['UNKNOWN'] is None
    verdict = cross_check('UNKNOWN', {}, lambda request: {'mechanism': 'geometry'})
    assert verdict['verdict'] == 'disagree'


def test_reset_admission_needs_a_certificate_and_an_uncontested_mechanism():
    agreed = {'verdict': 'agree'}
    contested = {'verdict': 'disagree', 'mechanism': 'PLANNING', 'reviewer_mechanism': 'tracking'}
    certified = {'object': 'red_cube'}
    kw = dict(legal_is_empty=True, reset_allowed=True, resets=0, max_resets=1)
    assert admit_reset(certified, agreed, **kw)[0] is True
    offered, why = admit_reset(certified, contested, **kw)
    assert offered is False and 'contested' in why
    assert admit_reset(None, agreed, **kw)[0] is False
    assert admit_reset(certified, agreed, **{**kw, 'legal_is_empty': False})[0] is False


def _fallen_cube_outcome(executor, plan):
    """A cube that fell below the table: the requested object exists (so a retry can compile) but the
    scene now has an observed, unrecoverable cause - the one condition that certifies a reset."""
    for obj in executor.state['objects']:
        if obj['id'] == 'red_cube':
            obj['position'] = [0.44, 0.15, 0.90]
    return {'success': False, 'status': 'FAILED',
            'skills': [{'skill': 'approach_object', 'arm': 'left', 'object': 'red_cube',
                        'success': False, 'error': 'IK_FAILURE', 'action_index': 0, 'step_index': 0,
                        'evidence': {'position_error': 0.02, 'rotation_error': 0.05}}],
            'verification': {'success': False, 'failure_category': 'IK_FAILURE'}}


def test_a_contested_mechanism_keeps_the_layout_instead_of_resetting():
    """The end-to-end guard: a reviewer disagreement must stop the task with the scene intact."""
    executor = FakeExecutor(WORLD, [_fallen_cube_outcome] * 4)
    result = TaskAgent(executor, AgentPlanner(), mechanism_reviewer=lambda request: {'mechanism': 'contact'},
                       max_attempts=3).run('赤いキューブを左手で取って')
    assert executor.resets == 0, 'a contested mechanism must not restart the scene'
    decision = result['reset_decisions'][-1]
    assert decision['offered'] is False and 'contested' in decision['reason']
    assert result['mechanism_reviews'][-1]['verdict'] == 'disagree'


def test_an_agreeing_reviewer_leaves_the_reset_available():
    # The rule table attributes this shape to TRACKING (position residual 0.02 over tolerance), so an
    # agreeing reviewer must name that, not "planning": a second opinion that names a different mechanism
    # withholds the reset by design.
    executor = FakeExecutor(WORLD, [_fallen_cube_outcome] * 4)
    result = TaskAgent(executor, AgentPlanner(),
                       mechanism_reviewer=lambda request: {'mechanism': 'tracking'}, max_attempts=3).run(
        '赤いキューブを左手で取って')
    assert result['reset_decisions'][-1]['offered'] is True
    assert executor.resets == 1


def test_without_a_reviewer_the_rule_table_still_decides():
    executor = FakeExecutor(WORLD, [_fallen_cube_outcome] * 4)
    result = TaskAgent(executor, AgentPlanner(), max_attempts=3).run('赤いキューブを左手で取って')
    assert result['mechanism_reviews'] == [], 'an absent reviewer must leave no verdict behind'
    assert result['reset_decisions'][-1]['offered'] is True
