"""A learned policy may propose, but only a proposal that passes the same checks becomes a plan."""
import pytest

from backend.app.planning import policy_provider
from backend.app.planning.agent_planner import AgentPlanner
from backend.app.planning.policy_provider import (LeRobotPolicyBackend, PolicyBackend, PolicyBackendError,
                                                  policy_backend_from_env)
from backend.app.planning.task_program import TaskProgram

WORLD = {'objects': [{'id': 'red_cube', 'type': 'cube', 'position': [.44, .15, 1.031]},
                     {'id': 'blue_cube', 'type': 'cube', 'position': [.50, -.22, 1.031]}],
         'targets': [], 'robot': {'available_arms': ['left', 'right'],
                                  'left_arm': {'held_object': None}, 'right_arm': {'held_object': None}}}


class FakeBackend(PolicyBackend):
    name = 'fake'

    def __init__(self, program):
        self.program = program
        self.calls = 0

    def propose(self, instruction, state):
        self.calls += 1
        return self.program


@pytest.fixture(autouse=True)
def no_llm(monkeypatch):
    for key in ('AI_AND_API_KEY', 'AI_AND_BASE_URL', 'AI_AND_MODEL', 'OPENARM_LOCAL_PLANNER',
                'OPENARM_FAST_LANGUAGE', 'OPENARM_CAPABILITY_MODE', 'OPENARM_POLICY_BACKEND'):
        monkeypatch.delenv(key, raising=False)


def test_no_backend_is_configured_by_default():
    assert policy_backend_from_env({}) is None


def test_an_unknown_backend_name_is_a_configuration_error():
    with pytest.raises(PolicyBackendError, match='POLICY_BACKEND_UNKNOWN'):
        policy_backend_from_env({'OPENARM_POLICY_BACKEND': 'nonexistent'})


def test_the_lerobot_backend_refuses_raw_action_output():
    """A joint-chunk policy cannot be checked against a contract, so this seam refuses it."""
    backend = LeRobotPolicyBackend(model_path='/tmp/weights', output='actions')
    with pytest.raises(PolicyBackendError, match='POLICY_OUTPUT_NOT_VERIFIABLE'):
        backend.propose('x', WORLD)


def test_the_lerobot_backend_never_downloads_weights_itself():
    with pytest.raises(PolicyBackendError, match='POLICY_WEIGHTS_ABSENT'):
        LeRobotPolicyBackend(model_path='').propose('x', WORLD)


def test_a_valid_proposal_is_used_and_recorded(monkeypatch):
    backend = FakeBackend(TaskProgram(actions=[{'op': 'pick', 'object': 'red_cube', 'arm': 'left'}]))
    policy_provider.register('fake', lambda: backend)
    monkeypatch.setenv('OPENARM_POLICY_BACKEND', 'fake')
    plan = AgentPlanner().plan('赤いキューブを左手で取って', WORLD)
    assert plan.provider == 'policy:fake'
    assert backend.calls == 1
    assert plan.planning_evidence['policy_backend'] == {'ok': True, 'backend': 'fake'}
    assert plan.planning_evidence['grounding']['instruction']


def test_a_proposal_that_invents_an_object_never_becomes_a_plan(monkeypatch):
    backend = FakeBackend(TaskProgram(actions=[{'op': 'pick', 'object': 'green_cube', 'arm': 'left'}]))
    policy_provider.register('fake', lambda: backend)
    monkeypatch.setenv('OPENARM_POLICY_BACKEND', 'fake')
    plan = AgentPlanner().plan('赤いキューブを左手で取って', WORLD)
    assert plan.provider != 'policy:fake'
    attempt = plan.planning_evidence['policy_backend']
    assert attempt['ok'] is False and 'green_cube' in attempt['error']


def test_a_proposal_that_drops_a_requested_operation_is_refused(monkeypatch):
    """A proposer that silently drops half the task must be refused, not executed."""
    backend = FakeBackend(TaskProgram(actions=[{'op': 'pick', 'object': 'red_cube', 'arm': 'left'},
                                               {'op': 'place', 'object': 'red_cube', 'arm': 'left',
                                                'target': 'left_side'}]))
    policy_provider.register('fake', lambda: backend)
    monkeypatch.setenv('OPENARM_POLICY_BACKEND', 'fake')
    plan = AgentPlanner().plan('赤いキューブを左手で左においてから、青いキューブを右手で右において', WORLD)
    assert plan.provider != 'policy:fake'
    attempt = plan.planning_evidence['policy_backend']
    assert attempt['ok'] is False and 'blue_cube' in attempt['error']


def test_a_proposal_that_drops_a_handover_cannot_stand_in_for_one(monkeypatch):
    """The obligation check applies to a learned proposer exactly as it applies to the grammar."""
    backend = FakeBackend(TaskProgram(actions=[{'op': 'pick', 'object': 'blue_cube', 'arm': 'right'}]))
    policy_provider.register('fake', lambda: backend)
    monkeypatch.setenv('OPENARM_POLICY_BACKEND', 'fake')
    plan = AgentPlanner().plan('青いキューブを右手で取って左手に渡して', WORLD)
    assert plan.provider != 'policy:fake'
    attempt = plan.planning_evidence['policy_backend']
    assert attempt['ok'] is False
    assert 'handover' in attempt['error'].lower() or 'objects' in attempt['error'].lower()


def test_an_unset_backend_is_never_called(monkeypatch):
    backend = FakeBackend(TaskProgram(actions=[{'op': 'pick', 'object': 'red_cube', 'arm': 'left'}]))
    policy_provider.register('fake', lambda: backend)
    AgentPlanner().plan('赤いキューブを左手で取って', WORLD)
    assert backend.calls == 0
