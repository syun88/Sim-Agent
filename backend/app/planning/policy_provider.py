"""A seam for a learned policy backend, with the verifier still in charge.

LeRobot-style VLAs (e.g. Galaxea's G0.5, shown 2026-11-03 emitting a subtask + object boxes and then an
action chunk) solve the perception/grounding problem this repo keeps patching by hand. They do **not**
produce this repo's typed task program, and they cannot certify their own success. So the seam is narrow
on purpose:

* A backend returns a ``TaskProgram`` - symbolic subtasks, not joint chunks. A raw action-chunk output is
  refused here, because nothing in this repo can check a 7-dimensional chunk against a contract.
* The proposal goes through the same grounding, relation checks, obligation checks and `validate_plan`
  as every other planner. It has no way to bypass them.
* Genesis measures success. A policy that says it succeeded is a claim, not evidence.

Nothing is enabled by default: ``OPENARM_POLICY_BACKEND`` must name a registered backend, and a backend
that needs weights refuses to start without them rather than silently planning from nothing.
"""

from __future__ import annotations

import os
from typing import Callable

from backend.app.planning.task_program import TaskProgram


class PolicyBackendError(RuntimeError):
    """The backend could not produce a proposal; the caller reports it and continues."""


class PolicyBackend:
    """One learned proposer. Implementations must not touch the simulator."""

    name = 'unnamed'

    def propose(self, instruction: str, state: dict) -> TaskProgram:
        raise NotImplementedError


class LeRobotPolicyBackend(PolicyBackend):
    """Placeholder for a LeRobot policy (``--policy.type=g05`` style).

    Refused unless both the weights path and an explicit opt-in are present. It also refuses when the
    policy is configured to emit raw actions: this seam can only carry symbolic subtasks, and pretending
    otherwise would let an unverifiable chunk reach the motors.
    """

    name = 'lerobot'

    def __init__(self, model_path: str | None = None, output: str = 'subtasks'):
        self.model_path = model_path or os.getenv('OPENARM_POLICY_MODEL_PATH', '').strip()
        self.output = output

    def propose(self, instruction: str, state: dict) -> TaskProgram:
        if self.output != 'subtasks':
            raise PolicyBackendError(
                'POLICY_OUTPUT_NOT_VERIFIABLE: this seam accepts symbolic subtasks only; '
                f'the configured output is {self.output!r}')
        if not self.model_path:
            raise PolicyBackendError('POLICY_WEIGHTS_ABSENT: set OPENARM_POLICY_MODEL_PATH to a local '
                                     'checkpoint; the backend never downloads weights by itself')
        raise PolicyBackendError('POLICY_NOT_WIRED: no local LeRobot runtime is installed on this host; '
                                 'install it and implement propose() before enabling this backend')


_BACKENDS: dict[str, Callable[[], PolicyBackend]] = {
    'lerobot': LeRobotPolicyBackend,
}


def register(name: str, factory: Callable[[], PolicyBackend]) -> None:
    _BACKENDS[name] = factory


def policy_backend_from_env(environ=None) -> PolicyBackend | None:
    """The configured backend, or None. An unknown name is a configuration error, not a silent no-op."""
    environ = os.environ if environ is None else environ
    name = (environ.get('OPENARM_POLICY_BACKEND') or '').strip().lower()
    if not name:
        return None
    factory = _BACKENDS.get(name)
    if factory is None:
        raise PolicyBackendError(f'POLICY_BACKEND_UNKNOWN: {name!r} is not registered '
                                 f'(known: {", ".join(sorted(_BACKENDS))})')
    return factory()
