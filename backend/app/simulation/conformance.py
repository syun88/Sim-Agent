"""What any body has to answer before the skills may drive it, checked instead of assumed.

The skills talk to a body through one small set of operations (`object_pose`, `object_contacts`,
`arm_qpos`, `step`, ...). Genesis implements them today; a real arm would implement them through a bridge.
Rather than trusting that a backend "has the methods", this module states the contract each one has to keep
and checks it. The same suite is what a real backend would have to pass, so a hardware port is validated by
evidence instead of by a successful-looking demo.

Two properties matter beyond the shapes: the read-only operations must not move anything (so a planner can ask
questions freely), and `step` must actually advance the engine (so a backend that silently ignores stepping
cannot report motion it did not make).
"""

from __future__ import annotations

import inspect

import numpy as np

#: operation -> (arguments, what it must return, why the skills need it)
READ_ONLY = {
    'frame': ((), 'an object with objects, zones, ee, grippers, contacts',
              'the planner reads the whole world through this'),
    'object_pose': (('name',), '(3,) position and (3,3) rotation', 'grasp and placement targets'),
    'object_bounds': (('name',), '(2,3) axis-aligned bounds', 'support faces and clearance'),
    'object_contacts': (('name', 'arm'), 'a sequence of contact names', 'grasp evidence'),
    'object_linear_speed': (('name',), 'a float', 'detecting an object that is still moving'),
    'ee_pose': (('arm',), '(3,) position and (3,3) rotation', 'every motion target'),
    'arm_qpos': (('arm',), 'a 1-D joint array', 'where the arm actually is'),
    'home_arm_qpos': (('arm',), 'a 1-D joint array of the same length', 'homing'),
    'arm_limits': (('arm',), 'an (n,2) array of joint limits', 'admissible IK solutions'),
    'arm_kinematics': (('arm',), 'an object exposing evaluate(q)',
                       'private forward kinematics for planning'),
}

#: Commands: they may move the body, and they are how the skills act.
COMMANDS = {
    'arm_qpos': 'read the commanded/actual joints',
    'set_arm': 'command joints, optionally with velocities',
    'step': 'advance the engine by n steps',
    'set_gripper': 'command one gripper',
    'gripper_position': 'read the measured finger position',
    'gripper_target': 'the open/closed target for this body',
}


def _shape(value):
    try:
        return tuple(np.shape(value))
    except Exception:
        return None


def check(env, *, arms=('left', 'right'), timestep_advances=True) -> list[dict]:
    """Every deviation from the contract, as `{operation, problem, detail}`. Empty means conforming."""
    problems = []

    def note(operation, problem, detail=''):
        problems.append({'operation': operation, 'problem': problem, 'detail': str(detail)})

    frame = None
    if callable(getattr(env, 'frame', None)):
        try:
            frame = env.frame()
        except Exception as error:
            note('frame', 'raised', f'{type(error).__name__}: {error}')
    else:
        note('frame', 'missing', 'no frame() on this body')

    for name, (arguments, returned, _why) in READ_ONLY.items():
        operation = getattr(env, name, None)
        if not callable(operation):
            note(name, 'missing', f'expected {returned}')
            continue
        try:
            inspect.signature(operation).bind(*arguments)
        except TypeError:
            note(name, 'wrong_arguments', f'expected {arguments}')

    if frame is not None:
        for attribute in ('objects', 'zones', 'ee', 'grippers', 'contacts'):
            if not hasattr(frame, attribute):
                note('frame', 'missing_field', attribute)
        if frame is not None and getattr(frame, 'objects', None):
            if not isinstance(getattr(frame, 'zones', None), dict):
                note('frame', 'wrong_type', 'zones must be a mapping of id to position')

    names = list(getattr(frame, 'objects', {}) or {})
    for name in names[:1]:
        try:
            position, rotation = env.object_pose(name)
            if _shape(position) != (3,) or _shape(rotation) != (3, 3):
                note('object_pose', 'wrong_shape', f'{_shape(position)} / {_shape(rotation)}')
        except Exception as error:
            note('object_pose', 'raised', f'{type(error).__name__}: {error}')
        try:
            bounds = np.asarray(env.object_bounds(name), float)
            if bounds.shape != (2, 3) or np.any(bounds[0] > bounds[1]):
                note('object_bounds', 'wrong_shape_or_order', bounds.shape)
        except Exception as error:
            note('object_bounds', 'raised', f'{type(error).__name__}: {error}')

    available = list(getattr(frame, 'ee', {}) or {}) or list(arms)
    for arm in available[:2]:
        try:
            position, rotation = env.ee_pose(arm)
            if _shape(position) != (3,) or _shape(rotation) != (3, 3):
                note('ee_pose', 'wrong_shape', f'{_shape(position)} / {_shape(rotation)}')
        except Exception as error:
            note('ee_pose', 'raised', f'{type(error).__name__}: {error}')
        try:
            qpos = np.asarray(env.arm_qpos(arm), float)
            home = np.asarray(env.home_arm_qpos(arm), float)
            limits = np.asarray(env.arm_limits(arm), float)
            if qpos.ndim != 1:
                note('arm_qpos', 'not_1d', qpos.shape)
            if home.shape != qpos.shape:
                note('home_arm_qpos', 'length_mismatch', f'{home.shape} vs {qpos.shape}')
            if limits.shape != (qpos.size, 2):
                note('arm_limits', 'wrong_shape', f'{limits.shape} for {qpos.size} joints')
            kinematics = env.arm_kinematics(arm)
            if not callable(getattr(kinematics, 'evaluate', None)):
                note('arm_kinematics', 'no_evaluate', type(kinematics).__name__)
            else:
                evaluated = kinematics.evaluate(qpos)
                if len(evaluated) < 2 or _shape(evaluated[0]) != (3,) or _shape(evaluated[1]) != (3, 3):
                    note('arm_kinematics', 'wrong_evaluate_result', str(evaluated)[:80])
        except Exception as error:
            note('arm_state', 'raised', f'{type(error).__name__}: {error}')

    if timestep_advances and callable(getattr(env, 'step', None)) and frame is not None:
        before = getattr(frame, 'sim_time', None)
        try:
            env.step(1)
            after = env.frame().sim_time if callable(getattr(env, 'frame', None)) else None
            if before is not None and after is not None and float(after) <= float(before):
                note('step', 'did_not_advance', f'{before} -> {after}')
        except Exception as error:
            note('step', 'raised', f'{type(error).__name__}: {error}')

    return problems


def describe() -> list[dict]:
    """The contract itself, for a backend author or a prompt."""
    rows = [{'operation': name, 'arguments': arguments, 'returns': returned, 'why': why,
             'kind': 'read_only'} for name, (arguments, returned, why) in READ_ONLY.items()]
    rows += [{'operation': name, 'why': why, 'kind': 'command'} for name, why in COMMANDS.items()]
    return rows
