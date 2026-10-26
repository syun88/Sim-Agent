"""Forward kinematics must anchor on the profile's live root transform.

IK (solve_profile_ik) and the collision queries read
``profile['root_transform']`` at call time, and a mobile base re-anchors
that entry to wherever it has driven (the placement search swaps in a
hypothetical one). ``arm_kinematics`` used the construction-time copy
instead, so after a drive the closure follow check compared world targets
against spawn-frame FK and reported a phantom ~0.3 m error on LeKiwi.
"""
import numpy as np

from backend.app.robots.urdf import URDFModel
from backend.app.simulation.genesis_backend.generic_environment import GenericURDFEnvironment


class _Tensor:
    def __init__(self, values):
        self._values = np.asarray(values, dtype=float)

    def cpu(self):
        return self

    def numpy(self):
        return self._values


class _Entity:
    def __init__(self, values):
        self._values = values

    def get_dofs_position(self, dofs):
        return _Tensor(self._values)


def _env(tmp_path):
    source = tmp_path / 'one.urdf'
    source.write_text('<robot name="one"><link name="base"/><link name="tip"/>'
                      '<joint name="a" type="revolute"><parent link="base"/><child link="tip"/>'
                      '<axis xyz="0 0 1"/><limit lower="-3" upper="3" effort="1" velocity="1"/>'
                      '</joint></robot>')
    env = GenericURDFEnvironment.__new__(GenericURDFEnvironment)
    env.model = URDFModel(source)
    env.profile = {'chains': {'left': {'joints': ['a'], 'end_link': 'tip',
                                       'tool_offset': [.1, 0., 0.]}},
                   'root_transform': np.eye(4).tolist()}
    env._all_names = ['a']
    env._all_dofs = [0]
    env._entity = _Entity([0.])
    env._root_transform = np.eye(4)  # the construction-time spawn anchor
    # A desk robot: arm_kinematics re-anchors through _sync_root_transform, which a real
    # instance always has and which returns immediately when there is no mobile base.
    env._mobile_base = None
    return env


def test_arm_kinematics_follows_the_live_profile_root(tmp_path):
    env = _env(tmp_path)
    point, *_ = env.arm_kinematics('left').evaluate([0.])
    np.testing.assert_allclose(point, [.1, 0., 0.], atol=1e-9)
    moved = np.eye(4)
    moved[:3, 3] = [1., 2., 3.]
    env.profile['root_transform'] = moved.tolist()
    point, *_ = env.arm_kinematics('left').evaluate([0.])
    np.testing.assert_allclose(point, [1.1, 2., 3.], atol=1e-9)
