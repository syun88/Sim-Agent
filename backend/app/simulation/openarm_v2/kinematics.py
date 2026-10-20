"""Read-only IK evaluation of the attached official OpenArm model.

The compiled model is shared with the metadata reader, but every solve owns a private
MjData snapshot. Only forward kinematics and Jacobians are evaluated here; Genesis owns
all dynamics, live poses, contacts and motor execution.
"""
from __future__ import annotations

import numpy as np

from backend.app.simulation.openarm_v2.definition import ARM_JOINTS, EE_SITES


class OfficialArmKinematics:
    def __init__(self, model, qpos: np.ndarray, side: str):
        import mujoco

        self._mj = mujoco
        self._model = model
        self._data = mujoco.MjData(model)
        self._data.qpos[:] = qpos
        joints = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                           for name in ARM_JOINTS[side]])
        self._positions = model.jnt_qposadr[joints]
        self._dofs = model.jnt_dofadr[joints]
        self._site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, EE_SITES[side])
        self._linear = np.zeros((3, model.nv))
        self._angular = np.zeros((3, model.nv))

    def evaluate(self, joints: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        self._data.qpos[self._positions] = joints
        self._mj.mj_kinematics(self._model, self._data)
        self._mj.mj_comPos(self._model, self._data)
        self._mj.mj_jacSite(self._model, self._data, self._linear, self._angular, self._site)
        return (self._data.site_xpos[self._site].copy(),
                self._data.site_xmat[self._site].reshape(3, 3).copy(),
                self._linear[:, self._dofs].copy(), self._angular[:, self._dofs].copy())
