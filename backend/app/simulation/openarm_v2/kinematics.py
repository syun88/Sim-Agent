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


class OfficialGravityKinematics:
    """Gravity lever arms from the official model; no dynamics solver is stepped.

    This avoids querying every live link Jacobian on both CPU and GPU. Input is
    always the actual Genesis configuration; only the known robot's own mass is
    compensated, so payloads and fluid still exert their real forces in Genesis.
    """
    def __init__(self, model):
        import mujoco
        self._mj, self._model = mujoco, model
        self._data = mujoco.MjData(model)
        self._jac = np.zeros((3, model.nv))
        self._dofs = {}
        self._bodies = {}
        for side in ARM_JOINTS:
            joints = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in ARM_JOINTS[side]]
            self._dofs[side] = model.jnt_dofadr[joints]
            self._bodies[side] = [i for i in range(model.nbody)
                                  if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or "").startswith(f"openarm_{side}_")]

    def evaluate(self, qpos: np.ndarray, gravity: np.ndarray) -> dict[str, np.ndarray]:
        self._data.qpos[:] = qpos
        self._mj.mj_kinematics(self._model, self._data)
        self._mj.mj_comPos(self._model, self._data)
        torques = {}
        for side, bodies in self._bodies.items():
            torque = np.zeros(7)
            for body in bodies:
                self._mj.mj_jacBodyCom(self._model, self._data, self._jac, None, body)
                torque -= self._jac[:, self._dofs[side]].T @ (self._model.body_mass[body] * gravity)
            torques[side] = torque
        return torques
