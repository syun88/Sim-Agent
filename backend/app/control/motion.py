from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import mujoco
import numpy as np

from backend.app.simulation.openarm_v2.environment import (
    ARM_JOINTS,
    EE_SITES,
    GRIPPER_CLOSED,
    GRIPPER_OPEN,
    OpenArmV2Environment,
)

GRASP_OFFSET = np.array([0.083, 0.0, -0.019])
GRASP_ROTATION = np.array([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])


@dataclass
class MotionResult:
    success: bool
    error: str | None = None
    final_position: list[float] | None = None
    position_error: float | None = None


class OpenArmV2Controller:
    def __init__(self, env: OpenArmV2Environment):
        self.env = env
        self.model = env.model
        self.data = env.data
        self.cancelled = False

    def cancel(self) -> None:
        self.cancelled = True

    def _ids(self, side: str) -> tuple[list[int], np.ndarray, np.ndarray]:
        joints = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in ARM_JOINTS[side]]
        qadr = np.array([self.model.jnt_qposadr[j] for j in joints])
        dof = np.array([self.model.jnt_dofadr[j] for j in joints])
        return joints, qadr, dof

    def orientation_error(self, current: np.ndarray, desired: np.ndarray) -> np.ndarray:
        qc = np.zeros(4)
        qd = np.zeros(4)
        qerr = np.zeros(4)
        vel = np.zeros(3)
        mujoco.mju_mat2Quat(qc, current.flatten())
        mujoco.mju_mat2Quat(qd, desired.flatten())
        mujoco.mju_negQuat(qc, qc)
        mujoco.mju_mulQuat(qerr, qd, qc)
        mujoco.mju_quat2Vel(vel, qerr, 1.0)
        return vel

    def solve_ik(
        self,
        side: str,
        target_position: np.ndarray,
        target_rotation: np.ndarray | None = None,
        q_init: np.ndarray | None = None,
        max_iterations: int = 350,
    ) -> tuple[np.ndarray | None, float, float]:
        site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, EE_SITES[side])
        _, qadr, dof = self._ids(side)
        limits = self.model.jnt_range[[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in ARM_JOINTS[side]]]
        if q_init is None:
            q = self.data.qpos[qadr].copy()
        else:
            q = np.clip(np.asarray(q_init, float), limits[:, 0], limits[:, 1])
        if target_rotation is None:
            target_rotation = self.data.site_xmat[site].reshape(3, 3).copy()
        qpos_backup = self.data.qpos.copy()
        best_q = q.copy()
        best_error = np.inf
        for iteration in range(max_iterations):
            self.data.qpos[qadr] = q
            mujoco.mj_kinematics(self.model, self.data)
            mujoco.mj_comPos(self.model, self.data)
            position = self.data.site_xpos[site].copy()
            rotation = self.data.site_xmat[site].reshape(3, 3).copy()
            perr = np.asarray(target_position, float) - position
            oerr = self.orientation_error(rotation, target_rotation)
            error = float(np.linalg.norm(perr) + np.linalg.norm(oerr) * 0.2)
            if error < best_error:
                best_error = error
                best_q = q.copy()
            if np.linalg.norm(perr) < 0.0012 and np.linalg.norm(oerr) < 0.015:
                break
            jacp = np.zeros((3, self.model.nv))
            jacr = np.zeros((3, self.model.nv))
            mujoco.mj_jac(self.model, self.data, jacp, jacr, position, self.model.site_bodyid[site])
            J = np.vstack([jacp[:, dof], jacr[:, dof]])
            task = np.concatenate([perr, oerr])
            damping = 0.0007
            dq = J.T @ np.linalg.solve(J @ J.T + damping * np.eye(6), task)
            step = np.linalg.norm(dq)
            if step > 0.18:
                dq *= 0.18 / step
            q = np.clip(q + dq, limits[:, 0], limits[:, 1])
        self.data.qpos[:] = qpos_backup
        mujoco.mj_forward(self.model, self.data)
        if best_error > 0.02:
            return None, best_error, float(best_error)
        return best_q, best_error, float(best_error)

    def execute_joint_path(
        self,
        side: str,
        path: list[np.ndarray],
        seconds_per_segment: float = 0.06,
        progress: Callable[[str], None] | None = None,
    ) -> MotionResult:
        ids = self.env._actuator_ids[side]
        _, qadr, _ = self._ids(side)
        steps_per = max(1, int(seconds_per_segment / self.model.opt.timestep))
        _, _, dof = self._ids(side)
        for i, target in enumerate(path):
            if self.cancelled:
                return MotionResult(False, "STOPPED")
            if progress and i % 8 == 0:
                progress(f"joint_path_{i}/{len(path)}")
            q_start = self.data.qpos[qadr].copy()
            for alpha in np.linspace(0, 1, max(2, int(np.linalg.norm(target - q_start) / 0.005))):
                qcmd = (1 - alpha) * q_start + alpha * target
                self.env.set_arm(side, qcmd)
                self.data.qfrc_applied[dof] = self.data.qfrc_bias[dof]
                self.env.step(1)
                if self.cancelled:
                    return MotionResult(False, "STOPPED")
            self.env.set_arm(side, target)
            self.data.qfrc_applied[dof] = self.data.qfrc_bias[dof]
            self.env.step(steps_per)
        pos, _ = self.env.ee_pose(side)
        return MotionResult(True, final_position=pos.tolist(), position_error=0.0)

    def move_to_site(
        self,
        side: str,
        position: np.ndarray,
        rotation: np.ndarray | None = None,
        progress: Callable[[str], None] | None = None,
    ) -> MotionResult:
        site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, EE_SITES[side])
        if rotation is None:
            rotation = GRASP_ROTATION.copy()
        q, pos_err, rot_err = self.solve_ik(side, position, rotation)
        if q is None:
            return MotionResult(False, "IK_FAILURE", position.tolist(), pos_err)
        start = self.data.qpos[self._ids(side)[1]].copy()
        path = [(1-a) * start + a * q for a in np.linspace(0, 1, 8)]
        return self.execute_joint_path(side, path, 0.08, progress)

    def grasp_target(self, object_position: np.ndarray) -> np.ndarray:
        return np.asarray(object_position, float) - GRASP_OFFSET

    def set_gripper(self, side: str, closed: bool) -> None:
        self.env.set_gripper(side, GRIPPER_CLOSED[side] if closed else GRIPPER_OPEN[side])

    def wait_gripper(self, side: str, closed: bool, seconds: float = 0.45) -> MotionResult:
        self.set_gripper(side, closed)
        for _ in range(max(1, int(seconds / self.model.opt.timestep))):
            self.env.step(1)
        return MotionResult(True)
