from __future__ import annotations

import time
from dataclasses import asdict
from typing import Callable

import mujoco
import numpy as np

from backend.app.control.motion import GRASP_OFFSET, OpenArmV2Controller
from backend.app.simulation.openarm_v2.environment import OpenArmV2Environment


class SkillError(Exception):
    def __init__(self, category: str, message: str):
        super().__init__(message)
        self.category = category


class SkillRunner:
    def __init__(self, env: OpenArmV2Environment, controller: OpenArmV2Controller):
        self.env = env
        self.controller = controller

    def _weld(self, arm: str, obj: str) -> int:
        return mujoco.mj_name2id(
            self.env.model, mujoco.mjtObj.mjOBJ_EQUALITY, f"grasp_{arm}_{obj}"
        )

    def _set_weld(self, arm: str, obj: str, active: bool) -> None:
        eid = self._weld(arm, obj)
        if eid >= 0:
            if active:
                arm_body = mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_BODY, f"openarm_{arm}_ee_base_link")
                obj_body = self.env._object_bodies[obj]
                relative_rotation = np.zeros(4)
                inverse_arm_rotation = np.zeros(4)
                mujoco.mju_negQuat(inverse_arm_rotation, self.env.data.xquat[arm_body])
                mujoco.mju_mulQuat(relative_rotation, inverse_arm_rotation, self.env.data.xquat[obj_body])
                desired_world = GRASP_OFFSET.copy()
                desired_local = self.env.data.xmat[arm_body].reshape(3, 3).T @ desired_world
                self.env.model.eq_data[eid, 3:6] = desired_local
                self.env.model.eq_data[eid, 6:10] = relative_rotation
                self.env.model.eq_data[eid, 10] = 1.0
            self.env.data.eq_active[eid] = bool(active)
            mujoco.mj_forward(self.env.model, self.env.data)

    def _skill(self, name: str, run: Callable[[], dict], started: float) -> dict:
        try:
            result = run()
            return {"skill": name, "success": True, "duration": time.perf_counter() - started, **result}
        except SkillError as exc:
            return {"skill": name, "success": False, "duration": time.perf_counter() - started, "error": exc.category, "detail": str(exc)}

    def approach_object(self, arm: str, obj: str, progress=None, fault_offset: float | None = None, approach_offset: float | None = None) -> dict:
        started = time.perf_counter()
        return self._skill(
            "approach_object",
            lambda: self._approach(arm, obj, progress, fault_offset, approach_offset),
            started,
        )

    def _approach(self, arm: str, obj: str, progress=None, fault_offset: float | None = None, approach_offset: float | None = None) -> dict:
        obj_pos, _ = self.env.object_pose(obj)
        x_offset = GRASP_OFFSET[0] if approach_offset is None else approach_offset
        if fault_offset is not None:
            x_offset = fault_offset
        target = obj_pos - np.array([x_offset, 0.0, -GRASP_OFFSET[2]])
        result = self.controller.move_to_site(arm, target, progress=progress)
        if not result.success:
            raise SkillError(result.error or "IK_FAILURE", "Unable to reach object")
        return {"target": target.tolist(), "final": result.final_position}

    def grasp_object(self, arm: str, obj: str) -> dict:
        started = time.perf_counter()
        return self._skill(
            "grasp_object",
            lambda: self._grasp(arm, obj),
            started,
        )

    def _grasp(self, arm: str, obj: str) -> dict:
        self.controller.wait_gripper(arm, False, 0.35)
        before = self.env.object_pose(obj)[0].copy()
        self.controller.wait_gripper(arm, True, 0.65)
        contacts = self.env.object_contacts(obj, arm)
        self._set_weld(arm, obj, True)
        after = self.env.object_pose(obj)[0].copy()
        return {
            "grasp_mode": "contact-assisted-weld",
            "physical_contacts": contacts,
            "position_before": before.tolist(),
            "position_after": after.tolist(),
        }

    def lift_object(self, arm: str, obj: str, height: float = 0.10) -> dict:
        started = time.perf_counter()
        return self._skill("lift_object", lambda: self._move_relative(arm, [0, 0, height]), started)

    def move_to_target(self, arm: str, target: str) -> dict:
        started = time.perf_counter()
        return self._skill("move_to_target", lambda: self._move_to_zone(arm, target), started)

    def _move_to_zone(self, arm: str, target: str) -> dict:
        zone = self.env.zone_position(target)
        desired_object = zone.copy()
        desired_object[2] = 1.032
        site_target = desired_object - GRASP_OFFSET
        result = self.controller.move_to_site(arm, site_target)
        if not result.success:
            raise SkillError(result.error or "IK_FAILURE", "Unable to reach target")
        return {"target": site_target.tolist(), "final": result.final_position}

    def release_object(self, arm: str, obj: str) -> dict:
        started = time.perf_counter()
        return self._skill("release_object", lambda: self._release(arm, obj), started)

    def _release(self, arm: str, obj: str) -> dict:
        pos_before = self.env.object_pose(obj)[0].copy()
        self._set_weld(arm, obj, False)
        self.controller.wait_gripper(arm, False, 0.45)
        return {"position_before": pos_before.tolist(), "position_after": self.env.object_pose(obj)[0].tolist()}

    def _move_relative(self, arm: str, delta: list[float]) -> dict:
        pos, rotation = self.env.ee_pose(arm)
        result = self.controller.move_to_site(arm, pos + np.asarray(delta), rotation)
        if not result.success:
            raise SkillError(result.error or "IK_FAILURE", "Relative move failed")
        return {"final": result.final_position}

    def retreat(self, arm: str) -> dict:
        started = time.perf_counter()
        return self._skill("retreat", lambda: self._move_relative(arm, [-0.08, 0, 0.10]), started)

    def return_home(self, arm: str) -> dict:
        started = time.perf_counter()
        def run():
            target = np.array([0.401, 0.1535 if arm == "left" else -0.1535, 1.120])
            result = self.controller.move_to_site(arm, target)
            if not result.success:
                raise SkillError(result.error or "IK_FAILURE", "Home pose unreachable")
            return {"final": result.final_position}
        return self._skill("return_home", run, started)

    def execute_plan(self, plan: list[dict], progress=None) -> list[dict]:
        results = []
        for step in plan:
            skill = step["skill"]
            arm = step.get("arm", "left")
            if progress:
                progress(skill)
            if skill == "approach_object":
                result = self.approach_object(arm, step["object"], progress, step.get("fault_offset"), step.get("approach_offset"))
            elif skill == "grasp_object":
                result = self.grasp_object(arm, step["object"])
            elif skill == "lift_object":
                result = self.lift_object(arm, step["object"])
            elif skill == "move_to_target":
                result = self.move_to_target(arm, step["target"])
            elif skill == "release_object":
                result = self.release_object(arm, step["object"])
            elif skill == "retreat":
                result = self.retreat(arm)
            elif skill == "return_home":
                result = self.return_home(arm)
            else:
                result = {"skill": skill, "success": False, "duration": 0.0, "error": "INVALID_PLAN", "detail": "Unknown skill"}
            results.append(result)
            if not result.get("success", False):
                break
        return results
