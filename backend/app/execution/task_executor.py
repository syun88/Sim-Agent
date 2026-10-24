"""One synchronous, stateful execution path for native viewer, tests, and async wrapper.

Each skill observes the scene again. Recovery is bounded and resumes from actual state;
it never resets the scene or replays completed handovers to manufacture success.
"""
from __future__ import annotations

import time
import numpy as np

from backend.app.agent.world_state import build_world_state
from backend.app.control.motion import OpenArmV2Controller
from backend.app.planning.agent_planner import validate_plan
from backend.app.skills.robot_skills import SkillRunner
from backend.app.verification.goal_verification import verify_task_goal


class TaskExecutor:
    def __init__(self, env):
        self.env = env
        self.controller = OpenArmV2Controller(env)
        self.skills = SkillRunner(env, self.controller)
        self.discourse: dict = {}

    def world_state(self):
        return {**build_world_state(self.env), "discourse": self.discourse.copy()}

    def reset_context(self):
        self.discourse.clear()
        self.controller.target_rotations.clear()

    def home_empty_arms(self, *, emit=None, should_stop=None, clear_contacts=False):
        """Return free arms to a clear home posture, preserving objects/discourse.

        A holding task retains its physical grasp and loaded arm. Contact
        clearance is a separately recorded recovery, never a scene reset.
        """
        self.controller.should_stop=should_stop
        self.controller.cancelled=False
        before=self.world_state()
        results=[]
        for arm in before['robot'].get('available_arms',['left','right']):
            if should_stop and should_stop():
                return {'success':False,'status':'STOPPED','arms':results}
            held=self.skills.held_object(arm)
            if held:
                results.append({'arm':arm,'success':True,'skipped':'holding_requested_object','object':held})
                continue
            contacts=[o['id'] for o in self.world_state()['objects'] if self.env.object_contacts(o['id'],arm)]
            if contacts and clear_contacts:
                opening=self.controller.wait_gripper(arm,False)
                if not opening.success:
                    return {'success':False,'status':'FAILED','arms':results,'failure_category':opening.error}
                retreat=self.skills.retreat(arm)
                results.append({'arm':arm,'phase':'clear_contacts',**retreat})
                if not retreat['success']:
                    return {'success':False,'status':'FAILED','arms':results,'failure_category':retreat.get('error')}
            target=self.env.home_arm_qpos(arm)
            if np.max(abs(self.env.arm_qpos(arm)-target)) <= .025 and not any(
                    self.env.object_contacts(o['id'],arm) for o in self.world_state()['objects']):
                results.append({'arm':arm,'success':True,'already_home':True,
                    'maximum_joint_error_rad':float(np.max(abs(self.env.arm_qpos(arm)-target)))})
                continue
            if emit:
                emit('task_homing',status='HOMING',detail=f'{arm}: 物体の配置を維持してホームへ戻します。')
            result=self.skills.park_empty_arm(arm)
            results.append({'arm':arm,**result})
            if not result['success']:
                return {'success':False,'status':'STOPPED' if self.controller.stop_requested() else 'FAILED',
                    'arms':results,'failure_category':result.get('error')}
        after=self.world_state()
        held={v.get('held_object') for k,v in before['robot'].items() if k in {'left_arm','right_arm'}}
        positions={o['id']:o['position'] for o in before['objects']}
        displacement={o['id']:float(np.linalg.norm(np.asarray(o['position'])-positions[o['id']]))
                      for o in after['objects'] if o['id'] not in held}
        preserved=all(d<.006 for d in displacement.values())
        return {'success':preserved,'status':'SUCCEEDED' if preserved else 'FAILED','arms':results,
                'stationary_object_displacements_m':displacement,
                'failure_category':None if preserved else 'HOMING_DISTURBED_SCENE'}

    def run(self, plan, *, emit=None, should_stop=None, force_failure=False) -> dict:
        if plan.goal_program:
            from backend.app.execution.goal_executor import run_goal_program
            return run_goal_program(self, plan, emit=emit, should_stop=should_stop, force_failure=force_failure)
        started = time.perf_counter()
        self.controller.cancelled = False
        self.controller.should_stop = should_stop
        initial = self.world_state()
        valid, errors = validate_plan(plan, initial)
        if not valid:
            return {"success": False, "status": "FAILED", "detail": "INVALID_PLAN: " + "; ".join(errors),
                    "plan": plan.model_dump(), "skills": [], "verification": {"success": False, "failure_category": "INVALID_PLAN"},
                    "initial_world_state": initial, "final_world_state": initial, "retry_count": 0,
                    "total_duration": time.perf_counter() - started}
        results = []
        retry_count = 0
        for index, original in enumerate(plan.steps):
            if self.controller.stop_requested():
                break
            step = original.model_dump()
            if force_failure and index == 0 and step["skill"] == "approach_object":
                step["fault_offset"] = -.12
            before = build_world_state(self.env)
            if emit:
                emit("skill_started", step_index=index, step=step, world_state=before)
            attempts = []
            result = self.skills.execute_plan([step])[0]
            attempts.append(result)
            # Reacquire a failed table pinch only while neither hand holds the object.
            # A lost transport or a partly completed handover needs explicit recovery.
            if not result["success"] and result.get("error") in {"GRASP_FAILED", "IK_FAILURE"} and step["skill"] in {"approach_object", "grasp_object"} and not self.controller.stop_requested():
                arm, obj = step["arm"], step["object"]
                if not self.skills.held_object(arm) and not any(self.env.object_contacts(obj, side) for side in ("left", "right")):
                    retry_count += 1
                    if emit:
                        emit("skill_retry", step_index=index, reason=result.get("error"))
                    self.controller.wait_gripper(arm, False)
                    retry = self.skills.approach_object(arm, obj, grasp_offset_x=step.get("grasp_offset_x", 0.))
                    if retry["success"] and step["skill"] == "grasp_object":
                        retry = self.skills.grasp_object(arm, obj)
                    result = retry
                    attempts.append(retry)
            result = {**result, "step_index": index, "action_index": step.get("action_index"), "arm": step["arm"],
                      "object": step.get("object"), "attempts": attempts, "world_state": build_world_state(self.env)}
            results.append(result)
            if result["success"] and step.get("action_index") is not None and plan.program:
                action = plan.program["actions"][step["action_index"]]
                if action.get("object"):
                    self.discourse.update(last_object=action["object"], last_arm=action.get("receiver") or action["arm"])
                if action.get("target") and step["skill"] == "release_object":
                    self.discourse["last_target"] = action["target"]
            if not result["success"]:
                result["contacts_after_failure"] = self.env.contact_names()
            if emit:
                emit("skill_finished", step_index=index, result=result)
            if not result["success"]:
                break
        stopped = self.controller.stop_requested()
        if emit:
            emit("verifying")
        verification = verify_task_goal(self.env, plan, results, initial_state=initial)
        stopped = self.controller.stop_requested() or stopped
        success = verification["success"] and not stopped
        return {"success": bool(success), "status": "STOPPED" if stopped else "SUCCEEDED" if success else "FAILED",
                "plan": plan.model_dump(), "skills": results, "verification": verification,
                "initial_world_state": initial, "final_world_state": build_world_state(self.env),
                "retry_count": retry_count, "total_duration": time.perf_counter() - started}
