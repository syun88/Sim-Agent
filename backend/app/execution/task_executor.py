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

    #: Measured on this host 2026-10-27 with obvious questions: noul is P(YES) - "is the sky blue" 0.92,
    #: "does a robot arm have joints" 0.97, "can a cube fly by itself" 0.13. The first version of this read it
    #: backwards and would have recorded every agreement as a disagreement.
    MONITOR_TIMEOUT = 3.
    #: Steps whose outcome is "the hand is still holding the object". A release or a throw ends the hold, so
    #: asking whether the report agrees with holding something confuses the question rather than testing it.
    MONITORED_STEPS = {"grasp_object", "lift_object", "move_to_object", "move_to_target", "handover_object"}

    def _monitor_step(self, step, result):
        """Ask the local decision model one closed question about this step, if it is running and fast enough."""
        from backend.app.planning import decision_model

        if not decision_model.available() or step.get("skill") not in self.MONITORED_STEPS:
            return None
        if getattr(self, '_monitor_latency', 0.) > self.MONITOR_TIMEOUT:
            # A monitor that costs more than the step it watches is not a reflex; record why it was skipped
            # instead of silently paying for it on every step.
            return {"step": step.get("skill"), "asked": False, "skipped_because": "measured_latency_too_high",
                    "advisory": True}
        obj = step.get("object")
        if not obj:
            return None
        try:
            held = bool(self.skills.held_object(step.get("arm", "left")))
        except Exception:
            held = None
        state = (f"Robot step just finished: {step.get('skill')} on {obj}. Reported success: "
                 f"{bool(result.get('success'))}. The arm reports holding an object: {held}. "
                 f"Failure category from the skill, if any: {result.get('error') or 'none'}.")
        questions = {"agrees": {
            "type": "noul",
            "instructions": "Does the skill's reported outcome agree with the robot's own report of holding "
                            "an object after this step? Answer yes if they agree."}}
        started = __import__('time').perf_counter()
        answer = decision_model.ask(state, questions, timeout=self.MONITOR_TIMEOUT)
        self._monitor_latency = __import__('time').perf_counter() - started
        if not answer:
            return None
        judgement = (answer.get("answers") or {}).get("agrees") or {}
        if "noul" not in judgement:
            return None
        p_yes = float(judgement["noul"])
        return {"step": step.get("skill"), "asked": True, "agrees": p_yes >= .5, "p_yes": round(p_yes, 3),
                "latency_seconds": round(self._monitor_latency, 2), "advisory": True}

    def reset_context(self):
        self.discourse.clear()
        self.controller.target_rotations.clear()
        # A pinch frame belongs to the approach that planned it, so it must not survive a scene reset: the
        # next task would pinch with the previous task's frame. Measured 2026-10-27 on OpenManipulator-X: the
        # second task in one process closed its jaws fully with a single pad touching (q=0.999, one pad)
        # while the first task in the same process pinched normally, with identical object poses.
        for holder in (self, getattr(self, 'skills', None)):
            if holder is not None:
                holder.__dict__.pop('_pinch_frames', None)

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
            # System 1, per step: a fast typed judgement about whether the hold is still there, recorded next
            # to the outcome. Advisory only - it cannot turn a failure into a success or the reverse; a
            # monitor that is absent, slow or wrong simply leaves the field out.
            monitored = self._monitor_step(step, result)
            if monitored is not None:
                result["monitor"] = monitored
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
