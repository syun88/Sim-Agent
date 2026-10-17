"""Robot skills executing a physical pick and place.

The grasp is a sequence the official OpenArm v2 hand makes possible and the skills only orchestrate:

1. open the gripper;
2. IK the gripper so the hand's *fingertip midpoint* (the midpoint between the two fingertip pad
   centroids of the official model, `FINGERTIP_MIDPOINT`) lands on the object centre, approaching
   through a high waypoint so the open fingers never sweep through the object;
3. close the fingers - the position-controlled fingertip pads press onto the object's side faces,
   and the official finger mimic equality keeps the two fingers symmetric;
4. verify the contact: both finger links must touch the object, or the grasp fails;
5. verify the hold physically: a short lift must carry the object with the gripper, or the grasp
   fails - no engine state is written to make that happen;
6. lift, transport, release and retreat.

Nothing here writes an object's pose: the only object position writes are scene reset and setup,
and `SkillError` is raised whenever the physical evidence (contacts, the object following the
gripper) contradicts the intended action.
"""

from __future__ import annotations

import time
from typing import Callable

import numpy as np

from backend.app.control.motion import OpenArmV2Controller
from backend.app.simulation.base import SimulationBackend
from backend.app.simulation.openarm_v2.definition import (
    FINGERTIP_MIDPOINT,
    GRASP_ROTATION,
    RELEASE_HEIGHT,
)


class SkillError(Exception):
    def __init__(self, category: str, message: str):
        super().__init__(message)
        self.category = category


#: Height of the in-grasp verification lift (m). The object must rise at least
#: `VERIFICATION_LIFT * VERIFICATION_FOLLOW_RATIO` with the gripper for the hold to count.
VERIFICATION_LIFT = 0.03
VERIFICATION_FOLLOW_RATIO = 0.6
#: How far the gripper pulls back along the approach axis before the straight slide-in (m).
APPROACH_STANDOFF = 0.06
#: Height of the swing waypoint above the approach-axis start (m), keeping the open fingers
#: clear of the object during the swing from the staging altitude.
APPROACH_OVERFLY = 0.10
#: Altitude the arm rises to above its current position before any lateral swing (m, world z).
#: The spawn pose holds the hand extended almost horizontally, and a joint-interpolated swing to a
#: lateral target yaws and folds the whole arm through the workspace; rising first puts that sweep
#: above the tallest object. 1.24 m is the highest the official arm reaches with the documented
#: horizontal approach orientation (measured: 1.0 mm IK residual at 1.24, 12.7 mm at 1.26), which
#: already holds the fingertips 15 cm above a resting cube.
STAGING_ALTITUDE = 1.24
#: A rise shorter than this is not worth a waypoint.
MIN_STAGING_RISE = 0.02
#: The approach may not displace the object by more than this (m); the object is only pinched
#: after this check, so a shove here would fake nothing anyway, but the plan still fails loudly.
APPROACH_DISTURBANCE_TOLERANCE = 0.005


class SkillRunner:
    def __init__(self, env: SimulationBackend, controller: OpenArmV2Controller):
        self.env = env
        self.controller = controller

    # -- grasp geometry ----------------------------------------------------------

    def _pinch_site_target(
        self, arm: str, object_center: np.ndarray, world_x_offset: float | None = None
    ) -> np.ndarray:
        """Control-point position that puts the fingertip midpoint on the object centre.

        `FINGERTIP_MIDPOINT` is the hand's fingertip midpoint in the control-point frame, so the
        object centre sits at `GRASP_ROTATION @ FINGERTIP_MIDPOINT` relative to the control point.
        `world_x_offset` replaces that offset's world-x component; it exists only so a controlled
        fault can be injected for the retry path.
        """
        grasp_offset = GRASP_ROTATION @ FINGERTIP_MIDPOINT[arm]
        if world_x_offset is not None:
            grasp_offset = grasp_offset.copy()
            grasp_offset[0] = world_x_offset
        return np.asarray(object_center, float) - grasp_offset

    def _contact_fingers(self, obj: str, arm: str) -> set[str]:
        """Finger links currently touching the object, e.g. {'openarm_left_ee_inner_finger'}."""
        fingers = set()
        for contact in self.env.object_contacts(obj, arm):
            fingers.add(contact.split("|")[0])
        return fingers

    def _pinch_established(self, obj: str, arm: str) -> bool:
        """A pinch needs both the inner and the outer finger on the object."""
        fingers = self._contact_fingers(obj, arm)
        inner = any("inner" in finger for finger in fingers)
        outer = any("outer" in finger for finger in fingers)
        return inner and outer

    def _skill(self, name: str, run: Callable[[], dict], started: float) -> dict:
        try:
            result = run()
            return {"skill": name, "success": True, "duration": time.perf_counter() - started, **result}
        except SkillError as exc:
            return {"skill": name, "success": False, "duration": time.perf_counter() - started, "error": exc.category, "detail": str(exc)}

    # -- skills ------------------------------------------------------------------

    def approach_object(self, arm: str, obj: str, progress=None, fault_offset: float | None = None) -> dict:
        started = time.perf_counter()
        return self._skill(
            "approach_object",
            lambda: self._approach(arm, obj, progress, fault_offset),
            started,
        )

    def _approach(self, arm: str, obj: str, progress=None, fault_offset: float | None = None) -> dict:
        obj_pos, _ = self.env.object_pose(obj)
        pinch = self._pinch_site_target(arm, obj_pos, fault_offset)
        finger_direction = GRASP_ROTATION @ np.array([0.0, 0.0, -1.0])
        standoff = pinch - APPROACH_STANDOFF * finger_direction
        high = standoff + np.array([0.0, 0.0, APPROACH_OVERFLY])
        before = obj_pos.copy()
        # 1) open the gripper
        self.controller.wait_gripper(arm, False, 0.35)
        # 2) rise to the staging altitude above the current pose, so the lateral swing that
        #    follows happens above every object on the table;
        ee_position, _ = self.env.ee_pose(arm)
        rise = STAGING_ALTITUDE - ee_position[2]
        waypoints: list[tuple[str, np.ndarray]] = []
        if rise > MIN_STAGING_RISE:
            waypoints.append(("staging", ee_position + np.array([0.0, 0.0, rise])))
        # 3) swing high above the object, drop onto the approach axis, slide straight in
        waypoints += (("pregrasp_high", high), ("pregrasp", standoff), ("pinch", pinch))
        for name, target in waypoints:
            result = self.controller.move_to_site(arm, target, progress=progress)
            if not result.success:
                raise SkillError(result.error or "IK_FAILURE", f"Unable to reach the {name} pose")
        after = self.env.object_pose(obj)[0]
        disturbance = float(np.linalg.norm(after - before))
        if disturbance > APPROACH_DISTURBANCE_TOLERANCE:
            raise SkillError(
                "APPROACH_COLLISION",
                f"The approach displaced {obj} by {disturbance * 1000:.1f} mm",
            )
        midpoint_error = float(np.linalg.norm(after - (self.env.ee_pose(arm)[0] + GRASP_ROTATION @ FINGERTIP_MIDPOINT[arm])))
        return {
            "target": pinch.tolist(),
            "final": result.final_position,
            "fingertip_midpoint_error": midpoint_error,
            "object_displacement": disturbance,
        }

    def grasp_object(self, arm: str, obj: str) -> dict:
        started = time.perf_counter()
        return self._skill(
            "grasp_object",
            lambda: self._grasp(arm, obj),
            started,
        )

    def _grasp(self, arm: str, obj: str) -> dict:
        before = self.env.object_pose(obj)[0].copy()
        # 3) close the fingers onto the object
        self.controller.wait_gripper(arm, True, 0.65)
        contacts = self._contact_fingers(obj, arm)
        # 4) verify the contact: a pinch needs both fingers on the object
        if not self._pinch_established(obj, arm):
            raise SkillError(
                "GRASP_FAILED",
                f"No pinch contact with {obj}; finger links touching: {sorted(contacts)}",
            )
        # 5) verify the hold physically: a short lift must carry the object with the gripper
        ee_position = self.env.ee_pose(arm)[0]
        lift = self.controller.move_to_site(arm, ee_position + np.array([0.0, 0.0, VERIFICATION_LIFT]))
        if not lift.success:
            raise SkillError(lift.error or "IK_FAILURE", "The hold-verification lift failed")
        after = self.env.object_pose(obj)[0]
        rose = float(after[2] - before[2])
        fingers_after = self._contact_fingers(obj, arm)
        if rose < VERIFICATION_FOLLOW_RATIO * VERIFICATION_LIFT or not fingers_after:
            raise SkillError(
                "GRASP_LOST",
                f"{obj} did not follow the verification lift (rose {rose * 1000:.1f} mm)",
            )
        return {
            "grasp_mode": self.env.grasp_mode,
            "grasp_is_physical_contact": self.env.grasp_is_physical_contact,
            "pinch_contacts": sorted(contacts),
            "contacts_after_lift": sorted(fingers_after),
            "position_before": before.tolist(),
            "position_after": after.tolist(),
            "verification_lift": VERIFICATION_LIFT,
            "object_rose": rose,
        }

    def lift_object(self, arm: str, obj: str, height: float = 0.06) -> dict:
        """Lift the pinched object off the table for transport.

        The height is bounded by the arm's workspace, not by taste: with the fingertip grasp the
        control point sits 16 cm closer to the shoulder than the old kinematic hold, and 1.18 m is
        the ceiling the official arm reaches there with the horizontal approach orientation. A
        6 cm lift puts a resting cube's bottom face ~9 cm above the table, which the verified
        transport paths never come close to.
        """
        started = time.perf_counter()
        return self._skill("lift_object", lambda: self._move_relative(arm, [0, 0, height]), started)

    def move_to_target(self, arm: str, target: str) -> dict:
        started = time.perf_counter()
        return self._skill("move_to_target", lambda: self._move_to_zone(arm, target), started)

    def _move_to_zone(self, arm: str, target: str) -> dict:
        zone = self.env.zone_position(target)
        desired_object = zone.copy()
        desired_object[2] = RELEASE_HEIGHT
        site_target = self._pinch_site_target(arm, desired_object)
        result = self.controller.move_to_site(arm, site_target)
        if not result.success:
            raise SkillError(result.error or "IK_FAILURE", "Unable to reach target")
        return {"target": site_target.tolist(), "final": result.final_position}

    def release_object(self, arm: str, obj: str) -> dict:
        started = time.perf_counter()
        return self._skill("release_object", lambda: self._release(arm, obj), started)

    def _release(self, arm: str, obj: str) -> dict:
        pos_before = self.env.object_pose(obj)[0].copy()
        # Opening the fingers ends the pinch; gravity carries the object the rest of the way.
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
                result = self.approach_object(arm, step["object"], progress, step.get("fault_offset"))
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
