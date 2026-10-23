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
        self.evidence: dict = {}


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
#: World-frame deltas for human "right/left/forward/back" commands. Positive world y is the left
#: side of the official Cell spawn pose; positive world x is toward the table/front.
DIRECTION_VECTORS = {
    "up": np.array([0.0, 0.0, 1.0]),
    "down": np.array([0.0, 0.0, -1.0]),
    "right": np.array([0.0, -1.0, 0.0]),
    "left": np.array([0.0, 1.0, 0.0]),
    "forward": np.array([1.0, 0.0, 0.0]),
    "back": np.array([-1.0, 0.0, 0.0]),
}


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
        return self.env.pinch_established(obj, arm)

    def held_object(self, arm: str) -> str | None:
        return next((obj for obj in self.env.frame().objects if self._pinch_established(obj, arm)), None)

    def _require_hold(self, arm: str, obj: str) -> None:
        if not self._pinch_established(obj, arm):
            raise SkillError("GRASP_LOST", f"{arm} no longer has both finger contacts with {obj}")

    def _skill(self, name: str, run: Callable[[], dict], started: float) -> dict:
        try:
            result = run()
            return {"skill": name, "success": True, "duration": time.perf_counter() - started, **result}
        except SkillError as exc:
            return {"skill": name, "success": False, "duration": time.perf_counter() - started, "error": exc.category,
                    "detail": str(exc), **({"evidence": exc.evidence} if exc.evidence else {})}

    # -- skills ------------------------------------------------------------------

    def approach_object(self, arm: str, obj: str, progress=None, fault_offset: float | None = None, grasp_offset_x: float = 0.) -> dict:
        started = time.perf_counter()
        return self._skill(
            "approach_object",
            lambda: self._approach(arm, obj, progress, fault_offset, grasp_offset_x),
            started,
        )

    def _approach(self, arm: str, obj: str, progress=None, fault_offset: float | None = None, grasp_offset_x: float = 0.) -> dict:
        if self.held_object(arm):
            raise SkillError("HAND_OCCUPIED", f"Place the object held by {arm} before approaching another cube")
        other = "right" if arm == "left" else "left"
        if self._pinch_established(obj, other):
            raise SkillError("HANDOVER_REQUIRED", f"{obj} is held by {other}; use a handover")
        obj_pos, _ = self.env.object_pose(obj)
        pinch = self._pinch_site_target(arm, obj_pos + [grasp_offset_x, 0., 0.], fault_offset)
        finger_direction = GRASP_ROTATION @ np.array([0.0, 0.0, -1.0])
        standoff = pinch - APPROACH_STANDOFF * finger_direction
        high = standoff + np.array([0.0, 0.0, APPROACH_OVERFLY])
        before = obj_pos.copy()
        # Reject an unreachable pinch before starting the approach. Faster
        # staging must not turn an invalid target into a later contact failure.
        pinch_q, _, _ = self.controller.solve_ik(arm, pinch, GRASP_ROTATION)
        if pinch_q is None:
            raise SkillError("IK_FAILURE", "The requested fingertip pinch pose is unreachable")
        # 1) open the gripper
        self.controller.wait_gripper(arm, False, 0.35)
        # 2) rise to the staging altitude above the current pose, so the lateral swing that
        #    follows happens above every object on the table;
        ee_position, _ = self.env.ee_pose(arm)
        rise = STAGING_ALTITUDE - ee_position[2]
        waypoints: list[tuple[str, np.ndarray]] = []
        if obj == "wine_bottle":
            # The tall neck intersects a cube-style overflight. Pull the open
            # fingers behind the body before the lateral swing instead. These
            # clearances were checked against the official finger collision mesh.
            back = np.array([min(standoff[0] - .01, .23), ee_position[1], 1.16])
            waypoints += (("bottle_backoff", back), ("pregrasp", standoff), ("pinch", pinch))
        elif rise > MIN_STAGING_RISE:
            waypoints.append(("staging", ee_position + np.array([0.0, 0.0, rise])))
        # 3) swing high above the object, drop onto the approach axis, slide straight in
        if obj != "wine_bottle":
            waypoints += (("pregrasp_high", high), ("pregrasp", standoff), ("pinch", pinch))
        destinations = waypoints if self.controller.motion_speed == 1. else [waypoints[-1]]
        for name, target in destinations:
            via = [point for _, point in waypoints[:-1]] if self.controller.motion_speed != 1. else None
            result = self.controller.move_to_site(arm, target, progress=progress, via=via)
            if not result.success:
                raise SkillError(result.error or "IK_FAILURE", f"Unable to reach the {name} pose "
                                 f"(position error={result.position_error}, rotation error={result.rotation_error})")
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
            "grasp_offset_x": grasp_offset_x,
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
        lift = self.controller.move_to_site(arm, ee_position + np.array([0.0, 0.0, VERIFICATION_LIFT]),
                                            duration=.8 if obj == "wine_bottle" else None)
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
        def run():
            self._require_hold(arm, obj)
            result = self._move_relative(arm, [0, 0, height])
            self._require_hold(arm, obj)
            return result
        return self._skill("lift_object", run, started)

    def open_gripper(self, arm: str) -> dict:
        started = time.perf_counter()
        def run():
            self.controller.wait_gripper(arm, False, 0.45)
            measured = self.env.frame().grippers[arm]
            from backend.app.simulation.openarm_v2.definition import GRIPPER_OPEN
            if abs(measured - self.env.gripper_target(arm, False)) > .08:
                raise SkillError("GRIPPER_BLOCKED", "The gripper did not physically open")
            return {"closed": False, "measured_joint": measured}
        return self._skill("open_gripper", run, started)

    def close_gripper(self, arm: str) -> dict:
        started = time.perf_counter()
        def run():
            self.controller.wait_gripper(arm, True, 0.65)
            from backend.app.simulation.openarm_v2.definition import GRIPPER_CLOSED
            measured = self.env.frame().grippers[arm]
            if abs(measured - self.env.gripper_target(arm, True)) > .08 and self.held_object(arm) is None:
                raise SkillError("GRIPPER_BLOCKED", "The empty gripper could not close")
            return {"closed": True, "measured_joint": measured}
        return self._skill("close_gripper", run, started)

    def move_relative(self, arm: str, direction: str, distance_cm: float) -> dict:
        started = time.perf_counter()
        def run():
            if direction not in DIRECTION_VECTORS:
                raise SkillError("INVALID_PLAN", f"Unknown move direction: {direction}")
            if not 1.0 <= distance_cm <= 30.0:
                raise SkillError("INVALID_PLAN", "Relative move must be between 1 and 30 cm")
            return self._move_relative(arm, DIRECTION_VECTORS[direction] * distance_cm / 100.0)
        return self._skill("move_relative", run, started)

    def move_to_target(self, arm: str, target: str) -> dict:
        started = time.perf_counter()
        return self._skill("move_to_target", lambda: self._move_to_zone(arm, target), started)

    def _move_to_zone(self, arm: str, target: str) -> dict:
        obj = self.held_object(arm)
        if obj is None:
            raise SkillError("GRASP_LOST", f"{arm} has no physical pinch to transport")
        zone = self.env.zone_position(target)
        for other, position in self.env.frame().objects.items():
            if other != obj and np.linalg.norm(np.asarray(position)[:2] - zone[:2]) < .075 and abs(position[2] - zone[2]) < .06:
                raise SkillError("TARGET_OCCUPIED", f"{other} occupies {target}; refusing to stack cubes accidentally")
        desired_object = zone.copy()
        desired_object[2] = RELEASE_HEIGHT
        self._align_for_placement(arm, obj)
        travel = desired_object.copy()
        travel[2] = 1.14
        self._carry_to(arm, obj, travel)
        return self._carry_to(arm, obj, desired_object)

    def _align_for_placement(self, arm: str, obj: str) -> None:
        _, rotation = self.env.ee_pose(arm)
        if np.linalg.norm(rotation - GRASP_ROTATION) > .1:
            # The handover attitude fits the shared centre workspace. Return to the
            # calibrated table attitude while the cube is still high and clear.
            ee, rotation = self.env.ee_pose(arm)
            center = self.env.object_pose(obj)[0]
            offset = rotation.T @ (center - ee)
            from scipy.spatial.transform import Rotation, Slerp
            arc = Slerp([0., 1.], Rotation.from_matrix(np.stack([rotation, GRASP_ROTATION])))
            orientations = [arc(alpha).as_matrix() for alpha in np.linspace(0., 1., 9)[1:]]
            positions = [center - attitude @ offset for attitude in orientations]
            for index in range(len(positions)) if self.controller.motion_speed == 1. else [len(positions) - 1]:
                result = self.controller.move_to_site(arm, positions[index], orientations[index],
                    via=positions[:-1] if self.controller.motion_speed != 1. else None,
                    via_rotations=orientations[:-1] if self.controller.motion_speed != 1. else None,
                    on_sample=lambda: self._require_hold(arm, obj))
                if not result.success:
                    raise SkillError(result.error or "IK_FAILURE", "Cannot reorient the physical grasp for table placement")
                self._require_hold(arm, obj)

    def move_to_object(self, arm, obj, target, relation, distance_cm=None):
        def run():
            from backend.app.skills.spatial_placement import relative_destination
            self._require_hold(arm, obj)
            if target == obj or target not in self.env.frame().objects:
                raise SkillError("INVALID_TARGET", "Choose another observed cube as the reference")
            if self.env.object_contacts(target) or self.env.object_linear_speed(target) > .02:
                raise SkillError("REFERENCE_MOVING", "The placement reference must be resting on the table")
            self._align_for_placement(arm, obj)
            desired = relative_destination(self.env, obj, target, relation, arm, distance_cm)
            travel = desired.copy()
            travel[2] = max(1.14, desired[2] + .035)
            self._carry_to(arm, obj, travel)
            placed = self._carry_to(arm, obj, desired)
            return {**placed, "reference_object": target, "relation": relation,
                    "support_bounds": self.env.object_bounds(target).tolist()}
        return self._skill("move_to_object", run, time.perf_counter())

    def _carry_to(self, arm: str, obj: str, desired: np.ndarray) -> dict:
        self._require_hold(arm, obj)
        ee, rotation = self.env.ee_pose(arm)
        start = self.env.object_pose(obj)[0].copy()
        offset = rotation.T @ (start - ee)
        commanded_rotation = self.controller.target_rotations.get(arm, rotation)
        target = np.asarray(desired) - commanded_rotation @ offset
        # Keep the carried cube on a Cartesian clearance path. A joint interpolation
        # between two high endpoints can still dip the palm through a table object.
        count = max(1, int(np.ceil(np.linalg.norm(np.asarray(desired) - start) / .025)))
        via = [(1 - alpha) * start + alpha * np.asarray(desired) - commanded_rotation @ offset
               for alpha in np.linspace(0., 1., count + 1)[1:-1]]
        if self.controller.motion_speed == 1.:
            for waypoint in via:
                result = self.controller.move_to_site(arm, waypoint, commanded_rotation)
                if not result.success:
                    raise SkillError(result.error or "IK_FAILURE", "Cartesian carry was blocked")
                self._require_hold(arm, obj)
            via = []
        for attempt in range(3):
            result = self.controller.move_to_site(arm, target, commanded_rotation,
                via=via if attempt == 0 else None, on_sample=lambda: self._require_hold(arm, obj))
            if not result.success:
                raise SkillError(result.error or "IK_FAILURE", f"Cannot carry {obj} to {np.round(desired, 4).tolist()} (EE {np.round(target, 4).tolist()}, position residual {result.position_error}, rotation residual {result.rotation_error})")
            self._require_hold(arm, obj)
            actual = self.env.object_pose(obj)[0]
            error = float(np.linalg.norm(actual - desired))
            if error <= .008:
                return {"object": obj, "target": np.asarray(desired).tolist(), "position": actual.tolist(), "position_error": error}
            target += np.asarray(desired) - actual
        raise SkillError("GRASP_SLIP", f"{obj} deviated from carry target by {error * 1000:.1f} mm")

    def handover_object(self, arm: str, obj: str, receiver: str) -> dict:
        return self._skill("handover_object", lambda: self._handover(arm, obj, receiver), time.perf_counter())

    def _handover(self, arm: str, obj: str, receiver: str) -> dict:
        if receiver not in {"left", "right"} or receiver == arm:
            raise SkillError("INVALID_PLAN", "Handover requires different giver and receiver arms")
        self._require_hold(arm, obj)
        if self.held_object(receiver):
            raise SkillError("HAND_OCCUPIED", f"The receiving {receiver} hand is occupied")
        from backend.app.simulation.openarm_v2.handover import plan_handover_geometry
        # A rearward table pinch leaves the other part of the face available to the receiver.
        # For a previously held cube, adapt the receiving depth to the measured grasp instead.
        ee, rotation = self.env.ee_pose(arm)
        cube, cube_rotation = self.env.object_pose(obj)
        pad_bias = float((cube_rotation.T @ (ee + rotation @ FINGERTIP_MIDPOINT[arm] - cube))[0])
        bias = float(np.clip(.03 + pad_bias, .015, .025))
        try:
            geometry = plan_handover_geometry(self.env, arm, obj, receiver_bias=bias)
        except ValueError as exc:
            raise SkillError("HANDOVER_UNREACHABLE", str(exc)) from None
        offset = rotation.T @ (cube - ee)
        donor_site = geometry.position - geometry.donor_rotation @ offset
        donor_positions, donor_rotations = [], []
        if self.controller.motion_speed != 1.:
            from scipy.spatial.transform import Rotation, Slerp
            arc = Slerp([0., 1.], Rotation.from_matrix(np.stack([rotation, geometry.donor_rotation])))
            angle = Rotation.from_matrix(geometry.donor_rotation @ rotation.T).magnitude()
            count = max(2, int(np.ceil(max(angle / .25, np.linalg.norm(geometry.position - cube) / .025))))
            for alpha in np.linspace(0., 1., count + 1)[1:-1]:
                attitude = arc(alpha).as_matrix()
                center = (1. - alpha) * cube + alpha * geometry.position
                donor_positions.append(center - attitude @ offset)
                donor_rotations.append(attitude)
        moved = self.controller.move_to_site(arm, donor_site, geometry.donor_rotation,
            q_init=geometry.donor_seed, via=donor_positions, via_rotations=donor_rotations,
            on_sample=lambda: self._require_hold(arm, obj))
        if not moved.success:
            raise SkillError(moved.error or "IK_FAILURE", "Donor could not reach the coupled handover pose")
        self.env.step(max(1, round(600 / self.controller.motion_speed)))
        self._require_hold(arm, obj)
        self.controller.wait_gripper(receiver, False, .5)
        cube, cube_rotation = self.env.object_pose(obj)
        receiving_rotation = cube_rotation @ geometry.receiver_relative_rotation
        site = cube + receiving_rotation @ np.array([0., 0., bias]) - receiving_rotation @ FINGERTIP_MIDPOINT[receiver]
        pregrasp = site + receiving_rotation[:, 2] * .12
        # A joint interpolation straight to the cube sweeps an open finger through it.
        # First stay outside the donor, then insert along a measured Cartesian line.
        result = self.controller.move_to_site(receiver, pregrasp, receiving_rotation, q_init=geometry.receiver_seed)
        if not result.success:
            raise SkillError(result.error or "IK_FAILURE", "Receiver could not reach the handover standoff")
        self.env.step(max(1, round(200 / self.controller.motion_speed)))
        self._require_hold(arm, obj)
        insertion = [(1 - alpha) * pregrasp + alpha * site for alpha in np.linspace(0., 1., 13)[1:]]
        destinations = insertion if self.controller.motion_speed == 1. else [site]
        for target in destinations:
            result = self.controller.move_to_site(receiver, target, receiving_rotation,
                via=insertion[:-1] if self.controller.motion_speed != 1. else None,
                on_sample=lambda: self._require_hold(arm, obj))
            if not result.success:
                raise SkillError(result.error or "IK_FAILURE", "Receiver was blocked during the straight insertion")
            self._require_hold(arm, obj)
        displacement = float(np.linalg.norm(self.env.object_pose(obj)[0] - cube))
        if displacement > .006:
            raise SkillError("HANDOVER_APPROACH_COLLISION", f"Receiver displaced the held cube by {displacement * 1000:.1f} mm")
        self.controller.wait_gripper(receiver, True, .65)
        receiving_contacts = sorted(self._contact_fingers(obj, receiver))
        if not self._pinch_established(obj, receiver):
            raise SkillError("HANDOVER_NO_RECEIVER_PINCH", f"No two-finger pinch by {receiver}: {receiving_contacts}")
        donor_contacts = sorted(self._contact_fingers(obj, arm))
        self.controller.wait_gripper(arm, False, .4)
        donor_ee, donor_rotation = self.env.ee_pose(arm)
        result = self.controller.move_to_site(arm, donor_ee + donor_rotation[:, 2] * .08, donor_rotation)
        if not result.success:
            raise SkillError(result.error or "MOTION_BLOCKED", "Donor failed to withdraw after releasing")
        if self.env.object_contacts(obj, arm):
            raise SkillError("HANDOVER_DONOR_CONTACT", "Donor still touches the transferred cube")
        self._require_hold(receiver, obj)
        before = self.env.object_pose(obj)[0].copy()
        self._carry_to(receiver, obj, before + [0., 0., VERIFICATION_LIFT])
        after = self.env.object_pose(obj)[0]
        rose = float(after[2] - before[2])
        if rose < VERIFICATION_LIFT * VERIFICATION_FOLLOW_RATIO:
            raise SkillError("HANDOVER_LOST", "Cube did not follow the receiver after donor released")
        return {"object": obj, "giver": arm, "receiver": receiver,
                "giver_contacts_before_release": donor_contacts, "receiver_pinch_contacts": receiving_contacts,
                "giver_released": True, "receiver_contacts_after_lift": sorted(self._contact_fingers(obj, receiver)),
                "object_rose": rose, "approach_displacement": displacement,
                "kinematic_residual": geometry.residual, "receiver_grasp_bias": bias,
                "grasp_mode": self.env.grasp_mode}

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
        rotation = self.controller.target_rotations.get(arm, rotation)
        obj = self.held_object(arm)
        before = self.env.object_pose(obj)[0].copy() if obj else None
        result = self.controller.move_to_site(arm, pos + np.asarray(delta), rotation)
        if not result.success:
            raise SkillError(result.error or "IK_FAILURE", "Relative move failed")
        actual_delta = self.env.ee_pose(arm)[0] - pos
        if obj:
            self._require_hold(arm, obj)
            follow_error = float(np.linalg.norm((self.env.object_pose(obj)[0] - before) - actual_delta))
            if follow_error > .018:
                raise SkillError("GRASP_SLIP", f"{obj} did not follow the {arm} gripper")
        return {"start": pos.tolist(), "final": result.final_position, "measured_delta": actual_delta.tolist()}

    def retreat(self, arm: str) -> dict:
        started = time.perf_counter()
        def run():
            _, rotation = self.env.ee_pose(arm)
            return self._move_relative(arm, rotation[:, 2] * .08 + [0., 0., .04])
        return self._skill("retreat", run, started)

    def park_empty_arm(self, arm: str) -> dict:
        def run():
            if self.held_object(arm):
                raise SkillError('HAND_OCCUPIED', 'Only an empty hand can be parked')
            frame = self.env.frame()
            for obj in frame.objects:
                if self.env.object_contacts(obj, arm):
                    raise SkillError('HAND_CONTACT', 'Release and retreat before parking an empty arm')
            others = {other: self.held_object(other) for other in ('left', 'right') if other != arm}
            stationary = {obj: self.env.object_pose(obj)[0].copy() for obj in frame.objects
                          if obj not in others.values()}
            start, rotation = self.env.ee_pose(arm)
            target, target_rotation, _, _ = self.env.arm_kinematics(arm).evaluate(self.env.home_arm_qpos(arm))
            sizes = [float(np.max(self.env.object_bounds(obj)[1]-self.env.object_bounds(obj)[0]))
                     for obj in stationary if obj.endswith('_cube')]
            height = max(start[2], target[2]) + max(.08, 2*max(sizes, default=.04))
            high_start, high_target = start.copy(), target.copy()
            high_start[2] = high_target[2] = height
            def guard():
                for other, obj in others.items():
                    if obj:
                        self._require_hold(other, obj)
                for obj, position in stationary.items():
                    if (self.env.object_contacts(obj, arm)
                            or np.linalg.norm(self.env.object_pose(obj)[0]-position) > .002):
                        raise SkillError('PARK_COLLISION', f'Empty-arm parking disturbed {obj}')
            # Lift first, then move/rotate above table objects, then descend at the
            # model-derived home frame. Every move uses the normal IK and motors.
            for position, attitude in ((high_start, rotation), (high_target, target_rotation), (target, target_rotation)):
                moved = self.controller.move_to_site(arm, position, attitude, on_sample=guard)
                if not moved.success:
                    raise SkillError(moved.error or 'MOTION_BLOCKED', 'Empty arm could not clear the handover workspace')
                guard()
            return {'target': target.tolist(), 'clearance_height_m': height,
                    'stationary_object_displacements_m': {obj: float(np.linalg.norm(self.env.object_pose(obj)[0]-position)) for obj,position in stationary.items()}}
        return self._skill('park_empty_arm', run, time.perf_counter())

    def return_home(self, arm: str) -> dict:
        started = time.perf_counter()
        def run():
            target = np.array([0.401, 0.1535 if arm == "left" else -0.1535, 1.120])
            result = self.controller.move_to_site(arm, target)
            if not result.success:
                raise SkillError(result.error or "IK_FAILURE", "Home pose unreachable")
            return {"final": result.final_position}
        return self._skill("return_home", run, started)

    def throw_object(self, arm: str, obj: str, height_cm: float = 3.) -> dict:
        from backend.app.skills.dynamic_skills import throw_up
        return self._skill("throw_object", lambda: throw_up(self, arm, obj, height_cm), time.perf_counter())

    def pour_liquid(self, arm: str, obj: str, target: str, volume_ml: float) -> dict:
        from backend.app.skills.pouring import pour_wine
        return self._skill("pour_liquid", lambda: pour_wine(self, arm, obj, target, volume_ml), time.perf_counter())

    def execute_plan(self, plan: list[dict], progress=None) -> list[dict]:
        results = []
        for step in plan:
            skill = step["skill"]
            arm = step.get("arm", "left")
            if progress:
                progress(skill)
            if skill == "observe_scene":
                from backend.app.agent.world_state import build_world_state
                result = {"skill": skill, "success": True, "evidence": build_world_state(self.env)}
            elif skill == "park_empty_arm":
                result = self.park_empty_arm(arm)
            elif skill == "approach_object":
                result = self.approach_object(arm, step["object"], progress, step.get("fault_offset"), step.get("grasp_offset_x", 0.))
            elif skill == "grasp_object":
                result = self.grasp_object(arm, step["object"])
            elif skill == "lift_object":
                result = self.lift_object(arm, step["object"])
            elif skill == "move_to_target":
                result = self.move_to_target(arm, step["target"])
            elif skill == "move_to_object":
                result = self.move_to_object(arm, step["object"], step["target"], step["relation"], step.get("distance_cm"))
            elif skill == "release_object":
                result = self.release_object(arm, step["object"])
            elif skill == "retreat":
                result = self.retreat(arm)
            elif skill == "return_home":
                result = self.return_home(arm)
            elif skill == "open_gripper":
                result = self.open_gripper(arm)
            elif skill == "close_gripper":
                result = self.close_gripper(arm)
            elif skill == "move_relative":
                result = self.move_relative(
                    arm,
                    step.get("direction") or "",
                    float(step.get("distance_cm") or 0.0),
                )
            elif skill == "handover_object":
                result = self.handover_object(arm, step["object"], step.get("receiver") or "")
            elif skill == "throw_object":
                result = self.throw_object(arm, step["object"], float(step.get("height_cm") or 3.))
            elif skill == "pour_liquid":
                result = self.pour_liquid(arm, step["object"], step.get("target") or "wine_glass", float(step.get("volume_ml") or 20.))
            else:
                result = {"skill": skill, "success": False, "duration": 0.0, "error": "INVALID_PLAN", "detail": "Unknown skill"}
            results.append(result)
            if not result.get("success", False):
                break
        return results
