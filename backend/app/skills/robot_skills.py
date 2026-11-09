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
from backend.app.control.navigation import BasePose, WaypointController, clearance_to_aabbs
from backend.app.simulation.base import SimulationBackend
from backend.app.planning.capability_queries import attitude as attitude_for_yaw
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
#: How far an in-hand support's pinch is slid down before another cube is stacked on it (m).
#: A centred pinch leaves pad material 7.15 mm above the support's top face (measured 2026-11-07),
#: so the stacked cube cannot reach it; 15 mm clears the pads while keeping the object between them.
SUPPORT_REGRASP_DROP_M = 0.015
#: Relief steps (fraction of the travel to full open) tried during a regrasp, smallest first. The
#: largest step that still pinches is kept, because how much relief an object takes is not a constant.
SUPPORT_REGRASP_RELIEF_STEPS = (0.05, 0.10, 0.15, 0.20, 0.25)
#: A held support whose own fingers stand more than this above its top face cannot receive a cube.
#: Measured live 2026-11-07: 40.5 mm, and no regrasp depth changed it.
SUPPORT_FACE_BLOCKED_M = 0.004
#: Whether to try lowering the pinch before refusing. The live measurement says the pad bodies stay
#: ~40 mm above the face whatever the grip depth, so this is off by default and the refusal is immediate.
SUPPORT_REGRASP_ENABLED = False


def clearances_blocks(clearance_m: float) -> bool:
    """Whether a measured clearance means the support's top face is occupied by its own grip."""
    return clearance_m < -SUPPORT_FACE_BLOCKED_M


#: A regrasp must not walk the object sideways; this is the tolerated lateral shift (m).
SUPPORT_REGRASP_SHIFT_TOLERANCE = 0.010
#: How fast the approach transit is driven, as a fraction of the nominal trajectory speed. The trajectory's
#: speed and the servo's response are separate levers and only this one helps: at the default speed 6 a pad
#: entered the cube by 0.74-1.54 mm during the transit on the two measured layouts, while the same route at
#: 0.5 with the servo response unchanged cleared both with 10.18 and 15.01 mm of room (measured 2026-10-28).
#: Softening the servo instead deepens the penetration, because a softer servo lags further behind.
APPROACH_TRAJECTORY_SPEED = 0.5
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

    def _raise_support_face_blocked(self, arm: str, support: str, clearance: dict, regrasp) -> None:
        """Refuse a placement the hand's own geometry cannot allow, naming the measured reason."""
        measured = clearance.get('clearance_m')
        reason = ("the hand's own geometry was not measurable" if measured is None
                  else f"{arm} hand's fingers rise {-measured * 1000:.1f} mm above {support}'s top face")
        error = SkillError('SUPPORT_FACE_BLOCKED',
                           f"a cube cannot land on {support} while it is held: {reason}")
        error.evidence = {'clearance': clearance, 'support_regrasp': regrasp,
                          'reference_object': support, 'phase': 'support_face'}
        raise error

    def _support_face_clearance(self, arm: str, support: str) -> dict:
        """Measured room above a held support's top face, from the model's own finger geometry."""
        from backend.app.skills.pad_clearance import support_face_clearance
        try:
            bounds = self.env.object_bounds(support)
            centre = self.env.object_pose(support)[0]
            half = float(max(bounds[1][0] - bounds[0][0], bounds[1][1] - bounds[0][1])) / 2.
            return support_face_clearance(self.env, arm, centre, half, float(bounds[1][2]))
        except Exception as error:
            return {'clearance_m': None, 'unknown': f'{type(error).__name__}: {error}'}

    def _motion_evidence(self, result, **extra):
        """The residuals a blocked motion was judged with, so its mechanism can be named from numbers.

        `control/motion.py` decides success from these residuals against the environment's tolerances;
        copying them (and the tolerances that applied) into `SkillError.evidence` is what lets
        `verification/failure_attribution` separate servo lag from an unreachable or occupied attitude
        instead of repeating the error code.
        """
        evidence = {'position_error': getattr(result, 'position_error', None),
                    'rotation_error': getattr(result, 'rotation_error', None),
                    'position_tolerance_m': getattr(self.env, 'motion_position_tolerance', .012),
                    'rotation_tolerance_m': getattr(self.env, 'motion_rotation_tolerance', .15)}
        return {**{k: v for k, v in evidence.items() if v is not None}, **extra}

    # -- skills ------------------------------------------------------------------

    def approach_object(self, arm: str, obj: str, progress=None, fault_offset: float | None = None,
                        grasp_offset_x: float = 0., grasp_offset=None) -> dict:
        started = time.perf_counter()
        return self._skill(
            "approach_object",
            lambda: self._approach(arm, obj, progress, fault_offset, grasp_offset_x, grasp_offset),
            started,
        )

    def _approach(self, arm: str, obj: str, progress=None, fault_offset: float | None = None,
                  grasp_offset_x: float = 0., grasp_offset=None) -> dict:
        if self.held_object(arm):
            raise SkillError("HAND_OCCUPIED", f"Place the object held by {arm} before approaching another cube")
        other = "right" if arm == "left" else "left"
        if self._pinch_established(obj, other):
            raise SkillError("HANDOVER_REQUIRED", f"{obj} is held by {other}; use a handover")
        obj_pos, _ = self.env.object_pose(obj)
        # The offset may be given along any world direction (``grasp_offset``), because the correction a
        # missed pinch needs is along the jaws' closing axis, which is not world x. The scalar x form stays
        # for the plan language, which has always expressed the offset that way.
        shift = np.asarray(grasp_offset, float) if grasp_offset is not None else np.array([grasp_offset_x, 0., 0.])
        pinch = self._pinch_site_target(arm, obj_pos + shift, fault_offset)
        finger_direction = GRASP_ROTATION @ np.array([0.0, 0.0, -1.0])
        standoff = pinch - APPROACH_STANDOFF * finger_direction
        high = standoff + np.array([0.0, 0.0, APPROACH_OVERFLY])
        before = obj_pos.copy()
        # Reject an unreachable pinch before starting the approach. Faster
        # staging must not turn an invalid target into a later contact failure.
        if not obj.endswith('_cube'):
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
        approach_evidence = {}
        stationary = {name:self.env.object_pose(name)[0].copy() for name in self.env.frame().objects}
        initial_joints = ({side:self.env.arm_qpos(side).tolist() for side in ('left','right')}
                          if hasattr(self.env,'arm_qpos') else None)
        def guard_undisturbed():
            for name,position in stationary.items():
                displacement=float(np.linalg.norm(self.env.object_pose(name)[0]-position))
                if displacement > APPROACH_DISTURBANCE_TOLERANCE:
                    error=SkillError('APPROACH_COLLISION',
                        f'Approach displaced {name} by {displacement * 1000:.1f} mm; stopping the arm')
                    error.evidence={**approach_evidence,'object':name,'displacement_m':displacement,
                                    'initial_object_positions':{key:value.tolist() for key,value in stationary.items()},
                                    'observed_object_positions':{key:self.env.object_pose(key)[0].tolist()
                                                                 for key in stationary},
                                    'initial_arm_joints':initial_joints,'phase':'approach_trajectory'}
                    raise error
        if obj.endswith('_cube'):
            from backend.app.skills.approach import cube_approach_frames, plan_cube_approach
            frames = cube_approach_frames(arm, obj_pos + shift, APPROACH_STANDOFF,
                                          world_x_offset=fault_offset)
            route,approach_evidence=plan_cube_approach(self.env,self.controller,arm,obj,pinch,standoff,
                                                       APPROACH_OVERFLY,STAGING_ALTITUDE,frames=frames)
            if route is None:
                rejected=approach_evidence.get('rejected_routes',[])
                category='COLLISION_PATH' if rejected and all(r['error']=='COLLISION_PATH' for r in rejected) else 'IK_FAILURE'
                error=SkillError(category,'No reachable collision-screened cube approach in the bounded grasp-frame search')
                error.evidence=approach_evidence
                raise error
            destinations=[('pinch',np.asarray(approach_evidence['pinch_target']))]
        else:
            destinations = waypoints if self.controller.motion_speed == 1. else [waypoints[-1]]
        for name, target in destinations:
            via = [point for _, point in waypoints[:-1]] if self.controller.motion_speed != 1. and not approach_evidence else None
            result = self.controller.move_to_site(arm, target,
                                                  np.asarray(approach_evidence['grasp_rotation']) if approach_evidence else None,
                                                  progress=progress, via=via,
                                                  planned_joints=route if approach_evidence else None,
                                                  scale_speed=not approach_evidence or approach_evidence['grasp_yaw_deg']==0,
                                                  # The transit is where a trailing pad reaches the cube; drive it
                                                  # slowly while the servo keeps the response it was configured with.
                                                  trajectory_speed=APPROACH_TRAJECTORY_SPEED if approach_evidence else None,
                                                  on_sample=guard_undisturbed)
            guard_undisturbed()
            if not result.success:
                error=SkillError(result.error or "IK_FAILURE", f"Unable to reach the {name} pose "
                                 f"(position error={result.position_error}, rotation error={result.rotation_error})")
                error.evidence={**(approach_evidence or {}), **self._motion_evidence(result, phase='approach')}
                raise error
        after = self.env.object_pose(obj)[0]
        disturbance = float(np.linalg.norm(after - before))
        if disturbance > APPROACH_DISTURBANCE_TOLERANCE:
            error = SkillError(
                "APPROACH_COLLISION",
                f"The approach displaced {obj} by {disturbance * 1000:.1f} mm",
            )
            # The mechanism is named from these numbers, not from the code (see verification/failure_attribution).
            error.evidence = {'object': obj, 'displacement_m': disturbance,
                              'tolerance_m': APPROACH_DISTURBANCE_TOLERANCE, 'phase': 'approach'}
            raise error
        ee, measured_rotation = self.env.ee_pose(arm)
        midpoint_error = float(np.linalg.norm(after - (ee + measured_rotation @ FINGERTIP_MIDPOINT[arm])))
        return {
            "target": destinations[-1][1].tolist(),
            "final": result.final_position,
            "fingertip_midpoint_error": midpoint_error,
            "object_displacement": disturbance,
            "grasp_offset_x": grasp_offset_x, "grasp_offset": shift.tolist(),
            "planning":approach_evidence,
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
                                            self.env.ee_pose(arm)[1],
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

    def navigate_to(self, target, target_yaw: float | None = None, timeout: float = 60.,
                    exclude_object: str | None = None,
                    arrive_tolerance: float | None = None, yaw_tolerance: float | None = None) -> dict:
        """Drive the scene's mobile base to a world (x, y) target.

        The base moves only through its wheel actuators; arrival is decided
        from the measured base pose, and an obstacle the guard cannot pass
        fails the skill instead of being driven through. Holonomic bases are
        driven forward-only in this slice (no strafing yet).

        Callers whose target pose was *evaluated* at an exact placement
        (the grasp-placement retry) can tighten the arrival tolerances;
        the defaults are the waypoint controller's.
        """
        started = time.perf_counter()
        return self._skill("navigate_to", lambda: self._navigate_to(
            target, target_yaw, timeout, exclude_object, arrive_tolerance, yaw_tolerance), started)

    def _navigate_to(self, target, target_yaw, timeout, exclude_object=None,
                     arrive_tolerance=None, yaw_tolerance=None) -> dict:
        try:
            self.env.base_pose()
        except NotImplementedError:
            raise SkillError("NAVIGATION_UNSUPPORTED", "This scene's robot has no mobile base")
        # A base hold (engaged by a surrounding manipulation skill) servos
        # the wheels on the measured pose and would fight this drive; the
        # drive suspends it and resumes on exit, which re-anchors the hold
        # to the pose the base was deliberately driven to.
        suspend = getattr(self.env, "suspend_base_hold", None)
        if callable(suspend):
            suspend()

        def resume_hold():
            resume = getattr(self.env, "resume_base_hold", None)
            if callable(resume):
                resume()
        limits = getattr(self.env, "base_speed_limits", None)
        max_v, max_omega = limits() if callable(limits) else (0.5, 1.2)
        controller = WaypointController(
            max_v=float(max_v), max_omega=float(max_omega),
            **({"arrive_tolerance": float(arrive_tolerance)} if arrive_tolerance is not None else {}),
            **({"yaw_tolerance": float(yaw_tolerance)} if yaw_tolerance is not None else {}))
        target = np.asarray(target, dtype=float)
        # Command at ~50 Hz of simulation time, like the motor cadence.
        chunk = max(1, round(0.02 / self.env.timestep))
        elapsed = 0.
        final = None
        while elapsed <= timeout:
            measured = self.env.base_pose()
            pose = BasePose(float(measured[0]), float(measured[1]), float(measured[2]))
            aabbs = []
            for name in self.env.frame().objects:
                if name == exclude_object:
                    # The destination itself is not an obstacle to avoid.
                    continue
                try:
                    bounds = np.asarray(self.env.object_bounds(name), dtype=float)
                except NotImplementedError:
                    continue
                aabbs.append(bounds[:, :2])
            clearance = clearance_to_aabbs(pose, 0.25, np.array(aabbs)) if aabbs else float("inf")
            command = controller.update(pose, target, target_yaw=target_yaw, obstacle_clearance=clearance)
            final = pose
            if command.status == "arrived":
                self.env.command_base_velocity(0., 0., 0.)
                resume_hold()
                return {"target": target.tolist(),
                        "final_pose": [pose.x, pose.y, pose.yaw],
                        "final_distance_m": float(np.linalg.norm(target - pose.position()))}
            if command.status == "blocked":
                self.env.command_base_velocity(0., 0., 0.)
                resume_hold()
                raise SkillError("NAVIGATION_BLOCKED",
                                 f"Obstacle stopped the base {float(np.linalg.norm(target - pose.position())):.2f} m short of the target")
            self.env.command_base_velocity(command.v, 0., command.omega)
            self.env.step(chunk)
            elapsed += chunk * self.env.timestep
        self.env.command_base_velocity(0., 0., 0.)
        resume_hold()
        raise SkillError("NAVIGATION_TIMEOUT",
                         f"Base did not reach the target within {timeout:.0f} s of simulation time"
                         + (f"; last pose {final}" if final else ""))

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

    def _align_for_placement(self, arm: str, obj: str, rotation=None) -> None:
        """Rotate the held object's grasp to the placement attitude while it is still clear.

        The calibrated table attitude is the default. An in-hand support held by the other arm needs the
        opposite attitude instead: both hands approach a cube along their fingers' closing axis, so two
        grippers meeting the same cube from the same side collide. Mirroring the attitude (180 deg about
        world z) sends the placing hand in from the side the holder's palm is not on, which is the same
        opposed-approach rule the handover already uses.
        """
        target_rotation = GRASP_ROTATION if rotation is None else np.asarray(rotation, float)
        _, rotation = self.env.ee_pose(arm)
        if np.linalg.norm(rotation - target_rotation) <= .1:
            return
        # Return to the calibrated attitude while the cube is still high and clear.
        ee, rotation = self.env.ee_pose(arm)
        center = self.env.object_pose(obj)[0]
        offset = rotation.T @ (center - ee)
        from scipy.spatial.transform import Rotation, Slerp
        arc = Slerp([0., 1.], Rotation.from_matrix(np.stack([rotation, target_rotation])))
        orientations = [arc(alpha).as_matrix() for alpha in np.linspace(0., 1., 9)[1:]]
        positions = [center - attitude @ offset for attitude in orientations]
        for index in range(len(positions)) if self.controller.motion_speed == 1. else [len(positions) - 1]:
            result = self.controller.move_to_site(arm, positions[index], orientations[index],
                via=positions[:-1] if self.controller.motion_speed != 1. else None,
                via_rotations=orientations[:-1] if self.controller.motion_speed != 1. else None,
                on_sample=lambda: self._require_hold(arm, obj))
            if not result.success:
                raise SkillError(result.error or "IK_FAILURE", "Cannot reorient the physical grasp for placement")
            self._require_hold(arm, obj)

    def _stacking_attitude(self, arm: str, obj: str, holder_arm: str, support_center):
        """A grasp attitude that reaches the top of a held support from a different side than its holder.

        Two grippers cannot share one approach axis: the placing hand's fingers and the holding hand's
        fingers would occupy the same corridor. Measured 2026-11-07, the placing arm reached the top pose
        kinematically (3.2 mm, 0.8 deg) yet the live motion stopped with an 11.7 deg attitude residual
        because the holding hand was in that corridor. Sweeping the grasp yaw and keeping the reachable
        candidate whose end effector sits farthest from the holder picks the opposed side from the arms'
        own geometry, rather than hard-coding an angle for this cell.
        """
        from scipy.spatial.transform import Rotation

        try:
            holder_ee, _ = self.env.ee_pose(holder_arm)
        except Exception:
            return None
        solve = getattr(self.controller, 'solve_ik', None)
        if not callable(solve):
            return None
        offset_z = 0.
        try:
            bounds = self.env.object_bounds(obj)
            offset_z = float(bounds[1][2] - bounds[0][2])
        except Exception:
            pass
        support_top = np.asarray(support_center, float) + np.array([0., 0., offset_z])
        best, best_gap = None, -1.
        for yaw in (0., 90., 180., 270.):
            candidate = Rotation.from_euler('z', yaw, degrees=True).as_matrix() @ GRASP_ROTATION
            site = support_top - candidate @ FINGERTIP_MIDPOINT[arm]
            solved, error, _ = solve(arm, site, candidate)
            if solved is None or error > .012:
                continue
            gap = float(np.linalg.norm(site - np.asarray(holder_ee, float)))
            if gap > best_gap:
                best, best_gap = candidate, gap
        return best

    def move_to_object(self, arm, obj, target, relation, distance_cm=None):
        def run():
            from backend.app.skills.spatial_placement import relative_destination
            self._require_hold(arm, obj)
            if target == obj or target not in self.env.frame().objects:
                raise SkillError("INVALID_TARGET", "Choose another observed cube as the reference")
            other_arm = "right" if arm == "left" else "left"
            regrasp = None
            strategy = None
            planner_yaw = None
            is_held_stationary = (self._pinch_established(target, other_arm)
                                  and self.env.object_linear_speed(target) < .02)
            if not is_held_stationary:
                if self.env.object_contacts(target) or self.env.object_linear_speed(target) > .02:
                    raise SkillError("REFERENCE_MOVING", "The placement reference must be resting stationary on the table or held stationary by the other arm")
                self._align_for_placement(arm, obj)
            else:
                # Ask the model's own geometry, before moving anything, whether the holder's fingers leave
                # room above the face the cube must land on. Measured live 2026-11-07: the pads stand
                # 40.5 mm above a held cube's top face, so the placement would fail at the last step for a
                # reason that has nothing to do with the plan; naming it here is cheaper and honest.
                clearance = self._support_face_clearance(other_arm, target)
                blocked_by_grip = (clearance.get('clearance_m') is not None
                                   and clearances_blocks(clearance['clearance_m']))
                if blocked_by_grip and not SUPPORT_REGRASP_ENABLED:
                    self._raise_support_face_blocked(other_arm, target, clearance, None)
                # Bimanual in-hand stacking: stage the held support at the shared sweet spot
                # (x ≈ 0.45, y ≈ -0.04, z ≈ 1.06; both arms solve it to <3.2 mm and <0.8 deg),
                # then choose a placing attitude that reaches the top from the holder's opposite side.
                # Ask the body where this stack can happen, instead of inheriting the last measurement as
                # a constant: the staging point, the placing attitude and whether the holder's own fingers
                # cover the face are queries now (planning/strategy_search.py).
                from backend.app.planning.strategy_search import plan_stack_on_held_support
                strategy = plan_stack_on_held_support(self.env, self.controller, arm, target)
                if strategy.get('ok') is False:
                    if strategy['reason'] == 'SUPPORT_FACE_BLOCKED':
                        self._raise_support_face_blocked(other_arm, target,
                                                         {'clearance_m': strategy.get('clearance_m'),
                                                          'measured': strategy.get('measured')}, None)
                    error = SkillError(strategy['reason'], strategy.get('detail') or strategy['reason'])
                    error.evidence = {'strategy': strategy, 'reference_object': target}
                    raise error
                if strategy.get('ok') is True:
                    staging_pos = np.array(strategy['staging'], float)
                    planner_yaw = strategy['placer_yaw_deg']
                else:
                    # Unmeasured: keep the previous staging point rather than inventing a refusal.
                    staging_pos = np.array([0.45, 0.04 if arm == 'left' else -0.04, 1.06])
                    planner_yaw = None
                self._carry_to(other_arm, target, staging_pos, exclude_target=obj)
                # Free the support's top face before the placing arm arrives: the holder's pads sit above
                # the centre they pinch, so a centred grip would leave the stacked cube resting on them.
                # Best effort and recorded: a regrasp this robot cannot achieve leaves the grip as it was,
                # and the placement then has to prove itself against the verifier like any other.
                try:
                    regrasp = {'success': True, **self._regrasp(other_arm, target, -SUPPORT_REGRASP_DROP_M)}
                except SkillError as error:
                    regrasp = {'success': False, 'error': error.category, 'detail': str(error),
                               **({'evidence': error.evidence} if error.evidence else {})}
                if regrasp is not None and regrasp.get('success'):
                    after = self._support_face_clearance(other_arm, target)
                    if after.get('clearance_m') is not None and clearances_blocks(after['clearance_m']):
                        self._raise_support_face_blocked(other_arm, target, after, regrasp)
                staged_support = self.env.object_pose(target)[0]
                attitude = (attitude_for_yaw(planner_yaw) if planner_yaw is not None
                            else self._stacking_attitude(arm, obj, other_arm, staged_support))
                self._align_for_placement(arm, obj, rotation=attitude)
            desired = relative_destination(self.env, obj, target, relation, arm, distance_cm)
            travel = desired.copy()
            travel[2] = min(1.15, desired[2] + .02 if is_held_stationary else max(1.14, desired[2] + .035))
            try:
                self._carry_to(arm, obj, travel, exclude_target=target)
                placed = self._carry_to(arm, obj, desired, exclude_target=target)
            except SkillError as error:
                # Keep what the preparation step measured. A placement that fails later must not hide
                # whether the support regrasp happened (measured 2026-11-07: it did, and the report lost it).
                if regrasp is not None:
                    error.evidence = {**(error.evidence or {}), 'support_regrasp': regrasp,
                                      'reference_object': target, 'relation': relation}
                raise
            return {**placed, "reference_object": target, "relation": relation,
                    "support_bounds": self.env.object_bounds(target).tolist(),
                    "support_regrasp": regrasp if is_held_stationary else None,
                    "stack_strategy": strategy if is_held_stationary else None}
        return self._skill("move_to_object", run, time.perf_counter())

    def _carry_aim(self, arm: str, obj: str, desired, *, attitude: str = "pinned"):
        """The end-effector pose a carry to ``desired`` (the object's pose) commands, and its rotation.

        One owner for the grasp-offset arithmetic, because everything that asks whether a carry is
        possible must ask the same question the carry itself will command. Measured 2026-10-25 on
        OpenManipulator-X: the wind-up search asked the IK for the end effector to be at the *object*'s
        column, while the carry commanded the end effector held off from it by the grasp offset, so a
        wind-up passed the search and the carry stopped 92 mm short - the offset's own length.
        """
        ee, rotation = self.env.ee_pose(arm)
        offset = rotation.T @ (self.env.object_pose(obj)[0].copy() - ee)
        commanded_rotation = None if attitude == "free" else self.controller.target_rotations.get(arm, rotation)
        aim = rotation if commanded_rotation is None else commanded_rotation
        return np.asarray(desired) - aim @ offset, commanded_rotation, offset

    def _planned_carry(self, arm: str, obj: str, desired: np.ndarray, rotation):
        """A planner-solved joint path that carries the held object to ``desired``, and its attitude.

        Only for arms with fewer than six joints, where the controller's own damped local solver stops
        tens of millimetres short of poses the planner's IK solves with zero error: measured
        2026-10-25 on the four-joint OpenManipulator-X, the solver's position error on the wind-up
        candidates was 90-118 mm while ``launch_ready`` had already accepted the same targets below
        2 mm through the planner. The decision stays with ``move_to_site``, which validates the path it
        is handed (start continuity, joint limits, final pose) and refuses a plan that does not hold.
        """
        from backend.app.robots.urdf import solve_profile_ik

        env = self.env
        names = env.profile['chains'][arm]['joints']
        ee, held = env.ee_pose(arm)
        offset = held.T @ (env.object_pose(obj)[0].copy() - ee)
        aim = held if rotation is None else np.asarray(rotation, float)
        start = env.object_pose(obj)[0].copy()
        goal = np.asarray(desired, float)
        seed = env.arm_qpos(arm)
        path = [seed]
        count = max(1, int(np.ceil(np.linalg.norm(goal - start) / .025)))
        for alpha in np.linspace(0., 1., count + 1)[1:]:
            solution = solve_profile_ik(env.profile, arm, (1 - alpha) * start + alpha * goal - aim @ offset,
                                        None, model=env.model, q_init=seed, snapshot=env.joint_snapshot())
            if not solution['success']:
                return None
            seed = env.model.nearest_continuous(names, solution['q_rad'], seed)
            path.append(seed)
        return path, np.asarray(env.arm_kinematics(arm).evaluate(seed)[1], float)

    def _carry_to(self, arm: str, obj: str, desired: np.ndarray, *, attitude: str = "pinned",
                  exclude_target: str | None = None) -> dict:
        """Carry the held object to a pose. ``attitude="free"`` lets the arm choose its orientation.

        A six-joint arm carries the attitude its pinch was validated in, and that is the default. An arm
        whose pinch frame is only achievable near the object cannot carry it any higher while keeping
        that frame - measured 2026-10-25 on OpenManipulator-X, whose wind-up arrived but whose carry was
        refused with the held attitude. Free travel still requires the verified hold at every waypoint,
        the same fallback the grasp planning already uses below six joints.
        """
        self._require_hold(arm, obj)
        ee, rotation = self.env.ee_pose(arm)
        start = self.env.object_pose(obj)[0].copy()
        target, commanded_rotation, offset = self._carry_aim(arm, obj, desired, attitude=attitude)
        aim = rotation if commanded_rotation is None else commanded_rotation
        ignored = {obj}
        if exclude_target:
            ignored.add(exclude_target)
        stationary = {name: self.env.object_pose(name)[0].copy()
                      for name in self.env.frame().objects if name not in ignored}
        def guard_carry():
            self._require_hold(arm, obj)
            for name, position in stationary.items():
                displacement = float(np.linalg.norm(self.env.object_pose(name)[0] - position))
                if displacement > APPROACH_DISTURBANCE_TOLERANCE:
                    error = SkillError("CARRY_COLLISION",
                                       f"Carrying {obj} disturbed resting {name} by {displacement * 1000:.1f} mm; stopping")
                    error.evidence = {'object': name, 'carrying': obj, 'displacement_m': displacement,
                                      'tolerance_m': APPROACH_DISTURBANCE_TOLERANCE, 'phase': 'carry'}
                    raise error
        # Keep the carried cube on a Cartesian clearance path. A joint interpolation
        # between two high endpoints can still dip the palm through a table object.
        count = max(1, int(np.ceil(np.linalg.norm(np.asarray(desired) - start) / .025)))
        via = [(1 - alpha) * start + alpha * np.asarray(desired) - aim @ offset
               for alpha in np.linspace(0., 1., count + 1)[1:-1]]
        if self.controller.motion_speed == 1.:
            for waypoint in via:
                result = self.controller.move_to_site(arm, waypoint, commanded_rotation, on_sample=guard_carry)
                if not result.success:
                    raise SkillError(result.error or "IK_FAILURE", "Cartesian carry was blocked")
                guard_carry()
            via = []
        # An imported arm with fewer than six joints is carried along a planner-solved path: its own
        # solver cannot reach the wind-up candidates the planner accepts (see ``_planned_carry``).
        # Six-joint arms - the official cell - keep exactly the motion they had, and so does an
        # environment that does not expose a profile at all: an unknown arm must not be assumed to be
        # the short one, or the official environment is silently handed a plan it never had.
        profile = getattr(self.env, 'profile', None)
        chain_joints = profile.get('chains', {}).get(arm, {}).get('joints') if isinstance(profile, dict) else None
        planned = (self._planned_carry(arm, obj, np.asarray(desired, float), commanded_rotation)
                   if chain_joints is not None and len(chain_joints) < 6 else None)
        for attempt in range(3):
            if attempt == 1 and planned is not None:
                result = self.controller.move_to_site(arm, target, planned[1], planned_joints=planned[0],
                                                      on_sample=guard_carry)
            else:
                result = self.controller.move_to_site(arm, target, commanded_rotation,
                    via=via if attempt == 0 else None, on_sample=guard_carry)
            guard_carry()
            if not result.success:
                # The controller's own solve failing is what the planner-solved path is for: without this
                # the fallback sits behind a raise that never lets it run (measured: the residuals were
                # bit-identical with and without the plan, which is how the dead path was found).
                if attempt == 0 and planned is not None:
                    continue
                error = SkillError(result.error or "IK_FAILURE", f"Cannot carry {obj} to {np.round(desired, 4).tolist()} (EE {np.round(target, 4).tolist()}, position residual {result.position_error}, rotation residual {result.rotation_error})")
                error.evidence = self._motion_evidence(result, object=obj, phase='carry')
                raise error
            self._require_hold(arm, obj)
            actual = self.env.object_pose(obj)[0]
            error = float(np.linalg.norm(actual - desired))
            if error <= .008:
                return {"object": obj, "target": np.asarray(desired).tolist(), "position": actual.tolist(), "position_error": error}
            target += np.asarray(desired) - actual
        raise SkillError("GRASP_SLIP", f"{obj} deviated from carry target by {error * 1000:.1f} mm")

    def regrasp_object(self, arm: str, obj: str, drop_m: float = -SUPPORT_REGRASP_DROP_M) -> dict:
        """Slide this arm's pinch down the held object, so its top face becomes reachable.

        A cube gripped at its centre is held by pads that stand above that centre, so a second cube
        placed on top cannot reach the top face: measured 2026-11-07, the stacked cube stopped 7.15 mm
        above the support and rested on the holder's own fingers (`support_contact=false`). Gripping the
        support low is not possible while it rests on the table (the pads would hit the table), so the
        grip is lowered **after** the object is in the air.

        The jaws are opened only part way, so the object stays between the pads; the arm then slides
        down and closes again. Every step is verified: if the pinch is lost while opening, the skill stops
        before moving, and if the object shifts or does not follow a short lift, it raises rather than
        reporting a regrasp it did not achieve.
        """
        return self._skill("regrasp_object", lambda: self._regrasp(arm, obj, drop_m), time.perf_counter())

    def _regrasp(self, arm: str, obj: str, drop_m: float) -> dict:
        self._require_hold(arm, obj)
        if drop_m >= 0.:
            raise SkillError("INVALID_PLAN", "A regrasp lowers the pinch; the offset must be negative")
        before = self.env.object_pose(obj)[0].copy()
        closed_aperture = self.env.gripper_position(arm)
        open_aperture = self.env.gripper_target(arm, False)
        # How far the jaws may be relieved is a property of this object and this grip, not a constant:
        # measured 2026-11-07, relieving 35% of the travel dropped the cube from the pinch entirely.
        # Search upward in small steps and keep the last relief that still pinched.
        kept, kept_fraction = closed_aperture, 0.
        for step in SUPPORT_REGRASP_RELIEF_STEPS:
            candidate = closed_aperture + step * (open_aperture - closed_aperture)
            self.env.set_gripper(arm, candidate)
            self.env.step(max(1, round(120 / self.controller.motion_speed)))
            if not self._pinch_established(obj, arm):
                break
            kept, kept_fraction = candidate, step
        self.env.set_gripper(arm, kept)
        self.env.step(max(1, round(120 / self.controller.motion_speed)))
        if kept_fraction == 0. or not self._pinch_established(obj, arm):
            # Never slide on a lost pinch: close again and report, leaving the object where it was.
            self.controller.wait_gripper(arm, True, .5)
            raise SkillError("REGRASP_LOST", f"{obj} left the pinch when the jaws were relieved; not sliding")
        partial = kept
        ee, rotation = self.env.ee_pose(arm)
        slide = self.controller.target_rotations.get(arm, rotation)
        result = self.controller.move_to_site(arm, ee + np.array([0., 0., drop_m]), slide,
                                              on_sample=lambda: self._require_hold(arm, obj))
        if not result.success:
            raise SkillError(result.error or "IK_FAILURE", "Could not slide the pinch along the object")
        self.controller.wait_gripper(arm, True, .5)
        if not self._pinch_established(obj, arm):
            raise SkillError("REGRASP_NO_PINCH", f"{obj} is no longer pinched after closing at the lower point")
        after = self.env.object_pose(obj)[0]
        shifted = float(np.linalg.norm((after - before)[:2]))
        if shifted > SUPPORT_REGRASP_SHIFT_TOLERANCE:
            raise SkillError("REGRASP_MOVED", f"{obj} moved {shifted * 1000:.1f} mm sideways during the regrasp")
        dropped = float(before[2] - after[2])
        if dropped > SUPPORT_REGRASP_SHIFT_TOLERANCE:
            # The object rode down with the hand: the pads never slid along it, so the grip point on the
            # object is unchanged and the top face is still blocked.
            raise SkillError("REGRASP_NO_SLIDE",
                             f"{obj} followed the hand down {dropped * 1000:.1f} mm instead of staying put")
        lift = self.controller.move_to_site(arm, self.env.ee_pose(arm)[0] + np.array([0., 0., VERIFICATION_LIFT]),
                                           self.env.ee_pose(arm)[1], on_sample=lambda: self._require_hold(arm, obj))
        if not lift.success:
            raise SkillError(lift.error or "IK_FAILURE", "The hold did not survive the regrasp verification lift")
        self.controller.move_to_site(arm, self.env.ee_pose(arm)[0] - np.array([0., 0., VERIFICATION_LIFT]),
                                     self.env.ee_pose(arm)[1], on_sample=lambda: self._require_hold(arm, obj))
        self._require_hold(arm, obj)
        return {"object": obj, "arm": arm, "drop_m": drop_m, "relief_fraction": kept_fraction,
                "aperture_before": closed_aperture, "aperture_relieved": partial, "aperture_open": open_aperture,
                "sideways_shift_m": shifted, "vertical_shift_m": dropped,
                "pinch_contacts": sorted(self._contact_fingers(obj, arm))}

    def handover_object(self, arm: str | None, obj: str, receiver: str | None = None) -> dict:
        if arm is None:
            arm = next((side for side in ("left", "right") if self.held_object(side) == obj), None)
            if arm is None:
                raise SkillError("HOLD_NOT_FOUND", f"Neither hand holds {obj} for handover")
        if receiver is None:
            available = ["left", "right"]
            candidates = [side for side in available if side != arm and not self.held_object(side)]
            if not candidates:
                raise SkillError("NO_RECEIVER_AVAILABLE", f"No empty arm available opposite {arm} to receive {obj}")
            receiver = candidates[0]
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
            error = SkillError(moved.error or "IK_FAILURE", "Donor could not reach the coupled handover pose")
            error.evidence = self._motion_evidence(moved, object=obj, phase='handover_donor')
            raise error
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
            error = SkillError(result.error or "IK_FAILURE", "Receiver could not reach the handover standoff")
            error.evidence = self._motion_evidence(result, object=obj, phase='handover_receiver')
            raise error
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
        # Adaptive pinch recovery for irregular contact geometry:
        # If the initial pinch touches only one finger pad (or misses slightly due to angle/offset),
        # do not rigidly abort. Perform local adaptive micro-adjustments in depth before giving up.
        if not self._pinch_established(obj, receiver):
            for retry in range(3):
                self.controller.wait_gripper(receiver, False, .35)
                self.env.step(max(1, round(50 / self.controller.motion_speed)))
                adjustment = receiving_rotation @ np.array([0., 0., 0.003 * (retry + 1)])
                self.controller.move_to_site(receiver, site + adjustment, receiving_rotation,
                                             on_sample=lambda: self._require_hold(arm, obj))
                self.controller.wait_gripper(receiver, True, .65)
                receiving_contacts = sorted(self._contact_fingers(obj, receiver))
                if self._pinch_established(obj, receiver):
                    break
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
            from backend.app.skills.homing import plan_empty_arm_home
            route, planning = plan_empty_arm_home(self.env, self.controller, arm)
            if route is None:
                error = SkillError(planning['error'], 'No reachable clear route to the model home posture')
                error.evidence = planning
                raise error
            def guard():
                for other, obj in others.items():
                    if obj:
                        self._require_hold(other, obj)
                for obj, position in stationary.items():
                    if (self.env.object_contacts(obj, arm)
                            or np.linalg.norm(self.env.object_pose(obj)[0]-position) > .002):
                        error = SkillError('PARK_COLLISION', f'Empty-arm parking disturbed {obj}')
                        error.evidence = {**planning,'object':obj,'contacts':self.env.object_contacts(obj,arm),
                                          'displacement_m':float(np.linalg.norm(self.env.object_pose(obj)[0]-position))}
                        raise error
            # Execute only an admitted route. Exact final joint targets preserve
            # the canonical home branch of a redundant arm; normal motors move it.
            for joints in route[1:]:
                kin = self.env.arm_kinematics(arm)
                position, attitude, _, _ = kin.evaluate(joints)
                check = getattr(kin, 'empty_arm_path_is_clear', None)
                edge = [self.env.arm_qpos(arm), joints]
                if check and not check(edge, {o:self.env.object_pose(o) for o in self.env.frame().objects}):
                    raise SkillError('PARK_COLLISION', 'Home route is no longer clear in the observed scene')
                moved = self.controller.move_to_site(arm, position, attitude, planned_joints=edge, on_sample=guard)
                if not moved.success:
                    raise SkillError(moved.error or 'MOTION_BLOCKED', 'Empty arm could not clear the handover workspace')
                guard()
            joint_error = float(np.max(abs(self.env.arm_qpos(arm)-route[-1])))
            if joint_error > .025:
                raise SkillError('HOME_POSTURE_NOT_REACHED', f'Home joint residual {joint_error:.4f} rad')
            target = self.env.arm_kinematics(arm).evaluate(route[-1])[0]
            return {'target': target.tolist(), **planning, 'maximum_joint_error_rad':joint_error,
                    'stationary_object_displacements_m': {obj: float(np.linalg.norm(self.env.object_pose(obj)[0]-position)) for obj,position in stationary.items()}}
        return self._skill('park_empty_arm', run, time.perf_counter())

    def return_home(self, arm: str) -> dict:
        # Home is the model's joint posture, with clearance above current
        # objects, rather than a fixed point near the original cube locations.
        return {**self.park_empty_arm(arm),'skill':'return_home'}

    def throw_object(self, arm: str, obj: str, height_cm: float = 3.,
                     stroke_cm: float | None = None, release_speed_m_s: float | None = None,
                     target: str | None = None, flight_time_s: float | None = None,
                     back_off_cm: float | None = None) -> dict:
        from backend.app.skills.dynamic_skills import throw_up
        # An aimed throw names its target; the skill needs the point, and it reads it from the engine
        # rather than from the language.
        point = None if target is None else self.env.object_pose(target)[0]
        return self._skill("throw_object", lambda: throw_up(self, arm, obj, height_cm,
                                                           stroke_cm=stroke_cm,
                                                           release_speed_m_s=release_speed_m_s,
                                                           target_point=point,
                                                           flight_time_s=flight_time_s or .30,
                                                           back_off_cm=back_off_cm),
                           time.perf_counter())

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
                result = self.throw_object(arm, step["object"], float(step.get("height_cm") or 3.),
                                           stroke_cm=step.get("stroke_cm"),
                                           release_speed_m_s=step.get("release_speed_m_s"),
                                           target=step.get("target"),
                                           flight_time_s=step.get("flight_time_s"),
                                           back_off_cm=step.get("back_off_cm"))
            elif skill == "pour_liquid":
                result = self.pour_liquid(arm, step["object"], step.get("target") or "wine_glass", float(step.get("volume_ml") or 20.))
            elif skill == "navigate_to":
                result = self.navigate_to(step["target_xy"], step.get("target_yaw"),
                                          exclude_object=step.get("object"))
            else:
                result = {"skill": skill, "success": False, "duration": 0.0, "error": "INVALID_PLAN", "detail": "Unknown skill"}
            results.append(result)
            if not result.get("success", False):
                break
        return results
