"""Screen cube-grasp frames and joint paths against the current physical scene."""
import numpy as np
from scipy.spatial.transform import Rotation

from backend.app.control.motion import joint_spline
from backend.app.planning.angle_search import refine
from backend.app.simulation.openarm_v2.definition import FINGERTIP_MIDPOINT, GRASP_ROTATION
from backend.app.simulation.transform import orientation_error


# Bounded exploration, not proof that every cube face can be pinched. The physical
# finger-contact and follow checks after approach remain the success authority.
CUBE_GRASP_YAWS_DEG = (0, -15, 15, -30, 30, -45, 45, -60, 60, -75, 75, -90, 90, 180)
# Task 57efa442 moved a cube 248 mm after a no-margin path passed; with 8 mm
# clearance task 22d8e6c7 still moved red 15.8 mm between motion samples.
# Keep 20 mm around NON-contact geometry; target fingertip contact is exempt.
NONCONTACT_CLEARANCE_M = .020
# The target is approached intentionally; its non-fingertip geometry still needs
# space, but a 20 mm exclusion there would forbid a normal close-in pinch.
TARGET_BODY_CLEARANCE_M = .003
# Above this the path is obviously not tight; the search does not need an exact distance.
CLEARANCE_CEILING_M = .05


def two_axis_frames(arm, center, standoff_m, *, world_x_offset=None):
    """The same candidates as ``cube_approach_frames``, plus a lean of the approach axis.

    Rotating about the vertical changes which side the hand comes from; leaning in the vertical plane
    changes the angle it comes down at, and with it how high the hand sits while the fingers are beside the
    object. Measured 2026-10-27: rotating about the frame's own approach axis was refused at every angle
    from 15 to 90 degrees because the hand keeps the same height while it spins, so the lean is the axis
    worth searching. Every candidate keeps the invariant that makes it the same grasp: the fingertip
    midpoint still lands on the object centre.
    """
    from backend.app.skills.pinch_frames import pinch_frames

    frames = []
    for tilt, azimuth, rotation in pinch_frames():
        offset = rotation @ FINGERTIP_MIDPOINT[arm]
        if world_x_offset is not None:
            offset = offset.copy()
            offset[0] = world_x_offset
        pinch = np.asarray(center, float) - offset
        finger_direction = rotation @ np.array([0., 0., -1.])
        frames.append({'yaw_deg': azimuth, 'tilt_deg': tilt, 'rotation': rotation, 'pinch': pinch,
                       'standoff': pinch - standoff_m * finger_direction})
    return frames


def frame_at(arm, center, standoff_m, yaw_deg, *, world_x_offset=None, tilt_deg=0.):
    """The grasp frame at one azimuth and lean. Any angle is a real frame, not an interpolation."""
    from backend.app.skills.pinch_frames import grasp_rotation

    rotation = grasp_rotation(tilt_deg, yaw_deg)
    offset = rotation @ FINGERTIP_MIDPOINT[arm]
    if world_x_offset is not None:
        offset[0] = world_x_offset
    pinch = np.asarray(center, float) - offset
    finger_direction = rotation @ np.array([0., 0., -1.])
    return {'yaw_deg':float(yaw_deg), 'tilt_deg':float(tilt_deg), 'rotation':rotation, 'pinch':pinch,
            'standoff':pinch - standoff_m * finger_direction}


def cube_approach_frames(arm, center, standoff_m, *, world_x_offset=None):
    frames = [frame_at(arm, center, standoff_m, yaw, world_x_offset=world_x_offset)
              for yaw in CUBE_GRASP_YAWS_DEG]
    return frames


def plan_cube_approach(env, controller, arm, obj, pinch, standoff, overfly, staging_altitude, *, frames=None):
    start = env.arm_qpos(arm)
    point = env.ee_pose(arm)[0]
    kin = env.arm_kinematics(arm)
    objects = {name:env.object_pose(name) for name in env.frame().objects}
    check = getattr(kin,'empty_arm_path_is_clear',None)
    # The clearance query ranks candidates; the route screen still decides. Where the two disagree the
    # trace says so (rejected_by), because a measured win the screen refuses is not an improvement.
    measure = getattr(kin,'path_minimum_clearance',None)
    allowed = {obj:{f'openarm_{arm}_ee_inner_finger',f'openarm_{arm}_ee_outer_finger'}}
    rejected = []
    # Older callers screen only the calibrated nominal frame. The runtime
    # explicitly supplies candidate frames, each with its own fingertip offset.
    candidates = frames if frames is not None else [
        {'yaw_deg':0, 'rotation':GRASP_ROTATION, 'pinch':pinch, 'standoff':standoff}]
    def route_for(frame, rise, staged):
        """Solve the joint route for one attitude and rise, reusing the accepted frame's offsets."""
        attitude = frame['rotation']
        high = frame['standoff']+np.array([0., 0., rise])
        poses = ([np.r_[point[:2],staging_altitude]] if staged else [])+[
            high, frame['standoff'], frame['pinch']]
        route = [start]
        for target in poses:
            solved = None
            for seed in (route[-1],env.home_arm_qpos(arm)):
                q,_,_=controller.solve_ik(arm,target,attitude,q_init=seed)
                if q is None:continue
                actual,rotation,_,_=kin.evaluate(q)
                if (np.linalg.norm(actual-target)<.004
                        and np.linalg.norm(orientation_error(rotation,attitude))<.04):
                    solved=q;break
            if solved is None:
                return None
            route.append(solved)
        return route, joint_spline(route,env.arm_velocity_limits(arm))

    def clearance_of(frame, rise, staged):
        """How much room this route leaves, in metres. None when it cannot be built or measured."""
        if measure is None:
            return None
        built = route_for(frame, rise, staged)
        if built is None:
            return None
        route, curve = built
        return measure(route,objects,allowed_contacts=allowed,curve=curve,ceiling=CLEARANCE_CEILING_M)

    found = None
    for frame in candidates:
        for rise in np.linspace(overfly,0.,5):
            for staged in (True,False) if staging_altitude-point[2]>.02 else (False,):
                built = route_for(frame, rise, staged)
                if built is None:
                    rejected.append({'yaw_deg':frame['yaw_deg'],'overfly_m':float(rise),
                                     'staging':staged,'error':'IK_FAILURE'})
                    continue
                route, curve = built
                if check is None or check(route,objects,allowed_contacts=allowed,curve=curve,
                                          minimum_object_clearance=NONCONTACT_CLEARANCE_M,
                                          object_clearances={obj:TARGET_BODY_CLEARANCE_M}):
                    found = (frame, float(rise), staged, route, curve)
                    break
                rejected.append({'yaw_deg':frame['yaw_deg'],'overfly_m':float(rise),
                                 'staging':staged,'error':'COLLISION_PATH'})
            if found: break
        if found: break
    if not found:
        return None, {'error':'APPROACH_PATH_UNAVAILABLE','rejected_routes':rejected,
                      'attempted_grasp_yaws_deg':[frame['yaw_deg'] for frame in candidates]}

    # The listed yaws and the five rise values only got us a route that passes. Now the angle and the height
    # are derived from the measurement: the same clearance query, maximised. A yaw or rise that cannot be
    # reached scores None and loses, so the search cannot drift somewhere unattainable.
    frame, rise, staged, route, curve = found
    accepted_clearance = clearance_of(frame, rise, staged)
    refinements = []
    if measure is not None:
        centre = frame['pinch'] + frame['rotation'] @ FINGERTIP_MIDPOINT[arm]
        standoff_distance = float(np.linalg.norm(frame['standoff'] - frame['pinch']))
        yaw_now = float(frame['yaw_deg'])
        tilt_now = float(frame.get('tilt_deg', 0.))

        def frame_with(yaw_deg, tilt_deg):
            return frame_at(arm, centre, standoff_distance, yaw_deg, tilt_deg=tilt_deg)

        # Lean and side are separate axes because they move the hand differently: the azimuth spins it about
        # the vertical at one height, the lean changes that height. Each is searched against the same
        # measurement, so whichever the body can actually reach better wins.
        for name, axis, from_value, span, step in (
                ('refine_grasp_yaw', 'yaw', yaw_now, 15., 5.),
                ('refine_grasp_tilt', 'tilt', tilt_now, 15., 5.),
                ('refine_overfly', 'overfly', rise, max(rise/2., .01), max(rise/4., .005))):
            def rebuild(value, axis=axis):
                if axis == 'yaw':
                    return frame_with(value, tilt_now), rise
                if axis == 'tilt':
                    return frame_with(yaw_now, value), rise
                return frame, value

            trace = refine(lambda value: clearance_of(*rebuild(value), staged),
                           from_value, span, step=step)
            entry = {'query': name, **trace, 'accepted': False}
            refinements.append(entry)
            if trace['best_value'] is None or trace['best_score'] is None:
                entry['rejected_by'] = 'UNMEASURED'
                continue
            if accepted_clearance is not None and trace['best_score'] <= accepted_clearance:
                entry['rejected_by'] = 'NOT_BETTER_THAN_ACCEPTED'
                continue
            candidate_frame, candidate_rise = rebuild(trace['best_value'])
            built = route_for(candidate_frame, candidate_rise, staged)
            if built is None:
                entry['rejected_by'] = 'IK_FAILURE'
                continue
            candidate_route, candidate_curve = built
            if check is not None and not check(candidate_route, objects, allowed_contacts=allowed,
                                               curve=candidate_curve,
                                               minimum_object_clearance=NONCONTACT_CLEARANCE_M,
                                               object_clearances={obj:TARGET_BODY_CLEARANCE_M}):
                entry['rejected_by'] = 'COLLISION_PATH'
                continue
            entry['accepted'] = True
            frame, rise, route, curve = candidate_frame, candidate_rise, candidate_route, candidate_curve
            accepted_clearance = trace['best_score']
            yaw_now = float(frame['yaw_deg'])
            tilt_now = float(frame.get('tilt_deg', 0.))

    attitude = frame['rotation']
    return route, {'approach_overfly_m':float(rise),'staging':staged,
                   'noncontact_clearance_m':NONCONTACT_CLEARANCE_M,
                   'target_body_clearance_m':TARGET_BODY_CLEARANCE_M,
                   'grasp_yaw_deg':frame['yaw_deg'],
                   'grasp_tilt_deg':frame.get('tilt_deg', 0.),
                   'grasp_rotation':attitude.tolist(),
                   'pinch_target':frame['pinch'].tolist(),
                   'measured_clearance_m':accepted_clearance,
                   'refinements':refinements,
                   'rejected_routes':rejected}
