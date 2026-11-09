"""Screen cube-grasp frames and joint paths against the current physical scene."""
import numpy as np
from scipy.spatial.transform import Rotation

from backend.app.control.motion import joint_spline
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


def cube_approach_frames(arm, center, standoff_m, *, world_x_offset=None):
    frames = []
    for yaw in CUBE_GRASP_YAWS_DEG:
        rotation = Rotation.from_euler('z', yaw, degrees=True).as_matrix() @ GRASP_ROTATION
        offset = rotation @ FINGERTIP_MIDPOINT[arm]
        if world_x_offset is not None:
            offset[0] = world_x_offset
        pinch = np.asarray(center, float) - offset
        finger_direction = rotation @ np.array([0., 0., -1.])
        frames.append({'yaw_deg':yaw, 'rotation':rotation, 'pinch':pinch,
                       'standoff':pinch - standoff_m * finger_direction})
    return frames


def plan_cube_approach(env, controller, arm, obj, pinch, standoff, overfly, staging_altitude, *, frames=None):
    start = env.arm_qpos(arm)
    point = env.ee_pose(arm)[0]
    kin = env.arm_kinematics(arm)
    objects = {name:env.object_pose(name) for name in env.frame().objects}
    check = getattr(kin,'empty_arm_path_is_clear',None)
    allowed = {obj:{f'openarm_{arm}_ee_inner_finger',f'openarm_{arm}_ee_outer_finger'}}
    rejected = []
    # Older callers screen only the calibrated nominal frame. The runtime
    # explicitly supplies candidate frames, each with its own fingertip offset.
    candidates = frames if frames is not None else [
        {'yaw_deg':0, 'rotation':GRASP_ROTATION, 'pinch':pinch, 'standoff':standoff}]
    for frame in candidates:
        attitude = frame['rotation']
        pinch_target = frame['pinch']
        standoff_target = frame['standoff']
        for rise in np.linspace(overfly,0.,5):
            high = standoff_target+np.array([0.,0.,rise])
            for staged in (True,False) if staging_altitude-point[2]>.02 else (False,):
                poses = ([np.r_[point[:2],staging_altitude]] if staged else [])+[high,standoff_target,pinch_target]
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
                    if solved is None:break
                    route.append(solved)
                else:
                    curve=joint_spline(route,env.arm_velocity_limits(arm))
                    if check is None or check(route,objects,allowed_contacts=allowed,curve=curve,
                                              minimum_object_clearance=NONCONTACT_CLEARANCE_M,
                                              object_clearances={obj:TARGET_BODY_CLEARANCE_M}):
                        return route,{'approach_overfly_m':float(rise),'staging':staged,
                                      'noncontact_clearance_m':NONCONTACT_CLEARANCE_M,
                                      'target_body_clearance_m':TARGET_BODY_CLEARANCE_M,
                                      'grasp_yaw_deg':frame['yaw_deg'],
                                      'grasp_tilt_deg':frame.get('tilt_deg', 0.),
                                      'grasp_rotation':attitude.tolist(),
                                      'pinch_target':pinch_target.tolist(),
                                      'rejected_routes':rejected}
                    rejected.append({'yaw_deg':frame['yaw_deg'],'overfly_m':float(rise),
                                     'staging':staged,'error':'COLLISION_PATH'})
                    continue
                rejected.append({'yaw_deg':frame['yaw_deg'],'overfly_m':float(rise),
                                 'staging':staged,'error':'IK_FAILURE'})
    return None,{'error':'APPROACH_PATH_UNAVAILABLE','rejected_routes':rejected,
                 'attempted_grasp_yaws_deg':[frame['yaw_deg'] for frame in candidates]}
