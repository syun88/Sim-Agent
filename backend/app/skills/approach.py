"""Screen approach height and joint branches against the current physical scene."""
import numpy as np

from backend.app.control.motion import joint_spline
from backend.app.simulation.openarm_v2.definition import GRASP_ROTATION
from backend.app.simulation.transform import orientation_error


def plan_cube_approach(env, controller, arm, obj, pinch, standoff, overfly, staging_altitude):
    start = env.arm_qpos(arm)
    point = env.ee_pose(arm)[0]
    kin = env.arm_kinematics(arm)
    objects = {name:env.object_pose(name) for name in env.frame().objects}
    check = getattr(kin,'empty_arm_path_is_clear',None)
    allowed = {obj:{f'openarm_{arm}_ee_inner_finger',f'openarm_{arm}_ee_outer_finger'}}
    rejected = []
    # Elevated objects need a different overflight than objects on the table.
    # Start with the nominal clearance, then admit only reachable, clear routes.
    for rise in np.linspace(overfly,0.,5):
        high = standoff+np.array([0.,0.,rise])
        for staged in (True,False) if staging_altitude-point[2]>.02 else (False,):
            poses = ([np.r_[point[:2],staging_altitude]] if staged else [])+[high,standoff,pinch]
            route = [start]
            for target in poses:
                solved = None
                for seed in (route[-1],env.home_arm_qpos(arm)):
                    q,_,_=controller.solve_ik(arm,target,GRASP_ROTATION,q_init=seed)
                    if q is None:continue
                    actual,rotation,_,_=kin.evaluate(q)
                    if (np.linalg.norm(actual-target)<.004
                            and np.linalg.norm(orientation_error(rotation,GRASP_ROTATION))<.04):
                        solved=q;break
                if solved is None:break
                route.append(solved)
            else:
                curve=joint_spline(route,env.arm_velocity_limits(arm))
                if check is None or check(route,objects,allowed_contacts=allowed,curve=curve):
                    return route,{'approach_overfly_m':float(rise),'staging':staged,'rejected_routes':rejected}
                rejected.append({'overfly_m':float(rise),'staging':staged,'error':'COLLISION_PATH'})
                continue
            rejected.append({'overfly_m':float(rise),'staging':staged,'error':'IK_FAILURE'})
    return None,{'error':'APPROACH_PATH_UNAVAILABLE','rejected_routes':rejected}
