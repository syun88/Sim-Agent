"""Check the real fingertip geometry, attitude and approach before cube setup."""
import numpy as np

from backend.app.control.motion import OpenArmV2Controller
from backend.app.simulation.openarm_v2.definition import FINGERTIP_MIDPOINT, GRASP_ROTATION
from backend.app.simulation.transform import orientation_error
from backend.app.simulation.randomization import resolve_seed,desk_candidates,changed_position,SpawnSamplingError


def private_spawn_environment():
    """Official-model FK/IK only, without constructing a Genesis world."""
    import mujoco
    from backend.app.simulation.genesis_backend.environment import _OfficialModelMetadata
    from backend.app.simulation.openarm_v2.scene import resolved_scene_xml
    from backend.app.simulation.openarm_v2.definition import ARM_JOINTS,SPAWN_ARM_QPOS
    from backend.app.simulation.openarm_v2.kinematics import OfficialArmKinematics
    meta=_OfficialModelMetadata(resolved_scene_xml());snapshot=meta.qpos_with_arm_pose(SPAWN_ARM_QPOS)
    class PrivateEnvironment:
        def arm_kinematics(self,side):return OfficialArmKinematics(meta.model,snapshot,side)
        def arm_limits(self,side):
            ids=[mujoco.mj_name2id(meta.model,mujoco.mjtObj.mjOBJ_JOINT,n) for n in ARM_JOINTS[side]]
            return meta.model.jnt_range[ids]
        def home_arm_qpos(self,side):return SPAWN_ARM_QPOS.copy()
    return PrivateEnvironment()


def random_spawn_layout(env,*,seed=None,shuffle_sides=True):
    """Search the attached cell's desk, admitting complete approach IK poses."""
    import mujoco
    from backend.app.simulation.openarm_v2.scene import INITIAL_OBJECT_POSES
    seed=resolve_seed(seed);rng=np.random.default_rng(seed)
    kin=env.arm_kinematics('left');model=kin._model
    table=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,'cell_table_col')
    cube=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,'red_cube_geom')
    half=float(max(model.geom_size[cube]));margin=half+.03
    lower=model.geom_pos[table,:2]-model.geom_size[table,:2]+margin
    upper=model.geom_pos[table,:2]+model.geom_size[table,:2]-margin
    names=['red_cube','blue_cube']
    if shuffle_sides and rng.integers(2):names.reverse()
    positions={};report={'objects':{},'method':'private official-model IK before scene setup'}
    bounds={};attempts={}
    for name,side in zip(names,('left','right')):
        lo,hi=lower.copy(),upper.copy()
        if side=='left':lo[1]=max(lo[1],margin)
        else:hi[1]=min(hi[1],-margin)
        bounds[side]=[lo.tolist(),hi.tolist()];attempts[name]=0
        for xy in desk_candidates(lo,hi,rng):
            if not changed_position(xy,INITIAL_OBJECT_POSES[name]):continue
            if any(np.linalg.norm(xy-p[:2])<.10 for p in positions.values()):continue
            point=np.r_[xy,INITIAL_OBJECT_POSES[name][2]]
            if getattr(env,'planning_pulse',None) and env.planning_pulse():
                raise InterruptedError('Desk randomization stopped')
            attempts[name]+=1
            try:entry=validate_spawn_positions(env,{name:point})['objects'][name]
            except ValueError:continue
            positions[name]=point;report['objects'][name]=entry;break
        else:raise SpawnSamplingError(f'No changed reachable desk spawn for {name} after {attempts[name]} candidates')
    report['randomization']={'enabled':True,'seed':seed,'shuffle_sides':shuffle_sides,
        'sampling':'stratified_reachable_workspace','minimum_change_m':.04,'change_reference':'default_layout',
        'candidate_attempts':attempts,'candidate_bounds_m':bounds}
    return positions,report


def setup_cube_is_clear(kin,name,point):
    """A private collision query checks a prop against the cell and home robot."""
    import mujoco
    model=kin._model;data=mujoco.MjData(model);data.qpos[:]=kin._data.qpos
    body=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,name)
    if body<0:return True  # The wine-only scene has no cube body.
    joint=model.body_jntadr[body];address=model.jnt_qposadr[joint]
    data.qpos[address:address+3]=point
    mujoco.mj_kinematics(model,data);mujoco.mj_collision(model,data)
    cubes={mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,n+'_geom') for n in ('red_cube','blue_cube')}
    for c in data.contact:
        a,b=int(c.geom1),int(c.geom2)
        if c.dist>=-.0001:continue
        if int(model.geom_bodyid[a])==body and b not in cubes:return False
        if int(model.geom_bodyid[b])==body and a not in cubes:return False
    return True


def approach_pose_is_clear(kin,name,point,side):
    import mujoco
    model=kin._model;data=mujoco.MjData(model);data.qpos[:]=kin._data.qpos
    cube=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,name)
    address=model.jnt_qposadr[model.body_jntadr[cube]];data.qpos[address:address+3]=point
    mujoco.mj_kinematics(model,data);mujoco.mj_collision(model,data)
    allowed={f'openarm_{side}_ee_inner_finger',f'openarm_{side}_ee_outer_finger'}
    for c in data.contact:
        if c.dist>=-.0001:continue
        names=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,int(model.geom_bodyid[g])) or 'world'
               for g in (c.geom1,c.geom2)]
        if any(n in ('red_cube','blue_cube') and n!=name for n in names):continue
        if name in names:
            other=names[1-names.index(name)]
            if other not in allowed:return False
        elif any(n.startswith(f'openarm_{side}_') for n in names):return False
    return True


def validate_spawn_positions(env, placements):
    controller = OpenArmV2Controller(env, configure_motors=False)
    cache = getattr(env, '_spawn_ik_cache', {})
    report = {}
    for name, point in placements.items():
        point = np.asarray(point, float)
        if point.shape != (3,) or not np.isfinite(point).all():
            raise ValueError(f'{name}: spawn needs three finite coordinates')
        side = 'left' if name == 'wine_bottle' or point[1] >= 0. else 'right'
        key = (side, *point)
        if key not in cache:
            pinch = point - GRASP_ROTATION @ FINGERTIP_MIDPOINT[side]
            standoff = pinch - .06 * (GRASP_ROTATION @ np.array([0., 0., -1.]))
            goals = [('pinch', pinch), ('pregrasp', standoff)]
            if name.endswith('_cube'):
                goals.append(('pregrasp_high', standoff + [0., 0., .10]))
            checks, seed = [], env.home_arm_qpos(side)
            kin = env.arm_kinematics(side)
            if name.endswith('_cube') and not setup_cube_is_clear(kin,name,point):
                raise ValueError(f'{name}: spawn intersects the cell or robot')
            for label, target in goals:
                q, _, _ = controller.solve_ik(side, target, GRASP_ROTATION, q_init=seed)
                if q is None:
                    raise ValueError(f'{name}: {side} {label} is outside IK reach')
                actual, rotation, _, _ = kin.evaluate(q)
                if name.endswith('_cube') and not approach_pose_is_clear(kin,name,point,side):
                    raise ValueError(f'{name}: {side} {label} intersects the cell or robot')
                error = float(np.linalg.norm(actual - target))
                angular_error = float(np.linalg.norm(orientation_error(rotation, GRASP_ROTATION)))
                if error > .003 or angular_error > .03:
                    raise ValueError(f'{name}: {side} {label} fails precise spawn IK ({error:.4f}m)')
                checks.append({'pose': label, 'target_position': target.tolist(), 'q_rad': q.tolist(),
                               'position_error': error, 'rotation_error': angular_error})
                seed = q
            cache[key] = {'position': point.tolist(), 'arm': side, 'checks': checks,
                          'scope': 'official fingertip pinch and approach IK; physical grasp verified during task'}
        report[name] = cache[key]
    env._spawn_ik_cache = cache
    return {'objects': report, 'method': 'private official-model IK before scene setup'}
