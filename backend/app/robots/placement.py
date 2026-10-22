"""Loaded transport with the observed grip and a geometrically checked payload.

The box here is a private prediction for path planning. Only Genesis contact
forces move the real payload, and execution independently verifies its follow.
"""
import time
import numpy as np
from scipy.spatial.transform import Rotation

from backend.app.control.motion import joint_spline
from backend.app.robots.collision_planner import PrivateCollisionQueries
from backend.app.robots.urdf import solve_profile_ik
from backend.app.skills.robot_skills import SkillError


def boxes_overlap(a,ra,ha,b,rb,hb,skin=.0003):
    delta=np.asarray(b)-a
    axes=[*np.asarray(ra).T,*np.asarray(rb).T]
    axes.extend(np.cross(x,y) for x in np.asarray(ra).T for y in np.asarray(rb).T)
    for axis in axes:
        length=np.linalg.norm(axis)
        if length<1e-8:continue
        axis=axis/length
        radius=np.sum(abs(np.asarray(ra).T@axis)*ha)+np.sum(abs(np.asarray(rb).T@axis)*hb)
        if abs(delta@axis)>radius+skin:return False
    return True


def stable_cube_rotations():
    bases={tuple(np.round(r[2]).astype(int)):r for r in Rotation.create_group('O').as_matrix()}
    return [Rotation.from_euler('z',yaw,degrees=True).as_matrix()@r
            for r in bases.values() for yaw in range(0,360,15)]


class LoadedBoxQueries(PrivateCollisionQueries):
    def __init__(self,env,side,obj):
        self.base=env.collision_queries;self.env=env;self.side=side
        self.names=env.profile['chains'][side]['joints'];self.kinematics=env.arm_kinematics(side)
        point,rotation=env.ee_pose(side);center,attitude=env.object_pose(obj)
        self.offset=rotation.T@(center-point);self.relative_rotation=rotation.T@attitude
        self.half=np.repeat(env.cube_size/2.,3)
        fingers={n for group in env.grippers[side]['contact_groups'] for n in group}
        self.forbidden=set(env.model.links)-fingers
        self.robot_links=set(env.model.links)
        self.obstacles=[env.object_pose(name) for name in env._objects if name!=obj]

    def clear(self,values):
        if self.base.contacts(values,tolerance=-.0001):return False
        q=np.array([values[n] for n in self.names]);point,rotation,*_=self.kinematics.evaluate(q)
        center=point+rotation@self.offset;attitude=rotation@self.relative_rotation
        if center[2]-np.sum(abs(attitude[2])*self.half)<self.env.table_height+.0003:return False
        if any(boxes_overlap(center,attitude,self.half,p,r,self.half,skin=.002) for p,r in self.obstacles):return False
        if self.base.box_contact_points(values,center,attitude,self.env.cube_size,self.forbidden,distance=.002):return False
        return not any(self.base.box_contact_points(values,p,r,self.env.cube_size,self.robot_links,distance=.002)
                       for p,r in self.obstacles)


class EmptyHandQueries(PrivateCollisionQueries):
    def __init__(self,env,side):
        self.base=env.collision_queries;self.env=env
        self.robot_links=set(env.model.links)
        self.obstacles=[env.object_pose(name) for name in env._objects]
        self.names=env.profile['chains'][side]['joints'];self.start=env.arm_qpos(side)
        fingers={n for group in env.grippers[side]['contact_groups'] for n in group}
        self.initial=[]
        for p,r in self.obstacles:
            contacts=self.base.box_contact_points(env.joint_snapshot(),p,r,env.cube_size,self.robot_links,distance=.001)
            self.initial.append({name:min(v['distance_m'] for v in contacts if v['link']==name)
                for name in fingers if any(v['link']==name for v in contacts)})

    def clear(self,values):
        if not self.base.clear(values):return False
        small=np.max(abs(np.array([values[n] for n in self.names])-self.start))<=.06
        for (p,r),allowed in zip(self.obstacles,self.initial):
            for contact in self.base.box_contact_points(values,p,r,self.env.cube_size,self.robot_links,distance=.001):
                if (not small or contact['link'] not in allowed
                        or contact['distance_m']<allowed[contact['link']]-.00005):return False
        return True


def cleared_empty_path(env,arm,goal,stop):
    query=EmptyHandQueries(env,arm);snapshot=env.joint_snapshot();names=env.profile['chains'][arm]['joints']
    path=query.path(names,env.arm_qpos(arm),goal,snapshot,env.ik_joint_limits(arm),stop)
    if path is None:return None
    for spacing in (.05,.025,.01):
        dense=[path[0]]
        for a,b in zip(path,path[1:]):
            count=max(1,int(np.ceil(np.max(abs(b-a))/spacing)))
            dense.extend(a+(b-a)*t for t in np.linspace(0.,1.,count+1)[1:])
        curve=joint_spline(dense,env.arm_velocity_limits(arm))
        count=max(40,int(np.sum(np.linalg.norm(np.diff(dense,axis=0),axis=1))/.025))
        if all(query.clear({**snapshot,**dict(zip(names,curve(t)))})
               for t in np.unique(np.r_[np.linspace(0.,1.,count+1),curve.x])):return dense
    return None


def shorten_loaded_path(env,arm,obj,path,stop=lambda:False):
    """Shortcut a transit only when the robot AND real payload geometry fit.

    A single held cube may change orientation during free transit. Keeping an
    unnecessary straight Cartesian tool line can produce large redundant wrist
    excursions and slow down a route which has a short, clear joint path.
    """
    query=LoadedBoxQueries(env,arm,obj);snapshot=env.joint_snapshot();names=query.names
    compact=[np.asarray(path[0])];index=0
    while index<len(path)-1:
        if stop():return None
        end=len(path)-1
        while end>index and not query.edge_clear(names,path[index],path[end],snapshot):end-=1
        if end==index:return None
        compact.append(np.asarray(path[end]));index=end
    original=joint_spline(path,env.arm_velocity_limits(arm))
    from backend.app.control.path_timing import PathTiming
    old_time=PathTiming(query.kinematics,original,path[0],path[-1],env.arm_velocity_limits(arm),env.cartesian_motion_limits).duration
    for spacing in (.05,.025,.01):
        dense=[compact[0]]
        for a,b in zip(compact,compact[1:]):
            count=max(1,int(np.ceil(np.max(abs(b-a))/spacing)))
            dense.extend(a+(b-a)*t for t in np.linspace(0.,1.,count+1)[1:])
        curve=joint_spline(dense,env.arm_velocity_limits(arm))
        count=max(40,int(np.sum(np.linalg.norm(np.diff(dense,axis=0),axis=1))/.025))
        if all(query.clear({**snapshot,**dict(zip(names,curve(t)))})
               for t in np.unique(np.r_[np.linspace(0.,1.,count+1),curve.x])):
            new_time=PathTiming(query.kinematics,curve,dense[0],dense[-1],env.arm_velocity_limits(arm),env.cartesian_motion_limits).duration
            if new_time<old_time:return dense
            break
    # A failed shortcut must not replace the original validated transit.
    count=max(40,int(np.sum(np.linalg.norm(np.diff(path,axis=0),axis=1))/.025))
    if all(query.clear({**snapshot,**dict(zip(names,original(t)))})
           for t in np.unique(np.r_[np.linspace(0.,1.,count+1),original.x])):return path
    return None


def plan_placement(skills,arm,obj,xy,surface,reference=None,timeout=90.):
    env=skills.env;c=skills.controller;deadline=time.monotonic()+timeout
    loaded=LoadedBoxQueries(env,arm,obj);snapshot=env.joint_snapshot();start=env.arm_qpos(arm)
    object_rotation=env.object_pose(obj)[1]
    group=stable_cube_rotations()
    preferred=np.asarray(env.grippers[arm]['grasp_rotation'])
    # All 24 cube symmetries are equivalent stable faces, including a cube
    # received with its original top face between the two fingers.
    attitudes=sorted(group,key=lambda r:
        .6*np.linalg.norm(Rotation.from_matrix(r@object_rotation.T).as_rotvec())+
        .4*np.linalg.norm(Rotation.from_matrix(r@loaded.relative_rotation.T@preferred.T).as_rotvec()))
    attempts=0
    for release_gap in (.006,.010,.016):
        center=np.r_[np.asarray(xy)[:2],surface+env.cube_size/2.+release_gap]
        for attitude in attitudes:
            if c.stop_requested():raise SkillError('STOPPED','Loaded placement planning stopped')
            if time.monotonic()>deadline:break
            rotation=attitude@loaded.relative_rotation.T
            point=center-rotation@loaded.offset
            for guess in (start,env.home_arm_qpos(arm)):
                attempts+=1
                solution=solve_profile_ik(env.profile,arm,point,rotation,model=env.model,q_init=guess,
                    snapshot=snapshot,joint_limits=env.ik_joint_limits(arm))
                if not solution['success'] or solution['position_error']>.0005 or solution['rotation_error']>.01:continue
                goal=env.model.nearest_continuous(loaded.names,solution['q_rad'],start)
                if not loaded.clear({**snapshot,**dict(zip(loaded.names,goal))}):continue
                path=loaded.path(loaded.names,start,goal,snapshot,env.ik_joint_limits(arm),c.stop_requested)
                if path is None:continue
                for spacing in (.05,.025,.01):
                    dense=[path[0]]
                    for a,b in zip(path,path[1:]):
                        count=max(1,int(np.ceil(np.max(abs(b-a))/spacing)))
                        dense.extend(a+(b-a)*t for t in np.linspace(0.,1.,count+1)[1:])
                    curve=joint_spline(dense,env.arm_velocity_limits(arm))
                    count=max(40,int(np.sum(np.linalg.norm(np.diff(dense,axis=0),axis=1))/.025))
                    samples=np.unique(np.r_[np.linspace(0.,1.,count+1),curve.x])
                    if all(loaded.clear({**snapshot,**dict(zip(loaded.names,curve(t)))}) for t in samples):
                        return {'path':dense,'target':point,'rotation':rotation,'object_target':center,
                                'object_rotation':attitude,'candidate_count':attempts,'release_gap_m':release_gap}
        if time.monotonic()>deadline:break
    error=SkillError('COLLISION_PATH','No loaded path can place the cube on a stable face without robot, desk or payload collisions')
    error.evidence={'candidate_count':attempts,'reference':reference}
    raise error
