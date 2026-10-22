"""Plan a physical exchange against both mechanisms' measured geometry.

The private oracle predicts collision clearance only. Genesis contacts and an
independent receiving-hand lift decide whether an exchange actually succeeded.
"""
from collections import Counter
import time

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from backend.app.control.motion import joint_spline
from backend.app.robots.urdf import solve_profile_ik
from backend.app.skills.robot_skills import SkillError


class ExchangePlanner:
    def __init__(self,skills,timeout=180.):
        self.skills=skills;self.env=skills.env;self.controller=skills.controller
        self.deadline=time.monotonic()+timeout
        self.attempts=Counter()

    def check(self):
        if self.controller.stop_requested():raise SkillError('STOPPED','Exchange planning stopped')
        if time.monotonic()>self.deadline:
            error=SkillError('HANDOVER_UNREACHABLE','No complete exchange path found within the planning budget')
            error.evidence={'planning_attempts':dict(self.attempts)}
            raise error
        if self.env.step_observer:self.env.step_observer()

    def receiver(self,donor,receiver,center,cube_rotation,snapshot,completion=None,preferred=None,preferred_only=False):
        env=self.env;c=self.controller;g=env.grippers[receiver]
        names=env.profile['chains'][receiver]['joints']
        # Keep a finite contact patch on the cube while making room for the
        # other finger bodies. Their thickness is read from the inferred pads.
        overlap=min(.004,env.cube_size*.15)
        rng=np.random.default_rng(9)
        home=env.arm_kinematics(receiver).evaluate(env.home_arm_qpos(receiver))[0]
        toward=center-home;toward/=max(np.linalg.norm(toward),1e-9)
        local_forward=np.asarray(g['grasp_rotation']).T@g['approach_axis']
        attitudes=[]
        for yaw in (0.,90.,180.,270.):
            for roll in (0.,90.,180.,270.):
                for pitch in (-15.,0.,-30.,15.,30.):
                    rotation=(cube_rotation@Rotation.from_euler('z',yaw,degrees=True).as_matrix()
                        @Rotation.from_euler('x',roll,degrees=True).as_matrix()
                        @Rotation.from_euler('y',pitch,degrees=True).as_matrix()@np.asarray(g['grasp_rotation']))
                    forward=rotation@local_forward
                    angle=np.linalg.norm(Rotation.from_matrix(rotation@env.ee_pose(receiver)[1].T).as_rotvec())
                    attitudes.append((float(forward@toward)-angle*.08,rotation))
        rotations=[r for score,r in sorted(attitudes,key=lambda item:-item[0])]
        if preferred:rotations.insert(0,cube_rotation@np.asarray(preferred['object_rotation']))
        if preferred_only:rotations=rotations[:1]
        for index,rotation in enumerate(rotations):
            direction=rotation@(np.asarray(g['grasp_rotation']).T@g['approach_axis'])
            pad=g['pad_region']
            height_axis=rotation@np.asarray(pad['height_axis'])
            radius=env.cube_size*.5*np.sum(abs(cube_rotation.T@height_axis))
            height_limit=radius+g['pad_height']/2.-overlap
            bite_travel=(pad['depth_max']-pad['depth_min'])/2.+env.cube_size/2.-overlap
            biases=[-cube_rotation.T@direction*bite_travel*depth+cube_rotation.T@height_axis*height_limit*f*sign
                    for depth in (.6,.85,.3,0.) for f in (.55,.8,.3,0.) for sign in (1.,-1.)]
            if preferred and index==0:biases.insert(0,np.asarray(preferred['bias'],float))
            if preferred_only:biases=biases[:1]
            for bias in biases:
                target=center+cube_rotation@bias
                seeds=[env.arm_qpos(receiver),rng.normal(0.,.5,len(names))]
                if preferred and index==0:seeds.insert(0,np.asarray(preferred['open_arm_q']))
                for seed in seeds:
                    self.check();self.attempts['receiver_pose']+=1
                    goal=solve_profile_ik(env.profile,receiver,target,rotation,model=env.model,
                        q_init=seed,snapshot=snapshot,joint_limits=env.ik_joint_limits(receiver))
                    if not goal['success']:continue
                    self.attempts['receiver_ik']+=1
                    q=env.model.nearest_continuous(names,goal['q_rad'],env.arm_qpos(receiver))
                    # Check the contact geometry with one loaded-aperture IK
                    # solve before spending 49 solves on a complete jaw sweep.
                    rows=sorted(g['sweep'],key=lambda row:row['q'])
                    loaded=min(rows,key=lambda row:abs(row['gap']-(env.cube_size-.0004)))['q']
                    centers=[(v['face_a']+v['face_b'])/2. for v in rows];jaws=[v['q'] for v in rows]
                    loaded_target=target-rotation@np.asarray(pad['normal'])*(np.interp(loaded,jaws,centers)-np.interp(g['open_q'],jaws,centers))
                    loaded_snapshot={**snapshot,g['joint']:loaded,**{n:loaded*k for n,k in g['followers'].items()}}
                    predicted=solve_profile_ik(env.profile,receiver,loaded_target,rotation,model=env.model,
                        q_init=q,snapshot=loaded_snapshot,joint_limits=env.ik_joint_limits(receiver))
                    if not predicted['success'] or not self.grasp_geometry(receiver,center,cube_rotation,snapshot,
                            [{'jaw_q':loaded,'arm_q':np.asarray(predicted['q_rad'])}]):continue
                    self.attempts['geometric_pinch']+=1
                    collisions=env.collision_queries.contacts({**snapshot,**dict(zip(names,q))},tolerance=-.0001)
                    if collisions:
                        # A large body intersection is a different candidate,
                        # not a near-contact refinement problem.
                        if max(v['penetration_m'] for v in collisions)>.003:continue
                        q=env.collision_queries.refine_pose(env.profile,receiver,target,rotation,q,snapshot,minimum_clearance=.0001)
                    if q is None:continue
                    closure=c.plan_closure(receiver,target,rotation,q,snapshot=snapshot,check_lift=False,
                        minimum_gap=env.cube_size-.001)
                    if closure is None:continue
                    if not self.grasp_geometry(receiver,center,cube_rotation,snapshot,closure):continue
                    self.attempts['receiver_closure']+=1
                    distance=max(.06,g['finger_length']*1.1)
                    count=max(1,int(np.ceil(distance/.005)))
                    corridor=c.plan_poses(receiver,[target-direction*t for t in np.linspace(0.,distance,count+1)[1:]],
                        [rotation]*count,snapshot=snapshot,start=closure[0]['arm_q'])
                    if corridor is None:continue
                    self.attempts['receiver_insertion']+=1
                    transit=env.collision_queries.path(names,env.arm_qpos(receiver),corridor[-1],snapshot,
                        env.arm_limits(receiver),self.controller.stop_requested)
                    if transit is None:continue
                    path=[*transit,*corridor[-2::-1]]
                    curve=joint_spline(path,env.arm_velocity_limits(receiver))
                    if not env.joint_curve_is_clear(receiver,curve,path,snapshot=snapshot):continue
                    result={'rotation':rotation,'bias':bias,'closure':closure,'path':path,'target':target}
                    if completion is not None and not completion(result):continue
                    return result
        return None

    def grasp_geometry(self,side,center,cube_rotation,snapshot,closure):
        env=self.env;g=env.grippers[side];pad=g['pad_region']
        rows=sorted(g['sweep'],key=lambda row:row['q'])
        loaded=min(closure,key=lambda row:abs(np.interp(row['jaw_q'],[r['q'] for r in rows],
            [r['gap'] for r in rows])-(env.cube_size-.0004)))
        future={**snapshot,**dict(zip(env.profile['chains'][side]['joints'],loaded['arm_q'])),
                g['joint']:loaded['jaw_q'],**{n:loaded['jaw_q']*k for n,k in g['followers'].items()}}
        poses,_=env.model.forward(future,env.profile['root_transform'])
        ref=poses[g['reference_link']]
        links={n for group in g['contact_groups'] for n in group}
        contacts=env.collision_queries.box_contact_points(future,center,cube_rotation,env.cube_size,links)
        good=set()
        for contact in contacts:
            point=ref[:3,:3].T@(np.asarray(contact['position'])-ref[:3,3])-pad['pivot']
            if (pad['depth_min']<=point@pad['forward']<=pad['depth_max']
                    and abs(point@pad['height_axis']-pad['height_center'])<=pad['height_half_extent']):
                good.add(contact['link'])
        return all(good.intersection(group) for group in g['contact_groups'])

    def completion(self,donor,receiver,donor_q,donor_rotation,snapshot,recipient):
        env=self.env;g=env.grippers[receiver]
        rows=sorted(g['sweep'],key=lambda row:row['q'])
        jaw=[v['q'] for v in rows];gap=[v['gap'] for v in rows]
        loaded=min(recipient['closure'],key=lambda row:abs(np.interp(row['jaw_q'],jaw,gap)-env.cube_size))
        future={**snapshot,**dict(zip(env.profile['chains'][receiver]['joints'],loaded['arm_q'])),
                g['joint']:loaded['jaw_q'],**{n:loaded['jaw_q']*k for n,k in g['followers'].items()}}
        giver=env.grippers[donor]
        for q in np.linspace(snapshot[giver['joint']],giver['open_q'],33):
            future.update({giver['joint']:q,**{n:q*k for n,k in giver['followers'].items()}})
            if env.collision_queries.contacts(future,tolerance=-.0001):return False
        point=env.arm_kinematics(donor).evaluate(donor_q)[0]
        retreat=withdrawal(self.skills,donor,receiver,snapshot=future,start=point,
            rotation=donor_rotation,start_q=donor_q)
        if retreat is None:return False
        future.update(dict(zip(env.profile['chains'][donor]['joints'],retreat['path'][-1])))
        point=env.arm_kinematics(receiver).evaluate(loaded['arm_q'])[0]
        lift=self.controller.plan_poses(receiver,[point+[0.,0.,z] for z in np.arange(.005,.0301,.005)],
            [recipient['rotation']]*6,snapshot=future,start=loaded['arm_q'])
        return lift is not None

    def plan(self,donor,receiver,obj):
        env=self.env;c=self.controller
        start,rotation=env.ee_pose(donor)
        other=env.arm_kinematics(receiver).evaluate(env.home_arm_qpos(receiver))[0]
        object_position,object_rotation=env.object_pose(obj)
        offset=rotation.T@(object_position-start)
        forward=rotation@(np.asarray(env.grippers[donor]['grasp_rotation']).T@env.grippers[donor]['approach_axis'])
        toward=other-start
        yaw=np.arctan2(toward[1],toward[0])-np.arctan2(forward[1],forward[0])
        yaw=float(np.arctan2(np.sin(yaw),np.cos(yaw)))
        yaw=np.round(yaw/(np.pi/12.))*(np.pi/12.)
        length=max(g['finger_length'] for g in env.grippers.values())
        snapshot=env.joint_snapshot();names=env.profile['chains'][donor]['joints']
        homes=[env.arm_kinematics(s).evaluate(env.home_arm_qpos(s))[0] for s in (donor,receiver)]
        home_mid=np.mean(homes,axis=0)
        xy=home_mid[:2]
        height=max(env.table_height+2.45*length,home_mid[2]+length*.75)
        from backend.app.robots.skill_recipes import load_exchange_recipe
        recipe=load_exchange_recipe(env.profile,donor,receiver)
        if recipe:
            center=np.asarray(recipe['center']);final=np.asarray(recipe['donor_rotation'])
            angle=np.linalg.norm(Rotation.from_matrix(final@rotation.T).as_rotvec())
            count=max(12,int(np.ceil(np.linalg.norm(center-object_position)/.01)),int(np.ceil(angle/.09)))
            arc=Slerp([0.,1.],Rotation.from_matrix([rotation,final]));fractions=np.linspace(0.,1.,count+1)[1:]
            rotations=[arc(t).as_matrix() for t in fractions]
            path=c.plan_poses(donor,[object_position+(center-object_position)*t-r@offset for t,r in zip(fractions,rotations)],rotations)
            if path is not None:
                from backend.app.robots.placement import shorten_loaded_path
                path=shorten_loaded_path(env,donor,obj,path,c.stop_requested)
            if path is not None:
                future={**snapshot,**dict(zip(names,path[-1]))}
                predicted=final@rotation.T@object_rotation
                recipient=self.receiver(donor,receiver,center,predicted,future,
                    lambda p:self.completion(donor,receiver,path[-1],final,future,p),
                    preferred=recipe['predicted_receiver'],preferred_only=True)
                if recipient is not None:
                    self.attempts['recipe_cache_hit']+=1
                    return {'center':center,'donor_path':path,'donor_rotation':final,'receiver':recipient,
                        'predicted_cube_rotation':predicted,'observed_receiver_hint':recipe['receiver'],
                        'planning_attempts':dict(self.attempts)}
            self.attempts['recipe_cache_rejected']+=1
        if (np.linalg.norm(object_position[:2]-home_mid[:2])<length*1.8
                and object_position[2]>env.table_height+env.cube_size*2.):
            self.check()
            recipient=self.receiver(donor,receiver,object_position,object_rotation,snapshot,
                lambda p:self.completion(donor,receiver,env.arm_qpos(donor),rotation,snapshot,p))
            if recipient is not None:
                return {'center':object_position,'donor_path':[env.arm_qpos(donor)]*2,
                    'donor_rotation':rotation,'receiver':recipient,'predicted_cube_rotation':object_rotation,
                    'planning_attempts':dict(self.attempts)}
        for dz in (0.,-.5*length,.5*length,-length,length):
            for dx in (0.,length*.6,-length*.6):
                center=np.r_[xy+[dx,0.],height+dz]
                for extra in (0.,-30.,30.,-60.,60.):
                    self.check();self.attempts['donor_route']+=1
                    turn=Rotation.from_euler('z',yaw+np.deg2rad(extra)).as_matrix()
                    final=turn@rotation
                    count=max(12,int(np.ceil(np.linalg.norm(center-object_position)/.01)),
                              int(np.ceil(abs(yaw+np.deg2rad(extra))/.09)))
                    arc=Slerp([0.,1.],Rotation.from_matrix([rotation,final]))
                    fractions=np.linspace(0.,1.,count+1)[1:]
                    rotations=[arc(t).as_matrix() for t in fractions]
                    points=[object_position+(center-object_position)*t-r@offset for t,r in zip(fractions,rotations)]
                    path=c.plan_poses(donor,points,rotations)
                    if path is None:continue
                    from backend.app.robots.placement import shorten_loaded_path
                    path=shorten_loaded_path(env,donor,obj,path,c.stop_requested)
                    if path is None:continue
                    self.attempts['donor_clear']+=1
                    future={**snapshot,**dict(zip(names,path[-1]))}
                    recipient=self.receiver(donor,receiver,center,turn@object_rotation,future,
                        lambda plan:self.completion(donor,receiver,path[-1],final,future,plan))
                    if recipient is not None:
                        return {'center':center,'donor_path':path,'donor_rotation':final,
                                'receiver':recipient,'predicted_cube_rotation':turn@object_rotation,
                                'planning_attempts':dict(self.attempts)}
        error=SkillError('HANDOVER_UNREACHABLE','No shared posture passed loaded travel, receiver closure and insertion checks')
        error.evidence={'planning_attempts':dict(self.attempts)}
        raise error


def withdrawal(skills,donor,receiver,*,snapshot=None,start=None,rotation=None,start_q=None):
    """Separate the fixed pad, then withdraw along the measured approach axis."""
    env=skills.env;c=skills.controller;g=env.grippers[donor]
    actual,attitude=env.ee_pose(donor)
    start=actual if start is None else np.asarray(start)
    rotation=attitude if rotation is None else np.asarray(rotation)
    normal=rotation@np.asarray(g['pad_region']['normal'])
    direction=rotation@(np.asarray(g['grasp_rotation']).T@g['approach_axis'])
    for separation in (.003,.0015):
        separated=start-normal*min(separation,env.cube_size*.12)
        end=separated-direction*max(.055,g['finger_length']*.8)
        points=[]
        for a,b in ((start,separated),(separated,end)):
            count=max(1,int(np.ceil(np.linalg.norm(b-a)/.005)))
            points.extend(a+(b-a)*t for t in np.linspace(0.,1.,count+1)[1:])
        path=c.plan_poses(donor,points,[rotation]*len(points),snapshot=snapshot,start=start_q)
        if path is not None:return {'path':path,'target':end,'rotation':rotation}
    return None
