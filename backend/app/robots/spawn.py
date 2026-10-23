"""Desk props are admitted by private IK before any live object pose is written."""
import numpy as np

from backend.app.robots.urdf import URDFModel, URDFKinematics, solve_profile_ik
from backend.app.simulation.randomization import resolve_seed,desk_candidates,changed_position,SpawnSamplingError


class ReachableDeskSpawns:
    half_size = .026
    clearance = .002
    #: How far an achieved jaw frame may sit from the inferred one when the arm cannot hold
    #: the inferred frame exactly (measured 2026-10-24: OpenManipulator-X has four arm joints,
    #: so a fixed six-constraint grasp pose is unreachable for 75 of 99 desk probes, while a
    #: free attitude reaches all of them collision-free). Cubes still pinch with the pad faces
    #: a few degrees off, and the physical check remains two contacts plus lift-follow.
    attitude_tolerance_deg = 25.

    def __init__(self, profile, scene):
        self.profile, self.scene = profile, scene
        self._collision_queries = None
        self._validated = {}
        self._joint_limits={}
        self.half_size = float(scene.get("cube_size", .052)) / 2.
        self.model = URDFModel(profile['urdf'])
        self.preferred = {}
        self.bounds={}
        _,axes=self.model.forward(profile['home'],profile['root_transform'])
        for side, chain in profile['chains'].items():
            if profile.get('grippers') and side not in profile['grippers']:
                continue
            kin = URDFKinematics(self.model, chain, profile['home'], profile['root_transform'], chain['tool_offset'])
            q = np.asarray([profile['home'][name] for name in chain['joints']])
            self.preferred[side] = kin.evaluate(q)[0][:2]
            chain_joints=self.model.chain(chain['end_link'])
            first=next((i for i,j in enumerate(chain_joints) if j['name'] in chain['joints']),0)
            shoulder=axes[chain_joints[first]['name']][0] if chain_joints else np.asarray(profile['root_transform'])[:3,3]
            reach=sum(np.linalg.norm(np.asarray(j['origin'])[:3,3]) for j in chain_joints[first+1:])+np.linalg.norm(chain['tool_offset'])
            reach+=sum(max(abs(j['lower']),abs(j['upper'])) for j in chain_joints[first:]
                       if j['type']=='prismatic')
            radius=np.sqrt(max(0.,reach**2-(scene['table_top']+self.half_size-shoulder[2])**2))
            edge=np.asarray(scene['table_size'][:2])/2.-self.half_size-.02
            self.bounds[side]=(np.maximum(-edge,shoulder[:2]-radius),np.minimum(edge,shoulder[:2]+radius))

    @property
    def collision_queries(self):
        if self._collision_queries is None:
            from backend.app.robots.collision_planner import PrivateCollisionQueries
            self._collision_queries = PrivateCollisionQueries(self.profile,self.scene['table_top'])
        return self._collision_queries

    def arm_joint_limits(self,side):
        if side not in self._joint_limits:
            self._joint_limits[side]=self.collision_queries.adjacent_joint_intervals(
                self.profile['chains'][side]['joints'],self.profile['home'])
        return self._joint_limits[side]

    def validate(self, position, side):
        point = np.asarray(position, float)
        if point.shape != (3,) or not np.isfinite(point).all():
            raise ValueError('Spawn position must have three finite coordinates')
        half_table = np.asarray(self.scene['table_size'][:2]) / 2.
        if np.any(abs(point[:2]) + self.half_size + .02 > half_table):
            raise ValueError('Object spawn is outside the usable desk')
        expected_z = self.scene['table_top'] + self.half_size + self.clearance
        if abs(point[2] - expected_z) > .003:
            raise ValueError('Object spawn must be supported by the desk')
        key = (side,*point)
        if key in self._validated:
            import copy
            return copy.deepcopy(self._validated[key])
        modes = ('inferred', 'arm-achieved') if self.profile.get('grippers',{}).get(side) else ('inferred',)
        failure = None
        for mode in modes:
            try:
                entry = self._validated_pose(side, point, mode)
                break
            except ValueError as error:
                failure = error
        else:
            raise failure
        self._validated[key] = entry
        return entry

    def achieved_attitude(self, side, q, snapshot):
        chain = self.profile['chains'][side]
        kinematics = URDFKinematics(self.model, chain, snapshot, self.profile['root_transform'], chain['tool_offset'])
        return np.asarray(kinematics.evaluate(np.asarray(q))[1])

    def _validated_pose(self, side, point, mode):
        checks, seed = [], None
        names = self.profile['chains'][side]['joints']
        snapshot = dict(self.profile['home'])
        for g in self.profile.get('grippers',{}).values():
            snapshot.update({g['joint']:g['open_q'],**{n:g['open_q']*k for n,k in g['followers'].items()}})
        gripper = self.profile.get('grippers',{}).get(side)
        attitude = np.array(gripper['grasp_rotation']) if gripper else None
        joint_limits=self.arm_joint_limits(side)
        def checked_pose(target, initial):
            frame = attitude
            result = solve_profile_ik(self.profile,side,target,frame,model=self.model,q_init=initial,
                snapshot=snapshot,joint_limits=joint_limits)
            if mode == 'arm-achieved' and not result['success']:
                # A fixed jaw frame is a six-constraint request; an arm with fewer joints can
                # still pinch, holding whatever frame its own joints reach. Re-solve with that
                # frame fixed, so every offset below stays consistent with what the arm holds.
                free = solve_profile_ik(self.profile,side,target,None,model=self.model,q_init=initial,
                    snapshot=snapshot,joint_limits=joint_limits)
                if free['success']:
                    frame = self.achieved_attitude(side, free['q_rad'], snapshot)
                    off = self.attitude_offset_deg(frame, attitude, gripper)
                    if off > self.attitude_tolerance_deg:
                        raise ValueError(f'{side}: the arm can only hold a jaw frame {off:.0f} deg off the inferred one')
                    result = solve_profile_ik(self.profile,side,target,frame,model=self.model,q_init=free['q_rad'],
                        snapshot=snapshot,joint_limits=joint_limits)
            if not result['success']:
                raise ValueError(f'{side}: object or approach is outside IK reach')
            q = np.array(result['q_rad'])
            collisions=self.collision_queries.contacts({**snapshot,**dict(zip(names,q))},tolerance=-.0001)
            if collisions:
                if max(c['penetration_m'] for c in collisions)>.003:
                    raise ValueError(f'{side}: spawn IK posture deeply intersects the robot or desk')
                q = (self.collision_queries.refine_pose(self.profile,side,target,attitude,q,snapshot,
                        minimum_clearance=.0001,joint_limits=joint_limits)
                     if attitude is not None else None)
                if q is None:
                    raise ValueError(f'{side}: spawn grasp intersects the robot or desk')
                kin = URDFKinematics(self.model,self.profile['chains'][side],snapshot,
                    self.profile['root_transform'],self.profile['chains'][side]['tool_offset'])
                actual,rotation,_,_ = kin.evaluate(q)
                from scipy.spatial.transform import Rotation
                result = {**result,'q_rad':q.tolist(),'position_error':float(np.linalg.norm(actual-target)),
                    'rotation_error':float(np.linalg.norm(Rotation.from_matrix(frame@rotation.T).as_rotvec()))}
            return result,q,frame
        # Plan at the supported height. The object rests on the desk, so that height is the
        # desk's, not the caller's z minus the setup clearance: re-validating a settled object
        # (which sits exactly on the desk) otherwise plans 2mm below it and a marginal pose
        # fails (measured 2026-10-24: blue_cube settled 2.03mm lower and was refused, while
        # the same pose validated at the spawn height).
        supported = np.r_[point[:2], self.scene['table_top'] + self.half_size]
        if self.collision_queries.box_contact_points(snapshot,supported,np.eye(3),self.half_size*2.,
                set(self.model.links),distance=.0001):
            raise ValueError(f'{side}: spawn cube intersects the robot in its setup posture')
        frame = None
        for dz in (0., .04, .08):
            result,seed,frame = checked_pose(supported+[0.,0.,dz],seed)
            checks.append({'height_above_object': dz, **result})
        closure,lifted,holding_lifts = [],None,[]
        if gripper:
            rows = sorted(gripper['sweep'],key=lambda row:row['q'])
            qs = [row['q'] for row in rows]
            centers = [(row['face_a']+row['face_b'])/2. for row in rows]
            seed = np.array(checks[0]['q_rad'])
            opened = np.interp(gripper['open_q'],qs,centers)
            # Offsets follow the frame the arm actually holds, not the inferred one.
            pad_normal = frame @ np.array(gripper['pad_region']['normal'])
            for jaw in np.linspace(gripper['open_q'],gripper['closed_q'],5):
                snapshot.update({gripper['joint']:float(jaw),**{n:float(jaw*k) for n,k in gripper['followers'].items()}})
                target = supported-pad_normal*(np.interp(jaw,qs,centers)-opened)
                result,seed,frame = checked_pose(target,seed)
                closure.append({'jaw_q':float(jaw),**result})
            lifted,seed,frame = checked_pose(target+[0.,0.,.03],seed)
            for factor in (.95,1.,1.05):
                row = min(rows,key=lambda row:abs((row['face_b']-row['face_a'])-gripper['cube_size']*factor))
                jaw = row['q']
                snapshot.update({gripper['joint']:jaw,**{n:float(jaw*k) for n,k in gripper['followers'].items()}})
                target = supported-pad_normal*(np.interp(jaw,qs,centers)-opened)
                result,seed,frame = checked_pose(target,np.array(checks[0]['q_rad']))
                for dz in (.01,.02,.03):
                    result,seed,frame = checked_pose(target+[0.,0.,dz],seed)
                    holding_lifts.append({'jaw_q':jaw,'rise_m':dz,**result})
        entry = {'arm': side, 'position': point.tolist(), 'checks': checks,'closure_checks':closure,'lift_check':lifted,
                'holding_lift_checks':holding_lifts,
                'attitude_mode': mode,
                'attitude_offset_deg': self.attitude_offset_deg(frame, attitude, gripper),
                'scope': 'Supported-height grasp/approach IK, robot/desk clearance and sampled jaw closure; execution separately checks the full path and physical contact',
                'cube_size': self.half_size * 2.}
        return entry

    @staticmethod
    def attitude_offset_deg(frame, attitude, gripper):
        """How far the held jaw frame's pad facing sits from the inferred one."""
        if frame is None or attitude is None or not gripper:
            return 0.
        normal = np.asarray(gripper['pad_region']['normal'])
        ideal = np.asarray(attitude) @ normal
        held = np.asarray(frame) @ normal
        cosine = abs(float(ideal @ held)) / max(float(np.linalg.norm(ideal) * np.linalg.norm(held)), 1e-12)
        return float(np.degrees(np.arccos(np.clip(cosine, -1., 1.))))


    def measured_candidates(self, side, samples=(13, 16)):
        """Object positions the arm can actually reach, measured at the spawn height.

        ``bounds`` is a sphere approximation of reach - measured 2026-10-24 for a 0.3 m arm:
        x -0.69..0.22, y +/-0.45 - so random sampling inside it spends nearly every candidate
        on positions no IK can reach and the layout then omits the prop. Radii and angles are
        sampled around the base and kept only where the private position-only IK succeeds and
        the posture is free of deep contact; the caller validates them in the normal way.
        """
        chain = self.profile['chains'][side]
        root = np.asarray(self.profile['root_transform'])[:3, 3]
        z = self.scene['table_top'] + self.half_size + self.clearance
        half_table = np.asarray(self.scene['table_size'][:2]) / 2.
        snapshot = dict(self.profile['home'])
        for gripper in self.profile.get('grippers', {}).values():
            snapshot.update({gripper['joint']: gripper['open_q'],
                             **{n: gripper['open_q'] * k for n, k in gripper['followers'].items()}})
        names = chain['joints']
        limits = self.arm_joint_limits(side)
        reach = float(np.sum([np.linalg.norm(np.asarray(j['origin'])[:3, 3]) for j in self.model.chain(chain['end_link'])]))
        found = []
        for radius in np.linspace(.06, max(.10, min(.34, reach)), samples[0]):
            for angle in np.linspace(-np.pi, np.pi, samples[1], endpoint=False):
                point = np.r_[root[:2] + radius * np.array([np.cos(angle), np.sin(angle)]), z]
                if np.any(abs(point[:2]) + self.half_size + .02 > half_table):
                    continue
                result = solve_profile_ik(self.profile, side, point, None, model=self.model,
                                          snapshot=snapshot, joint_limits=limits)
                if not result['success']:
                    continue
                posture = {**snapshot, **dict(zip(names, result['q_rad']))}
                if any(c['penetration_m'] > .003
                       for c in self.collision_queries.contacts(posture, tolerance=-.0001)):
                    continue
                found.append(point.tolist())
        return found

    def layout(self, *, randomize=False, object_positions=None, base=None, rng=None,seed=None,
               shuffle_sides=True,stop=lambda:False):
        actual_seed=resolve_seed(seed) if rng is None and randomize else seed
        if rng is None:rng=np.random.default_rng(actual_seed)
        objects, omitted = {}, {}
        if randomize and base is not None and not base:
            raise SpawnSamplingError('The current scene has no reachable cubes to randomize')
        assignments = [('red_cube', 'left'), ('blue_cube', 'right')]
        if randomize and shuffle_sides and rng.integers(2):
            assignments=[('red_cube','right'),('blue_cube','left')]
        attempted={}
        measured_sampling=False
        for name, preferred_side in assignments:
            if base is not None and name not in base:
                omitted[name]='Object is absent from the current scene; reset cannot add a new entity'
                continue
            side = preferred_side if preferred_side in self.preferred else next(iter(self.preferred))
            explicit = (object_positions or {}).get(name)
            if explicit is not None and (np.asarray(explicit).shape != (3,) or not np.isfinite(explicit).all()):
                raise ValueError(f'{name}: spawn needs three finite coordinates')
            home_xy = np.asarray((base or {}).get(name, {}).get('position', [*self.preferred[side], 0.]))[:2]
            step = max(.008,min(.025,self.half_size))
            fine = [(x,y) for r in (step,1.5*step,2*step) for x,y in
                    [(-r,0.),(r,0.),(0.,-r),(0.,r),(-r,-r),(-r,r),(r,-r),(r,r)]]
            offsets = [(0.,0.),*fine,(.10, 0.), (-.10, 0.), (0., .10), (0., -.10),
                       (.07, .07), (.07, -.07), (-.07, .07), (-.07, -.07)]
            if randomize:
                lower,upper=self.bounds[side]
                try:xy=desk_candidates(lower,upper,rng)
                except ValueError as error:raise SpawnSamplingError(f'{side}: {error}') from None
                previous_point=(base or {}).get(name,{}).get('position')
                xy=[p for p in xy if changed_position(p,previous_point)]
            candidates = [np.asarray(explicit)] if explicit is not None else [
                np.r_[point, self.scene['table_top'] + self.half_size + self.clearance]
                for point in (xy if randomize else [home_xy+offset for offset in offsets])]
            attempted[name]=0
            for position in candidates:
                if stop():raise InterruptedError('Desk randomization stopped')
                if any(np.linalg.norm(position[:2] - np.asarray(other['position'])[:2]) < .10 for other in objects.values()):
                    continue
                attempted[name]+=1
                try:
                    objects[name] = self.validate(position, side)
                    break
                except ValueError:
                    continue
            if name not in objects and explicit is None and not (base and name in base):
                # The rectangular bounds can be mostly unreachable; fall back to the measured
                # band once, so a small arm still receives a prop instead of having none.
                measured = self.measured_candidates(side)
                order = rng.permutation(len(measured)) if randomize else range(len(measured))
                for index in order:
                    position = np.asarray(measured[index], float)
                    if stop():
                        raise InterruptedError('Desk randomization stopped')
                    if any(np.linalg.norm(position[:2] - np.asarray(other['position'])[:2]) < .10
                           for other in objects.values()):
                        continue
                    attempted[name] += 1
                    try:
                        objects[name] = self.validate(position.tolist(), side)
                        measured_sampling = True
                        break
                    except ValueError:
                        continue
            if name not in objects:
                if explicit is not None or base and name in base:
                    raise SpawnSamplingError(f'No changed, reachable, separated desk spawn for {name} after {attempted[name]} candidates')
                omitted[name] = 'No desk position reachable at object and approach heights; no object spawned'
        return {'objects': objects, 'omitted': omitted, 'openarm_cell': False,
                'randomization':{'enabled':randomize,'seed':actual_seed,'shuffle_sides':shuffle_sides,
                    'sampling': ('measured_reachable_band' if measured_sampling else
                                 'stratified_reachable_workspace' if randomize else 'default'),
                    'minimum_change_m':.04,'change_reference':'default_layout','candidate_attempts':attempted,
                    'candidate_bounds_m':{s:[a.tolist(),b.tolist()] for s,(a,b) in self.bounds.items()}},
                'method': 'private URDF IK and collision queries before scene setup', 'grasp_calibrated': False}
