"""Desk props are admitted by private IK before any live object pose is written."""
import numpy as np

from backend.app.robots.urdf import URDFModel, URDFKinematics, solve_profile_ik


class ReachableDeskSpawns:
    half_size = .026
    clearance = .002

    def __init__(self, profile, scene):
        self.profile, self.scene = profile, scene
        self._collision_queries = None
        self._validated = {}
        self.half_size = float(scene.get("cube_size", .052)) / 2.
        self.model = URDFModel(profile['urdf'])
        self.preferred = {}
        for side, chain in profile['chains'].items():
            if profile.get('grippers') and side not in profile['grippers']:
                continue
            kin = URDFKinematics(self.model, chain, profile['home'], profile['root_transform'], chain['tool_offset'])
            q = np.asarray([profile['home'][name] for name in chain['joints']])
            self.preferred[side] = kin.evaluate(q)[0][:2]

    @property
    def collision_queries(self):
        if self._collision_queries is None:
            from backend.app.robots.collision_planner import PrivateCollisionQueries
            self._collision_queries = PrivateCollisionQueries(self.profile,self.scene['table_top'])
        return self._collision_queries

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
        checks, seed = [], None
        names = self.profile['chains'][side]['joints']
        snapshot = dict(self.profile['home'])
        for g in self.profile.get('grippers',{}).values():
            snapshot.update({g['joint']:g['open_q'],**{n:g['open_q']*k for n,k in g['followers'].items()}})
        gripper = self.profile.get('grippers',{}).get(side)
        attitude = np.array(gripper['grasp_rotation']) if gripper else None
        def checked_pose(target, initial):
            result = solve_profile_ik(self.profile,side,target,attitude,model=self.model,q_init=initial,snapshot=snapshot)
            if not result['success']:
                raise ValueError(f'{side}: object or approach is outside IK reach')
            q = np.array(result['q_rad'])
            if self.collision_queries.contacts({**snapshot,**dict(zip(names,q))},tolerance=-.0001):
                q = (self.collision_queries.refine_pose(self.profile,side,target,attitude,q,snapshot,minimum_clearance=.0001)
                     if attitude is not None else None)
                if q is None:
                    raise ValueError(f'{side}: spawn grasp intersects the robot or desk')
                kin = URDFKinematics(self.model,self.profile['chains'][side],snapshot,
                    self.profile['root_transform'],self.profile['chains'][side]['tool_offset'])
                actual,rotation,_,_ = kin.evaluate(q)
                from scipy.spatial.transform import Rotation
                result = {**result,'q_rad':q.tolist(),'position_error':float(np.linalg.norm(actual-target)),
                    'rotation_error':float(np.linalg.norm(Rotation.from_matrix(attitude@rotation.T).as_rotvec()))}
            return result,q
        # Plan at the supported height, rather than the 2mm setup drop height.
        supported = point-[0.,0.,self.clearance]
        for dz in (0., .04, .08):
            result,seed = checked_pose(supported+[0.,0.,dz],seed)
            checks.append({'height_above_object': dz, **result})
        closure,lifted,holding_lifts = [],None,[]
        if gripper:
            rows = sorted(gripper['sweep'],key=lambda row:row['q'])
            qs = [row['q'] for row in rows]
            centers = [(row['face_a']+row['face_b'])/2. for row in rows]
            seed = np.array(checks[0]['q_rad'])
            opened = np.interp(gripper['open_q'],qs,centers)
            for jaw in np.linspace(gripper['open_q'],gripper['closed_q'],5):
                snapshot.update({gripper['joint']:float(jaw),**{n:float(jaw*k) for n,k in gripper['followers'].items()}})
                target = supported-attitude@np.array(gripper['pad_region']['normal'])*(np.interp(jaw,qs,centers)-opened)
                result,seed = checked_pose(target,seed)
                closure.append({'jaw_q':float(jaw),**result})
            lifted,seed = checked_pose(target+[0.,0.,.03],seed)
            for factor in (.95,1.,1.05):
                row = min(rows,key=lambda row:abs((row['face_b']-row['face_a'])-gripper['cube_size']*factor))
                jaw = row['q']
                snapshot.update({gripper['joint']:jaw,**{n:float(jaw*k) for n,k in gripper['followers'].items()}})
                target = supported-attitude@np.array(gripper['pad_region']['normal'])*(np.interp(jaw,qs,centers)-opened)
                result,seed = checked_pose(target,np.array(checks[0]['q_rad']))
                for dz in (.01,.02,.03):
                    result,seed = checked_pose(target+[0.,0.,dz],seed)
                    holding_lifts.append({'jaw_q':jaw,'rise_m':dz,**result})
        entry = {'arm': side, 'position': point.tolist(), 'checks': checks,'closure_checks':closure,'lift_check':lifted,
                'holding_lift_checks':holding_lifts,
                'scope': 'Supported-height grasp/approach IK, robot/desk clearance and sampled jaw closure; execution separately checks the full path and physical contact',
                'cube_size': self.half_size * 2.}
        self._validated[key] = entry
        return entry

    def layout(self, *, randomize=False, object_positions=None, base=None, rng=None):
        rng = rng if rng is not None else np.random.default_rng()
        objects, omitted = {}, {}
        assignments = [('red_cube', 'left'), ('blue_cube', 'right')]
        for name, preferred_side in assignments:
            side = preferred_side if preferred_side in self.preferred else next(iter(self.preferred))
            explicit = (object_positions or {}).get(name)
            if explicit is not None and (np.asarray(explicit).shape != (3,) or not np.isfinite(explicit).all()):
                raise ValueError(f'{name}: spawn needs three finite coordinates')
            home_xy = np.asarray((base or {}).get(name, {}).get('position', [*self.preferred[side], 0.]))[:2]
            step = max(.008,min(.025,self.half_size))
            fine = [(x,y) for r in (step,2*step) for x,y in
                    [(-r,0.),(r,0.),(0.,-r),(0.,r),(-r,-r),(-r,r),(r,-r),(r,r)]]
            offsets = [(0.,0.),*fine,(.10, 0.), (-.10, 0.), (0., .10), (0., -.10),
                       (.07, .07), (.07, -.07), (-.07, .07), (-.07, -.07)]
            if randomize:
                offsets = [*rng.uniform(-.035, .035, (24, 2)), (0., 0.), *offsets[1:]]
            candidates = [np.asarray(explicit)] if explicit is not None else [
                np.r_[home_xy + offset, self.scene['table_top'] + self.half_size + self.clearance] for offset in offsets]
            for position in candidates:
                if any(np.linalg.norm(position[:2] - np.asarray(other['position'])[:2]) < .10 for other in objects.values()):
                    continue
                try:
                    objects[name] = self.validate(position, side)
                    break
                except ValueError:
                    continue
            if name not in objects:
                if explicit is not None or base and name in base:
                    raise ValueError(f'No reachable, separated desk spawn for {name}')
                omitted[name] = 'No desk position reachable at object and approach heights; no object spawned'
        return {'objects': objects, 'omitted': omitted, 'openarm_cell': False,
                'method': 'private URDF IK and collision queries before scene setup', 'grasp_calibrated': False}
