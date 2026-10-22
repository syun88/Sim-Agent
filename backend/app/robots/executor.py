"""Motor-driven manipulation using gripper frames inferred from URDF geometry."""
import time
import numpy as np

from backend.app.execution.task_executor import TaskExecutor
from backend.app.skills.robot_skills import SkillRunner, SkillError
from backend.app.control.motion import OpenArmV2Controller


class ImportedRobotController(OpenArmV2Controller):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._ik_seeds = {}
        self._approach_seeds = {}

    def solve_ik(self, side, target_position, target_rotation=None, q_init=None, max_iterations=500,
                 snapshot=None, minimum_clearance=.0001,path_start=None,require_direct=False):
        from backend.app.robots.urdf import solve_profile_ik
        snapshot = self.env.joint_snapshot() if snapshot is None else snapshot
        names = self.env.profile['chains'][side]['joints']
        start = self.env.arm_qpos(side) if q_init is None else np.asarray(q_init)
        edge_start = start if path_start is None else np.asarray(path_start)
        limits = self.env.arm_limits(side)
        rng = np.random.default_rng(17)
        home = self.env.home_arm_qpos(side)
        continuation = q_init is not None or side in self.env._gripper_force_holds
        seeds = [start]
        if not continuation:
            seeds.extend([*self._ik_seeds.get(side, []), home])
            seeds.extend(np.clip(home+axis*shift, *limits.T) for axis in np.eye(len(start)) for shift in [-.5, .5, -1., 1.])
            seeds.extend(np.clip(start+rng.normal(0., scale, len(start)), *limits.T)
                         for scale in [.5]*4 + [1.3]*8)
        valid = None;error = float('inf')
        for seed in seeds:
            if self.stop_requested():
                return None, error, error
            result = solve_profile_ik(self.env.profile, side, target_position, target_rotation,
                model=self.env.model, q_init=seed, snapshot=snapshot)
            error = min(error, result['position_error']+result['rotation_error']*.2)
            q = self.env.model.nearest_continuous(names,result['q_rad'],start)
            if continuation and np.max(abs(q-start)/self.env.arm_velocity_limits(side))>.75:
                continue
            tolerance = -minimum_clearance if minimum_clearance else .0002
            collisions = self.env.collision_queries.contacts({**snapshot, **dict(zip(names, q))},tolerance=tolerance) if result['success'] else []
            if (result['success'] and collisions and seed is seeds[0]
                    and max(c['penetration_m'] for c in collisions)<.003 and target_rotation is not None):
                adjusted = self.env.collision_queries.refine_pose(self.env.profile, side,
                    np.asarray(target_position), target_rotation, q, snapshot,minimum_clearance=minimum_clearance)
                if adjusted is not None:
                    q = adjusted
                    collisions = []
                    point, attitude, _, _ = self.env.arm_kinematics(side).evaluate(q)
                    from backend.app.simulation.transform import orientation_error
                    result['position_error'] = float(np.linalg.norm(point-target_position))
                    result['rotation_error'] = float(np.linalg.norm(orientation_error(attitude,target_rotation)))
            if result['success'] and not collisions:
                if self.env.collision_queries.edge_clear(names, edge_start, q, snapshot,minimum_clearance=minimum_clearance):
                    self._ik_seeds[side] = [q.copy(), *self._ik_seeds.get(side, [])][:6]
                    return q, result['position_error'], result['rotation_error']
                if not require_direct and (valid is None or np.linalg.norm(q-edge_start)<np.linalg.norm(valid-edge_start)):
                    valid = q
        return valid, error, error

    def plan_closure(self, side, object_position, rotation, opened_seed):
        """Check arm motion throughout jaw closure, including changed jaw geometry."""
        g = self.env.grippers[side]
        rows = sorted(g['sweep'],key=lambda row:row['q'])
        qs = [row['q'] for row in rows]
        centers = [(row['face_a']+row['face_b'])/2. for row in rows]
        initial_center = np.interp(g['open_q'],qs,centers)
        normal = np.array(g['pad_region']['normal'])
        snapshot = self.env.joint_snapshot()
        samples = []
        snapshot[g['joint']] = g['closed_q']
        snapshot.update({name:float(g['closed_q']*k) for name,k in g['followers'].items()})
        target = object_position-rotation@normal*(np.interp(g['closed_q'],qs,centers)-initial_center)
        seed,_,_ = self.solve_ik(side,target,rotation,q_init=opened_seed,snapshot=snapshot,minimum_clearance=.0001)
        if seed is None:return None
        closed_seed = seed.copy()
        for dz in (.01,.02,.03):
            seed,_,_ = self.solve_ik(side,target+[0.,0.,dz],rotation,q_init=seed,
                snapshot=snapshot,minimum_clearance=.0001)
            if seed is None:return None
        seed = closed_seed
        for jaw_q in np.linspace(g['closed_q'],g['open_q'],49):
            snapshot[g['joint']] = float(jaw_q)
            snapshot.update({name:float(jaw_q*k) for name,k in g['followers'].items()})
            target = object_position-rotation@normal*(np.interp(jaw_q,qs,centers)-initial_center)
            seed,_,_ = self.solve_ik(side,target,rotation,q_init=seed,snapshot=snapshot,minimum_clearance=.0001)
            if seed is None:return None
            if samples:
                previous = samples[-1]
                for alpha in (.25,.5,.75):
                    q = previous['arm_q']+(seed-previous['arm_q'])*alpha
                    j = previous['jaw_q']+(jaw_q-previous['jaw_q'])*alpha
                    intermediate = {**snapshot,g['joint']:float(j),
                        **{name:float(j*k) for name,k in g['followers'].items()},
                        **dict(zip(self.env.profile['chains'][side]['joints'],q))}
                    if not self.env.collision_queries.clear(intermediate):return None
                    point,attitude,_,_ = self.env.arm_kinematics(side).evaluate(q)
                    expected = object_position-rotation@normal*(np.interp(j,qs,centers)-initial_center)
                    from backend.app.simulation.transform import orientation_error
                    if np.linalg.norm(point-expected)>.0008 or np.linalg.norm(orientation_error(attitude,rotation))>.01:
                        return None
            samples.append({'jaw_q':float(jaw_q),'arm_q':seed.copy()})
        if g.get('cube_size'):
            # The cube stops closure before the fully-closed command. Verify
            # liftability around that actual aperture, not only at closed_q.
            # A redundant branch can lift when fully closed but run into the
            # root casing with the wider, loaded aperture.
            names = self.env.profile['chains'][side]['joints']
            checked = set()
            for gap_factor in (.95,1.,1.05):
                row = min(samples,key=lambda row:abs(
                    np.interp(row['jaw_q'],qs,[r['face_b']-r['face_a'] for r in rows])-g['cube_size']*gap_factor))
                jaw = row['jaw_q']
                if jaw in checked:continue
                checked.add(jaw)
                snapshot.update({g['joint']:jaw,**{name:float(jaw*k) for name,k in g['followers'].items()}})
                target = object_position-rotation@normal*(np.interp(jaw,qs,centers)-initial_center)
                seed = row['arm_q']
                for dz in np.arange(.005,.0301,.005):
                    lifted,_,_ = self.solve_ik(side,target+[0.,0.,dz],rotation,q_init=seed,
                        snapshot=snapshot,minimum_clearance=.0001)
                    if lifted is None or not self.env.collision_queries.edge_clear(
                            names,seed,lifted,snapshot,minimum_clearance=.0001):return None
                    seed = lifted
        return samples[::-1]

    def plan_cartesian(self, side, positions, rotation):
        """One locally continuous branch, including every loaded descent."""
        path = [self.env.arm_qpos(side)]
        names = self.env.profile['chains'][side]['joints']
        snapshot = self.env.joint_snapshot()
        for point in positions:
            q, _, _ = self.solve_ik(side,point,rotation,q_init=path[-1])
            if q is None or not self.env.collision_queries.edge_clear(names,path[-1],q,snapshot,minimum_clearance=.0001):
                return None
            path.append(q)
        from backend.app.control.motion import joint_spline
        curve = joint_spline(path,self.env.arm_velocity_limits(side))
        return path if self.env.joint_curve_is_clear(side,curve,path) else None


class ImportedRobotSkills(SkillRunner):
    def _require_hold(self, arm, obj):
        evidence = self.env.pinch_evidence(obj, arm)
        if not evidence['established']:
            error = SkillError('GRASP_LOST', f'{arm} has no loaded opposed distal-pad pinch on {obj}')
            error.evidence = {'pinch': evidence, 'gripper_position': self.env.gripper_position(arm),
                              'object_position': self.env.object_pose(obj)[0].tolist(),
                              'raw_contacts': self.env.object_contacts(obj, arm),
                              'close_trace': getattr(self, '_close_trace', [])}
            raise error

    def _attitude(self, arm):
        return np.asarray(self.env.grippers[arm]['grasp_rotation'])

    def _move(self, arm, target, rotation=None, **kwargs):
        result = self.controller.move_to_site(arm, np.asarray(target), self._attitude(arm) if rotation is None else rotation, **kwargs)
        if not result.success:
            raise SkillError(result.error or 'IK_FAILURE', f'Cannot reach {np.round(target, 4).tolist()}: {result}')
        return result

    def _approach(self, arm, obj, progress=None, fault_offset=None, grasp_offset_x=0.):
        if arm not in self.env.grippers:
            raise SkillError('ROBOT_CAPABILITY', 'This arm has no detected opposed jaws')
        if self.held_object(arm):
            raise SkillError('HAND_OCCUPIED', 'Place the held object first')
        before = self.env.object_pose(obj)[0].copy()
        direction = np.asarray(self.env.grippers[arm]['approach_axis'])
        standoff = before - direction * .055
        high = standoff + [0., 0., .08]
        self.controller.wait_gripper(arm, False, .6)
        # Plan back from the insertion pose. Choosing a redundant elbow posture
        # for the high point in isolation can force a several-radian posture
        # change beside the cube, even for the last millimetres of approach.
        # All these poses and edges are checked on the private collision model.
        rotation = self._attitude(arm)
        insertion = [standoff + (before-standoff)*a for a in np.linspace(0.,1.,9)]
        from backend.app.robots.urdf import solve_profile_ik
        home = self.env.home_arm_qpos(arm)
        limits = self.env.arm_limits(arm)
        seed_key = (arm,*np.round(before/.002).astype(int))
        guesses = [*self.controller._approach_seeds.get(seed_key,[]), self.env.arm_qpos(arm), home]
        guesses.extend(np.clip(home+axis*shift,*limits.T) for axis in np.eye(len(home)) for shift in [-.5,.5,-1.,1.])
        reverse_seeds = None
        snapshot = self.env.joint_snapshot()
        names = self.env.profile['chains'][arm]['joints']
        for guess in guesses:
            if self.controller.stop_requested():raise SkillError('STOPPED','Approach planning stopped')
            goal = solve_profile_ik(self.env.profile,arm,before,rotation,model=self.env.model,q_init=guess,snapshot=snapshot)
            if not goal['success']:continue
            seed = self.env.model.nearest_continuous(names,goal['q_rad'],self.env.arm_qpos(arm))
            if not self.env.collision_queries.clear({**snapshot,**dict(zip(names,seed))}):continue
            closure = self.controller.plan_closure(arm,before,rotation,seed)
            if closure is None:continue
            seed = closure[0]['arm_q']
            corridor = [seed]
            for point in reversed(insertion[:-1]):
                seed, _, _ = self.controller.solve_ik(arm, point, rotation, q_init=seed)
                if seed is None:break
                corridor.append(seed)
            if seed is not None:
                # Reconfiguration is allowed in the high, empty-hand transit.
                # Only the final insertion needs a single Cartesian branch.
                high_seed = None
                for height in (.08,.05,max(.025,self.env.cube_size)):
                    candidate_high = standoff+[0.,0.,height]
                    high_seed,_,_ = self.controller.solve_ik(arm,candidate_high,rotation,q_init=seed,
                        path_start=self.env.arm_qpos(arm),require_direct=True)
                    if high_seed is None:
                        high_seed,_,_ = self.controller.solve_ik(arm,candidate_high,rotation,
                            path_start=self.env.arm_qpos(arm),require_direct=True)
                    if high_seed is not None and self.env.collision_queries.edge_clear(
                            names,high_seed,seed,snapshot,minimum_clearance=.0001):
                        high = candidate_high
                        break
                    high_seed = None
                if high_seed is None:continue
                corridor.append(high_seed)
                reverse_seeds = corridor
                self._closure_plans = {**getattr(self,'_closure_plans',{}),arm:closure}
                self.controller._approach_seeds[seed_key] = [corridor[0],*self.controller._approach_seeds.get(seed_key,[])][:4]
                break
        if reverse_seeds is None:
            raise SkillError('COLLISION_PATH','No grasp posture has a continuous collision-free approach corridor')
        self._move(arm, high, q_init=reverse_seeds[-1], progress=progress)
        self._move(arm, standoff, q_init=reverse_seeds[-2], progress=progress)
        result = self._move(arm,before,progress=progress,planned_joints=[
            self.env.arm_qpos(arm),*reversed(reverse_seeds[:-2])])
        after = self.env.object_pose(obj)[0]
        disturbance = float(np.linalg.norm(after - before))
        if disturbance > self.env.cube_size * .12:
            raise SkillError('APPROACH_COLLISION', f'Approach displaced the cube by {disturbance:.6f}m')
        return {'target': before.tolist(), 'final': result.final_position,
                'object_displacement': disturbance, 'cube_size': self.env.cube_size}

    def _grasp(self, arm, obj):
        before = self.env.object_pose(obj)[0].copy()
        rotation = self._attitude(arm)
        g = self.env.grippers[arm]
        rows = sorted(g['sweep'], key=lambda row: row['q'])
        commands = [row['q'] for row in rows]
        centers = [(row['face_a']+row['face_b'])/2. for row in rows]
        gaps = [row['gap'] for row in rows]
        open_center = np.interp(g['open_q'], commands, centers)
        normal = np.asarray(g['pad_region']['normal'])
        self.env.set_gripper_effort(arm, holding=False)
        quantum = max(1, round(.02/self.env.timestep))
        stable = 0
        pinch_goal = None
        arm_reference = self.env.arm_qpos(arm).copy()
        self._close_trace = []
        # Follow the changing aperture centre while closing, using observed
        # joints and motor commands. A fixed jaw must not be shoved through the
        # cube by a one-shot predicted closing offset.
        for _ in range(250):
            if self.controller.stop_requested():
                raise SkillError('STOPPED', 'Grasp stopped')
            measured = self.env.gripper_position(arm)
            jaw_q = g['open_q']+measured*(g['closed_q']-g['open_q'])
            direction = np.sign(g['closed_q']-g['open_q'])
            offset = normal*(np.interp(jaw_q, commands, centers)-open_center)
            target = self.env.object_pose(obj)[0]-rotation@offset
            from backend.app.simulation.transform import orientation_error
            actual_point, actual_rotation = self.env.ee_pose(arm)
            centered = (np.linalg.norm(target-actual_point) < min(.002, self.env.cube_size*.08)
                        and np.linalg.norm(orientation_error(actual_rotation, rotation)) < .025)
            if pinch_goal is None:
                gap_error = float(np.interp(jaw_q, commands, gaps)-self.env.cube_size)
                lever = g['finger_length'] if g['type']=='rotary_jaw' else 1.
                rate = min(self.env.model.by_name[g['joint']]['velocity']*.6, max(.05, gap_error/max(lever, .01)*5.))
                gains = self.env._grip_force_limits[arm]
                increment = max(.002, rate*(gains['kv']+gains['damping'])/gains['kp'])
                goal = jaw_q+direction*increment
                goal = np.clip(goal, min(g['open_q'],g['closed_q']), max(g['open_q'],g['closed_q']))
                self.env.command_gripper_joint(arm, goal if centered else jaw_q)
            # Maintain a complete pose target while the aperture changes. A
            # single small correction relative to measured q each tick erased
            # the arm's holding reference and let wrist attitude drift.
            closure = getattr(self,'_closure_plans',{}).get(arm,[])
            ordered = sorted(closure,key=lambda row:row['jaw_q'])
            seed = (np.array([np.interp(jaw_q,[r['jaw_q'] for r in ordered],
                    [r['arm_q'][i] for r in ordered]) for i in range(len(arm_reference))])
                    if ordered else arm_reference)
            solution,_,_ = self.controller.solve_ik(arm,target,rotation,q_init=seed,minimum_clearance=.0001)
            if solution is None:
                raise SkillError('IK_FAILURE', 'Cannot keep the aperture centered while closing')
            cap = self.env.arm_velocity_limits(arm)*quantum*self.env.timestep
            delta = solution-arm_reference
            commanded = arm_reference+delta/max(1.,float(np.max(abs(delta)/cap)))
            names = self.env.profile['chains'][arm]['joints']
            clashes = self.env.collision_queries.contacts({**self.env.joint_snapshot(), **dict(zip(names, commanded))})
            if clashes:
                error = SkillError('COLLISION_PATH', 'Aperture centering would collide with the robot or desk')
                error.evidence = {'collisions': clashes, 'close_trace': self._close_trace}
                raise error
            self.env.set_arm(arm, commanded)
            arm_reference = commanded
            self.env.step(quantum)
            evidence = self.env.pinch_evidence(obj, arm)
            dof = [self.env._gripper_dofs[arm][g['joint']]]
            table_name = self.env._table.links[0].name
            self._close_trace.append({'jaw_q': g['open_q']+self.env.gripper_position(arm)*(g['closed_q']-g['open_q']),
                'goal_q': self.env._gripper_goals[arm], 'command_q': self.env._gripper_commands[arm],
                'motor_effort': float(self.env._entity.get_dofs_control_force(dof).cpu().numpy()[0]),
                'table_contacts': [p for p in self.env.contact_names() if table_name in p.split('|')],
                'object_contacts': self.env.object_contacts(obj, arm),
                'aperture_center_error_m': float(np.linalg.norm(target-self.env.ee_pose(arm)[0]))})
            if any(c['on_distal_pad'] for c in evidence['candidate_contacts']):
                self.env.set_gripper_effort(arm, holding=True)
            if evidence['established']:
                if pinch_goal is None:
                    measured_q = g['open_q']+self.env.gripper_position(arm)*(g['closed_q']-g['open_q'])
                    pinch_goal = measured_q+np.sign(g['closed_q']-g['open_q'])*.003
                    self.env.hold_gripper_force(arm)
                stable += 1
                if stable >= 5:
                    break
            else:
                stable = 0
        contacts = self.env.object_contacts(obj, arm)
        self._require_hold(arm, obj)
        position, rotation = self.env.ee_pose(arm)
        self.controller.target_rotations[arm] = self._attitude(arm).copy()
        self._move_relative(arm,[0.,0.,.03])
        self._require_hold(arm, obj)
        after = self.env.object_pose(obj)[0]
        rise = float(after[2] - before[2])
        if rise < .021:
            raise SkillError('GRASP_LOST', f'Cube rose only {rise:.6f}m during 30mm verification lift')
        start_ee, start_rotation = self.env.ee_pose(arm)
        relative = start_rotation.T @ (after - start_ee)
        samples = []
        for _ in range(15):
            self.env.step(max(1, round(.02 / self.env.timestep)))
            ee, attitude = self.env.ee_pose(arm)
            drift = float(np.linalg.norm(attitude.T @ (self.env.object_pose(obj)[0] - ee) - relative))
            pinch = self.env.pinch_evidence(obj, arm)
            clearance = float(self.env.object_bounds(obj)[0, 2] - self.env.table_height)
            samples.append({'pinch': pinch['established'], 'relative_drift_m': drift,
                            'table_clearance_m': clearance, 'speed_m_s': self.env.object_linear_speed(obj)})
        if (sum(s['pinch'] for s in samples) < 14
                or max(s['relative_drift_m'] for s in samples) > max(.002, self.env.cube_size * .08)
                or min(s['table_clearance_m'] for s in samples) < .005):
            error = SkillError('GRASP_LOST', 'The cube did not maintain an opposed, unsupported pinch for 300ms')
            error.evidence = {'hold_samples': samples, 'pinch': self.env.pinch_evidence(obj, arm)}
            raise error
        evidence = {'grasp_mode': self.env.grasp_mode, 'grasp_is_physical_contact': True,
                    'pinch_contacts': contacts, 'contacts_after_lift': self.env.object_contacts(obj, arm),
                    'object_rose': rise, 'verification_lift': .03, 'cube_size': self.env.cube_size,
                    'position_before': before.tolist(), 'position_after': self.env.object_pose(obj)[0].tolist(),
                    'hold_duration_s': .3, 'hold_samples': samples, 'pinch': self.env.pinch_evidence(obj, arm),
                    'grip_force_limits': self.env._grip_force_limits[arm],
                    'self_collision': self.env.self_collision_report()}
        self.env.grippers[arm]['status'] = 'physical_grasp_verified'
        self.env.record_grasp_verification(arm, evidence)
        return evidence

    def _align_for_placement(self, arm, obj):
        ee, current = self.env.ee_pose(arm)
        center = self.env.object_pose(obj)[0]
        desired_rotation = self._attitude(arm)
        self._move(arm, center - desired_rotation @ (current.T @ (center - ee)), desired_rotation,
                   on_sample=lambda: self._require_hold(arm, obj))

    def lift_object(self, arm, obj, height=None):
        def run():
            self._require_hold(arm,obj)
            bottom = float(self.env.object_bounds(obj)[0,2])
            # The imported mechanism does not inherit OpenArm's extra 6cm
            # transport lift. Its grasp already includes a measured 3cm lift.
            # Only add the clearance still needed above the scene's cube size.
            minimum = self.env.table_height+self.env.cube_size+.006
            rise = max(0.,minimum-bottom) if height is None else float(height)
            result = self._move_relative(arm,[0.,0.,rise]) if rise>.001 else {}
            self._require_hold(arm,obj)
            actual = float(self.env.object_bounds(obj)[0,2])
            if height is None and actual<minimum-.002:
                raise SkillError('GRASP_SLIP','Transport lift did not reach its clearance')
            return {**result,'requested_rise_m':rise,'bottom_height_m':actual,
                    'minimum_bottom_height_m':minimum,'automatic_clearance':height is None}
        return self._skill('lift_object',run,time.perf_counter())

    def _move_relative(self, arm, delta):
        start,rotation = self.env.ee_pose(arm)
        rotation = self.controller.target_rotations.get(arm,rotation)
        obj = self.held_object(arm)
        before = self.env.object_pose(obj)[0].copy() if obj else None
        delta = np.asarray(delta,float)
        count = max(1,int(np.ceil(np.linalg.norm(delta)/.005)))
        points = [start+delta*t for t in np.linspace(0.,1.,count+1)[1:]]
        path = self.controller.plan_cartesian(arm,points,rotation)
        if path is None:raise SkillError('COLLISION_PATH','No continuous Cartesian path for the relative move')
        result = self._move(arm,start+delta,rotation,planned_joints=path,
            on_sample=(lambda:self._require_hold(arm,obj)) if obj else None)
        actual_delta = self.env.ee_pose(arm)[0]-start
        if obj:
            self._require_hold(arm,obj)
            error = float(np.linalg.norm((self.env.object_pose(obj)[0]-before)-actual_delta))
            if error>max(.002,self.env.cube_size*.08):
                raise SkillError('GRASP_SLIP',f'Object follow error during relative motion: {error:.6f}m')
        return {'start':start.tolist(),'final':result.final_position,'measured_delta':actual_delta.tolist()}

    def _move_to_zone(self, arm, target):
        obj = self.held_object(arm)
        if obj is None:
            raise SkillError('GRASP_LOST', 'No opposed jaw contacts')
        self._align_for_placement(arm, obj)
        ee, rotation = self.env.ee_pose(arm)
        offset = rotation.T@(self.env.object_pose(obj)[0]-ee)
        rotation = self.controller.target_rotations.get(arm,rotation)
        release_height = self.env.minimum_release_height(arm,obj,self.env.table_height)
        desired = None
        # A semantic desk region has many possible points. Select an admissible
        # one BEFORE travelling, including the final loaded descent. Exact
        # object-relative and metric destinations are never moved this way.
        for point in self.env.zone_candidates(target):
            point[2] = release_height
            for clearance in (.07,.04,.02):
                centers = [self.env.object_pose(obj)[0],point+[0.,0.,clearance],point]
                route = []
                for a,b in zip(centers,centers[1:]):
                    count = max(1,int(np.ceil(np.linalg.norm(b-a)/.012)))
                    route.extend(a+(b-a)*alpha for alpha in np.linspace(0.,1.,count+1)[1:])
                path = self.controller.plan_cartesian(arm,[p-rotation@offset for p in route],rotation)
                if path is not None:
                    desired = point
                    break
            if desired is not None:break
        if desired is None:
            raise SkillError('COLLISION_PATH', 'No reachable, collision-free placement point in the requested desk region')
        self.env.resolve_zone(target,desired)
        distance = sum(np.linalg.norm(b-a) for a,b in zip(centers,centers[1:]))
        result = self._move(arm,desired-rotation@offset,rotation,planned_joints=path,
            duration=max(.3,distance/.02),on_sample=lambda:self._require_hold(arm,obj))
        actual = self.env.object_pose(obj)[0]
        error = float(np.linalg.norm(actual-desired))
        if error>.008:raise SkillError('GRASP_SLIP',f'Placement tracking error: {error:.6f}m')
        return {'object':obj,'target':desired.tolist(),'position':actual.tolist(),'position_error':error,
                'planned_cartesian_samples':len(route),'trajectory_duration':result.trajectory_duration}

    def move_to_object(self, arm, obj, target, relation, distance_cm=None):
        def run():
            from backend.app.skills.spatial_placement import relative_destination
            self._require_hold(arm, obj)
            self._align_for_placement(arm, obj)
            desired = relative_destination(self.env, obj, target, relation, arm, distance_cm)
            surface = self.env.object_bounds(target)[1 if relation == "on" else 0, 2]
            desired[2] = max(desired[2], self.env.minimum_release_height(arm, obj, surface))
            travel = desired.copy()
            travel[2] = max(self.env.table_height + self.env.cube_size * 2. + .06, desired[2] + .04)
            self._carry_to(arm, obj, travel)
            return {**self._carry_to(arm, obj, desired), 'reference_object': target, 'relation': relation}
        return self._skill('move_to_object', run, time.perf_counter())

    def _release(self, arm, obj):
        self.env.set_gripper_effort(arm, holding=False)
        before = self.env.object_pose(obj)[0].copy()
        # Hold the already-checked arm pose and withdraw the moving jaw. Keeping
        # an empty aperture centred would add an unnecessary wrist translation
        # and could drive a root joint into its mechanical contact limit.
        self.controller.wait_gripper(arm,False,.6)
        self.env.step(round(.15 / self.env.timestep))
        if abs(self.env.gripper_position(arm))>.05 or self.env.pinch_established(obj,arm):
            raise SkillError('GRIPPER_BLOCKED','The jaws did not open and release the object')
        return {'position_before': before.tolist(), 'position_after': self.env.object_pose(obj)[0].tolist(),
                'contacts': self.env.object_contacts(obj, arm)}

    def retreat(self, arm):
        def run():
            g = self.env.grippers[arm]
            start,rotation = self.env.ee_pose(arm)
            before = {name:self.env.object_pose(name)[0].copy() for name in self.env._objects}
            # A fixed jaw can still touch a supported cube after opening. First
            # separate along its surface normal, then lift above the cube. A
            # joint-space shortcut can bow through the cube even when its final
            # tool pose is farther away.
            normal = rotation@np.asarray(g['pad_region']['normal'])
            links = {name for group in g['contact_groups'] for name in group}
            lowest = min(float(self.env._entity.get_link(name).get_AABB().cpu().numpy().reshape(2,3)[0,2])
                         for name in links)
            nearby = [self.env.object_bounds(name)[1,2] for name,p in before.items()
                      if np.linalg.norm(p[:2]-start[:2])<g['finger_length']+self.env.cube_size]
            rise = max(.015,max(nearby,default=lowest)-lowest+.006)
            path = None
            for separation in (.003,.0015):
                separated = start-normal*min(separation,self.env.cube_size*.12)
                end = separated+[0.,0.,rise]
                points = []
                for a,b in zip([start,separated],[separated,end]):
                    count = max(1,int(np.ceil(np.linalg.norm(b-a)/.005)))
                    points.extend(a+(b-a)*t for t in np.linspace(0.,1.,count+1)[1:])
                path = self.controller.plan_cartesian(arm,points,rotation)
                if path is not None:break
            if path is None:
                raise SkillError('COLLISION_PATH','No continuous separation and retreat path')
            result = self._move(arm,end,rotation,planned_joints=path)
            displaced = {name:float(np.linalg.norm(self.env.object_pose(name)[0]-p)) for name,p in before.items()}
            if max(displaced.values(),default=0.)>.002:
                error = SkillError('RETREAT_CONTACT','Retreat displaced a released object by more than 2mm')
                error.evidence = {'object_displacement_m':displaced}
                raise error
            return {'final':result.final_position,'object_displacement_m':displaced}
        return self._skill('retreat',run,time.perf_counter())

    def _handover(self, arm, obj, receiver):
        from scipy.spatial.transform import Rotation
        from backend.app.robots.urdf import solve_profile_ik
        if receiver == arm or receiver not in self.env.grippers:
            raise SkillError('ROBOT_CAPABILITY', 'Handover requires two detected grippers')
        self._require_hold(arm, obj)
        if self.held_object(receiver):
            raise SkillError('HAND_OCCUPIED', 'The receiving hand is occupied')
        # Search a common six-dimensional workspace using private FK only.
        # Both mechanisms use their inferred frames; no robot-specific joint names.
        flat = {s: Rotation.from_euler('y', -g['pitch_degrees'], degrees=True).as_matrix() @ g['grasp_rotation']
                for s, g in self.env.grippers.items()}
        xy = np.mean([v for v in self.env._spawn_sampler.preferred.values()], axis=0)
        rng = np.random.default_rng(9)
        candidate = None
        for height in (.34, .49, .24):
            if self.controller.stop_requested():
                raise SkillError('STOPPED', 'Handover workspace search stopped')
            center = np.r_[xy + [-.10, 0.], self.env.table_height + height]
            for yaw in (90., 120., 60., 0., 180.):
                for pitch in (0., 30., -30.):
                    if self.env.step_observer:
                        self.env.step_observer()
                    attitudes = {arm: Rotation.from_euler('zy', [yaw, pitch], degrees=True).as_matrix() @ flat[arm],
                                 receiver: Rotation.from_euler('zy', [yaw + 180., -pitch], degrees=True).as_matrix() @ flat[receiver]}
                    seeds = {}
                    for side in (arm, receiver):
                        for attempt in range(3):
                            solution = solve_profile_ik(self.env.profile, side, center, attitudes[side], model=self.env.model,
                                q_init=None if attempt == 0 else rng.uniform(-2., 2., len(self.env.arm_qpos(side))))
                            if solution['success']:
                                seeds[side] = np.asarray(solution['q_rad'])
                                break
                        if side not in seeds:
                            break
                    if len(seeds) == 2:
                        candidate = center, attitudes, seeds
                        break
                if candidate: break
            if candidate: break
        if candidate is None:
            raise SkillError('HANDOVER_UNREACHABLE', 'No shared opposed-jaw handover pose passed both full-pose IK checks')
        center, attitudes, seeds = candidate
        self.controller.wait_gripper(receiver, False, .6)
        # Lift clear of the desk before changing the donor's wrist attitude.
        position, rotation = self.env.ee_pose(arm)
        self._move(arm, position + [0., 0., .12], rotation)
        ee, rotation = self.env.ee_pose(arm)
        offset = rotation.T @ (self.env.object_pose(obj)[0] - ee)
        from scipy.spatial.transform import Slerp
        start = self.env.object_pose(obj)[0].copy()
        angle = np.linalg.norm(Rotation.from_matrix(attitudes[arm] @ rotation.T).as_rotvec())
        count = max(8, int(np.ceil(angle / .12)))
        arc = Slerp([0., 1.], Rotation.from_matrix(np.stack([rotation, attitudes[arm]])))
        fractions = np.linspace(0., 1., count + 1)[1:]
        rotations = [arc(t).as_matrix() for t in fractions]
        positions = [(1.-t)*start+t*center-r@offset for t,r in zip(fractions,rotations)]
        self._move(arm, positions[-1], rotations[-1], via=positions[:-1], via_rotations=rotations[:-1],
                   on_sample=lambda: self._require_hold(arm, obj))
        center = self.env.object_pose(obj)[0].copy()
        g = self.env.grippers[receiver]
        center = center + [0., 0., self.env.cube_size * .46]
        direction = attitudes[receiver] @ (self._attitude(receiver).T @ g['approach_axis'])
        self._move(receiver, center - direction * .09, attitudes[receiver], q_init=seeds[receiver])
        for distance in (.06, .03, .0):
            self._move(receiver, center - direction * distance, attitudes[receiver],
                       on_sample=lambda: self._require_hold(arm, obj))
        pinch_target = center - attitudes[receiver] @ np.asarray(g.get('pinch_offset', [0.,0.,0.]))
        self._move(receiver, pinch_target, attitudes[receiver], on_sample=lambda: self._require_hold(arm,obj))
        self.controller.wait_gripper(receiver, True, .8)
        self._require_hold(arm, obj)
        self._require_hold(receiver, obj)
        donor_contacts = self.env.object_contacts(obj, arm)
        receiver_contacts = self.env.object_contacts(obj, receiver)
        self.controller.wait_gripper(arm, False, .6)
        donor_direction = attitudes[arm] @ (self._attitude(arm).T @ self.env.grippers[arm]['approach_axis'])
        pos, rot = self.env.ee_pose(arm)
        donor_retreat = pos - donor_direction * .06 - [0., 0., .03]
        donor_rotation = rot.copy()
        self._require_hold(receiver, obj)
        before = self.env.object_pose(obj)[0].copy()
        pos, rot = self.env.ee_pose(receiver)
        self._move(receiver, pos + [0., 0., .03], rot)
        self._require_hold(receiver, obj)
        rise = float(self.env.object_pose(obj)[0][2] - before[2])
        if rise < .021 or self.env.object_contacts(obj, arm):
            raise SkillError('HANDOVER_FAILED', 'The receiving hand did not independently lift the released cube')
        self._move(arm, donor_retreat, donor_rotation)
        return {'object': obj, 'receiver': receiver, 'giver_contacts_before_release': donor_contacts,
                'receiver_pinch_contacts': receiver_contacts, 'receiver_lift': rise, 'handover_center': center.tolist()}

    def _joint_trajectory(self, arm, target, duration=1.):
        start = self.env.arm_qpos(arm)
        specs = {j['name']: j for j in self.env.profile['joints']}
        velocities = np.asarray([max(.01, specs[name]['velocity']) for name in self.env.profile['chains'][arm]['joints']])
        duration = max(duration, float(np.max(abs(target - start) / velocities)) * 1.875)
        steps = max(2, int(np.ceil(duration / self.env.timestep)))
        completed = False
        try:
            for index in range(0, steps, 2):
                if self.controller.stop_requested():
                    raise SkillError('STOPPED', 'Joint motion stopped')
                t = index / steps
                alpha = t ** 3 * (10. - 15. * t + 6. * t * t)
                rate = 30. * t * t * (1. - t) ** 2 / duration
                self.env.set_arm(arm, start + (target - start) * alpha, (target - start) * rate)
                self.env.step(min(2, steps - index))
            completed = True
        finally:
            self.env.set_arm(arm, target if completed else self.env.arm_qpos(arm))
        self.env.step(max(1, round(.2 / self.env.timestep)))
        actual = self.env.arm_qpos(arm)
        error = float(np.max(abs(actual - target)))
        if error > .025:
            raise SkillError('MOTION_BLOCKED', f'Actual joint error is {error:.4f} rad')
        return {'target_joints': target.tolist(), 'actual_joints': actual.tolist(), 'maximum_joint_error': error}

    def return_home(self, arm):
        return self._skill('return_home', lambda: self._joint_trajectory(arm, self.env.home_arm_qpos(arm)), time.perf_counter())

    def execute_plan(self, plan, progress=None):
        if plan and plan[0]['skill'] == 'move_joint':
            step = plan[0]
            def run():
                chain = self.env.profile['chains'][step['arm']]['joints']
                if step['joint'] not in chain:
                    g = self.env.grippers[step['arm']]
                    target = np.deg2rad(step['position_deg'])
                    duration = max(.8, abs(target - self.env._gripper_commands[step['arm']]) / self.env.model.by_name[g['joint']]['velocity'] + .3)
                    self.env.command_gripper_joint(step['arm'], target)
                    self.env.step(round(duration / self.env.timestep))
                    g = self.env.grippers[step['arm']]
                    actual = float(self.env._entity.get_dofs_position([self.env._gripper_dofs[step['arm']][g['joint']]]).cpu().numpy()[0])
                    target = np.deg2rad(step['position_deg'])
                    if abs(actual-target) > .025:
                        raise SkillError('MOTION_BLOCKED', f'Gripper joint reached {actual:.5f}, target {target:.5f}rad')
                    return {'target_joint': float(target), 'actual_joint': actual}
                target = self.env.arm_qpos(step['arm'])
                target[chain.index(step['joint'])] = np.deg2rad(step['position_deg'])
                return self._joint_trajectory(step['arm'], target)
            return [self._skill('move_joint', run, time.perf_counter())]
        return super().execute_plan(plan, progress)


class ImportedRobotExecutor(TaskExecutor):
    def __init__(self, env):
        super().__init__(env)
        self.controller = ImportedRobotController(env)
        self.skills = ImportedRobotSkills(env, self.controller)
