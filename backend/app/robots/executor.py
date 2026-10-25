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
                 snapshot=None, minimum_clearance=.0001,path_start=None,require_direct=False,relaxed=None):
        from backend.app.robots.urdf import solve_profile_ik
        snapshot = self.env.joint_snapshot() if snapshot is None else snapshot
        names = self.env.profile['chains'][side]['joints']
        start = self.env.arm_qpos(side) if q_init is None else np.asarray(q_init)
        edge_start = start if path_start is None else np.asarray(path_start)
        limits = (self.env.ik_joint_limits(side) if hasattr(self.env,'ik_joint_limits') else self.env.arm_limits(side))
        rng = np.random.default_rng(17)
        home = self.env.home_arm_qpos(side)
        continuation = q_init is not None or side in self.env._gripper_force_holds
        seeds = [start]
        if continuation and len(start)>6:
            # A redundant wrist can retain the tool pose while the elbow moves
            # away from a contact limit. A single nearest IK solution repeatedly
            # walks into that limit even when another local branch is clear.
            _,_,linear,angular = self.env.arm_kinematics(side).evaluate(start)
            jacobian = np.vstack([linear,angular*.15])
            _,singular,vectors = np.linalg.svd(jacobian,full_matrices=True)
            rank = int(np.sum(singular>1e-7))
            for null in vectors[rank:]:
                seeds.extend(np.clip(start+null*shift,*limits.T) for shift in (-.2,.2,-.5,.5,-1.,1.))
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
                model=self.env.model, q_init=seed, snapshot=snapshot,joint_limits=limits)
            # A transit waypoint only has to be collision-free and close: the caller asks for a looser
            # acceptance where the arm is allowed to reorient, and keeps the solver's own thresholds
            # for the grasp pose itself. Measured 2026-10-24 on UR5e + 2F-85: one insertion waypoint
            # 27.5 mm behind the object is reachable in position (error 0.00000) but misses the inferred
            # frame by about 1.7 degrees, which the strict acceptance refuses.
            solved = result['success'] or (relaxed is not None
                                           and result['position_error'] <= relaxed
                                           and result['rotation_error'] <= relaxed)
            error = min(error, result['position_error']+result['rotation_error']*.2)
            q = self.env.model.nearest_continuous(names,result['q_rad'],start)
            if continuation and np.max(abs(q-start)/self.env.arm_velocity_limits(side))>.75:
                continue
            tolerance = -minimum_clearance if minimum_clearance else .0002
            collisions = self.env.collision_queries.contacts({**snapshot, **dict(zip(names, q))},tolerance=tolerance) if solved else []
            if (solved and collisions
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
            if solved and not collisions:
                if self.env.collision_queries.edge_clear(names, edge_start, q, snapshot,minimum_clearance=minimum_clearance):
                    self._ik_seeds[side] = [q.copy(), *self._ik_seeds.get(side, [])][:6]
                    return q, result['position_error'], result['rotation_error']
                if not require_direct and (valid is None or np.linalg.norm(q-edge_start)<np.linalg.norm(valid-edge_start)):
                    valid = q
        return valid, error, error

    def plan_closure(self, side, object_position, rotation, opened_seed, *, snapshot=None, check_lift=True, minimum_gap=None):
        """Check arm motion throughout jaw closure, including changed jaw geometry."""
        g = self.env.grippers[side]
        rows = sorted(g['sweep'],key=lambda row:row['q'])
        qs = [row['q'] for row in rows]
        centers = [(row['face_a']+row['face_b'])/2. for row in rows]
        initial_center = np.interp(g['open_q'],qs,centers)
        closed_q=g['closed_q']
        if minimum_gap is not None:
            gap_rows=sorted(rows,key=lambda row:row['gap'])
            closed_q=float(np.interp(minimum_gap,[r['gap'] for r in gap_rows],[r['q'] for r in gap_rows]))
        normal = np.array(g['pad_region']['normal'])
        # Keeping the pads on the object's face while the jaws close only means anything to within a
        # fraction of that face, so the tolerances scale with the object: the nominal 52 mm cube keeps
        # the original 0.8 mm and 0.01 rad exactly. Measured 2026-10-24 on UR5e + 2F-85 (78 mm cube):
        # the first intermediate posture missed by 1.036 mm and 0.0101 rad - just outside limits
        # calibrated on a 25 mm object - and every closure plan failed, so the approach reported
        # COLLISION_PATH. More closure samples did not change it, because the gap comes from the IK's
        # own acceptance (2 mm, 0.03 rad), not from the chord between samples.
        slack = max(1., float(g.get('cube_size') or .052) / .052)
        snapshot = dict(self.env.joint_snapshot() if snapshot is None else snapshot)
        samples = []
        snapshot[g['joint']] = closed_q
        snapshot.update({name:float(closed_q*k) for name,k in g['followers'].items()})
        target = object_position-rotation@normal*(np.interp(closed_q,qs,centers)-initial_center)
        seed,_,_ = self.solve_ik(side,target,rotation,q_init=opened_seed,snapshot=snapshot,minimum_clearance=.0001)
        if seed is None:return None
        closed_seed = seed.copy()
        for dz in ((.01,.02,.03) if check_lift else ()):
            seed,_,_ = self.solve_ik(side,target+[0.,0.,dz],rotation,q_init=seed,
                snapshot=snapshot,minimum_clearance=.0001)
            if seed is None:return None
        seed = closed_seed
        for jaw_q in np.linspace(closed_q,g['open_q'],49):
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
                    if np.linalg.norm(point-expected)>.0008*slack or np.linalg.norm(orientation_error(attitude,rotation))>.01*slack:
                        return None
            samples.append({'jaw_q':float(jaw_q),'arm_q':seed.copy()})
        if check_lift and g.get('cube_size'):
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
        return self.plan_poses(side,positions,[rotation]*len(positions))

    def plan_poses(self,side,positions,rotations,*,snapshot=None,start=None):
        if len(positions)!=len(rotations):raise ValueError('Each position needs a rotation')
        path = [self.env.arm_qpos(side) if start is None else np.asarray(start,float)]
        names = self.env.profile['chains'][side]['joints']
        snapshot = dict(self.env.joint_snapshot() if snapshot is None else snapshot)
        for point,rotation in zip(positions,rotations):
            q, _, _ = self.solve_ik(side,point,rotation,q_init=path[-1],snapshot=snapshot)
            if q is None or not self.env.collision_queries.edge_clear(names,path[-1],q,snapshot,minimum_clearance=.0001):
                return None
            path.append(q)
        from backend.app.control.motion import joint_spline
        curve = joint_spline(path,self.env.arm_velocity_limits(side))
        return path if self.env.joint_curve_is_clear(side,curve,path,snapshot=snapshot) else None


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

    def _nominal_attitude(self, arm):
        return np.asarray(self.env.grippers[arm]['grasp_rotation'])

    def _attitude(self, arm):
        """The jaw frame this arm is pinching with - the inferred grasp rotation by default.

        A fixed jaw frame is a six-constraint request, so an arm with fewer joints pinches
        holding whatever frame its own joints reach. The approach plans with that frame when the
        nominal one leaves no corridor, and every later step - closing the jaws, holding through
        the lift, placing - must use the same frame: measured 2026-10-24 on a 4-DOF arm, the
        nominal frame is 6.6mm short at the object while the achieved frame solves to 1e-11, so
        closing with the nominal frame fails with 'Cannot keep the aperture centered while
        closing' even though the approach succeeded.
        """
        planned = getattr(self, '_pinch_frames', {}).get(arm)
        return self._nominal_attitude(arm) if planned is None else np.asarray(planned)

    def _move(self, arm, target, rotation=None, **kwargs):
        result = self.controller.move_to_site(arm, np.asarray(target), self._attitude(arm) if rotation is None else rotation, **kwargs)
        if not result.success:
            raise SkillError(result.error or 'IK_FAILURE', f'Cannot reach {np.round(target, 4).tolist()}: {result}')
        return result


    def _approach_rotations(self, arm, target, snapshot):
        """Jaw frames to plan the pinch with, the nominal grasp rotation first.

        A fixed jaw frame is a six-constraint request. An arm with fewer joints pinches holding
        whatever frame its own joints reach - the frame the spawn sampler validated
        (``attitude_mode='arm-achieved'``) - so planning only with the nominal frame can make an
        executable spawn unreachable. Measured 2026-10-24 on a 4-DOF arm: the nominal frame
        stopped 6.6 mm short of the object while the achieved frame solved to 1e-11, and the
        approach then reported COLLISION_PATH. Arms whose nominal frame works are unaffected:
        it is tried first, and the achieved frame is only added when it is within the sampler's
        own tolerance of the nominal one.
        """
        nominal = self._nominal_attitude(arm)
        gripper = self.env.profile.get('grippers', {}).get(arm)
        sampler = getattr(self.env, '_spawn_sampler', None)
        if not gripper or sampler is None:
            return [nominal]
        from backend.app.robots.urdf import solve_profile_ik
        free = solve_profile_ik(self.env.profile, arm, target, None, model=self.env.model,
                                q_init=self.env.home_arm_qpos(arm), snapshot=snapshot)
        if not free['success']:
            return [nominal]
        frame = sampler.achieved_attitude(arm, free['q_rad'], snapshot)
        if frame is None or sampler.attitude_offset_deg(frame, nominal, gripper) > sampler.attitude_tolerance_deg:
            return [nominal]
        if np.allclose(frame, nominal):
            # A capable arm holds the inferred frame itself, so the free solve reproduces it.
            # Planning twice with the same frame would only cost time.
            return [nominal]
        return [nominal, np.asarray(frame)]

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
        insertion = [standoff + (before-standoff)*a for a in np.linspace(0.,1.,9)]
        # The insertion is a straight push along the approach axis, so its intermediate poses are transit:
        # the final grasp pose is what must hold the jaw frame exactly, and the closure plan is checked on
        # its own. Measured 2026-10-24 on UR5e + 2F-85: the strict acceptance refuses one of these
        # waypoints over about 1.7 degrees of orientation while its position solves exactly.
        transit = .06
        from backend.app.robots.urdf import solve_profile_ik
        home = self.env.home_arm_qpos(arm)
        limits = self.env.arm_limits(arm)
        seed_key = (arm,*np.round(before/.002).astype(int))
        guesses = [*self.controller._approach_seeds.get(seed_key,[]), self.env.arm_qpos(arm), home]
        guesses.extend(np.clip(home+axis*shift,*limits.T) for axis in np.eye(len(home)) for shift in [-.5,.5,-1.,1.])
        reverse_seeds = None
        snapshot = self.env.joint_snapshot()
        names = self.env.profile['chains'][arm]['joints']
        rotations = self._approach_rotations(arm, before, snapshot)
        for rotation in rotations:
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
                    seed, _, _ = self.controller.solve_ik(arm, point, rotation, q_init=seed, relaxed=transit)
                    if seed is None:
                        # A single local solve can miss one waypoint of a Cartesian branch it has not
                        # explored, which drops an otherwise executable corridor: measured 2026-10-24
                        # on UR5e + 2F-85, every attempt died on exactly one insertion point. Passing
                        # no seed makes the controller try its whole seed set, the way the grasp itself
                        # already does for every guess. When the local solve works nothing changes.
                        seed, _, _ = self.controller.solve_ik(arm, point, rotation, relaxed=transit)
                    if seed is None:break
                    corridor.append(seed)
                if seed is not None:
                    # Reconfiguration is allowed in the high, empty-hand transit.
                    # Only the final insertion needs a single Cartesian branch.
                    high_seed = None
                    for height in (.08,.05,max(.025,self.env.cube_size)):
                        candidate_high = standoff+[0.,0.,height]
                        high_seed,_,_ = self.controller.solve_ik(arm,candidate_high,rotation,q_init=seed,
                            path_start=self.env.arm_qpos(arm),require_direct=True,relaxed=transit)
                        if high_seed is None:
                            high_seed,_,_ = self.controller.solve_ik(arm,candidate_high,rotation,
                                path_start=self.env.arm_qpos(arm),require_direct=True,relaxed=transit)
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
            if reverse_seeds is not None:break
        if reverse_seeds is None:
            raise SkillError('COLLISION_PATH','No grasp posture has a continuous collision-free approach corridor')
        # The frame that planned the corridor is the frame this arm pinches with from now on.
        self._pinch_frames = {**getattr(self, '_pinch_frames', {}), arm: np.asarray(rotation)}
        self._move(arm, high, rotation=rotation, q_init=reverse_seeds[-1], progress=progress)
        self._move(arm, standoff, rotation=rotation, q_init=reverse_seeds[-2], progress=progress)
        result = self._move(arm,before,rotation=rotation,progress=progress,planned_joints=[
            self.env.arm_qpos(arm),*reversed(reverse_seeds[:-2])])
        after = self.env.object_pose(obj)[0]
        disturbance = float(np.linalg.norm(after - before))
        if disturbance > self.env.cube_size * .12:
            raise SkillError('APPROACH_COLLISION', f'Approach displaced the cube by {disturbance:.6f}m')
        return {'target': before.tolist(), 'final': result.final_position,
                'object_displacement': disturbance, 'cube_size': self.env.cube_size}

    def _close_pinch(self, arm, obj, *, rotation=None, grasp_bias=None, monitor=None):
        rotation = self._attitude(arm) if rotation is None else np.asarray(rotation)
        bias = np.zeros(3) if grasp_bias is None else np.asarray(grasp_bias)
        g = self.env.grippers[arm]
        self.env.configure_grip_load(arm)
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
            if monitor:monitor()
            measured = self.env.gripper_position(arm)
            jaw_q = g['open_q']+measured*(g['closed_q']-g['open_q'])
            direction = np.sign(g['closed_q']-g['open_q'])
            offset = normal*(np.interp(jaw_q, commands, centers)-open_center)
            object_position,object_rotation = self.env.object_pose(obj)
            target = object_position+object_rotation@bias-rotation@offset
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
        if monitor:monitor()
        return contacts

    def _grasp(self, arm, obj):
        before = self.env.object_pose(obj)[0].copy()
        contacts = self._close_pinch(arm,obj)
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
        hold_steps=max(1,int(np.ceil(.02/self.env.timestep)))
        for _ in range(15):
            self.env.step(hold_steps)
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
                    'hold_duration_s':15*hold_steps*self.env.timestep,'hold_samples': samples, 'pinch': self.env.pinch_evidence(obj, arm),
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
        from scipy.spatial.transform import Rotation
        if np.linalg.norm(Rotation.from_matrix(self.env.ee_pose(arm)[1]@self._attitude(arm).T).as_rotvec())>.15:
            deadline=time.monotonic()+90.
            for point in self.env.zone_candidates(target):
                remaining=deadline-time.monotonic()
                if remaining<=0.:break
                try:
                    result=self._loaded_placement(arm,obj,point[:2],self.env.table_height,timeout=remaining)
                    self.env.resolve_zone(target,np.asarray(result['target']))
                    return result
                except SkillError as error:
                    if error.category!='COLLISION_PATH':raise
            raise SkillError('COLLISION_PATH','No loaded path reaches the requested table region')
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
            duration=max(.3,distance/.16)*self.controller.motion_speed,on_sample=lambda:self._require_hold(arm,obj))
        actual = self.env.object_pose(obj)[0]
        error = float(np.linalg.norm(actual-desired))
        if error>.008:raise SkillError('GRASP_SLIP',f'Placement tracking error: {error:.6f}m')
        return {'object':obj,'target':desired.tolist(),'position':actual.tolist(),'position_error':error,
                'planned_cartesian_samples':len(route),'trajectory_duration':result.trajectory_duration}

    def _loaded_placement(self,arm,obj,xy,surface,reference=None,timeout=90.):
        from backend.app.robots.placement import plan_placement
        self._require_hold(arm,obj)
        plan=plan_placement(self,arm,obj,xy,surface,reference=reference,timeout=timeout)
        start,rotation=self.env.ee_pose(arm)
        offset=rotation.T@(self.env.object_pose(obj)[0]-start)
        support_before=self.env.object_pose(reference)[0].copy() if reference else None
        original_observer=self.env.step_observer
        def observe():
            if original_observer:original_observer()
            if reference and set(self.env.object_contact_links(reference)).intersection(self.env.model.links):
                raise SkillError('SUPPORT_CONTACT','The robot contacted the reference cube during placement')
        def monitor():
            supported=(reference and reference in self.env.object_contact_links(obj)
                and np.linalg.norm(self.env.object_pose(obj)[0][:2]-plan['object_target'][:2])<.006)
            if not supported:
                self._require_hold(arm,obj)
                point,r=self.env.ee_pose(arm)
                error=np.linalg.norm(r.T@(self.env.object_pose(obj)[0]-point)-offset)
                if error>max(.003,self.env.cube_size*.12):
                    raise SkillError('GRASP_SLIP',f'The payload moved {error:.6f}m inside the grip')
        self.env.step_observer=observe
        try:
            result=self._move(arm,plan['target'],plan['rotation'],planned_joints=plan['path'],on_sample=monitor)
        finally:
            self.env.step_observer=original_observer
        actual=self.env.object_pose(obj)[0]
        error=float(np.linalg.norm(actual-plan['object_target']))
        if error>.008:raise SkillError('GRASP_SLIP',f'Loaded placement position error: {error:.6f}m')
        if reference and np.linalg.norm(self.env.object_pose(reference)[0]-support_before)>.002:
            raise SkillError('SUPPORT_DISTURBED','The reference cube moved by more than 2mm')
        self._placement_retreat={**getattr(self,'_placement_retreat',{}),arm:obj}
        return {'object':obj,'target':plan['object_target'].tolist(),'position':actual.tolist(),'position_error':error,
            'placement_candidates':plan['candidate_count'],'release_gap_m':plan['release_gap_m'],
            'trajectory_duration':result.trajectory_duration,'reference_object':reference}

    def move_to_object(self, arm, obj, target, relation, distance_cm=None):
        def run():
            from backend.app.skills.spatial_placement import relative_destination
            self._require_hold(arm,obj)
            desired=relative_destination(self.env,obj,target,relation,arm,distance_cm)
            surface=float(self.env.object_bounds(target)[1 if relation=='on' else 0,2])
            return {**self._loaded_placement(arm,obj,desired[:2],surface,reference=target),
                    'relation':relation,'reference_object':target}
        return self._skill('move_to_object',run,time.perf_counter())

    def _release(self, arm, obj):
        from backend.app.robots.gripper_envelope import release_command
        self.env.set_gripper_effort(arm, holding=False)
        before = self.env.object_pose(obj)[0].copy()
        # Hold the already-checked arm pose and withdraw the moving jaw. Keeping
        # an empty aperture centred would add an unnecessary wrist translation
        # and could drive a root joint into its mechanical contact limit.
        # Open only as far as the object needs: a full-aperture release sweeps the jaw into whatever sits
        # beside it, which is what made an adjacent placement need 71.5 mm of clearance.
        target = release_command(self.env, arm, obj)
        self.env.set_gripper(arm, target)
        # Settle without re-commanding: controller.wait_gripper() would send its own full-open target and
        # undo the partial opening that this release depends on. Wait for the TARGET, not for the motion
        # to look small: a jaw that has not started moving yet looks settled, and breaking on that left
        # the cube still pinched (measured 2026-10-25).
        quantum = max(1, int(round(.01 / self.env.timestep)))
        budget = max(1, int(round(.8 / self.env.timestep)))
        for _ in range(budget // quantum):
            self.env.step(quantum)
            if abs(self.env.gripper_position(arm) - target) < .02:
                break
        self.env.step(round(.15 / self.env.timestep))
        # Judge against the opening that was COMMANDED, not against a full-open zero: with a partial
        # release the jaw legitimately stops at ~0.45 of the stroke, and the old absolute test read that
        # as "did not open" (measured 2026-10-25, every partial release reported GRIPPER_BLOCKED).
        if self.env.gripper_position(arm) > target + .05 or self.env.pinch_established(obj,arm):
            raise SkillError('GRIPPER_BLOCKED','The jaws did not open and release the object')
        return {'position_before': before.tolist(), 'position_after': self.env.object_pose(obj)[0].tolist(),
                'contacts': self.env.object_contacts(obj, arm)}

    def retreat(self, arm):
        def run():
            if arm in getattr(self,'_placement_retreat',{}):return self._retreat_loaded_placement(arm)
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

    def _retreat_loaded_placement(self,arm):
        from backend.app.robots.placement import cleared_empty_path
        before={name:self.env.object_pose(name)[0].copy() for name in self.env._objects}
        g=self.env.grippers[arm];start,rotation=self.env.ee_pose(arm)
        normal=rotation@np.asarray(g['pad_region']['normal'])
        separated=None
        for distance in (.003,.006,.009):
            target=start-normal*distance
            path=self.controller.plan_cartesian(arm,[target],rotation)
            if path is None:continue
            self._move(arm,target,rotation,planned_joints=path)
            if not any(self.env.object_contacts(name,arm) for name in self.env._objects):
                separated=self.env.ee_pose(arm)[0].copy();break
        # A Cartesian translation can be unavailable at a mechanical limit.
        # The empty-hand planner may instead rotate away, allowing only the
        # existing pad contact without increasing its initial penetration.
        path=cleared_empty_path(self.env,arm,self.env.home_arm_qpos(arm),self.controller.stop_requested)
        if path is None:raise SkillError('COLLISION_PATH','No empty-hand retreat avoids the placed objects')
        point,attitude,*_=self.env.arm_kinematics(arm).evaluate(path[-1])
        observer=self.env.step_observer
        start_q=self.env.arm_qpos(arm)
        fingers={n for group in g['contact_groups'] for n in group}
        allowed={name:set(self.env.object_contact_links(name)).intersection(fingers) for name in self.env._objects}
        cleared=False
        def observe():
            nonlocal cleared
            if observer:observer()
            touching={name:set(self.env.object_contact_links(name)).intersection(self.env.model.links) for name in self.env._objects}
            if not any(touching.values()):cleared=True
            for name,links in touching.items():
                if (links and (cleared or not links<=allowed[name] or np.max(abs(self.env.arm_qpos(arm)-start_q))>.06)
                        or np.linalg.norm(self.env.object_pose(name)[0]-before[name])>.002):
                    raise SkillError('RETREAT_CONTACT','The empty arm failed to separate without moving the placed objects')
        self.env.step_observer=observe
        try:self._move(arm,point,attitude,planned_joints=path)
        finally:self.env.step_observer=observer
        displaced={name:float(np.linalg.norm(self.env.object_pose(name)[0]-p)) for name,p in before.items()}
        if max(displaced.values(),default=0.)>.002:
            error=SkillError('RETREAT_CONTACT','Retreat displaced a placed object by more than 2mm')
            error.evidence={'object_displacement_m':displaced};raise error
        self._placement_retreat.pop(arm,None)
        return {'final':self.env.ee_pose(arm)[0].tolist(),'object_displacement_m':displaced,'empty_hand_path':True}

    def _handover(self, arm, obj, receiver):
        from backend.app.robots.handover import ExchangePlanner, withdrawal
        if receiver == arm or receiver not in self.env.grippers:
            raise SkillError('ROBOT_CAPABILITY','Handover requires two detected grippers')
        self._require_hold(arm,obj)
        if self.held_object(receiver):raise SkillError('HAND_OCCUPIED','The receiving hand is occupied')
        self.controller.wait_gripper(receiver,False,.6)
        planner=ExchangePlanner(self)
        planning_started=time.perf_counter()
        plan=planner.plan(arm,receiver,obj)
        initial_planning_s=time.perf_counter()-planning_started
        point,rotation,*_=self.env.arm_kinematics(arm).evaluate(plan['donor_path'][-1])
        self._move(arm,point,rotation,planned_joints=plan['donor_path'],on_sample=lambda:self._require_hold(arm,obj))
        # Reobserve after physical travel: the cube's small compliance/slip is
        # measured, never replaced by the planner's predicted object pose.
        center,cube_rotation=self.env.object_pose(obj)
        snapshot=self.env.joint_snapshot();donor_q=self.env.arm_qpos(arm);donor_rotation=self.env.ee_pose(arm)[1]
        planner=ExchangePlanner(self)
        receiver_hint=plan.get('observed_receiver_hint') or {
            'object_rotation':(plan['predicted_cube_rotation'].T@plan['receiver']['rotation']).tolist(),
            'bias':plan['receiver']['bias'].tolist(),'open_arm_q':plan['receiver']['closure'][0]['arm_q'].tolist()}
        planning_started=time.perf_counter()
        recipient=planner.receiver(arm,receiver,center,cube_rotation,snapshot,
            lambda p:planner.completion(arm,receiver,donor_q,donor_rotation,snapshot,p),
            preferred=receiver_hint)
        observed_planning_s=time.perf_counter()-planning_started
        if recipient is None:raise SkillError('HANDOVER_UNREACHABLE','The measured exchange pose has no complete receiving path')
        self._move(receiver,recipient['target'],recipient['rotation'],planned_joints=recipient['path'],
                   on_sample=lambda:self._require_hold(arm,obj))
        self._closure_plans={**getattr(self,'_closure_plans',{}),receiver:recipient['closure']}
        self._close_pinch(receiver,obj,rotation=recipient['rotation'],grasp_bias=recipient['bias'],
                          monitor=lambda:self._require_hold(arm,obj))
        self.controller.target_rotations[receiver]=recipient['rotation'].copy()
        self._require_hold(arm,obj);self._require_hold(receiver,obj)
        donor_contacts=self.env.object_contacts(obj,arm)
        receiver_contacts=self.env.object_contacts(obj,receiver)
        # Check opening and withdrawal with the recipient's ACTUAL loaded jaw.
        future=self.env.joint_snapshot();g=self.env.grippers[arm]
        for jaw in np.linspace(future[g['joint']],g['open_q'],33):
            future.update({g['joint']:jaw,**{n:jaw*k for n,k in g['followers'].items()}})
            if self.env.collision_queries.contacts(future,tolerance=-.0001):
                raise SkillError('HANDOVER_BLOCKED','The actual loaded fingers leave no collision-free donor opening')
        if withdrawal(self,arm,receiver,snapshot=future) is None:
            raise SkillError('HANDOVER_BLOCKED','The actual loaded fingers leave no collision-free donor withdrawal')
        self.controller.wait_gripper(arm,False,.6)
        self.env.configure_grip_load(receiver,float(np.linalg.norm(recipient['bias'])))
        self.env.hold_gripper_force(receiver)
        self.env.step(max(1,round(.08/self.env.timestep)))
        self._require_hold(receiver,obj)
        retreat=withdrawal(self,arm,receiver)
        if retreat is None:raise SkillError('HANDOVER_BLOCKED','No withdrawal path after measured donor opening')
        self._move(arm,retreat['target'],retreat['rotation'],planned_joints=retreat['path'],
                   on_sample=lambda:self._require_hold(receiver,obj))
        if self.env.object_contacts(obj,arm):raise SkillError('HANDOVER_FAILED','The donor still contacts the cube')
        before=self.env.object_pose(obj)[0].copy()
        self._move_relative(receiver,[0.,0.,.03])
        self._require_hold(receiver,obj)
        rise=float(self.env.object_pose(obj)[0][2]-before[2])
        if rise<.021:raise SkillError('HANDOVER_FAILED',f'The receiver lifted the released cube only {rise:.6f}m')
        ee,attitude=self.env.ee_pose(receiver);relative=attitude.T@(self.env.object_pose(obj)[0]-ee)
        samples=[]
        hold_steps=max(1,int(np.ceil(.02/self.env.timestep)))
        allowed={name for group in self.env.grippers[receiver]['contact_groups'] for name in group}
        for _ in range(15):
            self.env.step(hold_steps)
            point,rot=self.env.ee_pose(receiver)
            samples.append({'pinch':self.env.pinch_established(obj,receiver),
                'relative_drift_m':float(np.linalg.norm(rot.T@(self.env.object_pose(obj)[0]-point)-relative)),
                'other_support_contacts':sorted(set(self.env.object_contact_links(obj))-allowed)})
        if (sum(s['pinch'] for s in samples)<14 or max(s['relative_drift_m'] for s in samples)>max(.002,self.env.cube_size*.08)
                or any(s['other_support_contacts'] for s in samples)):
            error=SkillError('HANDOVER_FAILED','The receiver did not maintain an independent unsupported pinch for 300ms')
            error.evidence={'hold_samples':samples};raise error
        evidence={'grasp_mode':self.env.grasp_mode,'grasp_is_physical_contact':True,'object':obj,'receiver':receiver,
            'giver_contacts_before_release':donor_contacts,'receiver_pinch_contacts':receiver_contacts,
            'receiver_lift':rise,'hold_samples':samples,'hold_duration_s':15*hold_steps*self.env.timestep,'handover_center':center.tolist(),
            'receiver_grasp_bias_m':recipient['bias'].tolist(),'planning_attempts':plan['planning_attempts'],
            'initial_planning_s':initial_planning_s,'observed_planning_s':observed_planning_s,
            'observed_planning_attempts':dict(planner.attempts),
            'self_collision':self.env.self_collision_report()}
        self.env.record_grasp_verification(receiver,evidence)
        from backend.app.robots.skill_recipes import record_exchange_recipe
        def hint(value,object_rotation):
            return {'object_rotation':(object_rotation.T@value['rotation']).tolist(),
                'bias':value['bias'].tolist(),'open_arm_q':value['closure'][0]['arm_q'].tolist()}
        evidence['recipe_saved']=record_exchange_recipe(self.env.profile,arm,receiver,{
            'center':plan['center'].tolist(),'donor_rotation':plan['donor_rotation'].tolist(),
            'predicted_receiver':hint(plan['receiver'],plan['predicted_cube_rotation']),
            'receiver':hint(recipient,cube_rotation),'verification_lift_m':rise,'hold_duration_s':15*hold_steps*self.env.timestep})
        return evidence

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
