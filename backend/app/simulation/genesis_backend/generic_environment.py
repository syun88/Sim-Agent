"""A selected URDF robot and a desk. No OpenArm model or Cell is instantiated."""
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from backend.app.robots.registry import get_profile
from backend.app.robots.urdf import URDFModel, URDFKinematics
from backend.app.robots.spawn import ReachableDeskSpawns
from backend.app.simulation.base import SimulationBackend, Frame
from backend.app.simulation.genesis_backend.environment import _ensure_genesis_init, simulation_speed_from_environment


class GenericURDFEnvironment(SimulationBackend):
    name = "genesis"
    grasp_mode = "unconfigured-gripper"
    grasp_is_physical_contact = False
    scene_profile = "cubes"
    ik_position_tolerance = .0003
    ik_rotation_tolerance = .004
    motion_position_tolerance = .004
    motion_rotation_tolerance = .025
    cartesian_motion_limits = {'linear_velocity_m_s':.25,'linear_acceleration_m_s2':1.,
        'angular_velocity_rad_s':2.5,'angular_acceleration_rad_s2':12.,'minimum_duration_s':.2}

    def __init__(self, robot_id, show_viewer=False, prefer_gpu=False):
        self.profile = get_profile(robot_id)
        self.robot_id, self.robot_name = self.profile['id'], self.profile["name"]
        self.model = URDFModel(self.profile["urdf"])
        self.available_arms = list(self.profile["chains"])
        self.grippers = self.profile.get("grippers", {})
        self.grasp_is_physical_contact = bool(self.grippers)
        self.grasp_mode = "physical-geometry-inferred-pinch" if self.grippers else "no-opposed-jaws-detected"
        self.simulation_speed = simulation_speed_from_environment()
        self._gs = gs = _ensure_genesis_init(prefer_gpu)
        self.compute_backend = gs.backend.name
        scene_config = json.loads((Path(self.profile['directory']) / 'scene.json').read_text())
        self.table_height = scene_config["table_top"]
        self.cube_size = scene_config.get("cube_size", .052)
        self.cube_mass = scene_config.get("cube_mass", .05)
        self._spawn_sampler = ReachableDeskSpawns(self.profile, scene_config)
        self.spawn_validation = self._spawn_sampler.layout()
        self._dt = float(scene_config["physics_dt"])
        self._scene = gs.Scene(sim_options=gs.options.SimOptions(dt=self._dt, substeps=2),
            rigid_options=gs.options.RigidOptions(constraint_solver=gs.constraint_solver.Newton,
                friction_cone=gs.friction_cone.elliptic, enable_torsional_friction=True, enable_rolling_friction=True,
                enable_self_collision=True, enable_neutral_collision=True,
                enable_adjacent_collision=True, max_collision_pairs=max(512, sum(
                    len(link.findall('collision')) for link in self.model.links.values()) * 4)),
            viewer_options=gs.options.ViewerOptions(refresh_rate=60, realtime_factor=self.simulation_speed,
                camera_pos=(1.1, -1.3, 1.5), camera_lookat=(0., 0., 1.)), show_viewer=show_viewer)
        self._scene.add_entity(gs.morphs.Plane(), material=gs.materials.Rigid(friction=.6))
        self._table = self._scene.add_entity(gs.morphs.Box(size=tuple(scene_config["table_size"]),
            pos=(0., 0., scene_config["table_top"] - scene_config["table_size"][2] / 2.), fixed=True),
            material=gs.materials.Rigid(friction=.6), surface=gs.surfaces.Default(color=(.62, .58, .51)))
        self._root_transform = np.asarray(self.profile["root_transform"])
        xyzw = Rotation.from_matrix(self._root_transform[:3, :3]).as_quat()
        self._entity = self._scene.add_entity(gs.morphs.URDF(file=self.profile["urdf"], fixed=True,
            pos=tuple(self._root_transform[:3, 3]), quat=tuple(xyzw[[3, 0, 1, 2]]),
            merge_fixed_links=False, recompute_inertia=False, default_armature=None,
            convexify=False, decimate=False, watertighten=None),
            material=gs.materials.Rigid(friction=.6, gravity_compensation=1.))
        self._objects = {}
        self._object_start = self.spawn_validation['objects']
        for name, point in self._object_start.items():
            point = np.asarray(point['position'])
            color = "0.8 0.08 0.08 1" if name == "red_cube" else "0.08 0.2 0.85 1"
            self._objects[name] = self._scene.add_entity(gs.morphs.MJCF(default_armature=None, file=f'<mujoco><worldbody><body name="{name}" pos="{point[0]} {point[1]} {point[2]}"><freejoint/><geom type="box" size="{self.cube_size/2} {self.cube_size/2} {self.cube_size/2}" mass="{self.cube_mass}" rgba="{color}"/></body></worldbody></mujoco>'))
        self._camera = self._scene.add_camera(res=(720, 480), pos=(1.1, -1.3, 1.5), lookat=(0., 0., 1.), GUI=False)
        self._scene.build()
        self._all_names = [j["name"] for j in self.profile["joints"]]
        self._all_dofs = np.array([self._entity.get_joint(name).dofs_idx_local[0] for name in self._all_names])
        self._dofs = {side: np.array([self._entity.get_joint(name).dofs_idx_local[0] for name in chain["joints"]])
                      for side, chain in self.profile["chains"].items()}
        self._limits = {j["name"]: [-np.inf if j["lower"] is None else j["lower"],
                                   np.inf if j["upper"] is None else j["upper"]] for j in self.profile["joints"]}
        self._entity.set_dofs_position(np.array([self.profile["home"][name] for name in self._all_names]), self._all_dofs)
        inertia = np.maximum(np.diag(self._entity.get_mass_mat().cpu().numpy())[self._all_dofs], 1e-6)
        kp = np.clip(inertia * 400., 10., 500.)
        self._entity.set_dofs_kp(kp, self._all_dofs)
        kv = 2. * np.sqrt(kp * inertia)
        self._entity.set_dofs_kv(kv, self._all_dofs)
        effort = np.asarray([max(0., j["effort"]) for j in self.profile["joints"]])
        self._entity.set_dofs_force_range(-effort, effort, self._all_dofs)
        self._link_names = {link.idx: link.name for entity in [self._entity, self._table, *self._objects.values()] for link in entity.links}
        self._gripper_dofs = {side: {name: self._entity.get_joint(name).dofs_idx_local[0]
                              for name in [g['joint'], *g['followers']]} for side, g in self.grippers.items()}
        self._gripper_force_holds = set()
        self._grip_force_limits = {}
        for side, g in self.grippers.items():
            force = max(.25, self.cube_mass * 9.81 * 4. / (2. * .6))
            effort = force if g['type'] == 'sliding_jaws' else force * g['finger_length']
            cap = min(effort, self.model.by_name[g['joint']]['effort'])
            dofs = list(self._gripper_dofs[side].values())
            damping = float(self._entity.get_dofs_damping([dofs[0]]).cpu().numpy()[0])
            index = self._all_names.index(g['joint'])
            transit = min(self.model.by_name[g['joint']]['effort'], cap + (damping+kv[index])*self.model.by_name[g['joint']]['velocity'])
            self._grip_force_limits[side] = {'normal_force_target_N': force, 'motor_effort_limit': cap,
                'transit_effort_limit': float(transit), 'kp': float(kp[index]), 'kv': float(kv[index]), 'damping': damping}
            self.set_gripper_effort(side, holding=False)
        self._gripper_commands, self._gripper_goals = {}, {}
        self._sim_step = 0
        self._contact_evidence_cache, self._contact_visuals = {}, {}
        self.reset()
        self._private_collisions = None

    @property
    def collision_queries(self):
        if self._private_collisions is None:
            self._private_collisions = self._spawn_sampler.collision_queries
        return self._private_collisions

    def joint_snapshot(self):
        return dict(zip(self._all_names, self._entity.get_dofs_position(self._all_dofs).cpu().numpy().astype(float)))

    def ik_joint_limits(self,side):
        if not hasattr(self,'_geometric_joint_limits'):self._geometric_joint_limits={}
        if side not in self._geometric_joint_limits:
            names=self.profile['chains'][side]['joints']
            self._geometric_joint_limits[side]=self.collision_queries.adjacent_joint_intervals(names,self.profile['home'])
        return self._geometric_joint_limits[side]

    def plan_joint_waypoints(self, side, waypoints, stop=lambda: False):
        names = self.profile['chains'][side]['joints']
        snapshot = self.joint_snapshot()
        result = [waypoints[0]]
        for a, b in zip(waypoints, waypoints[1:]):
            segment = self.collision_queries.path(names, a, b, snapshot, self.arm_limits(side), stop)
            if segment is None:
                return None
            result.extend(segment[1:])
        return result

    def joint_waypoints_are_clear(self,side,waypoints):
        names = self.profile['chains'][side]['joints'];snapshot = self.joint_snapshot()
        return all(self.collision_queries.edge_clear(names,a,b,snapshot) for a,b in zip(waypoints,waypoints[1:]))

    def joint_curve_is_clear(self, side, curve, waypoints, *, snapshot=None):
        names = self.profile['chains'][side]['joints']
        snapshot = self.joint_snapshot() if snapshot is None else snapshot
        count = max(16, int(np.sum(np.linalg.norm(np.diff(waypoints, axis=0), axis=1))/.035))
        samples = np.unique(np.r_[np.linspace(0.,1.,count+1),curve.x])
        return all(self.collision_queries.clear({**snapshot, **dict(zip(names, curve(t)))}) for t in samples)

    @property
    def timestep(self):
        return self._dt

    def model_summary(self):
        return {'engine': self.name, 'robot_links': self._entity.n_links, 'degrees_of_freedom': self._entity.n_dofs,
                'collision_geoms': self._entity.n_geoms, 'openarm_cell': False, 'robot_id': self.robot_id}

    def reset(self, randomize=False, object_positions=None, seed=None):
        self._resolved_zones = {}
        placements = self._spawn_sampler.layout(randomize=randomize, object_positions=object_positions,
                                                 base=self._object_start,rng=np.random.default_rng(seed))
        self.spawn_validation = placements
        self._sim_step = 0
        self._contact_evidence_cache.clear()
        q = np.array([self.profile["home"][name] for name in self._all_names])
        self._entity.set_dofs_position(q, self._all_dofs)
        self._entity.set_dofs_velocity(np.zeros_like(q), self._all_dofs)
        self._entity.control_dofs_position(q, self._all_dofs)
        for side, g in self.grippers.items():
            self.set_gripper_effort(side, holding=False)
            for name, multiplier in {g['joint']: 1., **g['followers']}.items():
                dof = [self._gripper_dofs[side][name]]
                self._entity.set_dofs_position(np.array([g['open_q'] * multiplier]), dof)
                self._entity.control_dofs_position(np.array([g['open_q'] * multiplier]), dof)
            self._gripper_commands[side] = g["open_q"]
            self.set_gripper(side, 0.)
        for name, entry in placements['objects'].items():
            position = np.asarray(entry['position'])
            self._objects[name].set_qpos(np.r_[position, 1., 0., 0., 0.], zero_velocity=True)
        self.refresh()

    def step(self, steps=1):
        for _ in range(steps):
            for side, target in self._gripper_goals.items():
                g = self.grippers[side]
                multipliers = {g['joint']: 1., **g['followers']}
                if side in self._gripper_force_holds:
                    # A constant motor preload survives contact compliance and
                    # small object motion. A frozen position target can unload
                    # completely after first contact. This is motor torque,
                    # with the original effort cap, never an object constraint.
                    actual = self.gripper_position(side)
                    for name, multiplier in multipliers.items():
                        dof = [self._gripper_dofs[side][name]]
                        if actual < 1.:
                            force = min(self.model.by_name[name]['effort'],
                                        self._grip_force_limits[side]['motor_effort_limit'])
                            direction = np.sign((g['closed_q']-g['open_q'])*multiplier)
                            self._entity.control_dofs_force(np.array([direction*force]), dof)
                        else:
                            self._entity.control_dofs_position(np.array([g['closed_q']*multiplier]), dof)
                    continue
                cap = min(self.model.by_name[name]['velocity'] / abs(k) for name, k in multipliers.items())
                command = self._gripper_commands[side]
                updated = float(command + np.clip(target - command, -cap*self._dt, cap*self._dt))
                if updated != command:
                    for name, multiplier in multipliers.items():
                        self._entity.control_dofs_position(np.array([updated*multiplier]), [self._gripper_dofs[side][name]])
                    self._gripper_commands[side] = updated
            self._scene.step()
            self._sim_step += 1
            if self.step_observer:
                self.step_observer()

    def refresh_viewer(self):
        viewer = self._scene.viewer
        if viewer is not None and viewer.is_alive():
            viewer.update(auto_refresh=True, force=False)

    def close(self):
        self._scene.destroy()

    def refresh(self):
        self._scene.rigid_solver.update_forward_pos()

    kinematics = refresh

    def render(self, width=720, height=480):
        from PIL import Image
        image = np.asarray(self._camera.render(rgb=True)[0])
        return np.asarray(Image.fromarray(image).resize((width, height)))

    def object_pose(self, name):
        link = self._objects[name].get_link(name)
        q = link.get_quat().cpu().numpy()
        return link.get_pos().cpu().numpy().astype(float), Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()

    def object_bounds(self, name):
        return self._objects[name].get_link(name).get_AABB().cpu().numpy().astype(float).reshape(2, 3)

    def record_grasp_verification(self, side, evidence):
        import time
        path = Path(self.profile['directory']) / 'gripper_runtime.json'
        recorded = json.loads(path.read_text()) if path.is_file() else {}
        signature = self.profile.get('configuration_sha256')
        if recorded.get('configuration_sha256') != signature:
            recorded = {'configuration_sha256': signature, 'arms': {}}
        recorded['arms'][side] = {'status': 'physical_grasp_verified', 'observed_at': time.time(), **evidence}
        temporary = path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(recorded, indent=2) + '\n')
        temporary.replace(path)
        self.profile['grasp_calibrated'] = all(s in recorded['arms'] for s in self.grippers)

    def minimum_release_height(self, side, obj, surface):
        center = self.object_pose(obj)[0]
        links = {name for group in self.grippers[side]["contact_groups"] for name in group}
        links.update(self.model.by_name[name]["child"] for name in self.profile["chains"][side]["joints"][-3:])
        lowest = min(float(self._entity.get_link(name).get_AABB().cpu().numpy().reshape(2,3)[0,2]) for name in links if self._entity.get_link(name).n_geoms)
        cube_bottom = self.object_bounds(obj)[0,2]
        return surface + center[2] - min(lowest, cube_bottom) + .012

    def zone_position(self, name):
        key = {'blue_zone':'left_side','green_zone':'right_side'}.get(name,name)
        if key in self._resolved_zones:
            return self._resolved_zones[key].copy()
        return self._default_zone_position(name)

    def _default_zone_position(self, name):
        positions = {entry['arm']: np.asarray(entry['position']) for entry in self._object_start.values()}
        side = 'left' if name in ('left_side', 'blue_zone') else 'right'
        point = positions.get(side, next(iter(positions.values()))).copy()
        if name == 'center':
            point = np.mean(list(positions.values()), axis=0)
        point[0] += .06
        point[2] = self.table_height
        return point

    def zone_candidates(self, name):
        preferred = self._default_zone_position(name)
        half_table = np.array([.8,.7])
        candidates = []
        for dx,dy in [(0.,0.),(-.03,0.),(-.09,0.),(0.,.04),(0.,-.04),(-.03,.04),(-.03,-.04)]:
            point = preferred + [dx,dy,0.]
            if np.any(abs(point[:2])+self.cube_size/2.+.02>half_table):
                continue
            if name in ('left_side','blue_zone') and point[1]<=0.:continue
            if name in ('right_side','green_zone') and point[1]>=0.:continue
            candidates.append(point)
        return candidates

    def resolve_zone(self, name, point):
        key = {'blue_zone':'left_side','green_zone':'right_side'}.get(name,name)
        self._resolved_zones[key] = np.r_[np.asarray(point)[:2], self.table_height]

    def object_velocity(self, name):
        return self._objects[name].get_link(name).get_vel().cpu().numpy().astype(float)

    def object_linear_speed(self, name):
        return float(np.linalg.norm(self.object_velocity(name)))

    def object_contacts(self, object_name, side=None):
        groups = [g for arm, g in self.grippers.items() if side is None or arm == side]
        fingers = {name for g in groups for group in g['contact_groups'] for name in group}
        return [link + '|' + object_name for link in self.object_contact_links(object_name) if link in fingers]

    def pinch_established(self, name, side):
        return bool(side in self.grippers and self.pinch_evidence(name, side)['established'])

    def _contact_surface(self, link_name):
        """Authored finger surfaces, with their URDF origin and scale applied."""
        if link_name not in self._contact_visuals:
            import trimesh
            from backend.app.robots.urdf import mesh_path, origin, vector
            body = self.model.links[link_name]
            parts = []
            for item in body.findall('visual') or body.findall('collision'):
                geometry = item.find('geometry')
                mesh, box, cylinder, sphere = [geometry.find(k) for k in ('mesh', 'box', 'cylinder', 'sphere')]
                if mesh is not None:
                    part = trimesh.load(mesh_path(mesh.get('filename'), self.model.path), force='mesh')
                    part.vertices *= vector(mesh.get('scale'), [1., 1., 1.])
                elif box is not None:
                    part = trimesh.creation.box(extents=vector(box.get('size'), [0., 0., 0.]))
                elif cylinder is not None:
                    part = trimesh.creation.cylinder(radius=float(cylinder.get('radius')), height=float(cylinder.get('length')))
                elif sphere is not None:
                    part = trimesh.creation.icosphere(radius=float(sphere.get('radius')))
                else:
                    continue
                part.apply_transform(origin(item.find('origin')))
                parts.append(part)
            self._contact_visuals[link_name] = trimesh.util.concatenate(parts) if parts else None
        return self._contact_visuals[link_name]

    def pinch_evidence(self, name, side):
        from backend.app.robots.contacts import evaluate_pinch_contacts
        import trimesh
        key = (self._sim_step, name, side)
        if key in self._contact_evidence_cache:
            return self._contact_evidence_cache[key]
        g = self.grippers[side]
        pad = g['pad_region']
        reference = self._entity.get_link(g['reference_link'])
        ref_pos = reference.get_pos().cpu().numpy()
        ref_quat = reference.get_quat().cpu().numpy()
        ref_rotation = Rotation.from_quat(ref_quat[[1, 2, 3, 0]]).as_matrix()
        raw = {k: v.detach().cpu().numpy() for k, v in self._entity.get_contacts(with_entity=self._objects[name]).items()}
        contacts = []
        finger_links = {n for group in g['contact_groups'] for n in group}
        for i, (a, b) in enumerate(zip(raw['link_a'], raw['link_b'])):
            link_name = self._link_names.get(int(a))
            object_is_b = link_name in finger_links
            if not object_is_b:
                link_name = self._link_names.get(int(b))
            if link_name not in finger_links:
                continue
            point = raw['position'][i]
            normal = raw['normal'][i] * (1. if object_is_b else -1.)
            penetration = max(0., float(raw['penetration'][i]))
            local = (point - ref_pos) @ ref_rotation - pad['pivot']
            depth = float(local @ pad['forward'])
            height = float(local @ pad['height_axis'])
            distal = pad['depth_min'] <= depth <= pad['depth_max'] and abs(height - pad['height_center']) <= pad['height_half_extent']
            surface_error = None
            if distal:
                link = self._entity.get_link(link_name)
                quat = link.get_quat().cpu().numpy()
                rotation = Rotation.from_quat(quat[[1, 2, 3, 0]]).as_matrix()
                positions = point + np.array([[-.5], [0.], [.5]]) * normal * penetration
                points = (positions - link.get_pos().cpu().numpy()) @ rotation
                surface = self._contact_surface(link_name)
                if surface is not None:
                    surface_error = float(trimesh.proximity.closest_point(surface, points)[1].min())
            force = raw['force_b' if object_is_b else 'force_a'][i]
            contacts.append({'robot_link': link_name, 'position': point.tolist(), 'normal': normal.tolist(),
                'force_on_object_N': force.tolist(), 'normal_force_N': abs(float(force @ normal)),
                'penetration_m': penetration, 'on_distal_pad': bool(distal), 'surface_error_m': surface_error})
        result = evaluate_pinch_contacts(contacts, g, self.cube_size, self.cube_mass)
        # Keep only this step; a reset or another physics step must never reuse old contacts.
        self._contact_evidence_cache = {k: v for k, v in self._contact_evidence_cache.items() if k[0] == self._sim_step}
        self._contact_evidence_cache[key] = result
        return result

    def self_collision_report(self):
        data = self._entity.get_contacts(with_entity=self._entity)
        pairs = {}
        for a, b, depth in zip(data['link_a'], data['link_b'], data['penetration']):
            names = tuple(sorted([self._link_names[int(a)], self._link_names[int(b)]]))
            pairs[names] = max(pairs.get(names, 0.), float(depth))
        return {'adjacent_collision': True, 'maximum_penetration_m': max(pairs.values(), default=0.),
                'pairs': [{'links': list(pair), 'penetration_m': value} for pair, value in pairs.items()]}

    def contact_names(self):
        pairs = set()
        for entity in [self._entity, *self._objects.values()]:
            try:
                data = entity.get_contacts()
            except KeyError:
                continue
            for a, b in zip(data.get('link_a', []), data.get('link_b', [])):
                pairs.add('|'.join(sorted([self._link_names.get(int(a), str(a)), self._link_names.get(int(b), str(b))])))
        return sorted(pairs)

    def object_contact_links(self, name):
        return [pair.replace(name, '').strip('|') for pair in self.contact_names() if name in pair.split('|')]

    def arm_qpos(self, side):
        return self._entity.get_dofs_position(self._dofs[side]).cpu().numpy().astype(float)

    def arm_limits(self, side):
        return np.array([self._limits[name] for name in self.profile['chains'][side]['joints']])

    def home_arm_qpos(self, side):
        return np.array([self.profile['home'][name] for name in self.profile['chains'][side]['joints']])

    def arm_velocity_limits(self, side):
        specs = {j['name']: j for j in self.profile['joints']}
        return np.array([specs[name]['velocity'] for name in self.profile['chains'][side]['joints']])

    def set_arm(self, side, targets, velocities=None):
        values = np.clip(np.asarray(targets), *self.arm_limits(side).T)
        if velocities is None:
            self._entity.control_dofs_position(values, self._dofs[side])
        else:
            self._entity.control_dofs_position_velocity(values, velocities, self._dofs[side])

    def set_arm_qpos(self, side, targets):
        self._entity.set_dofs_position(np.asarray(targets), self._dofs[side])

    def arm_kinematics(self, side):
        actual = self._entity.get_dofs_position(self._all_dofs).cpu().numpy()
        return URDFKinematics(self.model, self.profile['chains'][side], dict(zip(self._all_names, actual)),
                              self._root_transform, self.profile['chains'][side]['tool_offset'])

    def ee_pose(self, side):
        link = self._entity.get_link(self.profile['chains'][side]['end_link'])
        q = link.get_quat().cpu().numpy()
        rotation = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
        return link.get_pos().cpu().numpy().astype(float) + rotation @ self.profile['chains'][side]['tool_offset'], rotation

    def ee_jacobian(self, side):
        return self.arm_kinematics(side).evaluate(self.arm_qpos(side))[2:]

    def set_gripper(self, side, value):
        self._gripper_force_holds.discard(side)
        g = self.grippers.get(side)
        if g is None:
            raise ValueError('No opposed actuated jaws detected on this arm')
        if value <= 0.:
            self.set_gripper_effort(side, holding=False)
        target = g['open_q'] + np.clip(value, 0., 1.) * (g['closed_q'] - g['open_q'])
        self._gripper_goals[side] = float(target)

    def set_gripper_effort(self, side, *, holding):
        if not holding:
            self._gripper_force_holds.discard(side)
        limits = self._grip_force_limits[side]
        g = self.grippers[side]
        multipliers = {g['joint']: 1., **g['followers']}
        lower, upper, dofs = [], [], []
        for name, multiplier in multipliers.items():
            source = self.model.by_name[name]['effort']
            transit = min(source, limits['transit_effort_limit'])
            preload = min(source, limits['motor_effort_limit'])
            direction = (g['closed_q']-g['open_q'])*multiplier
            # Limit squeezing, while retaining enough reverse torque to brake
            # the finger's inertia at first contact.
            lo = transit if not holding or direction>0 else preload
            hi = transit if not holding or direction<0 else preload
            lower.append(-lo);upper.append(hi);dofs.append(self._gripper_dofs[side][name])
        self._entity.set_dofs_force_range(np.asarray(lower), np.asarray(upper), dofs)

    def configure_grip_load(self,side,offset_m=0.,mass=None):
        """Budget normal preload for payload weight and an off-centre grip."""
        g=self.grippers[side];mass=self.cube_mass if mass is None else float(mass)
        radius=max(.0001,min(g['pad_height'],self.cube_size)*.25)
        force=max(.25,mass*9.81*(4./(2.*.6)+abs(offset_m)/(.6*radius)))
        lever=1. if g['type']=='sliding_jaws' else g['finger_length']
        source=self.model.by_name[g['joint']]
        cap=min(force*lever,source['effort'])
        limits=self._grip_force_limits[side]
        limits.update(normal_force_target_N=force,motor_effort_limit=cap,
            transit_effort_limit=min(source['effort'],cap+(limits['kv']+limits['damping'])*source['velocity']),
            load_offset_m=float(offset_m),payload_mass_kg=mass,contact_radius_m=radius)

    def command_gripper_joint(self, side, target):
        self._gripper_force_holds.discard(side)
        self._gripper_goals[side] = float(target)

    def hold_gripper_force(self, side):
        self.set_gripper_effort(side, holding=True)
        self._gripper_force_holds.add(side)

    def gripper_target(self, side, closed):
        return float(closed)

    def gripper_position(self, side):
        if side not in self.grippers:
            return 0.
        g = self.grippers[side]
        actual = float(self._entity.get_dofs_position([self._gripper_dofs[side][g['joint']]]).cpu().numpy()[0])
        return (actual - g['open_q']) / (g['closed_q'] - g['open_q'])

    def apply_gravity_compensation(self, side):
        pass  # Genesis's native per-link gravity compensation is enabled on the robot only.

    def frame(self):
        ee = {side: self.ee_pose(side)[0].tolist() if side in self._dofs else [0., 0., 0.]
              for side in ('left', 'right')}
        return Frame(self._sim_step * self._dt, self._entity.get_qpos().cpu().numpy().tolist(),
                     {name: self.object_pose(name)[0].tolist() for name in self._objects}, {}, ee,
                     {side: self.gripper_position(side) for side in ('left', 'right')}, self.contact_names())
