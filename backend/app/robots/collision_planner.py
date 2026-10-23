"""Private geometric collision queries. Genesis alone executes robot dynamics.

MuJoCo compiles the URDF collision solids into a read-only geometric oracle.
Candidate joint positions affect this private data only, never the live robot.
No pair of adjacent links is excluded. Small CAD assembly residuals are bounded
by a metric tolerance, rather than ignoring a whole connected link pair.
"""
import xml.etree.ElementTree as ET
import time
import numpy as np
from scipy.spatial.transform import Rotation
from backend.app.robots.urdf import URDFModel, mesh_path, origin, vector


class PrivateCollisionQueries:
    def __init__(self, profile, table_top=None):
        import mujoco
        self.mujoco = mujoco
        self.urdf = URDFModel(profile['urdf'])
        root = ET.Element('mujoco', model='private_collision_queries')
        ET.SubElement(root, 'compiler', angle='radian', fusestatic='false')
        option = ET.SubElement(root, 'option')
        ET.SubElement(option, 'flag', filterparent='disable')
        asset = ET.SubElement(root, 'asset')
        world = ET.SubElement(root, 'worldbody')
        # MuJoCo already owns the body name 'world'. A URDF whose root link carries
        # that name (OpenManipulator-X, measured 2026-10-24) failed to compile with
        # "repeated name 'world' in body", so robot bodies get one safe name and the
        # class maps back to URDF link names at every query boundary.
        self.body_names = {link: ('robot_world' if link == 'world' else link) for link in self.urdf.links}
        self.link_names = {body: link for link, body in self.body_names.items()}
        if table_top is not None:
            ET.SubElement(world, 'geom', name='desk', type='box', size='.8 .7 .025',
                          pos=f'0 0 {table_top-.025}')
        assets = {}
        def pose_args(transform):
            quat = Rotation.from_matrix(transform[:3, :3]).as_quat()[[3, 0, 1, 2]]
            return {'pos': ' '.join(map(str, transform[:3, 3])), 'quat': ' '.join(map(str, quat))}
        def link_body(link, parent, transform, joint=None):
            body = ET.SubElement(parent, 'body', name=self.body_names[link], **pose_args(transform))
            # Mass is irrelevant to collision queries; mj_step is never called.
            ET.SubElement(body, 'inertial', pos='0 0 0', mass='1', diaginertia='1 1 1')
            if joint is not None and joint['type'] != 'fixed':
                ET.SubElement(body, 'joint', name=joint['name'], limited='false',
                    type='slide' if joint['type'] == 'prismatic' else 'hinge',
                    axis=' '.join(map(str, joint['axis'])))
            for index, collision in enumerate(self.urdf.links[link].findall('collision')):
                geometry = collision.find('geometry')
                # Geometric queries also retain near contacts for IK clearance
                # optimisation. No dynamics are integrated in this model.
                args = {'name': f'{link}_shape_{index}', 'margin':'.0005',
                        **pose_args(origin(collision.find('origin')))}
                mesh = geometry.find('mesh')
                if mesh is not None:
                    file = str(mesh_path(mesh.get('filename'), self.urdf.path))
                    scale = tuple(vector(mesh.get('scale'), [1., 1., 1.]))
                    key = (file, scale)
                    if key not in assets:
                        name = f'mesh_{len(assets)}';assets[key] = name
                        ET.SubElement(asset, 'mesh', name=name, file=file, scale=' '.join(map(str, scale)))
                    args.update(type='mesh', mesh=assets[key])
                elif geometry.find('box') is not None:
                    args.update(type='box', size=' '.join(map(str, vector(geometry.find('box').get('size'), [0, 0, 0])/2)))
                elif geometry.find('sphere') is not None:
                    args.update(type='sphere', size=geometry.find('sphere').get('radius'))
                elif geometry.find('cylinder') is not None:
                    cylinder = geometry.find('cylinder')
                    args.update(type='cylinder', size=f"{cylinder.get('radius')} {float(cylinder.get('length'))/2}")
                else:
                    raise ValueError('Unsupported collision shape in private collision queries')
                ET.SubElement(body, 'geom', **args)
            for child in self.urdf.joints:
                if child['parent'] == link:
                    link_body(child['child'], body, np.asarray(child['origin']), child)
        link_body(self.urdf.root, world, np.asarray(profile['root_transform']))
        ET.SubElement(world,'geom',name='grasp_probe',type='box',size='.01 .01 .01',contype='0',conaffinity='0')
        # A robot whose source meshes overlap at adjacent joints declares those pairs and the
        # queries must not collide them either, or the planner and the engine disagree.
        excluded = (profile.get('collision_config') or {}).get('runtime_excluded_adjacent_pairs') or []
        if excluded:
            contact = ET.SubElement(root, 'contact')
            for first, second in excluded:
                ET.SubElement(contact, 'exclude', body1=self.body_names.get(first, first),
                              body2=self.body_names.get(second, second))
        self.model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
        self.data = mujoco.MjData(self.model)
        self.addresses = {j['name']: self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, j['name'])]
                          for j in self.urdf.joints if j['type'] != 'fixed'}

    def box_contact_points(self,values,center,rotation,size,links,distance=.001):
        """Geometric grasp preflight only; this never asserts a loaded pinch."""
        mj,model,data=self.mujoco,self.model,self.data
        probe=mj.mj_name2id(model,mj.mjtObj.mjOBJ_GEOM,'grasp_probe')
        model.geom_pos[probe]=center
        model.geom_quat[probe]=Rotation.from_matrix(rotation).as_quat()[[3,0,1,2]]
        model.geom_size[probe]=np.broadcast_to(size,(3,))/2.
        model.geom_rbound[probe]=np.linalg.norm(model.geom_size[probe])
        for name,address in self.addresses.items():
            driver,k,offset=self.urdf.joint_driver(name);data.qpos[address]=values.get(driver,0.)*k+offset
        mj.mj_kinematics(model,data)
        # MuJoCo caches world-attached geom poses in MjData. Updating the
        # model's static pose alone leaves a query box at its compile-time pose.
        data.geom_xpos[probe]=center
        data.geom_xmat[probe]=np.asarray(rotation).reshape(9)
        result=[]
        if not hasattr(self,'_probe_body_names'):
            self._probe_body_names=[mj.mj_id2name(model,mj.mjtObj.mjOBJ_BODY,int(b)) for b in model.geom_bodyid]
        # Match on body names: MuJoCo's own world body also carries the desk geom, and a
        # URDF link named 'world' (OpenManipulator-X) would otherwise match the desk.
        wanted={self.body_names.get(link,link) for link in links}
        candidates=np.array([i for i,name in enumerate(self._probe_body_names) if name in wanted],int)
        near=(np.linalg.norm(data.geom_xpos[candidates]-center,axis=1)
              <=model.geom_rbound[candidates]+model.geom_rbound[probe]+distance)
        for geom in candidates[near]:
            name=self.link_names.get(self._probe_body_names[geom], self._probe_body_names[geom])
            line=np.zeros(6)
            gap=mj.mj_geomDistance(model,data,geom,probe,distance,line)
            if gap<distance:result.append({'link':name,'distance_m':float(gap),'position':((line[:3]+line[3:])/2.).tolist()})
        return result

    def contacts(self, values, tolerance=.0002):
        mj, model, data = self.mujoco, self.model, self.data
        for name, address in self.addresses.items():
            driver, multiplier, offset = self.urdf.joint_driver(name)
            data.qpos[address] = values.get(driver, 0.) * multiplier + offset
        mj.mj_kinematics(model, data)
        mj.mj_collision(model, data)
        pairs = {}
        for contact in data.contact:
            if contact.dist >= -tolerance:
                continue
            # Body names, so a pair indexes the same list refine_pose built. A URDF root
            # named 'world' is 'robot_world' here; MuJoCo's own world body is 'world'.
            a, b = [mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g]))
                    for g in (contact.geom1, contact.geom2)]
            pair = tuple(sorted([a or 'desk', b or 'desk']))
            pairs[pair] = min(pairs.get(pair, np.inf), float(contact.dist))
        return [{'links': list(pair), 'penetration_m': max(0., -distance), 'distance_m':distance}
                for pair, distance in pairs.items()]

    def clear(self, values):
        return not self.contacts(values)

    def adjacent_joint_intervals(self,names,reference,minimum_clearance=.0001):
        """Necessary IK bounds from each adjacent link pair's own geometry.

        Relative placement of a joint's parent/child solids depends only on that
        joint. Stay in the connected interval containing the setup posture.
        Sampling supplies optimisation bounds, not a collision certificate:
        every candidate and its complete path must still pass normal queries.
        """
        limits=[]
        for name in names:
            joint=self.urdf.by_name[name]
            low=-np.inf if joint['lower'] is None else joint['lower']
            high=np.inf if joint['upper'] is None else joint['upper']
            pair={joint['parent'],joint['child']}
            if not all(self.urdf.links[link].find('collision') is not None for link in pair):
                limits.append([low,high]);continue
            value=float(reference.get(name,0.))
            def free(q):
                return not any(set(c['links'])==pair for c in self.contacts(
                    {**reference,name:q},tolerance=-minimum_clearance))
            if not free(value):
                limits.append([low,high]);continue
            ends=[]
            for direction,bound in ((-1.,low),(1.,high)):
                span=min(abs(bound-value),2.*np.pi) if np.isfinite(bound) else 2.*np.pi
                spacing=.01 if joint['type']=='prismatic' else .05
                previous=value;end=bound
                for distance in np.linspace(0.,span,max(2,int(np.ceil(span/spacing))+1))[1:]:
                    probe=value+direction*distance
                    if not free(probe):
                        blocked=probe
                        for _ in range(12):
                            middle=(previous+blocked)/2.
                            if free(middle):previous=middle
                            else:blocked=middle
                        end=previous;break
                    previous=probe
                ends.append(end)
            limits.append(ends)
        return np.asarray(limits,float)

    def refine_pose(self, profile, side, position, rotation, q, snapshot, minimum_clearance=0.,joint_limits=None):
        """Resolve small collisions while retaining the requested tool pose.

        Pose-only IK can repeatedly converge to a colliding redundant posture.
        Include all link-pair penetrations in a fixed-size optimisation residual;
        the returned candidate must independently pass the usual collision gate.
        """
        from scipy.optimize import least_squares
        from backend.app.robots.urdf import URDFKinematics
        chain = profile['chains'][side]
        names = chain['joints']
        kin = URDFKinematics(self.urdf, chain, snapshot, profile['root_transform'], chain['tool_offset'])
        specs = [self.urdf.by_name[name] for name in names]
        lower = [j['lower'] if j['lower'] is not None else -np.inf for j in specs]
        upper = [j['upper'] if j['upper'] is not None else np.inf for j in specs]
        if joint_limits is not None:
            lower=np.maximum(lower,np.asarray(joint_limits)[:,0])
            upper=np.minimum(upper,np.asarray(joint_limits)[:,1])
        bodies = ['world', *self.body_names.values()]
        pairs = {tuple(sorted((a,b))): i for i,(a,b) in enumerate(
            ( (a,b) for i,a in enumerate(bodies) for b in bodies[i+1:] ))}
        initial = np.asarray(q)
        def residual(candidate):
            point, attitude, _, _ = kin.evaluate(candidate)
            orientation = Rotation.from_matrix(np.asarray(rotation)@attitude.T).as_rotvec()*.15
            depths = np.zeros(len(pairs))
            for contact in self.contacts({**snapshot, **dict(zip(names, candidate))}, tolerance=-.0005):
                depths[pairs[tuple(contact['links'])]] = max(0., minimum_clearance+.00015-contact['distance_m'])*10.
            return np.r_[point-position, orientation, (candidate-initial)*.0001, depths]
        result = least_squares(residual, initial, bounds=(lower,upper), max_nfev=60,
                               diff_step=.001, xtol=1e-9, ftol=1e-9, gtol=1e-9)
        if minimum_clearance:
            # Contact avoidance must not trade away tool height or attitude
            # while closing next to a table. Keep independent pose coordinates
            # as equality constraints, including underactuated URDF chains.
            from scipy.optimize import minimize, Bounds
            from scipy.linalg import qr
            _,_,linear,angular = kin.evaluate(initial)
            jacobian = np.vstack([linear,angular*.15])
            _, triangular, pivots = qr(jacobian.T,pivoting=True,mode='economic')
            rank = int(np.sum(abs(np.diag(triangular))>1e-8))
            active = pivots[:rank]
            def pose(candidate):
                p,r,_,_ = kin.evaluate(candidate)
                return np.r_[p-position,Rotation.from_matrix(r@np.asarray(rotation).T).as_rotvec()*.15][active]
            def separation(candidate):
                distances = np.full(len(pairs),.0005)
                for contact in self.contacts({**snapshot,**dict(zip(names,candidate))},tolerance=-.0005):
                    distances[pairs[tuple(contact['links'])]] = contact['distance_m']
                return distances-minimum_clearance-.00005
            result = minimize(lambda candidate:float(np.sum((candidate-initial)**2)),result.x,
                method='SLSQP',bounds=Bounds(lower,upper),constraints=[
                    {'type':'eq','fun':pose},{'type':'ineq','fun':separation}],
                options={'maxiter':80,'ftol':1e-11,'eps':1e-5})
        point, attitude, _, _ = kin.evaluate(result.x)
        if (np.linalg.norm(point-position)<(.0003 if minimum_clearance else .002)
                and np.linalg.norm(Rotation.from_matrix(np.asarray(rotation)@attitude.T).as_rotvec())<(.004 if minimum_clearance else .03)
                and not self.contacts({**snapshot, **dict(zip(names,result.x))},
                                      tolerance=-minimum_clearance if minimum_clearance else .0002)):
            return result.x
        return None

    def edge_clear(self, names, start, end, snapshot, resolution=.06, minimum_clearance=0.):
        steps = max(2, int(np.ceil(np.max(abs(np.asarray(end)-start))/resolution)) + 1)
        return all((not self.contacts({**snapshot, **dict(zip(names,q))},tolerance=-minimum_clearance)
                    if minimum_clearance else self.clear({**snapshot, **dict(zip(names, q))}))
                   for q in np.linspace(start, end, steps))

    def path(self, names, start, goal, snapshot, limits, stop=lambda: False):
        """RRT-connect when a direct joint segment intersects robot or desk."""
        start, goal = np.asarray(start), np.asarray(goal)
        if self.edge_clear(names, start, goal, snapshot):
            return [start, goal]
        if not self.clear({**snapshot, **dict(zip(names, goal))}):
            return None
        rng = np.random.default_rng(36)
        lower = np.maximum(limits[:, 0], np.minimum(start, goal)-1.2)
        upper = np.minimum(limits[:, 1], np.maximum(start, goal)+1.2)
        trees = [([start], [-1]), ([goal], [-1])]
        def extend(tree, target):
            nodes, parents = tree
            index = int(np.argmin(np.linalg.norm(np.asarray(nodes)-target, axis=1)))
            delta = target-nodes[index]
            q = nodes[index]+delta*min(1., .22/max(np.linalg.norm(delta), 1e-12))
            if not self.edge_clear(names, nodes[index], q, snapshot):
                return None
            nodes.append(q);parents.append(index)
            return len(nodes)-1
        def trace(tree, index):
            nodes, parents = tree;path = []
            while index != -1:
                path.append(nodes[index]);index = parents[index]
            return path[::-1]
        deadline = time.monotonic()+3.
        for iteration in range(500):
            if stop() or time.monotonic()>deadline:
                break
            side = iteration % 2;first, second = trees[side], trees[1-side]
            target = second[0][-1] if rng.random()<.25 else rng.uniform(lower, upper)
            a = extend(first, target)
            if a is None:
                continue
            for _ in range(80):
                b = extend(second, first[0][a])
                if b is None:
                    break
                if np.linalg.norm(first[0][a]-second[0][b])<1e-8:
                    p, q = trace(first, a), trace(second, b)
                    path = p+q[-2::-1] if side == 0 else q+p[-2::-1]
                    # Shortcut only when the entire replacement edge is clear.
                    compact = [path[0]];index = 0
                    while index<len(path)-1:
                        end = len(path)-1
                        while end>index+1 and not self.edge_clear(names,path[index],path[end],snapshot):end-=1
                        compact.append(path[end]);index=end
                    return compact
        return None
