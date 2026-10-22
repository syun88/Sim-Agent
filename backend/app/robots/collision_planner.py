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
        if table_top is not None:
            ET.SubElement(world, 'geom', name='desk', type='box', size='.8 .7 .025',
                          pos=f'0 0 {table_top-.025}')
        assets = {}
        def pose_args(transform):
            quat = Rotation.from_matrix(transform[:3, :3]).as_quat()[[3, 0, 1, 2]]
            return {'pos': ' '.join(map(str, transform[:3, 3])), 'quat': ' '.join(map(str, quat))}
        def link_body(link, parent, transform, joint=None):
            body = ET.SubElement(parent, 'body', name=link, **pose_args(transform))
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
        self.model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
        self.data = mujoco.MjData(self.model)
        self.addresses = {j['name']: self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, j['name'])]
                          for j in self.urdf.joints if j['type'] != 'fixed'}

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
            a, b = [mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g]))
                    for g in (contact.geom1, contact.geom2)]
            pair = tuple(sorted([a or 'desk', b or 'desk']))
            pairs[pair] = min(pairs.get(pair, np.inf), float(contact.dist))
        return [{'links': list(pair), 'penetration_m': max(0., -distance), 'distance_m':distance}
                for pair, distance in pairs.items()]

    def clear(self, values):
        return not self.contacts(values)

    def refine_pose(self, profile, side, position, rotation, q, snapshot, minimum_clearance=0.):
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
        bodies = ['world', *self.urdf.links]
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
