"""Read-only IK evaluation of the attached official OpenArm model.

The compiled model is shared with the metadata reader, but every solve owns a private
MjData snapshot. Only forward kinematics and Jacobians are evaluated here; Genesis owns
all dynamics, live poses, contacts and motor execution.
"""
from __future__ import annotations

import numpy as np

from backend.app.simulation.openarm_v2.definition import ARM_JOINTS, EE_SITES


class OfficialArmKinematics:
    def __init__(self, model, qpos: np.ndarray, side: str):
        import mujoco

        self._mj = mujoco
        self._model = model
        self._data = mujoco.MjData(model)
        self._data.qpos[:] = qpos
        joints = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                           for name in ARM_JOINTS[side]])
        self._positions = model.jnt_qposadr[joints]
        self._dofs = model.jnt_dofadr[joints]
        self._site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, EE_SITES[side])
        self._linear = np.zeros((3, model.nv))
        self._angular = np.zeros((3, model.nv))

    def evaluate(self, joints: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        self._data.qpos[self._positions] = joints
        self._mj.mj_kinematics(self._model, self._data)
        self._mj.mj_comPos(self._model, self._data)
        self._mj.mj_jacSite(self._model, self._data, self._linear, self._angular, self._site)
        return (self._data.site_xpos[self._site].copy(),
                self._data.site_xmat[self._site].reshape(3, 3).copy(),
                self._linear[:, self._dofs].copy(), self._angular[:, self._dofs].copy())

    def path_minimum_clearance(self, waypoints, objects, resolution=.025, *, allowed_contacts=None,
                               curve=None, ceiling=.05):
        """The smallest gap between this arm and the scene objects anywhere along a path.

        A boolean screen can only say pass or fail, so a search over candidate angles had nothing to rank
        them by. This returns the number: the minimum `mj_geomDistance` over sampled configurations, capped
        at `ceiling` - a path with nothing within the ceiling radius reports the ceiling, because "far from
        everything" and "0.05 m of room" are the same answer for a search that is ranking candidates.
        Negative means the arm is inside something. `None` means the scene held no object this arm could be
        measured against, which a caller must treat as unknown rather than clear.
        """
        from backend.app.simulation.transform import mat_to_quat
        waypoints = [np.asarray(point, float) for point in waypoints]
        mj, model = self._mj, self._model
        data = mj.MjData(model)
        data.qpos[:] = self._data.qpos
        placed = 0
        for name, (position, rotation) in objects.items():
            body = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, name)
            if body < 0 or not model.body_jntnum[body]:
                continue
            joint = model.body_jntadr[body]
            if model.jnt_type[joint] != mj.mjtJoint.mjJNT_FREE:
                continue
            address = model.jnt_qposadr[joint]
            data.qpos[address:address+3] = position
            data.qpos[address+3:address+7] = mat_to_quat(rotation)
            placed += 1
        if not placed:
            return None
        root = model.jnt_bodyid[model.dof_jntid[self._dofs[0]]]
        descendants = set()
        for body in range(model.nbody):
            ancestor = body
            while ancestor and ancestor != root:
                ancestor = model.body_parentid[ancestor]
            if ancestor == root:
                descendants.add(body)
        moving = np.array([int(b) in descendants for b in model.geom_bodyid])
        samples = []
        for start, end in zip(waypoints, waypoints[1:]):
            count = max(2, int(np.ceil(np.max(abs(end-start))/resolution))+1)
            samples.extend(np.linspace(start, end, count))
        if curve is not None:
            samples = curve(np.unique(np.r_[np.linspace(0., 1., max(30, 2*len(samples))), curve.x]))
        names = [mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, int(b)) or '' for b in model.geom_bodyid]
        allowed_contacts = allowed_contacts or {}
        prop_geoms = [i for i, name in enumerate(names) if name in objects]
        moving_geoms = np.flatnonzero(moving)
        if not len(moving_geoms):
            return None
        smallest = None
        for joints in samples:
            data.qpos[self._positions] = joints
            mj.mj_kinematics(model, data)
            for prop in prop_geoms:
                nearby = moving_geoms[np.linalg.norm(
                    data.geom_xpos[moving_geoms] - data.geom_xpos[prop], axis=1)
                    < model.geom_rbound[moving_geoms] + model.geom_rbound[prop] + ceiling]
                for geom in nearby:
                    a, b = names[int(geom)], names[prop]
                    if b in allowed_contacts.get(a, set()) or a in allowed_contacts.get(b, set()):
                        continue
                    gap = mj.mj_geomDistance(model, data, int(geom), prop, ceiling, None)
                    if smallest is None or gap < smallest:
                        smallest = gap
        # Nothing within the ceiling radius is not an unknown: it is the ceiling, saturated.
        return float(ceiling) if smallest is None else float(smallest)

    def path_clearance_detail(self, waypoints, objects, resolution=.025, *, allowed_contacts=None,
                              curve=None, ceiling=.05):
        """Where the path is tightest: the gap, which geoms, and which sample.

        A single number cannot answer "the model says 30 mm but the finger touched". This returns the pair and
        the configuration so the disagreement can be located instead of argued about.
        """
        from backend.app.simulation.transform import mat_to_quat
        waypoints = [np.asarray(point, float) for point in waypoints]
        mj, model = self._mj, self._model
        data = mj.MjData(model)
        data.qpos[:] = self._data.qpos
        placed = 0
        for name, (position, rotation) in objects.items():
            body = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, name)
            if body < 0 or not model.body_jntnum[body]:
                continue
            joint = model.body_jntadr[body]
            if model.jnt_type[joint] != mj.mjtJoint.mjJNT_FREE:
                continue
            address = model.jnt_qposadr[joint]
            data.qpos[address:address+3] = position
            data.qpos[address+3:address+7] = mat_to_quat(rotation)
            placed += 1
        if not placed:
            return None
        root = model.jnt_bodyid[model.dof_jntid[self._dofs[0]]]
        descendants = set()
        for body in range(model.nbody):
            ancestor = body
            while ancestor and ancestor != root:
                ancestor = model.body_parentid[ancestor]
            if ancestor == root:
                descendants.add(body)
        moving = np.array([int(b) in descendants for b in model.geom_bodyid])
        moving_geoms = np.flatnonzero(moving)
        if not len(moving_geoms):
            return None
        names = [mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, int(b)) or '' for b in model.geom_bodyid]
        geom_names = [mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, int(g)) or '' for g in range(model.ngeom)]
        allowed_contacts = allowed_contacts or {}
        prop_geoms = np.flatnonzero([name in objects for name in names])
        samples = []
        for start, end in zip(waypoints, waypoints[1:]):
            count = max(2, int(np.ceil(np.max(abs(end - start)) / resolution)) + 1)
            samples.extend(np.linspace(start, end, count))
        if curve is not None:
            samples = curve(np.unique(np.r_[np.linspace(0., 1., max(30, 2 * len(samples))), curve.x]))
        smallest, at_sample, at_pair = None, None, None
        for index, joints in enumerate(samples):
            data.qpos[self._positions] = joints
            mj.mj_kinematics(model, data)
            for geom in moving_geoms:
                for prop in prop_geoms:
                    if np.linalg.norm(data.geom_xpos[geom] - data.geom_xpos[prop]) > (
                            model.geom_rbound[geom] + model.geom_rbound[prop] + ceiling):
                        continue
                    a, b = names[int(geom)], names[int(prop)]
                    if b in allowed_contacts.get(a, set()) or a in allowed_contacts.get(b, set()):
                        continue
                    gap = float(mj.mj_geomDistance(model, data, int(geom), int(prop), ceiling, None))
                    if smallest is None or gap < smallest:
                        smallest = gap
                        at_sample = index
                        at_pair = (geom_names[int(geom)], geom_names[int(prop)],
                                   str(names[int(geom)]), str(names[int(prop)]))
        if smallest is None:
            return {'gap_m': float(ceiling), 'pair': None, 'sample': None, 'samples': len(samples)}
        return {'gap_m': float(smallest), 'pair': at_pair, 'sample': int(at_sample),
                'samples': len(samples)}

    def empty_arm_path_is_clear(self, waypoints, objects, resolution=.025, *, allowed_contacts=None, curve=None,
                                minimum_object_clearance=0., allow_clearance_exit=False,
                                object_clearances=None):
        """Screen joint segments against the current scene in private FK data.

        Only geometric queries are performed. Free object joints are unnamed in
        the attached model, so copy their *observed* poses explicitly instead of
        accidentally screening against the original layout.
        """
        from backend.app.simulation.transform import mat_to_quat
        mj, model = self._mj, self._model
        data = mj.MjData(model)
        data.qpos[:] = self._data.qpos
        for name, (position, rotation) in objects.items():
            body = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, name)
            if body < 0 or not model.body_jntnum[body]:
                continue  # Wine vessels are separate Genesis entities.
            joint = model.body_jntadr[body]
            if model.jnt_type[joint] != mj.mjtJoint.mjJNT_FREE:
                continue
            address = model.jnt_qposadr[joint]
            data.qpos[address:address+3] = position
            data.qpos[address+3:address+7] = mat_to_quat(rotation)
        # Both arms share the cell's root in some official assets; use actual
        # descendant bodies of this arm's first joint instead.
        root = model.jnt_bodyid[model.dof_jntid[self._dofs[0]]]
        descendants = set()
        for body in range(model.nbody):
            ancestor = body
            while ancestor and ancestor != root:
                ancestor = model.body_parentid[ancestor]
            if ancestor == root:
                descendants.add(body)
        moving = np.array([int(b) in descendants for b in model.geom_bodyid])
        samples = []
        for start, end in zip(waypoints, waypoints[1:]):
            count = max(2, int(np.ceil(np.max(abs(end-start))/resolution))+1)
            samples.extend(np.linspace(start, end, count))
        if curve is not None:
            samples = curve(np.unique(np.r_[np.linspace(0.,1.,max(30,2*len(samples))),curve.x]))
        names = [mj.mj_id2name(model,mj.mjtObj.mjOBJ_BODY,int(b)) or '' for b in model.geom_bodyid]
        allowed_contacts = allowed_contacts or {}
        object_clearances = object_clearances or {}
        prop_geoms = [i for i,name in enumerate(names) if name in objects]
        moving_geoms = np.flatnonzero(moving)
        initial_gaps = {}
        if minimum_object_clearance and allow_clearance_exit:
            data.qpos[self._positions] = waypoints[0]
            mj.mj_kinematics(model,data)
            for prop in prop_geoms:
                clearance = object_clearances.get(names[prop], minimum_object_clearance)
                for geom in moving_geoms:
                    if (names[prop] in allowed_contacts.get(names[int(geom)],set())
                            or names[int(geom)] in allowed_contacts.get(names[prop],set())):
                        continue
                    gap = mj.mj_geomDistance(model,data,int(geom),prop,clearance,None)
                    initial_gaps[int(geom),prop] = min(clearance,max(0.,gap-.0001))
        for index,joints in enumerate(samples):
            data.qpos[self._positions] = joints
            mj.mj_kinematics(model, data)
            mj.mj_collision(model, data)
            if minimum_object_clearance:
                # Endpoint tolerance and dynamic tracking need positive room,
                # not merely the absence of nominal mesh penetration.
                for prop in prop_geoms:
                    clearance = object_clearances.get(names[prop], minimum_object_clearance)
                    nearby = moving_geoms[np.linalg.norm(data.geom_xpos[moving_geoms]-data.geom_xpos[prop],axis=1)
                        < model.geom_rbound[moving_geoms]+model.geom_rbound[prop]+clearance]
                    if any(mj.mj_geomDistance(model,data,int(g),prop,clearance,None)
                           < (initial_gaps.get((int(g),prop),clearance)
                              if index<len(samples)-1 else clearance)
                           for g in nearby if not (names[prop] in allowed_contacts.get(names[int(g)],set())
                               or names[int(g)] in allowed_contacts.get(names[prop],set()))):
                        return False
            for contact in data.contact:
                if contact.dist >= -.0005 or not any(moving[int(g)] for g in contact.geom):
                    continue
                a,b = (names[int(g)] for g in contact.geom)
                if b in allowed_contacts.get(a,set()) or a in allowed_contacts.get(b,set()):
                    continue
                if contact.dist < -.0005:
                    return False
        return True


class OfficialGravityKinematics:
    """Gravity lever arms from the official model; no dynamics solver is stepped.

    This avoids querying every live link Jacobian on both CPU and GPU. Input is
    always the actual Genesis configuration; only the known robot's own mass is
    compensated, so payloads and fluid still exert their real forces in Genesis.
    """
    def __init__(self, model):
        import mujoco
        self._mj, self._model = mujoco, model
        self._data = mujoco.MjData(model)
        self._jac = np.zeros((3, model.nv))
        self._dofs = {}
        self._bodies = {}
        for side in ARM_JOINTS:
            joints = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in ARM_JOINTS[side]]
            self._dofs[side] = model.jnt_dofadr[joints]
            self._bodies[side] = [i for i in range(model.nbody)
                                  if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or "").startswith(f"openarm_{side}_")]

    def evaluate(self, qpos: np.ndarray, gravity: np.ndarray) -> dict[str, np.ndarray]:
        self._data.qpos[:] = qpos
        self._mj.mj_kinematics(self._model, self._data)
        self._mj.mj_comPos(self._model, self._data)
        torques = {}
        for side, bodies in self._bodies.items():
            torque = np.zeros(7)
            for body in bodies:
                self._mj.mj_jacBodyCom(self._model, self._data, self._jac, None, body)
                torque -= self._jac[:, self._dofs[side]].T @ (self._model.body_mass[body] * gravity)
            torques[side] = torque
        return torques
