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

    def empty_arm_path_is_clear(self, waypoints, objects, resolution=.025, *, allowed_contacts=None, curve=None,
                                minimum_object_clearance=0., allow_clearance_exit=False):
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
        prop_geoms = [i for i,name in enumerate(names) if name in objects]
        moving_geoms = np.flatnonzero(moving)
        initial_gaps = {}
        if minimum_object_clearance and allow_clearance_exit:
            data.qpos[self._positions] = waypoints[0]
            mj.mj_kinematics(model,data)
            for prop in prop_geoms:
                for geom in moving_geoms:
                    gap = mj.mj_geomDistance(model,data,int(geom),prop,minimum_object_clearance,None)
                    initial_gaps[int(geom),prop] = min(minimum_object_clearance,max(0.,gap-.0001))
        for index,joints in enumerate(samples):
            data.qpos[self._positions] = joints
            mj.mj_kinematics(model, data)
            mj.mj_collision(model, data)
            if minimum_object_clearance:
                # Endpoint tolerance and dynamic tracking need positive room,
                # not merely the absence of nominal mesh penetration.
                for prop in prop_geoms:
                    nearby = moving_geoms[np.linalg.norm(data.geom_xpos[moving_geoms]-data.geom_xpos[prop],axis=1)
                        < model.geom_rbound[moving_geoms]+model.geom_rbound[prop]+minimum_object_clearance]
                    if any(mj.mj_geomDistance(model,data,int(g),prop,minimum_object_clearance,None)
                           < (initial_gaps.get((int(g),prop),minimum_object_clearance)
                              if index<len(samples)-1 else minimum_object_clearance) for g in nearby):
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
