"""URDF tree, forward kinematics and analytic Jacobians, without a live simulator."""
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation


def vector(value, default):
    result = np.asarray([float(x) for x in value.split()] if value else default, float)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError("URDF vectors must contain three finite values")
    return result


def origin(element):
    transform = np.eye(4)
    if element is not None:
        transform[:3, 3] = vector(element.get("xyz"), [0, 0, 0])
        transform[:3, :3] = Rotation.from_euler("xyz", vector(element.get("rpy"), [0, 0, 0])).as_matrix()
    return transform


def mesh_path(filename, urdf):
    if filename.startswith("file://"):
        candidate = Path(filename[7:])
    elif filename.startswith("package://"):
        relative = Path(filename[10:])
        candidates = [parent / relative for parent in [urdf.parent, *urdf.parents]]
        candidates += [parent / Path(*relative.parts[1:]) for parent in [urdf.parent, *urdf.parents]]
        candidate = next((path for path in candidates if path.is_file()), candidates[0])
    else:
        candidate = urdf.parent / filename
    candidate = candidate.expanduser().resolve()
    if not candidate.is_file():
        raise ValueError(f"URDF mesh does not exist: {candidate}")
    return candidate


class URDFModel:
    def __init__(self, path):
        self.path = Path(path).expanduser().resolve()
        data = self.path.read_bytes()
        if b"<!ENTITY" in data or b"<!DOCTYPE" in data:
            raise ValueError("External XML entities are not supported")
        self.xml = ET.fromstring(data)
        if self.xml.tag != "robot":
            raise ValueError("Choose a URDF robot file")
        self.name = self.xml.get("name", self.path.stem)
        self.links = {link.get("name"): link for link in self.xml.findall("link")}
        self.joints = []
        self.by_child = {}
        for element in self.xml.findall("joint"):
            kind = element.get("type")
            if kind not in {"fixed", "revolute", "continuous", "prismatic"}:
                raise ValueError(f"Unsupported joint type {kind}; fixed-base serial/tree robots are supported")
            parent, child = element.find("parent").get("link"), element.find("child").get("link")
            if parent not in self.links or child not in self.links or child in self.by_child:
                raise ValueError("URDF must have known links and one parent per child")
            limit = element.find("limit")
            mimic = element.find("mimic")
            axis = vector(element.find("axis").get("xyz") if element.find("axis") is not None else None, [1, 0, 0])
            if np.linalg.norm(axis) < 1e-9:
                raise ValueError("A joint axis cannot be zero")
            axis /= np.linalg.norm(axis)
            joint = {"name": element.get("name"), "type": kind, "parent": parent, "child": child,
                     "origin": origin(element.find("origin")).tolist(), "axis": axis.tolist(),
                     "lower": float(limit.get("lower")) if kind in {"revolute", "prismatic"} and limit is not None and limit.get("lower") is not None else None,
                     "upper": float(limit.get("upper")) if kind in {"revolute", "prismatic"} and limit is not None and limit.get("upper") is not None else None,
                     "effort": float(limit.get("effort", "1")) if limit is not None else 1.,
                     "velocity": float(limit.get("velocity", "1")) if limit is not None else 1.,
                     "mimic": dict(mimic.attrib) if mimic is not None else None}
            if kind in {"revolute", "prismatic"} and (joint["lower"] is None or joint["upper"] is None):
                raise ValueError(f"Bounded joint {joint['name']} needs its actual lower and upper limits")
            if kind != 'fixed' and mimic is None and (not np.isfinite(joint['velocity']) or joint['velocity'] <= 0.):
                raise ValueError(f"Actuated joint {joint['name']} needs a positive source velocity limit")
            if joint['lower'] is not None and (not np.isfinite([joint['lower'], joint['upper']]).all()
                                               or joint['lower'] >= joint['upper']):
                raise ValueError(f"Invalid source limits for {joint['name']}")
            self.joints.append(joint)
            self.by_child[child] = joint
        roots = set(self.links) - set(self.by_child)
        if len(roots) != 1:
            raise ValueError("URDF must have exactly one root link")
        self.root = roots.pop()
        self.active = [joint for joint in self.joints if joint["type"] != "fixed" and not joint["mimic"]]
        self.by_name = {joint['name']: joint for joint in self.joints}
        self.forward({})  # Validate connectivity and cycles before generating files.

    def joint_driver(self, name, visited=()):
        if name in visited or name not in self.by_name:
            raise ValueError('Invalid or cyclic URDF mimic joint reference')
        mimic = self.by_name[name]['mimic']
        if not mimic:
            return name, 1., 0.
        driver, multiplier, offset = self.joint_driver(mimic['joint'], (*visited, name))
        scale = float(mimic.get('multiplier', '1'))
        return driver, multiplier * scale, offset * scale + float(mimic.get('offset', '0'))

    def chain(self, link):
        result, visited = [], set()
        while link in self.by_child:
            if link in visited:
                raise ValueError("URDF has a joint cycle")
            visited.add(link)
            joint = self.by_child[link]
            result.append(joint)
            link = joint["parent"]
        return list(reversed(result))

    def nearest_continuous(self, names, values, reference):
        """Use the nearest equivalent angle only when every driven link is periodic."""
        result = np.asarray(values,float).copy()
        for index,name in enumerate(names):
            if self.by_name[name]['type']!='continuous':continue
            related = [(joint,self.joint_driver(joint['name'])) for joint in self.joints]
            if any(driver==name and (joint['type']=='prismatic' or abs(multiplier-round(multiplier))>1e-9)
                   for joint,(driver,multiplier,_) in related):
                continue
            result[index] = reference[index]+(result[index]-reference[index]+np.pi)%(2.*np.pi)-np.pi
        return result

    def forward(self, values, root_transform=None):
        poses = {self.root: np.eye(4) if root_transform is None else np.asarray(root_transform, float)}
        axes, pending = {}, list(self.joints)
        while pending:
            remaining = []
            for joint in pending:
                if joint["parent"] not in poses:
                    remaining.append(joint)
                    continue
                transform = poses[joint["parent"]] @ np.asarray(joint["origin"])
                axes[joint["name"]] = (transform[:3, 3].copy(), transform[:3, :3] @ joint["axis"])
                driver, multiplier, offset = self.joint_driver(joint['name'])
                value = values.get(driver, 0.) * multiplier + offset
                motion = np.eye(4)
                if joint["type"] in {"revolute", "continuous"}:
                    motion[:3, :3] = Rotation.from_rotvec(np.asarray(joint["axis"]) * value).as_matrix()
                elif joint["type"] == "prismatic":
                    motion[:3, 3] = np.asarray(joint["axis"]) * value
                poses[joint["child"]] = transform @ motion
            if len(remaining) == len(pending):
                raise ValueError("URDF is cyclic or disconnected")
            pending = remaining
        return poses, axes


class URDFKinematics:
    def __init__(self, model, chain, snapshot=None, root_transform=None, offset=None):
        self.model, self.chain = model, chain
        self.snapshot = dict(snapshot or {})
        self.root_transform = root_transform
        self.offset = np.asarray(offset if offset is not None else [0., 0., 0.])
        self.names = chain["joints"]
        self.joints = {j["name"]: j for j in model.joints}

    def evaluate(self, values):
        poses, axes = self.model.forward({**self.snapshot, **dict(zip(self.names, values))}, self.root_transform)
        pose = poses[self.chain["end_link"]]
        point = pose[:3, 3] + pose[:3, :3] @ self.offset
        linear, angular = np.zeros((3, len(self.names))), np.zeros((3, len(self.names)))
        for joint in self.model.chain(self.chain['end_link']):
            if joint['type'] == 'fixed':
                continue
            driver, multiplier, _ = self.model.joint_driver(joint['name'])
            if driver not in self.names:
                continue
            index = self.names.index(driver)
            center, axis = axes[joint['name']]
            if joint["type"] == "prismatic":
                linear[:, index] += axis * multiplier
            else:
                linear[:, index] += np.cross(axis, point - center) * multiplier
                angular[:, index] += axis * multiplier
        return point.copy(), pose[:3, :3].copy(), linear, angular


def rotation_residual_jacobian(attitude,target,angular):
    error=Rotation.from_matrix(np.asarray(target)@attitude.T).as_rotvec()
    x,y,z=error
    skew=np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])
    theta=np.linalg.norm(error)
    coefficient=1./12. if theta<1e-5 else (1.-theta/(2.*np.tan(theta/2.)))/(theta*theta)
    return -(np.eye(3)+skew/2.+coefficient*(skew@skew))@angular


def solve_profile_ik(profile, side, position, rotation=None, *, model=None, q_init=None, snapshot=None, joint_limits=None):
    from scipy.optimize import least_squares
    model = model if model is not None else URDFModel(profile['urdf'])
    chain = profile['chains'][side]
    kinematics = URDFKinematics(model, chain, {**profile['home'], **(snapshot or {})}, profile['root_transform'], chain['tool_offset'])
    home = np.asarray([profile['home'][name] for name in chain['joints']])
    specs = {joint['name']: joint for joint in model.active}
    lower = [specs[name]['lower'] if specs[name]['lower'] is not None else -np.inf for name in chain['joints']]
    upper = [specs[name]['upper'] if specs[name]['upper'] is not None else np.inf for name in chain['joints']]
    if joint_limits is not None:
        lower = np.maximum(lower,np.asarray(joint_limits)[:,0])
        upper = np.minimum(upper,np.asarray(joint_limits)[:,1])
    target = np.asarray(position)
    start = np.clip(home if q_init is None else np.asarray(q_init), lower, upper)
    def residual(q):
        point, attitude, _, _ = kinematics.evaluate(q)
        orientation = [] if rotation is None else (Rotation.from_matrix(np.asarray(rotation) @ attitude.T).as_rotvec() * .15).tolist()
        return np.r_[point - target, orientation, (q - start) * .0001]
    def jacobian(q):
        _,attitude,linear,angular=kinematics.evaluate(q)
        pieces=[linear]
        if rotation is not None:pieces.append(rotation_residual_jacobian(attitude,rotation,angular)*.15)
        pieces.append(np.eye(len(q))*.0001)
        return np.vstack(pieces)
    result = least_squares(residual, start, jac=jacobian, bounds=(lower, upper), max_nfev=300, xtol=1e-10, gtol=1e-10, ftol=1e-10)
    point, attitude, _, _ = kinematics.evaluate(result.x)
    error = float(np.linalg.norm(point - target))
    orientation_error = float(np.linalg.norm(Rotation.from_matrix(np.asarray(rotation) @ attitude.T).as_rotvec())) if rotation is not None else 0.
    return {'success': error < .002 and orientation_error < .03, 'joint_names': chain['joints'],
            'q_rad': result.x.tolist(), 'target_position': target.tolist(), 'actual_fk_position': point.tolist(),
            'position_error': error, 'rotation_error': orientation_error, 'scope': 'private URDF kinematics; no live pose writes'}
