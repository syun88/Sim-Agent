"""Inspect a local URDF folder and generate a reproducible robot/scene/IK bundle."""
from pathlib import Path
import hashlib
import json
import re

import numpy as np
from scipy.spatial.transform import Rotation

from backend.app.robots.urdf import URDFModel, URDFKinematics, solve_profile_ik, mesh_path, origin, vector
from backend.app.robots.policy import IMPORTED_ROBOT_POLICY,policy_signature

GENERATED_ROOT = Path(__file__).resolve().parents[3] / "outputs" / "robots"
OPENARM = {"id": "openarm_v2", "name": "OpenArm v2", "source": "official openarm-mujoco",
           "builtin": True, "capabilities": ["motion", "physical_grasp", "handover", "stack", "pour"]}
PROFILE_VERSION = 16
_file_hash_cache = {}


def asset_signature(source):
    """Include mesh content: changing an OBJ must invalidate old grasp evidence."""
    import xml.etree.ElementTree as ET
    paths = {Path(source).resolve()}
    for name in ('initial_state.json','actuators.json'):
        initial = Path(source).with_name(name)
        if initial.is_file():paths.add(initial.resolve())
    for mesh in ET.parse(source).getroot().findall('.//mesh'):
        paths.add(mesh_path(mesh.get('filename'), source).resolve())
    records = []
    for path in sorted(paths):
        stat = path.stat()
        key = (str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        if key not in _file_hash_cache:
            _file_hash_cache[key] = hashlib.sha256(path.read_bytes()).hexdigest()
        records.append((str(path), _file_hash_cache[key]))
    return hashlib.sha256(json.dumps(records).encode()).hexdigest()


def available_robots():
    robots = {OPENARM['id']: dict(OPENARM)}
    for path in sorted(GENERATED_ROOT.glob("*/profile.json")):
        if json.loads(path.read_text()).get('redirect_to'):
            continue
        profile = get_profile(path.parent.name)
        robots[profile['id']] = profile
    return list(robots.values())


def get_profile(robot_id):
    if robot_id == "openarm_v2":
        return dict(OPENARM)
    if not re.fullmatch(r"[a-z0-9_-]+", robot_id):
        raise ValueError("Invalid robot ID")
    path = GENERATED_ROOT / robot_id / "profile.json"
    if not path.is_file():
        raise ValueError("Import the robot's URDF folder first")
    profile = json.loads(path.read_text())
    if profile.get('redirect_to'):
        return get_profile(profile['redirect_to'])
    source = Path(profile["source"])
    if not source.is_file():
        raise ValueError("The source URDF moved; choose its new location")
    if (profile.get("profile_version", 0) < PROFILE_VERSION
            or profile.get('control_policy_sha256')!=policy_signature()
            or profile.get('asset_sha256') != asset_signature(source)):
        updated = import_robot(source)
        if updated['id'] != robot_id:
            profile['redirect_to'] = updated['id']
            temporary = path.with_suffix('.json.tmp')
            temporary.write_text(json.dumps(profile, indent=2) + '\n')
            temporary.replace(path)
        return updated
    runtime_path = path.with_name('gripper_runtime.json')
    if runtime_path.is_file():
        runtime = json.loads(runtime_path.read_text())
        if runtime.get('configuration_sha256') == profile.get('configuration_sha256'):
            for side, result in runtime.get('arms', {}).items():
                if side in profile.get('grippers', {}):
                    profile['grippers'][side]['status'] = result['status']
            profile['grasp_calibrated'] = bool(profile.get('grippers')) and all(
                g['status'] == 'physical_grasp_verified' for g in profile['grippers'].values())
    return profile


def collision_declaration(source):
    """Read a robot package's declared collision semantics.

    A manufacturer URDF's meshes are visual-quality parts: adjacent links overlap at every
    joint by design (measured 2026-10-24 on OpenManipulator-X: 21.4-43.3 mm), so a robot
    whose source geometry does this must *declare* which adjacent pairs the runtime must not
    collide - the URDF cannot say it. Documentation of overlaps
    (``excluded_adjacent_pairs``, as the Scorpion package carries) is not a runtime request;
    ``runtime_excluded_adjacent_pairs`` is, and only that key changes behaviour.
    """
    for candidate in (source.parent / "collision_config.json", source.parent.parent / "collision_config.json"):
        if candidate.is_file():
            break
    else:
        return {"source": None, "runtime_excluded_adjacent_pairs": [], "covers_adjacent_pairs": None}
    settings = json.loads(candidate.read_text())
    pairs = [[str(a), str(b)] for a, b in settings.get("runtime_excluded_adjacent_pairs", [])]
    return {"source": str(candidate), "runtime_excluded_adjacent_pairs": pairs,
            "covers_adjacent_pairs": None,
            "documented_excluded_adjacent_pairs": len(settings.get("excluded_adjacent_pairs", []))}


def _link_wheel_radius(model, link_name):
    """Rolling radius of a wheel link from its own geometry, or None.

    A cylinder/sphere collision gives the radius directly; a mesh (the usual
    CAD export) gives it as half its largest extent, the wheel's diameter.
    """
    link = model.links[link_name]
    for kind in ("collision", "visual"):
        for element in link.findall(kind):
            cylinder = element.find("geometry/cylinder")
            if cylinder is not None and cylinder.get("radius"):
                return float(cylinder.get("radius"))
            sphere = element.find("geometry/sphere")
            if sphere is not None and sphere.get("radius"):
                return float(sphere.get("radius"))
            mesh = element.find("geometry/mesh")
            if mesh is not None:
                import trimesh
                geometry = trimesh.load(mesh_path(mesh.get("filename"), model.path), force="mesh", process=False)
                # CAD exports often model in millimetres and carry the unit
                # conversion in the mesh scale; apply it like the root fit.
                scale = vector(mesh.get("scale"), [1., 1., 1.])
                extents = (np.asarray(geometry.bounds[1]) - np.asarray(geometry.bounds[0])) * scale
                return float(extents.max()) / 2.
    return None


def _wheel_body_link(model, joint):
    """The link that actually rolls for a wheel joint.

    The joint's child is often just a mount (LeKiwi: omni_wheel_mount) and
    the wheel body hangs below it on a fixed joint; measuring the mount
    understates both the radius (~28 mm instead of ~52 mm) and the lever
    arm. Follow fixed joints down and keep the descendant whose own
    geometry gives the largest radius.
    """
    descendants = [joint["child"]]
    queue = [joint["child"]]
    while queue:
        parent = queue.pop()
        for other in model.joints:
            if other["parent"] == parent and other["type"] == "fixed":
                descendants.append(other["child"])
                queue.append(other["child"])
    return max(descendants, key=lambda name: _link_wheel_radius(model, name) or 0.)


def _detect_mobile_base(model, chains):
    """Detect a wheeled mobile base among the joints no arm chain claims.

    A wheel joint is an actuated joint with "wheel" in its name that is not
    part of any selected arm chain (LeKiwi: base_left_wheel, base_back_wheel,
    base_right_wheel). Three wheels make a holonomic kiwi drive, two a
    differential drive. Wheel positions come from the model's own forward
    kinematics at the zero pose, so the drive geometry is measured from the
    URDF, never assumed. Returns None when the robot has no wheeled base.
    """
    chain_joints = {name for chain in chains.values() for name in chain["joints"]}
    wheels = [j for j in model.active if "wheel" in j["name"].lower() and j["name"] not in chain_joints]
    if len(wheels) not in (2, 3):
        return None
    poses, axes = model.forward({})
    root_position = poses[model.root][:3, 3]
    bodies = [_wheel_body_link(model, j) for j in wheels]
    centres = np.array([poses[body][:3, 3] - root_position for body in bodies])
    # Sign of each wheel's spin, from the no-slip condition at the contact:
    # spinning about the axle a moves the contact patch along -(a x z), so
    # the dof sign that rolls the wheel along its tangent t is t . (a x z).
    # Measured from the model's own world axes, never assumed (LeKiwi's
    # axles all point radially outward and all three signs come out -1;
    # a differential pair with mirrored outward axles comes out (+1, -1)).
    forward_axis = poses[model.root][:3, :3] @ np.array([1., 0., 0.])
    signs = []
    for j, centre in zip(wheels, centres):
        if len(wheels) == 3:
            theta = np.arctan2(centre[1], centre[0])
            tangent = np.array([-np.sin(theta), np.cos(theta), 0.])
        else:
            tangent = forward_axis
        axis_world = np.asarray(axes[j["name"]][1], dtype=float)
        signs.append(1 if float(tangent @ np.cross(axis_world, np.array([0., 0., 1.]))) > 0 else -1)
    radii = [_link_wheel_radius(model, body) for body in bodies]
    wheel_radius = float(np.mean([r for r in radii if r])) if any(radii) else None
    if len(wheels) == 3:
        return {"drive": "kiwi", "wheel_joints": [j["name"] for j in wheels],
                "wheel_angles": [float(np.arctan2(c[1], c[0])) for c in centres],
                "wheel_signs": signs,
                "wheel_levers": [float(np.linalg.norm(c[:2])) for c in centres],
                "base_radius": float(np.mean(np.linalg.norm(centres[:, :2], axis=1))),
                "wheel_radius": wheel_radius}
    order = sorted(range(len(wheels)), key=lambda i: -centres[i, 1])
    return {"drive": "diff", "wheel_joints": [wheels[i]["name"] for i in order],
            "wheel_signs": [signs[i] for i in order],
            "track_width": float(abs(centres[order[0], 1] - centres[order[1], 1])),
            "wheel_radius": wheel_radius}


def _rpy_from_matrix(rotation):
    """URDF rpy (R = Rz @ Ry @ Rx) from a rotation matrix."""
    pitch = float(np.arcsin(np.clip(-rotation[2, 0], -1., 1.)))
    if abs(np.cos(pitch)) < 1e-6:
        return [float(np.arctan2(-rotation[0, 1], rotation[1, 1])), pitch, 0.]
    return [float(np.arctan2(rotation[2, 1], rotation[2, 2])), pitch,
            float(np.arctan2(rotation[1, 0], rotation[0, 0]))]


def _synthesize_omni_wheels(model, spec):
    """Rewrite fused CAD omni wheels as hub + free-spinning rollers.

    CAD exports fuse an omni wheel's rollers into one rigid body, so in
    physics the contact resists sliding along the axle with full friction:
    a kiwi base then cannot translate (measured: ~15-25% of the commanded
    speed, the rest lost to scrub). The real wheel's rollers spin freely
    about tangential axes and make the axle direction nearly free. This
    rewrite keeps the CAD visual on the hub, drops the hub's collision,
    and hangs K small roller cylinders around the rim on passive
    continuous joints, restoring that mechanism in the physics URDF.
    Only the serialized physics URDF changes; detection and the profile
    keep describing the source model.
    """
    import xml.etree.ElementTree as ET

    def fmt(values):
        return " ".join(f"{float(v):.6g}" for v in values)

    poses, axes = model.forward({})
    joints_by_name = {j["name"]: j for j in model.joints}
    elements = {el.get("name"): el for el in model.xml.findall("joint")}
    rollers_per_wheel = 8
    for name in spec["wheel_joints"]:
        joint = joints_by_name[name]
        body_name = _wheel_body_link(model, joint)
        body = model.links[body_name]
        radius = _link_wheel_radius(model, body_name) or spec["wheel_radius"]
        # Width = smallest mesh extent of the wheel body (its thickness).
        width = 0.6 * radius
        for element in body.findall("visual"):
            mesh = element.find("geometry/mesh")
            if mesh is not None:
                import trimesh
                geometry = trimesh.load(mesh_path(mesh.get("filename"), model.path), force="mesh", process=False)
                scale = vector(mesh.get("scale"), [1., 1., 1.])
                width = float(((np.asarray(geometry.bounds[1]) - np.asarray(geometry.bounds[0])) * scale).min())
                break
        # Axle direction in the body frame, from the measured world axis.
        world_axis = np.asarray(axes[name][1], dtype=float)
        axle = poses[body_name][:3, :3].T @ world_axis
        axle = axle / np.linalg.norm(axle)
        helper = np.array([0., 0., 1.]) if abs(axle[2]) < 0.9 else np.array([1., 0., 0.])
        u = np.cross(axle, helper)
        u = u / np.linalg.norm(u)
        v = np.cross(axle, u)
        # Keep the measured mass on the hub, light rollers.
        inertial = body.find("inertial")
        mass_el0 = inertial.find("mass") if inertial is not None else None
        total_mass = float(mass_el0.get("value")) if mass_el0 is not None and mass_el0.get("value") else 0.1
        if inertial is not None:
            mass_el0.set("value", fmt([total_mass * 0.65]))
            inertia_el = inertial.find("inertia")
            if inertia_el is not None:
                for attr in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz"):
                    if inertia_el.get(attr):
                        inertia_el.set(attr, fmt([float(inertia_el.get(attr)) * 0.65]))
        for collision in body.findall("collision"):
            body.remove(collision)
        roller_radius = float(np.clip(radius * 0.22, 0.006, 0.016))
        ring = radius - roller_radius
        roller_length = max(width * 0.92, 0.01)
        roller_mass = total_mass * 0.35 / rollers_per_wheel
        # Phase the rollers so one sits at the contact patch at spawn.
        down = poses[body_name][:3, :3].T @ np.array([0., 0., -1.])
        phase = float(np.arctan2(down @ v, down @ u))
        for k in range(rollers_per_wheel):
            theta = phase + 2 * np.pi * k / rollers_per_wheel
            radial = np.cos(theta) * u + np.sin(theta) * v
            tangent = -np.sin(theta) * u + np.cos(theta) * v
            rotation = np.column_stack([radial, axle, tangent])
            rpy = _rpy_from_matrix(rotation)
            roller_name = f"{body_name}_roller_{k}"
            link_el = ET.SubElement(model.xml, "link", {"name": roller_name})
            inertial_el = ET.SubElement(link_el, "inertial")
            ET.SubElement(inertial_el, "mass", {"value": fmt([roller_mass])})
            ixx = roller_mass * (3 * roller_radius ** 2 + roller_length ** 2) / 12.
            ET.SubElement(inertial_el, "inertia", {
                "ixx": fmt([ixx]), "ixy": "0", "ixz": "0",
                "iyy": fmt([ixx]), "iyz": "0", "izz": fmt([0.5 * roller_mass * roller_radius ** 2])})
            collision_el = ET.SubElement(link_el, "collision")
            geometry_el = ET.SubElement(collision_el, "geometry")
            ET.SubElement(geometry_el, "cylinder", {"radius": fmt([roller_radius]), "length": fmt([roller_length])})
            joint_el = ET.SubElement(model.xml, "joint", {"name": roller_name, "type": "continuous"})
            ET.SubElement(joint_el, "parent", {"link": body_name})
            ET.SubElement(joint_el, "child", {"link": roller_name})
            ET.SubElement(joint_el, "origin", {"xyz": fmt(ring * radial), "rpy": fmt(rpy)})
            ET.SubElement(joint_el, "axis", {"xyz": "0 0 1"})


def import_robot(folder):
    root = Path(folder).expanduser().resolve()
    if root.is_file() and root.suffix.lower() == ".urdf":
        source = root
    elif root.is_dir():
        preferred = sorted((root / "genesis").glob("*.urdf"))
        direct = sorted(root.glob("*.urdf"))
        candidates = preferred or direct or sorted(root.rglob("*.urdf"))
        candidates = [p for p in candidates if not p.stem.endswith("_ros")]
        if len(candidates) != 1:
            raise ValueError("Choose one URDF file explicitly; candidates: " + ", ".join(str(p) for p in candidates[:12]))
        source = candidates[0]
    else:
        raise ValueError("The local URDF folder/file does not exist")
    model = URDFModel(source)
    declaration = collision_declaration(source)
    adjacent = {tuple(sorted((joint['parent'], joint['child']))) for joint in model.joints}
    declared = {tuple(sorted(pair)) for pair in declaration['runtime_excluded_adjacent_pairs']}
    unknown = sorted({link for pair in declared for link in pair if link not in model.links})
    if unknown:
        raise ValueError('runtime_excluded_adjacent_pairs names unknown links: ' + ', '.join(unknown))
    if any(pair[0] == pair[1] for pair in declared):
        raise ValueError('runtime_excluded_adjacent_pairs pairs two identical links')
    if [pair for pair in declared if pair not in adjacent]:
        raise ValueError('A URDF path cannot filter non-adjacent link pairs: '
                         + json.dumps(sorted(pair for pair in declared if pair not in adjacent)))
    # Only pairs whose two links both carry collision geometry can ever collide; a virtual
    # frame (world, end_effector) cannot, so it needs no declaration. Coverage means every
    # such pair is declared - the engine can only turn adjacent collision off wholesale, and
    # doing that while an undeclared pair could collide would drop a real constraint.
    collidable = {pair for pair in adjacent
                  if all(len(model.links[link].findall('collision')) for link in pair)}
    declaration['covers_adjacent_pairs'] = bool(declared) and declared >= collidable
    if declared and not declaration['covers_adjacent_pairs']:
        # The engine can only turn adjacent collision off wholesale, so a partial runtime
        # declaration leaves the private queries ignoring pairs the engine still collides.
        # Measured 2026-10-24: with one of two overlapping adjacent pairs declared, the engine was
        # told adjacent_collision=True while the private model was silent on the declared pair, so a
        # posture colliding only there passes validation and then collides in physics. A partial set
        # belongs in the documentation-only key.
        raise ValueError(
            'runtime_excluded_adjacent_pairs must declare every collidable adjacent pair (%d of %d '
            'declared), because the engine turns adjacent collision off wholesale and cannot match a '
            'partial set; use excluded_adjacent_pairs to document an overlap instead.'
            % (len(declared & collidable), len(collidable)))
    declaration['adjacent_pair_count'] = len(collidable)
    declaration['non_collidable_adjacent_pairs'] = len(adjacent) - len(collidable)
    from backend.app.robots.actuators import apply_actuator_configuration
    actuator_configuration=apply_actuator_configuration(model,source)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    robot_id = (re.sub(r"[^a-z0-9_-]+", "-", model.name.lower()).strip("-") or "robot") + "-" + digest[:10]
    target = GENERATED_ROOT / robot_id
    target.mkdir(parents=True, exist_ok=True)
    meshes = []
    for element in model.xml.findall(".//mesh"):
        resolved = mesh_path(element.get("filename"), source)
        element.set("filename", str(resolved))
        meshes.append(str(resolved))
    parents = {j["parent"] for j in model.joints}
    leaves = [name for name in model.links if name not in parents]
    # Wheel joints are a base, not an arm: a chain that exists only to reach
    # a wheel (a standalone cart's only chains) must not be picked as an arm.
    wheel_joint_names = {j["name"] for j in model.active if "wheel" in j["name"].lower()}
    candidates = []
    for leaf in leaves:
        joints = [j["name"] for j in model.chain(leaf)
                  if j["type"] != "fixed" and not j["mimic"] and j["name"] not in wheel_joint_names]
        if joints:
            candidates.append({"end_link": leaf, "joints": joints, "tool_offset": [0., 0., 0.]})
    candidates.sort(key=lambda c: (-len(c["joints"]), c["end_link"]))
    chosen = candidates[:1]
    if len(candidates) > 1:
        # Distinct arms should win over two sibling finger leaves on the same arm.
        first = set(chosen[0]['joints'])
        other = max(candidates[1:], key=lambda c: (len(set(c['joints']) - first), len(c['joints'])))
        if len(set(other['joints']) - first) >= 3 and len(first - set(other['joints'])) >= 3:
            chosen.append(other)
    chains = {}
    for candidate in chosen:
        tokens = " ".join([candidate["end_link"], *candidate["joints"]])
        side = "right" if re.search(r"(?:^|[_\s])(?:r|right)(?:[_\s]|$)", tokens, re.I) else "left"
        if side in chains:
            side = "right" if side == "left" else "left"
        chains[side] = candidate
    poses, _ = model.forward({})
    yaw = 0.
    if len(chains) == 2:
        left, right = [poses[chains[side]["end_link"]][:3, 3] for side in ("left", "right")]
        delta = left - right
        if np.linalg.norm(delta[:2]) > .01:
            yaw = np.pi / 2. - np.arctan2(delta[1], delta[0])
    mobile_base = _detect_mobile_base(model, chains)
    if not chosen and not mobile_base:
        raise ValueError("No actuated chain found in this URDF")
    # Fit the root link to the desk using its actual visual mesh bounds.
    import trimesh
    minimum_z = 0.
    vertices = []
    for visual in model.links[model.root].findall("visual"):
        mesh = visual.find("geometry/mesh")
        if mesh is not None:
            geometry = trimesh.load(mesh_path(mesh.get("filename"), source), force="mesh", process=False)
            points = np.asarray(geometry.vertices) * vector(mesh.get("scale"), [1., 1., 1.])
            transform = origin(visual.find("origin"))
            vertices.append(points @ transform[:3, :3].T + transform[:3, 3])
    if vertices:
        minimum_z = float(np.concatenate(vertices)[:, 2].min())
    root_transform = np.eye(4)
    root_transform[:3, :3] = Rotation.from_euler("z", yaw).as_matrix()
    if mobile_base:
        # A wheeled base stands on the floor, not on the desk: rest the whole
        # robot's lowest measured point on z=0, using every link's visual
        # mesh transformed by the zero-pose forward kinematics.
        floor_z = minimum_z
        for link_name, pose in poses.items():
            for visual in model.links[link_name].findall("visual"):
                mesh = visual.find("geometry/mesh")
                if mesh is None:
                    continue
                geometry = trimesh.load(mesh_path(mesh.get("filename"), source), force="mesh", process=False)
                points = np.asarray(geometry.vertices) * vector(mesh.get("scale"), [1., 1., 1.])
                transform = origin(visual.find("origin"))
                world = points @ transform[:3, :3].T + transform[:3, 3]
                world = world @ pose[:3, :3].T + pose[:3, 3]
                floor_z = min(floor_z, float(world[:, 2].min()))
        root_transform[:3, 3] = [0., 0., -floor_z + .001]
    else:
        root_transform[:3, 3] = [-.25, 0., .76 - minimum_z + .001]
    initial = {}
    initial_file = source.with_name("initial_state.json")
    if initial_file.is_file():
        initial = json.loads(initial_file.read_text())
    home = {j["name"]: float(initial.get(j["name"], 0. if j["lower"] is None else np.clip(0., j["lower"], j["upper"])))
            for j in model.active}
    if mobile_base and mobile_base.get("drive") == "kiwi":
        _synthesize_omni_wheels(model, mobile_base)
    normalized = target / "robot.urdf"
    import xml.etree.ElementTree as ET
    ET.indent(model.xml)
    ET.ElementTree(model.xml).write(normalized, encoding="utf-8", xml_declaration=True)
    warnings = [
                "URDF effort, velocity, mass and continuous-joint status are preserved; these may be uncalibrated."]
    if declaration['covers_adjacent_pairs']:
        warnings.append('Source meshes overlap at adjacent joints: the engine does not collide adjacent links '
                        '(%d pairs) and the private queries exclude the same pairs; non-adjacent self-collision stays on.'
                        % len(declared))
    elif declaration['runtime_excluded_adjacent_pairs']:
        warnings.append('Declared adjacent-pair overlaps are partial (%d of %d): the private queries exclude them, '
                        'the engine keeps adjacent collision on for the undeclared rest.'
                        % (len(declared), declaration['adjacent_pair_count']))
    if len(candidates) > 2:
        warnings.append("Longest distinct chains selected; all candidate chains are recorded for configuration.")
    profile = {"id": robot_id, "name": model.name, "builtin": False, "source": str(source), "source_sha256": digest,
               "urdf": str(normalized), "directory": str(target), "root_link": model.root, "links": len(model.links),
               "joints": model.active, "chains": chains, "all_chains": candidates, "home": home,
               "root_transform": root_transform.tolist(), "capabilities": ["motion", "joint_control", "ik"] if chains else [],
               "profile_version": PROFILE_VERSION, "asset_sha256": asset_signature(source),
               "control_policy_sha256":policy_signature(),"control_policy":IMPORTED_ROBOT_POLICY,
               "grasp_calibrated": False, "warnings": warnings, "mesh_count": len(set(meshes))}
    if mobile_base:
        profile['mobile_base'] = mobile_base
        profile['capabilities'] += ['navigation']
    if actuator_configuration:
        profile['actuator_configuration']=actuator_configuration
        profile['warnings']=["Source masses, geometry and joint frames are preserved. Explicit actuators.json supplies motor effort and no-load speed limits."]
    scene = {"robot_id": robot_id, "openarm_cell": False,
             "table_top": 0. if mobile_base else .76, "table_size": [0., 0., 0.] if mobile_base else [1.6, 1.4, .05],
             "robot_root_transform": root_transform.tolist(), "physics_dt": IMPORTED_ROBOT_POLICY['physics_dt_max_s'],
             "physics_substeps":IMPORTED_ROBOT_POLICY['physics_substeps'],
             "robot_gravity_compensation": 1., "self_collision": True,
             # URDF cannot filter individual pairs, so the engine expresses only a declaration
             # that covers every adjacent pair (see collision_declaration).
             "adjacent_collision": not declaration['covers_adjacent_pairs'],
             "runtime_excluded_adjacent_pairs": declaration['runtime_excluded_adjacent_pairs'],
             "default_armature": None,
             "constraint_solver": "Newton", "friction_cone": "elliptic",
             "torsional_friction": True, "rolling_friction": True}
    from backend.app.robots.gripper_discovery import discover_grippers
    analysis = discover_grippers(model, profile)
    profile['grippers'] = analysis['grippers']
    for side, gripper in profile['grippers'].items():
        chains[side] = {'end_link': gripper['reference_link'], 'tool_offset': gripper['tool_offset'],
                        'joints': [j['name'] for j in model.chain(gripper['reference_link'])
                                   if j['type'] != 'fixed' and not j['mimic']]}
    if profile['grippers']:
        profile['capabilities'] += ['physical_grasp', 'place', 'stack']
        if len(profile['grippers']) > 1:
            profile['capabilities'].append('handover')
    scene['cube_size'] = min((g['cube_size'] for g in profile['grippers'].values()), default=.052)
    scene['cube_mass'] = .05 * (scene['cube_size'] / .052) ** 3
    scene['physics_dt']=min(IMPORTED_ROBOT_POLICY['physics_dt_max_s'],max(IMPORTED_ROBOT_POLICY['physics_dt_min_s'],
        scene['cube_size']*IMPORTED_ROBOT_POLICY['physics_dt_per_cube_meter']))
    profile['cube_size'] = scene['cube_size']
    profile['collision_config'] = declaration
    (target / 'gripper_analysis.json').write_text(json.dumps(analysis, indent=2) + '\n')
    seeds = {}
    for side, chain in chains.items():
        kinematics = URDFKinematics(model, chain, home, root_transform)
        q = np.array([home[name] for name in chain["joints"]])
        point, rotation, linear, angular = kinematics.evaluate(q)
        seeds[side] = {"joints": chain["joints"], "q": q.tolist(), "ee_position": point.tolist(),
                       "ee_rotation": rotation.tolist(), "jacobian_rank": int(np.linalg.matrix_rank(np.vstack([linear, angular])))}
    for name, data in [("scene.json", scene), ("ik_seeds.json", seeds)]:
        (target / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    ik_checks = {side: solve_profile_ik(profile, side, np.asarray(seed['ee_position']) + [0., 0., .005])
                 for side, seed in seeds.items()}
    (target / 'ik_validation.json').write_text(json.dumps(ik_checks, indent=2) + '\n')
    from backend.app.robots.spawn import ReachableDeskSpawns
    if mobile_base:
        # A floor robot's props are authored, not desk-sampled: an optional
        # floor_objects.json next to the URDF maps object names to floor
        # positions ([x, y] or [x, y, z]; z defaults to half the scene's
        # cube size, i.e. resting on the floor).
        spawn = {"objects": {}}
        authored = source.parent / "floor_objects.json"
        if authored.is_file():
            for name, entry in json.loads(authored.read_text()).get("objects", {}).items():
                position = [float(v) for v in entry["position"]]
                if len(position) == 2:
                    position.append(scene["cube_size"] / 2)
                spawn["objects"][name] = {"position": position}
    else:
        spawn = ReachableDeskSpawns(profile, scene).layout()
    scene['objects'] = spawn['objects']
    (target / 'scene.json').write_text(json.dumps(scene, indent=2) + '\n')
    (target / 'spawn_validation.json').write_text(json.dumps(spawn, indent=2) + '\n')
    project = Path(__file__).resolve().parents[3]
    (target / "solve_ik.py").write_text(
        '"""Private URDF IK; prints a solution without moving the live robot."""\n'
        'import argparse, json, sys\nfrom pathlib import Path\n'
        f'sys.path.insert(0, {str(project)!r})\n'
        'from backend.app.robots.urdf import solve_profile_ik\n'
        'parser=argparse.ArgumentParser()\nparser.add_argument("side", choices=["left","right"])\n'
        'parser.add_argument("position", type=float, nargs=3)\nargs=parser.parse_args()\n'
        'profile=json.loads(Path(__file__).with_name("profile.json").read_text())\n'
        'result=solve_profile_ik(profile,args.side,args.position)\n'
        'Path(__file__).with_name("ik_solution.json").write_text(json.dumps(result,indent=2)+"\\n")\n'
        'print(json.dumps(result,indent=2))\nraise SystemExit(0 if result["success"] else 1)\n')
    (target / "README.md").write_text(
        f"# {model.name}\n\nSource: `{source}`\n\n{len(model.active)} actuated joints; {len(model.links)} links.\n\n"
        "The source folder and meshes are attached in place. robot.urdf resolves their absolute paths.\n"
        "profile.json records chains, axes, limits, forces and tool frames; scene.json replaces the OpenArm Cell with a desk.\n"
        "ik_seeds.json contains private FK/Jacobian results, not simulation-runtime proof.\n"
        "spawn_validation.json admits desk props only when supported-height grasp/approach IK, robot/desk clearance and sampled jaw closure pass. Full paths and contacts are verified during execution.\n"
        "Opposed jaws and cube size are inferred from collision geometry. Every grasp still requires two contacts and lift-follow in Genesis. Continuous joints have no invented finite limits.\n")
    profile['configuration_sha256'] = hashlib.sha256(json.dumps(
        {'source': profile['asset_sha256'], 'version': PROFILE_VERSION, 'policy':policy_signature(),
         'adjacent_collision': not declaration['covers_adjacent_pairs'],
         **({'runtime_excluded_adjacent_pairs': declaration['runtime_excluded_adjacent_pairs']}
            if declaration['runtime_excluded_adjacent_pairs'] else {}),
         'chains': chains, 'grippers': profile['grippers'], 'cube_size': scene['cube_size'],
         'home': home, 'root': profile['root_transform'], 'mobile_base': mobile_base}, sort_keys=True).encode()).hexdigest()
    # The catalog sees only complete registrations, after all IK/setup files exist.
    temporary = target / 'profile.json.tmp'
    temporary.write_text(json.dumps(profile, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(target / 'profile.json')
    return profile


def selected_robot():
    path = GENERATED_ROOT / "selection.json"
    robot_id = json.loads(path.read_text()).get("robot_id", "openarm_v2") if path.is_file() else "openarm_v2"
    resolved = get_profile(robot_id)['id']
    if resolved != robot_id:
        remember_selection(resolved)
    return resolved


def remember_selection(robot_id):
    GENERATED_ROOT.mkdir(parents=True, exist_ok=True)
    (GENERATED_ROOT / "selection.json").write_text(json.dumps({"robot_id": robot_id}) + "\n")
