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
PROFILE_VERSION = 14
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
    # A URDF path cannot filter a pair that does not share a joint, but an engine model can exclude
    # any pair. A coupled linkage whose parts overlap across branches needs exactly that - the pair
    # does not overlap at one hinge, so no amount of collision-mesh preparation separates it.
    non_adjacent = sorted(pair for pair in declared if pair not in adjacent)
    # Only pairs whose two links both carry collision geometry can ever collide; a virtual
    # frame (world, end_effector) cannot, so it needs no declaration. Coverage means every
    # such pair is declared - the engine can only turn adjacent collision off wholesale, and
    # doing that while an undeclared pair could collide would drop a real constraint.
    collidable = {pair for pair in adjacent
                  if all(len(model.links[link].findall('collision')) for link in pair)}
    declaration['covers_adjacent_pairs'] = bool(declared) and declared >= collidable
    # A partial declaration is legitimate only when the engine can exclude exactly the declared pairs.
    # A URDF cannot filter individual pairs, which is why a complete set is normally required - but a
    # package whose overlaps live in its authored visuals has no complete set to declare, and no
    # amount of collision-mesh preparation removes them. Such a package ships an engine model whose
    # <contact><exclude> set is the declaration itself, so both the private queries and the engine
    # ignore exactly the declared pairs. See engine_model.engine_mjcf.
    engine_model_required = bool(declared) and (bool(non_adjacent) or not declaration['covers_adjacent_pairs'])
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
    candidates = []
    for leaf in leaves:
        joints = [j["name"] for j in model.chain(leaf) if j["type"] != "fixed" and not j["mimic"]]
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
    if not chosen:
        raise ValueError("No actuated chain found in this URDF")
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
    root_transform[:3, 3] = [-.25, 0., .76 - minimum_z + .001]
    initial = {}
    initial_file = source.with_name("initial_state.json")
    if initial_file.is_file():
        initial = json.loads(initial_file.read_text())
    home = {j["name"]: float(initial.get(j["name"], 0. if j["lower"] is None else np.clip(0., j["lower"], j["upper"])))
            for j in model.active}
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
        warnings.append('Declared adjacent-pair overlaps are partial (%d of %d): the private queries '
                        'exclude them and the package ships an engine model that excludes exactly the '
                        'same pairs, so collision stays on for the undeclared rest.'
                        % (len(declared), declaration['adjacent_pair_count']))
    if len(candidates) > 2:
        warnings.append("Longest distinct chains selected; all candidate chains are recorded for configuration.")
    profile = {"id": robot_id, "name": model.name, "builtin": False, "source": str(source), "source_sha256": digest,
               "urdf": str(normalized), "directory": str(target), "root_link": model.root, "links": len(model.links),
               "joints": model.active, "chains": chains, "all_chains": candidates, "home": home,
               "root_transform": root_transform.tolist(), "capabilities": ["motion", "joint_control", "ik"],
               "profile_version": PROFILE_VERSION, "asset_sha256": asset_signature(source),
               "control_policy_sha256":policy_signature(),"control_policy":IMPORTED_ROBOT_POLICY,
               "grasp_calibrated": False, "warnings": warnings, "mesh_count": len(set(meshes))}
    if actuator_configuration:
        profile['actuator_configuration']=actuator_configuration
        profile['warnings']=["Source masses, geometry and joint frames are preserved. Explicit actuators.json supplies motor effort and no-load speed limits."]
    scene = {"robot_id": robot_id, "openarm_cell": False, "table_top": .76, "table_size": [1.6, 1.4, .05],
             "robot_root_transform": root_transform.tolist(), "physics_dt": IMPORTED_ROBOT_POLICY['physics_dt_max_s'],
             "physics_substeps":IMPORTED_ROBOT_POLICY['physics_substeps'],
             "robot_gravity_compensation": 1., "self_collision": True,
             # URDF cannot filter individual pairs, so the engine expresses only a declaration
             # that covers every adjacent pair (see collision_declaration). A package that ships an
             # engine model instead relies on that model's own exclusions, which can name any pair.
             "adjacent_collision": bool(engine_model_required) or not declaration['covers_adjacent_pairs'],
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
    # The private model reads the declaration from the profile, so it must be recorded before the
    # engine model is built from it.
    profile['collision_config'] = declaration
    if engine_model_required:
        # Both the private queries and the engine must ignore the same pairs; the URDF cannot express
        # them, so the package carries an MJCF engine model built from the private collision model.
        from backend.app.robots.collision_planner import PrivateCollisionQueries
        from backend.app.robots.engine_model import engine_mjcf
        queries = PrivateCollisionQueries(profile, scene['table_top'])
        xml, restored_ranges = engine_mjcf(queries.xml, normalized)
        engine_file = target / 'engine_model.xml'
        engine_file.write_text(xml + '\n')
        profile['engine_model'] = {'morph': 'mjcf', 'file': str(engine_file),
                                   'excluded_adjacent_pairs': declaration['runtime_excluded_adjacent_pairs'],
                                   'restored_joint_ranges': restored_ranges}
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
    spawn = ReachableDeskSpawns(profile, scene).layout()
    if not spawn['objects'] and profile['grippers']:
        # An object shorter than the arm can descend above the desk is unpickable: no position is
        # reachable, so the layout omits every prop. Measure the height this arm needs and grow the
        # object to it while it still fits the jaws - a robot whose inferred cube already works never
        # reaches this branch, so nothing moves for the robots that spawn today. Measured 2026-10-24:
        # UR5e + 2F-85 places both props at an 84 mm cube and none at the inferred 33.6 mm.
        # See ReachableDeskSpawns.minimum_pinch_height.
        inferred_cube = scene['cube_size']
        side = min(profile['grippers'], key=lambda name: profile['grippers'][name]['cube_size'])
        gripper = profile['grippers'][side]
        needed = ReachableDeskSpawns(profile, scene).minimum_pinch_height(side)
        limit = float(gripper['maximum_aperture']) - .002

        def use_cube(size):
            scene['cube_size'] = float(size)
            scene['cube_mass'] = .05 * (scene['cube_size'] / .052) ** 3
            scene['physics_dt'] = min(IMPORTED_ROBOT_POLICY['physics_dt_max_s'],
                max(IMPORTED_ROBOT_POLICY['physics_dt_min_s'],
                    scene['cube_size'] * IMPORTED_ROBOT_POLICY['physics_dt_per_cube_meter']))
            profile['cube_size'] = scene['cube_size']
            for entry in profile['grippers'].values():
                entry['cube_size'] = scene['cube_size']

        # A height that admits some clean posture need not admit a whole layout - placing the second
        # prop also has to be reachable beside the first - so the layout itself decides, and the
        # largest object the jaws open to is the last resort (measured: 84 mm places both props for
        # UR5e + 2F-85 where 60 and 70 mm place one).
        taller = None
        for candidate in ([limit] if needed is None else [min(2. * needed, limit), limit]):
            if candidate <= inferred_cube + .0005:
                continue
            use_cube(candidate)
            attempt = ReachableDeskSpawns(profile, scene).layout()
            if not attempt['omitted']:
                spawn, taller = attempt, candidate
                break
        if taller is None:
            use_cube(inferred_cube)
            warnings.append('No prop can be pinched on this desk: this arm cannot reach an object centre '
                            'below %s above the surface, and the jaws open to only %.1f mm.'
                            % ((('%.1f mm' % (2. * needed * 1000.)) if needed is not None else 'the ladder'),
                               float(gripper['maximum_aperture']) * 1000.))
        else:
            warnings.append('The inferred cube (%.1f mm) is shorter than this arm can descend above the '
                            'desk, so no desk position was reachable; the scene uses %.1f mm%s.'
                            % (inferred_cube * 1000., scene['cube_size'] * 1000.,
                               (' (the height it needs is %.1f mm)' % (2. * needed * 1000.)
                                if needed is not None else ', the largest the jaws hold')))
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
         **({'engine_morph': profile['engine_model']['morph']} if profile.get('engine_model') else {}),
         'chains': chains, 'grippers': profile['grippers'], 'cube_size': scene['cube_size'],
         'home': home, 'root': profile['root_transform']}, sort_keys=True).encode()).hexdigest()
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
