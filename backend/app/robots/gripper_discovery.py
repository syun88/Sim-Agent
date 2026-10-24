"""Infer opposed jaws and a cube-sized pinch frame from authored URDF surfaces.

All sweeps evaluate private FK. Geometry inference proposes a grasp; actual finger
contacts and a motor-driven lift must still establish it in Genesis.
"""
import numpy as np
from scipy.spatial.transform import Rotation

from backend.app.robots.urdf import mesh_path, origin, vector


class Geometry:
    def __init__(self, model):
        self.model, self.cache = model, {}

    def points(self, link):
        import trimesh
        if link in self.cache:
            return self.cache[link]
        parts = []
        body = self.model.links[link]
        # Uniform surface samples keep the inferred size independent of how a
        # collision exporter subdivides the same finger into convex pieces.
        for item in body.findall('visual') or body.findall('collision'):
            shape = item.find('geometry')
            mesh, box, cylinder, sphere = [shape.find(name) for name in ('mesh', 'box', 'cylinder', 'sphere')]
            if mesh is not None:
                geometry = trimesh.load(mesh_path(mesh.get('filename'), self.model.path), force='mesh', process=False)
                geometry.vertices *= vector(mesh.get('scale'), [1., 1., 1.])
            elif box is not None:
                geometry = trimesh.creation.box(extents=vector(box.get('size'), [0., 0., 0.]))
                geometry = geometry.subdivide().subdivide().subdivide()
            elif cylinder is not None:
                geometry = trimesh.creation.cylinder(radius=float(cylinder.get('radius')), height=float(cylinder.get('length')))
            elif sphere is not None:
                geometry = trimesh.creation.icosphere(radius=float(sphere.get('radius')), subdivisions=2)
            else:
                continue
            geometry.apply_transform(origin(item.find('origin')))
            parts.append(geometry)
        surface = trimesh.util.concatenate(parts) if parts else None
        self.cache[link] = trimesh.sample.sample_surface(surface, 6000, seed=43)[0] if parts else np.empty((0, 3))
        return self.cache[link]

    def in_frame(self, links, poses, reference):
        arrays = []
        for link in links:
            points = self.points(link)
            t = reference @ poses[link]
            arrays.append(points @ t[:3, :3].T + t[:3, 3])
        return np.vstack(arrays) if arrays else np.empty((0, 3))


def _unit(v):
    return np.asarray(v) / max(np.linalg.norm(v), 1e-12)


def _coupled_fingers(model, geometry, joint):
    """Two pads a single motor drives through mimic joints, as in a finger-linkage gripper.

    A Robotiq 2F-85 declares one active joint and five mimics: the driver plus the knuckle and
    finger joints of both fingers, the finger joints at multiplier -1 (that is the parallelogram
    closure, so the mimics express it exactly). The sibling rule cannot see this - the opposing
    finger hangs off the palm as a *mimic*, which it skips by design - and neither finger *body*
    is a pad: the pads are the deepest driven link of each branch (measured on the 2F-85: pad
    centres 0.1mm either side of the hinge axis, 22mm wide, closest approach closing 86.0mm ->
    1.5mm monotonically, i.e. a parallel-jaw aperture rather than a pair of linkage plates).
    """
    driver = joint['name']
    followers, links = {}, set()
    for other in model.joints:
        if other['type'] == 'fixed' or other['name'] == driver:
            continue
        name, multiplier, _ = model.joint_driver(other['name'])
        if name == driver:
            followers[other['name']] = float(multiplier)
            links.add(other['child'])
    if len(followers) < 2:
        return None
    own, opposite = {joint['child']}, set()
    for link in links:
        ancestors = {entry['child'] for entry in model.chain(link)}
        (own if joint['child'] in ancestors or link == joint['child'] else opposite).add(link)
    if not own or not opposite:
        return None
    fixed_children = {}
    for other in model.joints:
        if other['type'] == 'fixed':
            fixed_children.setdefault(other['parent'], []).append(other['child'])
    def pad(branch):
        """The deepest link of the branch that carries collision surfaces of its own.

        Fixed children count: a linkage finger's pad is mounted on a fixed joint (2F-85:
        ``*_inner_finger`` --fixed--> ``*_inner_finger_pad``), so the driven links alone stop one
        link short of the surface that actually pinches.
        """
        links = [link for link in branch] + [child for link in branch for child in fixed_children.get(link, [])]
        for link in sorted(set(links), key=lambda name: -len(model.chain(name))):
            try:
                if len(geometry.points(link)) >= 12:
                    return link
            except ValueError:
                continue
        return None
    own_pad, opposite_pad = pad(own), pad(opposite)
    if own_pad is None or opposite_pad is None:
        return None
    return [own_pad], [opposite_pad], followers


def _candidate(model, geometry, profile, joint, cube_size=None):
    parent, child = joint['parent'], joint['child']
    followers = {}
    siblings = [j for j in model.joints if j['parent'] == parent and j['name'] != joint['name'] and j['type'] != 'fixed']
    opposite = next((j for j in siblings if j['type'] == joint['type']), None)
    if opposite and not opposite['mimic']:
        dot = np.dot(joint['axis'], opposite['axis'])
        if abs(dot) > .9:
            followers[opposite['name']] = -1. if dot > 0 else 1.
    # The sibling rule first, so a gripper whose opposing finger is a plain joint is inferred
    # exactly as before; the coupled rule is only reached when that produced no jaws.
    groups = [([child], [opposite['child']] if opposite else [parent], followers, 'sibling')]
    coupled = _coupled_fingers(model, geometry, joint)
    if coupled is not None and (coupled[0], coupled[1]) != (groups[0][0], groups[0][1]):
        groups.append((*coupled, 'coupled'))
    failures = []
    for moving_links, fixed_links, followers, rule in groups:
        try:
            return _candidate_for_fingers(model, geometry, profile, joint, cube_size,
                                          moving_links, fixed_links, followers, rule)
        except ValueError as error:
            failures.append(str(error))
    raise ValueError(failures[0] if failures else 'No opposed finger')


def _candidate_for_fingers(model, geometry, profile, joint, cube_size, moving_links, fixed_links, followers, rule):
    parent, child = joint['parent'], joint['child']
    home_poses, _ = model.forward(profile['home'])
    reference = np.linalg.inv(home_poses[parent])
    pivot = (reference @ home_poses[child])[:3, 3]
    axis = _unit((reference @ home_poses[child])[:3, :3] @ joint['axis'])
    moving = geometry.in_frame(moving_links, home_poses, reference) - pivot
    fixed = geometry.in_frame(fixed_links, home_poses, reference) - pivot
    if min(len(moving), len(fixed)) < 12:
        raise ValueError('No collision surfaces on both prospective jaws')
    radial = moving - np.outer(moving @ axis, axis)
    lengths = np.linalg.norm(radial, axis=1)
    forward = _unit(radial[lengths >= lengths.max() * .90].mean(axis=0))
    if rule == 'coupled':
        # A linkage finger is not a straight jaw: its farthest points from the hinge lie along the
        # finger, which is not perpendicular to the closing direction. Measured on the 2F-85, the
        # cross product points diagonally and reports a gap of -103.7mm where the pads' faces are
        # 98.7mm apart; the pads face each other along the line joining their centroids.
        normal = _unit(moving.mean(axis=0) - fixed.mean(axis=0))
        normal = _unit(normal - axis * float(normal @ axis))
    else:
        normal = axis if joint['type'] == 'prismatic' else _unit(np.cross(forward, axis))
    if np.median(moving @ normal) < np.median(fixed @ normal):
        normal = -normal
    # In a sliding gripper the third axis is the pad height; in a rotary
    # gripper it is the hinge axis. No name-specific finger mapping is used.
    height_axis = _unit(np.cross(forward, normal)) if joint['type'] == 'prismatic' else axis
    length = float(np.quantile(moving @ forward, .999))
    if length < .015 or np.max(fixed @ forward) < length * .55:
        raise ValueError('The parent/sibling has no opposed distal finger')
    home = profile['home'][joint['name']]
    follower_limits_inconsistent = False
    low = joint['lower'] if joint['lower'] is not None else home - .95
    high = joint['upper'] if joint['upper'] is not None else home + .95
    for follower, multiplier in followers.items():
        spec = model.by_name[follower]
        if spec['lower'] is not None:
            bounds = sorted([spec['lower'] / multiplier, spec['upper'] / multiplier])
            low, high = max(low, bounds[0]), min(high, bounds[1])
    if low >= high:
        # A mimic joint's own <limit> is often a copy of its driver's and then contradicts the
        # mimic relation: the 2F-85's inner fingers mimic the driver at -1 while declaring the
        # driver's [0, 0.8] range, so their projected interval is empty and the gripper used to be
        # refused outright. Keep the driver's own range, which the relation drives, and record that
        # the follower limits were unusable rather than hiding it.
        low = joint['lower'] if joint['lower'] is not None else home - .95
        high = joint['upper'] if joint['upper'] is not None else home + .95
        if low >= high:
            raise ValueError('The coupled fingers have no shared command interval')
        follower_limits_inconsistent = True
    samples = []
    for q in np.linspace(low, high, 49):
        values = {**profile['home'], joint['name']: float(q), **{name: float(q * k) for name, k in followers.items()}}
        poses, _ = model.forward(values)
        samples.append((float(q), geometry.in_frame(fixed_links, poses, reference) - pivot,
                        geometry.in_frame(moving_links, poses, reference) - pivot))
    provisional_edge = cube_size if cube_size is not None else min(.052, length * .35)

    def sweep(edge_cap):
        """Try every pad depth at one cube-size cap.

        Also reports the smallest cube the measured closure can squeeze
        (``minimum <= 0.8 * edge``), so a cap can be told apart from a closure.
        """
        choices, required = [], None
        for depth_fraction in (.70, .78, .85, .90):
            depth = length * depth_fraction
            rows = []
            for q, a, b in samples:
                a = a[abs(a @ forward - depth) < edge_cap * .5 + .001]
                b = b[abs(b @ forward - depth) < edge_cap * .5 + .001]
                if min(len(a), len(b)) < 8:
                    continue
                lo = max(np.quantile(a @ height_axis, .05), np.quantile(b @ height_axis, .05))
                hi = min(np.quantile(a @ height_axis, .95), np.quantile(b @ height_axis, .95))
                face_a, face_b = float(np.max(a @ normal)), float(np.min(b @ normal))
                if hi - lo < .004:
                    continue
                rows.append({'q': q, 'gap': face_b - face_a, 'face_a': face_a, 'face_b': face_b,
                             'height': float((lo + hi) / 2.), 'pad_height': float(hi - lo)})
            if len(rows) < 8:
                continue
            maximum = max(row['gap'] for row in rows)
            minimum = min(row['gap'] for row in rows)
            if not .012 < maximum < .20 or maximum - minimum < .012:
                continue
            edge = min(edge_cap, maximum * .45, max(row['pad_height'] for row in rows) * 3.5)
            required = max(required or 0., minimum / .8)
            if cube_size is not None:
                if cube_size > edge + .0001:
                    continue
                edge = cube_size
            if edge < .008 or minimum > edge * .8:
                continue
            holding = min(rows, key=lambda row: abs(row['gap'] - edge - .0007))
            opened = min((row for row in rows if row['gap'] >= edge + .012),
                         key=lambda row: -row['gap'], default=None)
            closed = min(rows, key=lambda row: abs(row['gap'] - edge * .9))
            if opened is None:
                continue
            offset = pivot + forward * depth + normal * ((opened['face_a'] + opened['face_b']) / 2.) + height_axis * holding['height']
            frame = np.column_stack([forward, _unit(np.cross(height_axis, forward)), height_axis])
            up_sign = 1. if (np.asarray(profile['root_transform'])[:3, :3] @ height_axis)[2] >= 0 else -1.
            world_frame = np.column_stack([[1., 0., 0.], [0., up_sign, 0.], [0., 0., up_sign]])
            base_rotation = world_frame @ frame.T
            for pitch in np.deg2rad([0., 15., 30., 45., 60., 75., 90.]):
                rotation = Rotation.from_rotvec([0., pitch, 0.]).as_matrix() @ base_rotation
                minimum_z = float('inf')
                for row in (holding, opened):
                    closest = min(samples, key=lambda entry: abs(entry[0] - row['q']))
                    points = np.vstack(closest[1:]) + pivot - offset
                    minimum_z = min(minimum_z, float((points @ rotation.T)[:, 2].min()))
                clearance = edge / 2. + minimum_z
                if clearance >= .003:
                    choices.append({'joint': joint['name'], 'followers': followers, 'reference_link': parent,
                        'contact_groups': [fixed_links, moving_links], 'tool_offset': offset.tolist(),
                        'pinch_offset': (normal * ((holding['face_a'] + holding['face_b'] - opened['face_a'] - opened['face_b']) / 2.)).tolist(),
                        'grasp_rotation': rotation.tolist(), 'approach_axis': (rotation @ forward).tolist(),
                        'open_q': opened['q'], 'closed_q': closed['q'], 'holding_q': holding['q'],
                        'cube_size': edge, 'maximum_aperture': maximum, 'minimum_aperture': minimum,
                        'pad_height': holding['pad_height'], 'finger_length': length,
                        'pad_region': {'pivot': pivot.tolist(), 'forward': forward.tolist(), 'normal': normal.tolist(),
                            'height_axis': height_axis.tolist(), 'depth_min': length * .45,
                            'depth_max': length * 1.10, 'height_center': holding['height'],
                            'height_half_extent': holding['pad_height'] * .5 + edge * .25},
                        'table_clearance': clearance, 'pitch_degrees': float(np.rad2deg(pitch)),
                        'type': 'sliding_jaws' if joint['type'] == 'prismatic' else 'rotary_jaw',
                        'sweep': rows, 'status': 'geometry_inferred',
                        **({'follower_limits_inconsistent': True} if follower_limits_inconsistent else {}),
                        'verification': 'two opposed contacts and lift-follow required during execution'})
                    break
        return choices, required

    choices, required = sweep(provisional_edge)
    if not choices and cube_size is None and required and required > provisional_edge:
        # A gripper can close wider than its finger length suggests: OMX closes to
        # 20mm while the length heuristic caps the cube at 22.7mm, so the squeeze
        # rule needs 25mm and every candidate was refused. Retry once with the cube
        # floor the measured closure requires; the aperture and pad caps still bind,
        # and a first-pass candidate is never replaced.
        choices, _ = sweep(min(required * 1.02, .20))
    if not choices:
        raise ValueError('No opposed aperture, object size and desk-clear grasp pose found')
    return max(choices, key=lambda candidate: (candidate['cube_size'], candidate['table_clearance']))


def discover_grippers(model, profile):
    geometry, grippers, failures = Geometry(model), {}, {}
    for side, chain in profile['chains'].items():
        # The last three *actuated* joints, not the last three joints: a linkage gripper ends in
        # fixed and mimic joints (2F-85: a fixed pad joint under a mimic finger joint), so counting
        # positions rather than actuators never reaches the driver.
        distal = [j for j in reversed(model.chain(chain['end_link'])) if j['type'] != 'fixed' and not j['mimic']]
        for joint in distal[:3]:
            try:
                gripper = _candidate(model, geometry, profile, joint)
                arm = [j['name'] for j in model.chain(gripper['reference_link']) if j['type'] != 'fixed' and not j['mimic']]
                if len(arm) < 3:
                    continue
                grippers[side] = gripper
                break
            except ValueError as error:
                failures.setdefault(side, []).append({'joint': joint['name'], 'reason': str(error)})
    if grippers:
        common_size = min(g['cube_size'] for g in grippers.values())
        for side, g in list(grippers.items()):
            grippers[side] = _candidate(model, geometry, profile, model.by_name[g['joint']], common_size)
    return {'grippers': grippers, 'failures': failures, 'method': 'Uniform authored URDF surfaces and private aperture sweeps',
            'physical_grasp_verified': False}
