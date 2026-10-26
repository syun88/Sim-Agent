"""Turn the private collision model into a model the physics engine can load as a robot.

The private model is built for collision queries: it carries a placeholder inertial on every body,
drops the URDF's joint ranges, includes the desk it was fitted against, and bakes its own query-time
placement into the root body. Shaping it into an engine model is what lets a package whose adjacent
overlaps live in its authored visuals - where no amount of collision-mesh preparation removes them -
express a per-pair exclusion that a URDF cannot.

Inertials are taken from the URDF where it has them and otherwise derived the way the engine derives
them itself (geometry volume x 1500 kg/m3), with the URDF's mesh scale as the authority on units.
"""
import pathlib
import xml.etree.ElementTree as ET

import numpy as np
import trimesh

from backend.app.robots.urdf import mesh_path

DENSITY = 1500.0
# A link whose frames carry no geometry still needs a positive mass and inertia; the engine gives
# such links a negligible mass of its own, and anything this small is equally irrelevant.
FRAME_MASS = 1e-6
FRAME_INERTIA = 1e-12


def _full_inertia(matrix):
    """MJCF takes six numbers in ixx iyy izz ixy ixz iyz order; trimesh gives a 3x3 tensor."""
    tensor = np.asarray(matrix, dtype=float)
    if tensor.shape != (3, 3):
        tensor = np.diag(tensor.ravel()[:3])
    return np.array([tensor[0, 0], tensor[1, 1], tensor[2, 2], tensor[0, 1], tensor[0, 2], tensor[1, 2]])


def _urdf_inertial(link):
    inertial = link.find('inertial')
    if inertial is None:
        return None
    inertia = inertial.find('inertia')
    mass = inertial.find('mass')
    if inertia is None or mass is None:
        return None
    origin = inertial.find('origin')
    position = np.array([float(v) for v in (origin.get('xyz', '0 0 0') if origin is not None else '0 0 0').split()])
    # The URDF lists ixx ixy ixz iyy iyz izz; MJCF wants ixx iyy izz ixy ixz iyz.
    tensor = np.array([float(inertia.get(key, 0.0)) for key in ('ixx', 'iyy', 'izz', 'ixy', 'ixz', 'iyz')])
    return position, float(mass.get('value')), tensor


def _derived_inertial(link, source):
    element = link.find('collision')
    if element is None:
        element = link.find('visual')
    geometry = element.find('geometry') if element is not None else None
    if geometry is None:
        return None
    mesh_element, box_element = geometry.find('mesh'), geometry.find('box')
    if mesh_element is not None:
        try:
            path = mesh_path(mesh_element.get('filename'), source)
        except Exception:
            return None
        if not pathlib.Path(path).is_file():
            return None
        parts = [float(value) for value in (mesh_element.get('scale') or '1').split()]
        scale = float(np.mean(parts)) if parts else 1.0
        volume_scale = float(np.prod(parts)) if parts else 1.0
        mesh = trimesh.load(str(path), force='mesh')
        volume = abs(float(mesh.volume)) * volume_scale
        if volume <= 0:
            return None
        mass = volume * DENSITY
        if mesh.is_volume:
            tensor = _full_inertia(np.asarray(mesh.moment_inertia) * volume_scale * scale ** 2 * DENSITY)
            position = np.asarray(mesh.center_mass) * scale
        else:
            # A mesh without a closed volume yields no inertia; its bounding box never lies about
            # being positive, and these parts are small enough for that to be a fair approximation.
            size = (mesh.bounds[1] - mesh.bounds[0]) * scale
            diagonal = np.array([size[1] ** 2 + size[2] ** 2, size[0] ** 2 + size[2] ** 2,
                                 size[0] ** 2 + size[1] ** 2]) * mass / 12.0
            tensor = np.concatenate([diagonal, np.zeros(3)])
            position = np.zeros(3)
        return position, mass, tensor
    if box_element is not None:
        size = np.array([float(value) for value in box_element.get('size').split()])
        if np.any(size <= 0):
            return None
        mass = float(np.prod(size)) * DENSITY
        diagonal = np.array([size[1] ** 2 + size[2] ** 2, size[0] ** 2 + size[2] ** 2,
                             size[0] ** 2 + size[1] ** 2]) * mass / 12.0
        return np.zeros(3), mass, np.concatenate([diagonal, np.zeros(3)])
    return None


def engine_mjcf(private_xml, source, *, root_body='robot_world'):
    """Shape a private collision model into an engine model and return its XML text.

    ``source`` is the normalized URDF the private model was built from; it supplies the joint ranges
    and the inertials the private model does not record.
    """
    source = pathlib.Path(source)
    links = {link.get('name'): link for link in ET.fromstring(source.read_text()).findall('link')}
    tree = ET.fromstring(private_xml)
    for geometry in list(tree.iter('geom')):
        if geometry.get('name') == 'desk':
            for parent in tree.iter():
                if geometry in list(parent):
                    parent.remove(geometry)
    ranges = {}
    for joint in ET.fromstring(source.read_text()).iter('joint'):
        limit = joint.find('limit')
        if limit is not None and limit.get('lower') is not None:
            ranges[joint.get('name')] = (float(limit.get('lower')), float(limit.get('upper')))
    seen_root, restored = False, 0
    for body in tree.iter('body'):
        name = body.get('name')
        for existing in body.findall('inertial'):
            body.remove(existing)
        data = _derived_inertial(links[name], source) if name in links else None
        if name in links:
            data = _urdf_inertial(links[name]) or data
        if data is None:
            # Diagonal only: equal off-diagonal values would make an eigenvalue zero, which MJCF rejects.
            position, mass, tensor = np.zeros(3), FRAME_MASS, np.array([FRAME_INERTIA] * 3 + [0.0] * 3)
        else:
            position, mass, tensor = data
        body.insert(0, ET.Element('inertial', {'pos': ' '.join(f'{v:.12g}' for v in position),
                                               'mass': f'{mass:.10g}',
                                               'fullinertia': ' '.join(f'{v:.12g}' for v in tensor)}))
        for joint in body.findall('joint'):
            if joint.get('name') in ranges:
                lower, upper = ranges[joint.get('name')]
                joint.set('limited', 'true')
                joint.set('range', f'{lower:.12g} {upper:.12g}')
                restored += 1
        if name == root_body and not seen_root:
            # The private model bakes its own query-time placement in; normalizing the root lets the
            # scene's placement (the profile's root transform) land the robot where the URDF path does.
            body.set('pos', '0 0 0')
            body.set('quat', '1 0 0 0')
            seen_root = True
    return ET.tostring(tree, encoding='unicode'), restored
