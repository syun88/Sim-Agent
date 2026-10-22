"""Physical pinch evidence independent of robot names and simulation wrappers."""
import numpy as np


def evaluate_pinch_contacts(contacts, gripper, cube_size, mass):
    tolerance = max(.0006, cube_size * .025)
    penetration_limit = max(.001, cube_size * .05)
    groups = []
    for links in gripper['contact_groups']:
        accepted = [c for c in contacts if c['robot_link'] in links and c['on_distal_pad']
                    and c['surface_error_m'] is not None and c['surface_error_m'] <= tolerance
                    and c['penetration_m'] <= penetration_limit
                    and c['normal_force_N'] >= max(.002, mass * 9.81 * .01)]
        force = sum(c['normal_force_N'] for c in accepted)
        normal = sum((np.asarray(c['normal']) * c['normal_force_N'] for c in accepted), np.zeros(3))
        normal /= max(np.linalg.norm(normal), 1e-12)
        groups.append({'links': links, 'contacts': accepted, 'normal_force_N': force, 'normal': normal.tolist()})
    opposed = float(np.dot(groups[0]['normal'], groups[1]['normal'])) if len(groups) == 2 else 1.
    return {'established': len(groups) == 2 and all(g['contacts'] for g in groups) and opposed < -.4,
            'normal_dot': opposed, 'groups': groups, 'surface_tolerance_m': tolerance,
            'penetration_limit_m': penetration_limit, 'candidate_contacts': contacts}
