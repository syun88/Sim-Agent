"""Physical pinch evidence independent of robot names and simulation wrappers."""
import numpy as np
from backend.app.robots.policy import IMPORTED_ROBOT_POLICY as POLICY


def evaluate_pinch_contacts(contacts, gripper, cube_size, mass,*,surface_tolerance_m=None):
    tolerance = max(POLICY['surface_floor_m'],cube_size*POLICY['surface_cube_fraction']) if surface_tolerance_m is None else surface_tolerance_m
    penetration_limit = max(POLICY['penetration_floor_m'],cube_size*POLICY['penetration_cube_fraction'])
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


class PinchContactTracker:
    """Schmitt trigger for contact-point/visual-surface numerical uncertainty.

    Entry always uses the strict surface test. Only an already established
    pinch gets a small exit band; force, opposition, pad and penetration tests
    remain unchanged. Opening or losing either loaded contact clears the latch.
    """
    def __init__(self):self._latched={}

    def reset(self,side=None):
        self._latched={key:value for key,value in self._latched.items() if side is not None and key[0]!=side}

    def evaluate(self,side,obj,contacts,gripper,cube_size,mass):
        strict=evaluate_pinch_contacts(contacts,gripper,cube_size,mass)
        result=strict;key=(side,obj)
        if not strict['established'] and self._latched.get(key,False):
            tolerance=strict['surface_tolerance_m']
            result=evaluate_pinch_contacts(contacts,gripper,cube_size,mass,
                surface_tolerance_m=min(tolerance*POLICY['surface_exit_scale'],tolerance+POLICY['surface_exit_extra_m']))
        result['strict_established']=strict['established']
        result['surface_hysteresis']=bool(result['established'] and not strict['established'])
        result['enter_surface_tolerance_m']=strict['surface_tolerance_m']
        self._latched[key]=result['established']
        return result
