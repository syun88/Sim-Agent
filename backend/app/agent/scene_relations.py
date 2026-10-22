"""Compact contact-backed support observations for task and manipulation planning.

These relations describe the measured scene. They do not assert that a stack
will survive lifting or that its upper members are pinched by a gripper.
"""
from __future__ import annotations

import numpy as np


def support_relations(objects: list[dict], contacts: list[str]) -> list[dict]:
    touching = {frozenset(pair.split('|')) for pair in contacts}
    cubes = [item for item in objects if item.get('type') == 'cube' and item.get('bounds') is not None]
    relations = []
    for upper in cubes:
        a = np.asarray(upper['bounds'],float)
        for lower in cubes:
            if lower['id'] == upper['id'] or frozenset([upper['id'],lower['id']]) not in touching:
                continue
            b = np.asarray(lower['bounds'],float)
            gap = float(a[0,2]-b[1,2])
            widths = np.maximum(0.,np.minimum(a[1,:2],b[1,:2])-np.maximum(a[0,:2],b[0,:2]))
            coverage = float(np.prod(widths)/max(np.prod(a[1,:2]-a[0,:2]),1e-12))
            # Side contact and bounding-box alignment without contact are both
            # insufficient evidence of vertical support.
            if -.003 <= gap <= .004 and coverage >= .35:
                relations.append({'subject':upper['id'],'relation':'on','object':lower['id'],
                    'horizontal_coverage':coverage,'surface_gap_m':gap,'method':'engine_contacts_and_bounds'})
    return relations


def support_anchor(members: list[str], relations: list[dict]) -> str | None:
    """One connected support tree needs one lowest object to pinch and lift."""
    wanted = set(members)
    if len(wanted) != len(members) or not wanted:
        return None
    parents = {item:set() for item in wanted}
    for relation in relations:
        if relation.get('relation') == 'on' and {relation['subject'],relation['object']} <= wanted:
            parents[relation['subject']].add(relation['object'])
    roots = [item for item,below in parents.items() if not below]
    if len(roots) != 1:return None
    root = roots[0]
    reached = {root}
    while True:
        extra = {item for item,below in parents.items() if below and below <= reached}
        if extra <= reached:break
        reached |= extra
    return root if reached == wanted else None
