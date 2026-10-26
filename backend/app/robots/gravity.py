"""Analytic gravity torques for an imported URDF arm (planning only).

The Genesis entity a desk import runs in cancels gravity for the whole
robot (material gravity_compensation=1), so a posture the planner likes is
a posture the motors hold for free. A mobile base cannot cancel gravity -
it must rest on its wheels - and its arm servos are small: on LeKiwi every
joint's peak effort is 1.91 N*m while a far-reaching grasp posture loads
the shoulder lift with -6.65 N*m of gravity torque (ratio 3.5). The
planner's IK + collision + closure gates all pass such a posture, and the
execution then sags to the floor (measured: tool tip at 0.006 m,
MOTION_BLOCKED with 0.169 m error). Seed selection therefore scores every
candidate posture by its gravity-torque margin, from the same URDF
inertials the engine integrates (the physics URDF is spawned with
recompute_inertia=False).

The computation is the standard Jacobian-transpose sum over the links in
each joint's subtree, evaluated on URDFModel.forward - no engine state is
read or written, so it is safe at planning time, including against the
hypothetical root transforms of the base-placement search. The base is
assumed upright on its floor plane (the only placements the search
generates), so gravity in the robot's root frame is -z regardless of the
base's yaw or position.

Validated against the project's engine measurements on LeKiwi: all-zero
posture elbow torque -3.517 N*m (engine: -3.52) and the tucked home's
worst ratio 0.396 (grid search: 0.39).
"""

from __future__ import annotations

import numpy as np

GRAVITY = np.array([0., 0., -9.81])

#: A seed whose worst joint gravity torque exceeds this fraction of the
#: joint's peak effort is treated as unholdable for planning preference:
#: the servo must spend that fraction of its peak just standing still,
#: before any motion, contact or payload. Seeds above the limit are not
#: forbidden - when nothing better exists the best-margin one is used and
#: the excess is surfaced in the result evidence - but they are only
#: planned (closure, corridor) after every in-limit candidate has failed.
GRAVITY_TORQUE_RATIO_LIMIT = .8


def _link_mass_properties(model):
    """{link name: (mass kg, COM in the link frame)} from URDF inertials."""
    cached = getattr(model, "_gravity_mass_cache", None)
    if cached is not None:
        return cached
    from backend.app.robots.urdf import vector
    properties = {}
    for name, element in model.links.items():
        inertial = element.find('inertial')
        if inertial is None:
            continue
        mass = float(inertial.find('mass').get('value'))
        if mass <= 0.:
            continue
        origin = inertial.find('origin')
        com = vector(origin.get('xyz'), [0., 0., 0.]) if origin is not None else np.zeros(3)
        properties[name] = (mass, com)
    model._gravity_mass_cache = properties
    return properties


def gravity_torques(model, joint_names, values, gravity=GRAVITY):
    """Generalized gravity force (N*m) at each of ``joint_names``.

    ``values`` is a full joint-value dict for the model (missing joints
    default to 0, as in URDFModel.forward). A joint is loaded by every
    link in its subtree; the returned vector is the torque gravity exerts
    about each joint, so a holding motor must apply its negation, and the
    *magnitude* against the joint's effort limit is the hold margin.
    """
    properties = _link_mass_properties(model)
    poses, axes = model.forward(values)
    g = np.asarray(gravity, dtype=float)
    torques = np.zeros(len(joint_names))
    chains = {name: [joint['name'] for joint in model.chain(name)] for name in properties}
    for link, (mass, com_local) in properties.items():
        com_world = poses[link][:3, :3] @ com_local + poses[link][:3, 3]
        force = mass * g
        for index, name in enumerate(joint_names):
            if name not in chains[link]:
                continue
            origin, axis = axes[name]
            if model.by_name[name]['type'] == 'prismatic':
                torques[index] += axis @ force
            else:
                torques[index] += axis @ np.cross(com_world - origin, force)
    return torques


def gravity_report(profile, model, side, q, snapshot=None):
    """Gravity-torque margin of one arm posture, against the effort limits.

    Returns {'torques_Nm', 'ratios', 'worst_ratio', 'worst_joint',
    'over_limit'} with the limit GRAVITY_TORQUE_RATIO_LIMIT, or None when
    the profile carries no effort data to score against (callers then
    treat the posture as unscored, never as in-limit by default).
    """
    chain = profile['chains'][side]['joints']
    efforts = []
    specs = {joint['name']: joint for joint in profile.get('joints', [])}
    for name in chain:
        effort = specs.get(name, {}).get('effort') or model.by_name[name]['effort']
        if not effort or effort <= 0.:
            return None
        efforts.append(float(effort))
    values = dict(snapshot) if snapshot is not None else dict(profile.get('home') or {})
    values.update(dict(zip(chain, np.asarray(q, dtype=float))))
    torques = gravity_torques(model, chain, values)
    ratios = np.abs(torques) / np.asarray(efforts)
    worst = int(np.argmax(ratios))
    return {'torques_Nm': torques.tolist(), 'ratios': ratios.tolist(),
            'worst_ratio': float(ratios[worst]), 'worst_joint': chain[worst],
            'limit': GRAVITY_TORQUE_RATIO_LIMIT,
            'over_limit': bool(ratios[worst] > GRAVITY_TORQUE_RATIO_LIMIT)}
