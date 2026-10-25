"""Object-relative placement driven by observed collision geometry."""
import numpy as np


def relative_destination(env, obj, target, relation, arm, distance_cm=None):
    bounds, support = env.object_bounds(obj), env.object_bounds(target)
    position = env.object_pose(obj)[0]
    desired = env.object_pose(target)[0].copy()
    bottom_offset = position[2] - bounds[0, 2]
    if relation == "on":
        desired[2] = support[1, 2] + bottom_offset + .003
    else:
        directions = {"left_of": [0., 1.], "right_of": [0., -1.], "front_of": [1., 0.],
                      "behind": [-1., 0.], "near": [0., 1. if arm == "left" else -1.]}
        direction = np.asarray(directions[relation])
        half_sizes = (bounds[1, :2] - bounds[0, :2] + support[1, :2] - support[0, :2]) / 2.
        gap = .03 if distance_cm is None else distance_cm / 100.
        # A placement beside a reference has to survive the jaws OPENING, not just the cube arriving: the
        # moving jaw sweeps out to the gripper's maximum aperture, and a reference inside that sweep is
        # pushed or blocks the release. Measured 2026-10-25 on OpenManipulator-X: a 30 mm gap put the
        # reference [red_cube] in contact with gripper_right_link and the release raised
        # GRIPPER_BLOCKED, with the gripper's own 82 mm aperture and 25.5 mm cube. The floor is physical,
        # so it wins over a smaller requested distance: a gap that cannot be released is not a gap.
        gripper = (getattr(env, "grippers", {}) or {}).get(arm, {}) or {}
        aperture = float(gripper.get("maximum_aperture") or 0.)
        if aperture:
            object_half = float(np.dot(abs(direction), (bounds[1, :2] - bounds[0, :2]) / 2.))
            # 0.02 m of margin, not 0.01: measured 2026-10-25, a 10 mm margin cleared the release but left
            # the jaw resting on the reference, so the retreat could not separate without moving it
            # (RETREAT_CONTACT). The margin has to cover the jaw body beyond the finger pad.
            gap = max(gap, aperture / 2. + .02 - object_half)
        desired[:2] += direction * (float(np.dot(abs(direction), half_sizes)) + gap)
        desired[2] = support[0, 2] + bottom_offset + .003
    return desired


def verify_relative_goal(env, obj, target, relation, arm="left", distance_cm=None):
    before = env.object_pose(obj)[0].copy()
    env.step(max(1, round(.15 / env.timestep)))
    bounds, support = env.object_bounds(obj), env.object_bounds(target)
    position, rotation = env.object_pose(obj)
    target_position = env.object_pose(target)[0]
    stable = (env.object_linear_speed(obj) < .02 and env.object_linear_speed(target) < .02
              and np.linalg.norm(position - before) < .002)
    released = not env.object_contacts(obj)
    if relation == "on":
        horizontal = float(np.linalg.norm(position[:2] - target_position[:2]))
        vertical = float(bounds[0, 2] - support[1, 2])
        contact = target in env.object_contact_links(obj)
        overlap = np.maximum(0., np.minimum(bounds[1, :2], support[1, :2]) - np.maximum(bounds[0, :2], support[0, :2]))
        supported_fraction = float(np.prod(overlap) / min(np.prod(bounds[1, :2] - bounds[0, :2]),
                                                         np.prod(support[1, :2] - support[0, :2])))
        success = (stable and released and contact and horizontal <= .012 and abs(vertical) < .004
                   and supported_fraction >= .65 and np.max(abs(rotation[2])) > np.cos(.20))
        return {"success": bool(success), "horizontal_position_error": horizontal, "support_gap": vertical,
                "support_contact": contact, "supported_fraction": supported_fraction, "stable": bool(stable),
                "released": bool(released), "position": position.tolist(), "support_position": target_position.tolist()}
    desired = relative_destination(env, obj, target, relation, arm, distance_cm)
    horizontal = float(np.linalg.norm(position[:2] - desired[:2]))
    return {"success": bool(stable and released and horizontal <= .015 and abs(position[2] - desired[2]) < .006),
            "horizontal_position_error": horizontal, "stable": bool(stable), "released": bool(released),
            "position": position.tolist(), "target_position": desired.tolist()}
