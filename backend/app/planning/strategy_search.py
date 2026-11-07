"""Choose how to perform a bimanual stack by *asking* the body, not by inheriting a constant.

Measured 2026-11-07: three constants in this path were each a past measurement - a staging point at
(0.45, -0.04, 1.06), a 15 mm regrasp, and a yaw-270 grasp attitude. The staging point was unreachable for
the placing arm in one scene, the regrasp could not clear the fingers, and the attitude only worked on the
side it was measured on. Each was a fact about one scene wearing the clothes of a rule.

`plan_stack_on_held_support` enumerates a small bounded set of strategies and evaluates them with the
read-only capability queries. It returns the first that passes, the ones it rejected and why, or a refusal
whose reason is the measurement that rejected every candidate. Nothing is executed here.
"""

from __future__ import annotations

import numpy as np

from backend.app.planning.capability_queries import (ATTITUDE_YAWS_DEG, pad_blocks_support_face,
                                                    pinch_point, reachable)
from backend.app.simulation.openarm_v2.definition import FINGERTIP_MIDPOINT

#: Where the holder can bring a support: the placing arm's side first (the placer need not cross the
#: body), then the middle, then the holder's own side.
STAGING_POSITIONS = ((0.45, 0.04, 1.06), (0.45, 0.0, 1.06), (0.45, -0.04, 1.06))
#: Support heights worth testing: the holder's arm must reach the staging point without the table.
STAGING_HEIGHTS = (1.06, 1.10)


def plan_stack_on_held_support(env, controller, placer_arm: str, support: str) -> dict:
    """Pick a staging point and a placing attitude the arms can actually reach, or say why not.

    `placer_arm` is the arm that will carry the cube onto `support`; the other arm holds the support.
    """
    holder_arm = 'right' if placer_arm == 'left' else 'left'
    sign = 1. if placer_arm == 'left' else -1.
    evaluations: list[dict] = []

    blocked = pad_blocks_support_face(env, holder_arm, support)
    if blocked.get('blocks'):
        # The holder's own fingers cover the face the cube must land on. No staging point or attitude
        # changes that: it is the hand's geometry, so the search stops here and says so.
        return {'ok': False, 'reason': 'SUPPORT_FACE_BLOCKED', 'mechanism': 'HAND_GEOMETRY',
                'clearance_m': blocked.get('clearance_m'), 'measured': blocked.get('measured'),
                'evaluations': [{'query': 'pad_blocks_support_face', **blocked}]}

    try:
        support_center = np.asarray(env.object_pose(support)[0], float)
        bounds = env.object_bounds(support)
        half_height = float(bounds[1][2] - bounds[0][2]) / 2.
    except Exception as error:
        return {'ok': False, 'reason': 'SUPPORT_NOT_MEASURABLE',
                'detail': f'{type(error).__name__}: {error}', 'evaluations': evaluations}

    for staging in [(x, y * sign if y else 0., z) for (x, y, _), z in
                    [(position, height) for position in STAGING_POSITIONS for height in STAGING_HEIGHTS]]:
        staging = np.asarray(staging, float)
        holder_reach = reachable(env, controller, holder_arm,
                                 pinch_point(holder_arm, staging), 0.)
        evaluations.append({'query': 'reachable_pose', 'arm': holder_arm,
                            'staging': staging.tolist(), **holder_reach})
        if holder_reach.get('ok') is not True:
            continue
        top = staging + np.array([0., 0., half_height])
        for yaw in ATTITUDE_YAWS_DEG:
            placer_reach = reachable(env, controller, placer_arm,
                                     pinch_point(placer_arm, top, yaw), yaw)
            evaluations.append({'query': 'reachable_pose', 'arm': placer_arm,
                                'staging': staging.tolist(), **placer_reach})
            if placer_reach.get('ok') is True:
                return {'ok': True, 'staging': staging.tolist(), 'placer_yaw_deg': yaw,
                        'holder_arm': holder_arm, 'evaluations': evaluations}

    decisive = [item for item in evaluations if item.get('ok') is not None]
    if not decisive:
        # Nothing could be measured (no IK solver, an imported arm without this query). Not a refusal:
        # the caller keeps its previous behaviour rather than refusing on an unknown.
        return {'ok': None, 'reason': 'UNMEASURED', 'evaluations': evaluations}
    return {'ok': False, 'reason': 'NO_REACHABLE_STACK_POSE',
            'detail': f'no staging point let {holder_arm} bring {support} within reach of {placer_arm}',
            'evaluations': evaluations}
