"""Where the hands meet is chosen by trying the candidates and keeping the best measured residual."""
import numpy as np
import pytest

from backend.app.simulation.openarm_v2 import handover


def test_the_candidates_include_the_point_the_cell_used_to_hard_code():
    points = dict(handover._meeting_points(np.array([0.30, 0.15, 1.05])))
    assert 'legacy_clip' in points
    # The legacy point is the old clip box, now expressed as a candidate rather than as the answer.
    assert points['legacy_clip'] == pytest.approx([0.40, 0.0, 1.10])
    assert any(name.startswith('offset_') for name in points)


def test_the_held_objects_own_position_is_a_candidate():
    cube = np.array([0.45, 0.02, 1.09])
    assert dict(handover._meeting_points(cube))['offset_+0.00_+0.00_+0.00'] == pytest.approx(cube)


def test_a_candidate_that_never_solved_cannot_win():
    solved = object()
    evaluations = [
        {'name': 'unsolvable', 'error': 0.001, 'solved': None},
        {'name': 'worse', 'error': 0.0025, 'solved': solved},
        {'name': 'best', 'error': 0.0009, 'solved': solved},
    ]
    winner = handover.select_winner(evaluations)
    assert winner['name'] == 'best'


def test_no_solvable_candidate_returns_none_instead_of_the_least_bad_label():
    assert handover.select_winner([{'name': 'a', 'error': 0.4, 'solved': None},
                                   {'name': 'b', 'error': None, 'solved': None}]) is None


def test_the_legacy_point_wins_only_if_it_measures_better():
    solved = object()
    evaluations = [{'name': 'legacy_clip', 'error': 0.002, 'solved': solved},
                   {'name': 'offset_+0.00_+0.03_+0.00', 'error': 0.0011, 'solved': solved}]
    assert handover.select_winner(evaluations)['name'] == 'offset_+0.00_+0.03_+0.00'


def test_the_meeting_evidence_can_be_returned_by_the_api():
    """A task failed once because this evidence held arrays: it has to serialise as it stands."""
    import json
    import numpy as np
    from backend.app.simulation.openarm_v2.handover import HandoverGeometry

    geometry = HandoverGeometry(np.zeros(3), np.eye(3), np.zeros(7), np.zeros(7), np.eye(3), .015, .001,
                                meeting={'point': [0.4, 0.0, 1.1], 'measured': True, 'chosen': 'legacy_clip',
                                         'residual': 0.001,
                                         'evaluations': [{'name': 'legacy_clip', 'point': [0.4, 0.0, 1.1],
                                                          'error': 0.001, 'collision_rejections': 0}]})
    assert json.loads(json.dumps(geometry.meeting))['chosen'] == 'legacy_clip'
