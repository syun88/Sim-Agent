from types import SimpleNamespace


def test_bounded_refusal_is_experimental_by_default(monkeypatch):
    from backend.app.planning.measurement_flags import enabled

    monkeypatch.delenv('OPENARM_MEASURE_REQUESTED_HAND_PREFLIGHT', raising=False)
    assert not enabled('requested_hand_preflight')


def test_one_unknown_hand_is_not_reported_as_neither_reachable(monkeypatch):
    from backend.app.agent.task_agent import _reachability_report
    from backend.app.planning import capability_queries

    def probe(env, controller, arm, obj, **kwargs):
        return {'reachable': False if arm == 'left' else None}

    monkeypatch.setattr(capability_queries, 'grasp_waypoints_reachable', probe)
    plan = SimpleNamespace(steps=[SimpleNamespace(skill='grasp_object', object='red_cube', arm='left')])
    executor = SimpleNamespace(env=object(), controller=object())
    assert _reachability_report(executor, plan)['verdict'] == 'UNMEASURED'
