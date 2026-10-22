"""Private collision and path checks; Genesis runtime acceptance is separate."""
import numpy as np
from backend.app.robots.collision_planner import PrivateCollisionQueries


def test_parent_child_collision_is_detected_without_mutating_source(tmp_path):
    source = tmp_path/'robot.urdf'
    source.write_text('''<robot name="adjacent"><link name="base"><collision><origin xyz=".2 0 0"/>
        <geometry><box size=".1 .1 .1"/></geometry></collision></link>
        <link name="arm"><collision><origin xyz=".2 0 0"/><geometry><box size=".1 .1 .1"/></geometry></collision></link>
        <joint name="turn" type="continuous"><parent link="base"/><child link="arm"/><axis xyz="0 0 1"/></joint></robot>''')
    before = source.read_bytes()
    query = PrivateCollisionQueries({'urdf': str(source), 'root_transform': np.eye(4).tolist()})
    assert query.contacts({'turn': 0.})[0]['links'] == ['arm', 'base']
    assert query.clear({'turn': np.pi/2.})
    assert source.read_bytes() == before


def test_path_goes_around_an_obstacle_instead_of_accepting_a_clear_endpoint_only():
    query = object.__new__(PrivateCollisionQueries)
    query.clear = lambda q: not (abs(q['x']) < .22 and abs(q['y']) < .65)
    names = ['x', 'y'];start = np.array([-1., 0.]);goal = np.array([1., 0.])
    assert not query.edge_clear(names, start, goal, {})
    path = query.path(names, start, goal, {}, np.array([[-2., 2.], [-2., 2.]]))
    assert path is not None
    np.testing.assert_allclose(path[0], start);np.testing.assert_allclose(path[-1], goal)
    assert all(query.edge_clear(names, a, b, {}) for a, b in zip(path, path[1:]))
def test_pose_refinement_uses_redundancy_to_avoid_collision_without_changing_tool_goal(tmp_path):
    import numpy as np
    from backend.app.robots.collision_planner import PrivateCollisionQueries
    from backend.app.robots.urdf import URDFModel, URDFKinematics
    source = tmp_path/'redundant.urdf'
    source.write_text('''<robot name="redundant"><link name="base"/>
      <link name="carriage"><collision><geometry><box size=".02 .02 .02"/></geometry></collision></link>
      <link name="tool"/>
      <joint name="up" type="prismatic"><parent link="base"/><child link="carriage"/>
        <origin xyz="0 0 .75"/><axis xyz="0 0 1"/><limit lower="0" upper=".2" effort="1" velocity="1"/></joint>
      <joint name="slide" type="prismatic"><parent link="carriage"/><child link="tool"/>
        <axis xyz="0 0 1"/><limit lower="0" upper=".2" effort="1" velocity="1"/></joint></robot>''')
    chain = {'joints':['up','slide'], 'end_link':'tool', 'tool_offset':[0.,0.,0.]}
    profile = {'urdf':str(source), 'root_transform':np.eye(4).tolist(), 'chains':{'left':chain}}
    query = PrivateCollisionQueries(profile, .76)
    q = np.array([.0185, .0315])
    assert not query.clear(dict(zip(chain['joints'],q)))
    goal = np.array([0.,0.,.8])
    result = query.refine_pose(profile,'left',goal,np.eye(3),q,{})
    assert result is not None
    assert query.clear(dict(zip(chain['joints'],result)))
    point = URDFKinematics(URDFModel(source),chain).evaluate(result)[0]
    np.testing.assert_allclose(point,goal,atol=.0001)
    separated = query.refine_pose(profile,'left',goal,np.eye(3),q,{},minimum_clearance=.0001)
    assert separated is not None
    values = dict(zip(chain['joints'],separated))
    assert query.edge_clear(chain['joints'],separated,separated,{},minimum_clearance=.0001)
    assert not query.contacts(values,tolerance=-.0001)
    point = URDFKinematics(URDFModel(source),chain).evaluate(separated)[0]
    np.testing.assert_allclose(point,goal,atol=1e-7)
