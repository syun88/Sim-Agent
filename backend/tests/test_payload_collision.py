import numpy as np
from scipy.spatial.transform import Rotation
from backend.app.robots.placement import boxes_overlap


def test_payload_boxes_use_all_separating_axes_for_rotated_clearances():
    p=np.zeros(3);r=np.eye(3);h=np.repeat(.02,3)
    assert boxes_overlap(p,r,h,[.039,0.,0.],r,h)
    assert not boxes_overlap(p,r,h,[.042,0.,0.],r,h)
    tilted=Rotation.from_euler('z',45,degrees=True).as_matrix()
    assert boxes_overlap(p,r,h,[.045,0.,0.],tilted,h)
    assert not boxes_overlap(p,r,h,[.052,0.,0.],tilted,h)
    assert not boxes_overlap(p,r,h,[0.,0.,.042],tilted,h)


def test_empty_retreat_allows_separating_contact_but_rejects_deeper_contact_and_crossing(tmp_path):
    from types import SimpleNamespace
    from backend.app.robots.collision_planner import PrivateCollisionQueries
    from backend.app.robots.placement import EmptyHandQueries
    source=tmp_path/'finger.urdf'
    source.write_text('''<robot name="finger"><link name="base"/><link name="finger">
      <collision><geometry><box size=".01 .01 .03"/></geometry></collision></link>
      <joint name="slide" type="prismatic"><parent link="base"/><child link="finger"/>
      <origin xyz=".02 0 0"/><axis xyz="1 0 0"/>
      <limit lower="-.1" upper=".1" effort="1" velocity="1"/></joint></robot>''')
    profile={'urdf':str(source),'root_transform':np.eye(4).tolist(),'chains':{'left':{'joints':['slide']}}}
    base=PrivateCollisionQueries(profile)
    env=SimpleNamespace(collision_queries=base,model=base.urdf,profile=profile,cube_size=.03,
        _objects={'cube':None},grippers={'left':{'contact_groups':[['finger']]}},
        arm_qpos=lambda side:np.array([0.]),joint_snapshot=lambda:{'slide':0.},
        object_pose=lambda name:(np.zeros(3),np.eye(3)))
    query=EmptyHandQueries(env,'left')
    assert query.clear({'slide':0.}) and query.clear({'slide':.002})
    assert not query.clear({'slide':-.001})
    assert not query.edge_clear(['slide'],np.array([0.]),np.array([-.07]),{})


def test_a_urdf_root_link_named_world_does_not_clash_with_the_engine_world(tmp_path):
    """Manufacturer URDFs name their root link 'world' - ROBOTIS's own OpenManipulator models do.

    MuJoCo owns that name for its own world body, so the private model compiles robot bodies under a
    safe name and keeps the mapping. Measured 2026-10-24: without it the model failed to compile
    ("repeated name 'world' in body") and that robot had no collision queries at all.
    """
    from backend.app.robots.collision_planner import PrivateCollisionQueries
    source = tmp_path / 'world_root.urdf'
    source.write_text('''<robot name="world_root">
      <link name="world"><collision><geometry><box size=".04 .04 .02"/></geometry></collision></link>
      <link name="tip"><collision><geometry><box size=".02 .02 .02"/></geometry></collision></link>
      <joint name="slide" type="prismatic"><parent link="world"/><child link="tip"/>
      <origin xyz="0 0 .01"/><axis xyz="0 0 1"/>
      <limit lower="-.05" upper=".05" effort="1" velocity="1"/></joint></robot>''')
    profile = {'urdf': str(source), 'root_transform': np.eye(4).tolist(),
               'chains': {'left': {'joints': ['slide']}}}
    queries = PrivateCollisionQueries(profile, table_top=0.)
    assert queries.body_names['world'] == 'robot_world'
    assert queries.link_names['robot_world'] == 'world'
    # Raised, the robot's own root link touches its tip: that pair names the robot's body, not
    # the engine's world. Lowered, the tip touches the desk, which is MuJoCo's own 'world' body.
    raised = {name for contact in queries.contacts({'slide': 0.}, tolerance=1e-6) for name in contact['links']}
    lowered = {name for contact in queries.contacts({'slide': -.03}, tolerance=1e-6) for name in contact['links']}
    assert raised == {'robot_world', 'tip'}, raised
    assert lowered == {'tip', 'world'}, lowered
