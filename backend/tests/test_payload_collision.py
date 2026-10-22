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
