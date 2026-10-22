import numpy as np

from backend.app.agent.scene_relations import support_anchor, support_relations


def cube(name,position):
    p = np.array(position)
    return {'id':name,'type':'cube','bounds':[list(p-.02),list(p+.02)]}


def test_support_requires_contact_height_and_overlap():
    bottom = cube('red_cube',[0.,0.,.02])
    top = cube('blue_cube',[0.,0.,.06])
    assert not support_relations([bottom,top],[])
    supports = support_relations([bottom,top],['blue_cube|red_cube'])
    assert [(r['subject'],r['object']) for r in supports]==[('blue_cube','red_cube')]
    assert support_anchor(['blue_cube','red_cube'],supports)=='red_cube'
    assert not support_relations([bottom,cube('blue_cube',[.04,0.,.02])],['blue_cube|red_cube'])
    assert not support_relations([bottom,cube('blue_cube',[.039,0.,.06])],['blue_cube|red_cube'])


def test_disconnected_or_cyclic_groups_have_no_support_anchor():
    assert support_anchor(['a','b'],[]) is None
    assert support_anchor(['a','b'],[{'subject':'a','object':'b','relation':'on'},
                                     {'subject':'b','object':'a','relation':'on'}]) is None
    graph = [{'subject':'a','object':'b','relation':'on'}, {'subject':'b','object':'c','relation':'on'}]
    assert support_anchor(['a','b','c'],graph)=='c'
    assert support_anchor(['a','a'],graph) is None
