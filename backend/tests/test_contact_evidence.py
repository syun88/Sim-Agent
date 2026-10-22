"""Motor-case contacts, unloaded overlaps and parallel normals cannot fake a pinch."""
from copy import deepcopy
from backend.app.robots.contacts import evaluate_pinch_contacts


def contact(link, normal):
    return {'robot_link': link, 'normal': normal, 'on_distal_pad': True,
            'surface_error_m': .0001, 'penetration_m': .0002, 'normal_force_N': .5}


def test_only_loaded_opposed_distal_contacts_establish_pinch():
    g = {'contact_groups': [['palm'], ['moving']]}
    contacts = [contact('palm', [0., 1., 0.]), contact('moving', [0., -1., 0.])]
    assert evaluate_pinch_contacts(contacts, g, .03, .01)['established']
    for field, value in [('on_distal_pad', False), ('surface_error_m', .02), ('surface_error_m', None),
                         ('penetration_m', .004), ('normal_force_N', 0.), ('normal', [0., 1., 0.])]:
        bad = deepcopy(contacts)
        bad[1][field] = value
        assert not evaluate_pinch_contacts(bad, g, .03, .01)['established'], field


def test_surface_exit_band_requires_an_existing_loaded_pinch_and_resets_on_open_or_loss():
    from backend.app.robots.contacts import PinchContactTracker
    g={'contact_groups':[['palm'],['moving']]}
    contacts=[contact('palm',[0.,1.,0.]),contact('moving',[0.,-1.,0.])]
    near=deepcopy(contacts);near[0]['surface_error_m']=.00076
    tracker=PinchContactTracker()
    assert not tracker.evaluate('left','cube',near,g,.03,.01)['established']
    assert tracker.evaluate('left','cube',contacts,g,.03,.01)['strict_established']
    held=tracker.evaluate('left','cube',near,g,.03,.01)
    assert held['established'] and held['surface_hysteresis']
    unloaded=deepcopy(near);unloaded[0]['normal_force_N']=0.
    assert not tracker.evaluate('left','cube',unloaded,g,.03,.01)['established']
    assert not tracker.evaluate('left','cube',near,g,.03,.01)['established']
    tracker.evaluate('left','cube',contacts,g,.03,.01);tracker.reset('left')
    assert not tracker.evaluate('left','cube',near,g,.03,.01)['established']
    tracker.evaluate('left','cube',contacts,g,.03,.01)
    far=deepcopy(contacts);far[0]['surface_error_m']=.0015
    assert not tracker.evaluate('left','cube',far,g,.03,.01)['established']
