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
