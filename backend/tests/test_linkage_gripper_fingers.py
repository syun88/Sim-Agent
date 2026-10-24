"""A single motor can drive the opposing finger through mimic joints, as a linkage gripper does.

The sibling rule pairs joints that share a parent. A Robotiq 2F-85 hangs both fingers off the palm
with the *same* hinge direction and declares one active joint plus five mimics, so the sibling rule
pairs the driver against a joint of its own finger and no aperture is found. Its finger joints mimic
the driver at -1, which is the parallelogram closure the linkage really has, and its pads hang off
those fingers on fixed joints.

These tests define the pads the way the hardware does - the two links that actually come together
over the motor's range - and hold the inference to that definition instead of to a list of names.
Measured on the 2F-85's official ROS-Industrial URDF 2026-10-24: pads 22mm wide at one depth, their
closest approach closing 86.0mm -> 1.5mm monotonically, inferred as a rotary_jaw whose aperture is
1.5-86.0mm, against the product's own 0-85mm stroke.
"""
import xml.etree.ElementTree as ET

import numpy as np

from backend.app.robots.gripper_discovery import Geometry, _coupled_fingers
from backend.app.robots.urdf import URDFModel


def linkage_robot(tmp_path, *, pads=True, single_mimic=False):
    """A three-joint arm whose palm carries two fingers driven by one mimic-coupled motor.

    ``single_mimic`` leaves only the opposing knuckle coupled (the plain opposed-finger case the
    sibling rule already handles), and ``pads=False`` leaves the pads without collision surfaces.
    """
    def box(link, xyz, size):
        collision = ET.SubElement(link, 'collision')
        ET.SubElement(collision, 'origin', xyz=xyz)
        ET.SubElement(ET.SubElement(collision, 'geometry'), 'box', size=size)

    def link(parent, name):
        return ET.SubElement(parent, 'link', name=name)

    root = ET.Element('robot', name='linkage')
    names = ['base', 'a', 'b', 'palm', 'knuckle_left', 'knuckle_right']
    if not single_mimic:
        names += ['inner_left', 'pad_left', 'inner_right', 'pad_right']
    for name in names:
        link(root, name)
    for name, parent, child, axis in (('axis0', 'base', 'a', '1 0 0'), ('axis1', 'a', 'b', '0 1 0'),
                                      ('axis2', 'b', 'palm', '0 0 1')):
        joint = ET.SubElement(root, 'joint', name=name, type='prismatic')
        ET.SubElement(joint, 'parent', link=parent)
        ET.SubElement(joint, 'child', link=child)
        ET.SubElement(joint, 'axis', xyz=axis)
        ET.SubElement(joint, 'limit', lower='-.4', upper='.4', effort='5', velocity='1')

    box(root.find("link[@name='palm']"), '0 0 0', '.02 .02 .01')
    box(root.find("link[@name='knuckle_left']"), '0 0 0', '.01 .006 .02')
    box(root.find("link[@name='knuckle_right']"), '0 0 0', '.01 .006 .02')
    if single_mimic:
        pass
    else:
        # The finger bodies barely move in the closing direction; the pads are what pinches.
        box(root.find("link[@name='inner_left']"), '0 0 0', '.02 .008 .004')
        box(root.find("link[@name='inner_right']"), '0 0 0', '.02 .008 .004')
        if pads:
            box(root.find("link[@name='pad_left']"), '0 0 -.025', '.02 .008 .05')
            box(root.find("link[@name='pad_right']"), '0 0 -.025', '.02 .008 .05')

    # Hinges sit either side of the palm so closing the motor swings the fingers towards each
    # other, with the right side rotating the other way (the model's own axis conventions).
    def revolute(name, parent, child, origin, multiplier=None):
        joint = ET.SubElement(root, 'joint', name=name, type='revolute')
        ET.SubElement(joint, 'parent', link=parent)
        ET.SubElement(joint, 'child', link=child)
        ET.SubElement(joint, 'origin', xyz=origin)
        ET.SubElement(joint, 'axis', xyz='1 0 0')
        ET.SubElement(joint, 'limit', lower='-1.2', upper='1.2', effort='5', velocity='1')
        if multiplier is not None:
            ET.SubElement(joint, 'mimic', joint='driver', multiplier=multiplier)

    revolute('driver', 'palm', 'knuckle_left', '0 .008 .048')
    root.find("joint[@name='driver']/limit").set('lower', '0')
    root.find("joint[@name='driver']/limit").set('upper', '1.2')
    revolute('knuckle_right_joint', 'palm', 'knuckle_right', '0 -.008 .048', '-1')
    if not single_mimic:
        revolute('inner_left_joint', 'knuckle_left', 'inner_left', '0 .041 .021', '-1')
        revolute('inner_right_joint', 'knuckle_right', 'inner_right', '0 -.041 .021', '-1')
    if not single_mimic:
        for name, parent, child in (('pad_left_joint', 'inner_left', 'pad_left'),
                                    ('pad_right_joint', 'inner_right', 'pad_right')):
            joint = ET.SubElement(root, 'joint', name=name, type='fixed')
            ET.SubElement(joint, 'parent', link=parent)
            ET.SubElement(joint, 'child', link=child)
    path = tmp_path / 'linkage.urdf'
    ET.ElementTree(root).write(path)
    return path


def driven_by_driver(model):
    """The driver joint and the multiplier of every joint it drives through mimic declarations."""
    followers = {}
    for other in model.joints:
        if other['type'] == 'fixed' or other['name'] == 'driver':
            continue
        name, multiplier, _ = model.joint_driver(other['name'])
        if name == 'driver':
            followers[other['name']] = float(multiplier)
    return model.by_name['driver'], followers


def range_gap(model, left_links, right_links, samples=13):
    """Closest distance between two link sets across the driver's range, from the kinematics."""
    driver, followers = driven_by_driver(model)
    geometry = Geometry(model)
    home = {j['name']: 0. for j in model.active}
    gaps = []
    for q in np.linspace(driver['lower'], driver['upper'], samples):
        values = {**home, 'driver': float(q), **{name: float(q * k) for name, k in followers.items()}}
        poses, _ = model.forward(values)
        left = geometry.in_frame(left_links, poses, np.linalg.inv(poses['palm']))
        right = geometry.in_frame(right_links, poses, np.linalg.inv(poses['palm']))
        gaps.append(min(float(np.linalg.norm(right - left[index], axis=1).min())
                        for index in range(0, len(left), 11)))
    return np.asarray(gaps)


def forms_an_aperture(gaps):
    """True when a pair closes monotonically without crossing and actually meets.

    A pinch is the pair that comes together; a pair that merely gets somewhat closer (a knuckle
    beside a finger, say) is not the gripper's jaw.
    """
    return bool((np.diff(gaps) <= 1e-9).all() and 5e-4 < gaps[-1] < .004)


def test_the_inferred_pads_are_the_links_that_actually_form_an_aperture(tmp_path):
    """The rule must find the pads the hardware pinches with, not its own finger's joints.

    The criterion is computed here, not read from the inference: on this fixture exactly one
    left/right link pair closes monotonically without crossing, and it is the pad pair.
    """
    model = URDFModel(linkage_robot(tmp_path))
    moving, fixed, _ = _coupled_fingers(model, Geometry(model), model.by_name['driver'])
    left_branch = ('knuckle_left', 'inner_left', 'pad_left')
    right_branch = ('knuckle_right', 'inner_right', 'pad_right')
    apertures = [(left, right) for left in left_branch for right in right_branch
                 if forms_an_aperture(range_gap(model, [left], [right]))]
    assert apertures == [(moving[0], fixed[0])], apertures


def test_the_pads_close_monotonically_and_keep_their_width(tmp_path):
    """A parallel-jaw aperture: the gap shrinks as the motor runs and the pads do not tilt."""
    model = URDFModel(linkage_robot(tmp_path))
    driver, followers = driven_by_driver(model)
    geometry = Geometry(model)
    home = {j['name']: 0. for j in model.active}
    widths = []
    for q in np.linspace(driver['lower'], driver['upper'], 7):
        values = {**home, 'driver': float(q), **{name: float(q * k) for name, k in followers.items()}}
        poses, _ = model.forward(values)
        points = geometry.in_frame(['pad_left'], poses, np.linalg.inv(poses['palm']))
        widths.append(float(np.ptp(points[:, 0])))
    gaps = range_gap(model, ['pad_left'], ['pad_right'])
    assert max(widths) - min(widths) < 1e-6, widths
    assert forms_an_aperture(gaps), gaps


def test_branches_without_pad_surfaces_fall_back_to_the_finger_bodies(tmp_path):
    model = URDFModel(linkage_robot(tmp_path, pads=False))
    moving, fixed, _ = _coupled_fingers(model, Geometry(model), model.by_name['driver'])
    assert moving == ['inner_left'] and fixed == ['inner_right']


def test_a_single_mimic_coupled_finger_is_left_to_the_sibling_rule(tmp_path):
    """One mimic is the plain opposed-finger case the sibling rule handles; this rule must not
    claim it."""
    model = URDFModel(linkage_robot(tmp_path, pads=False, single_mimic=True))
    assert _coupled_fingers(model, Geometry(model), model.by_name['driver']) is None


def test_a_motor_with_nothing_mimic_coupled_has_no_coupled_fingers(tmp_path):
    """Ordinary grippers must not be re-paired by this rule."""
    model = URDFModel(linkage_robot(tmp_path))
    assert _coupled_fingers(model, Geometry(model), model.by_name['axis0']) is None
