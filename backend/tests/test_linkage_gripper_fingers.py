"""A single motor can drive the opposing finger through mimic joints, as a linkage gripper does.

The sibling rule pairs joints that share a parent. A Robotiq 2F-85 hangs both fingers off the palm
with the *same* hinge direction and declares one active joint plus five mimics, so the sibling rule
pairs the driver against a joint of its own finger and no aperture is found. Measured on the 2F-85's
official ROS-Industrial URDF 2026-10-24: the pads are the deepest driven link of each branch (pad
centres 0.1mm either side of the hinge axis, 22mm wide, the same depth for both) and their closest
approach closes 86.0mm -> 1.5mm monotonically, i.e. a parallel-jaw aperture. The finger joints
mimic the driver at -1, which is the parallelogram closure the linkage really has.
"""
import xml.etree.ElementTree as ET

import numpy as np

from backend.app.robots.gripper_discovery import Geometry, _candidate, _coupled_fingers
from backend.app.robots.urdf import URDFModel


def linkage_robot(tmp_path, *, pad_on_each_branch=True):
    """A three-joint arm whose palms carry two fingers, both driven by one mimic-coupled motor."""
    def box(link, xyz, size='.02 .01 .02'):
        collision = ET.SubElement(link, 'collision')
        ET.SubElement(collision, 'origin', xyz=xyz)
        ET.SubElement(ET.SubElement(collision, 'geometry'), 'box', size=size)

    root = ET.Element('robot', name='linkage')
    for name in ('base', 'a', 'b', 'palm', 'knuckle_left', 'pad_left', 'knuckle_right', 'pad_right'):
        ET.SubElement(root, 'link', name=name)
    parent = 'base'
    for index, (child, axis) in enumerate([('a', '1 0 0'), ('b', '0 1 0'), ('palm', '0 0 1')]):
        joint = ET.SubElement(root, 'joint', name=f'axis{index}', type='prismatic')
        ET.SubElement(joint, 'parent', link=parent)
        ET.SubElement(joint, 'child', link=child)
        ET.SubElement(joint, 'axis', xyz=axis)
        ET.SubElement(joint, 'limit', lower='-.4', upper='.4', effort='5', velocity='1')
        parent = child
    box(root.find("link[@name='palm']"), '0 0 0', '.02 .02 .01')
    box(root.find("link[@name='knuckle_left']"), '0 .03 .03')
    box(root.find("link[@name='knuckle_right']"), '0 -.03 .03')
    for name, y in (('pad_left', .03), ('pad_right', -.03)):
        if pad_on_each_branch:
            box(root.find(f"link[@name='{name}']"), f'0 {y} -.02', '.02 .008 .05')

    driver = ET.SubElement(root, 'joint', name='driver', type='revolute')
    ET.SubElement(driver, 'parent', link='palm')
    ET.SubElement(driver, 'child', link='knuckle_left')
    ET.SubElement(driver, 'axis', xyz='1 0 0')
    ET.SubElement(driver, 'limit', lower='-.6', upper='0', effort='5', velocity='1')
    # Both knuckles hinge the same way off the palm, as on the 2F-85; the finger joints take the
    # -1 mimic that keeps the pads parallel to the palm.
    for name, parent_link, child, multiplier in (('knuckle_right_joint', 'palm', 'knuckle_right', '1'),
                                                 ('pad_left_joint', 'knuckle_left', 'pad_left', '-1'),
                                                 ('pad_right_joint', 'knuckle_right', 'pad_right', '-1')):
        joint = ET.SubElement(root, 'joint', name=name, type='revolute')
        ET.SubElement(joint, 'parent', link=parent_link)
        ET.SubElement(joint, 'child', link=child)
        ET.SubElement(joint, 'axis', xyz='1 0 0')
        ET.SubElement(joint, 'limit', lower='-.6', upper='.6', effort='5', velocity='1')
        ET.SubElement(joint, 'mimic', joint='driver', multiplier=multiplier)
    path = tmp_path / 'linkage.urdf'
    ET.ElementTree(root).write(path)
    return path


def test_mimic_driven_fingers_pair_the_two_pads_not_one_finger_s_own_joints(tmp_path):
    model = URDFModel(linkage_robot(tmp_path))
    geometry = Geometry(model)
    moving, fixed, followers = _coupled_fingers(model, geometry, model.by_name['driver'])
    assert moving == ['pad_left'] and fixed == ['pad_right']
    assert followers == {'knuckle_right_joint': 1., 'pad_left_joint': -1., 'pad_right_joint': -1.}


def test_branches_without_a_deeper_pad_fall_back_to_the_finger_bodies(tmp_path):
    model = URDFModel(linkage_robot(tmp_path, pad_on_each_branch=False))
    geometry = Geometry(model)
    moving, fixed, _ = _coupled_fingers(model, geometry, model.by_name['driver'])
    assert moving == ['knuckle_left'] and fixed == ['knuckle_right']


def test_a_motor_with_nothing_mimic_coupled_has_no_coupled_fingers(tmp_path):
    model = URDFModel(linkage_robot(tmp_path))
    assert _coupled_fingers(model, Geometry(model), model.by_name['axis0']) is None

