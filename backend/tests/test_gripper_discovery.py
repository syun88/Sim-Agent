"""Geometry inference is separate from contact-based runtime verification."""
import json
from pathlib import Path
import numpy as np
import pytest
from backend.app.robots import registry
from backend.app.robots.gripper_discovery import discover_grippers
from backend.app.robots.urdf import URDFModel


def jaw_robot(tmp_path, scale=1., kind='continuous'):
    def box(y):
        return f'<collision><origin xyz="{.055*scale} {y*scale} 0"/><geometry><box size="{.08*scale} {.008*scale} {.02*scale}"/></geometry></collision>'
    text='<robot name="unnamed_jaws"><link name="base"/><link name="a"/><link name="b"/>'
    text+=f'<link name="palm">{box(-.022)}</link><link name="tool">{box(.022)}</link>'
    for index,(parent,child,axis) in enumerate([('base','a','1 0 0'),('a','b','0 1 0'),('b','palm','0 0 1')]):
        text+=f'<joint name="axis{index}" type="prismatic"><parent link="{parent}"/><child link="{child}"/><axis xyz="{axis}"/><limit lower="-.4" upper=".4" effort="5" velocity="1"/></joint>'
    text+='<joint name="last" type="continuous"><parent link="palm"/><child link="tool"/><axis xyz="0 0 1"/><limit effort="1" velocity="2"/></joint></robot>'
    path=tmp_path/'robot.urdf';path.write_text(text)
    return path


def test_geometry_finds_unnamed_rotary_jaw_and_scales_cube_from_actual_surfaces(tmp_path,monkeypatch):
    monkeypatch.setattr(registry,'GENERATED_ROOT',tmp_path/'generated')
    original=jaw_robot(tmp_path)
    raw=original.read_bytes()
    p=registry.import_robot(original)
    assert original.read_bytes()==raw
    assert len(p['grippers'])==1
    g=p['grippers']['left']
    assert g['joint']=='last' and g['status']=='geometry_inferred'
    assert p['chains']['left']['joints']==['axis0','axis1','axis2']
    assert 0.008<g['cube_size']<g['maximum_aperture']
    scene=json.loads((Path(p['directory'])/'scene.json').read_text())
    assert scene['cube_mass']==pytest.approx(.05*(g['cube_size']/.052)**3)
    assert scene['cube_size']==g['cube_size']
    assert not p['grasp_calibrated']
    assert not json.loads((Path(p['directory'])/'gripper_analysis.json').read_text())['physical_grasp_verified']
    first=g['cube_size']
    jaw_robot(tmp_path,scale=.5)
    smaller=registry.import_robot(original)
    assert smaller['cube_size'] < first*.6


def test_plain_terminal_joint_does_not_become_a_gripper_without_opposed_geometry(tmp_path):
    path=jaw_robot(tmp_path)
    # Removing the palm's collision geometry leaves no opposed fixed jaw.
    import xml.etree.ElementTree as ET
    root=ET.fromstring(path.read_text())
    palm=root.find("link[@name='palm']")
    for item in list(palm):palm.remove(item)
    ET.ElementTree(root).write(path)
    model=URDFModel(path)
    report=discover_grippers(model,{'chains':{'left':{'end_link':'tool'}},
        'home':{j['name']:0. for j in model.active},'root_transform':np.eye(4).tolist()})
    assert report['grippers']=={}


def test_mimic_sliding_fingers_are_one_gripper_not_two_arms(tmp_path,monkeypatch):
    source=jaw_robot(tmp_path)
    import xml.etree.ElementTree as ET
    root=ET.fromstring(source.read_text())
    palm=root.find("link[@name='palm']")
    for item in list(palm):palm.remove(item)
    # Finger shafts point along +Z, their opposed sliding axes along +/-Y.
    tool=root.find("link[@name='tool']")
    for item in list(tool):tool.remove(item)
    def finger(link, y):
        c=ET.SubElement(link,'collision');ET.SubElement(c,'origin',xyz=f'0 {y} .045')
        g=ET.SubElement(c,'geometry');ET.SubElement(g,'box',size='.02 .008 .07')
    finger(tool,.008)
    other=ET.SubElement(root,'link',name='partner');finger(other,-.008)
    drive=root.find("joint[@name='last']");drive.set('type','prismatic')
    drive.find('axis').set('xyz','0 1 0');drive.find('limit').set('lower','0');drive.find('limit').set('upper','.04')
    follow=ET.SubElement(root,'joint',name='coupled',type='prismatic')
    ET.SubElement(follow,'parent',link='palm');ET.SubElement(follow,'child',link='partner')
    ET.SubElement(follow,'axis',xyz='0 -1 0');ET.SubElement(follow,'limit',lower='0',upper='.04',effort='1',velocity='1')
    ET.SubElement(follow,'mimic',joint='last',multiplier='1')
    ET.ElementTree(root).write(source)
    monkeypatch.setattr(registry,'GENERATED_ROOT',tmp_path/'generated')
    p=registry.import_robot(source)
    assert len(p['chains'])==1
    assert p['grippers']['left']['type']=='sliding_jaws'
    assert p['grippers']['left']['joint']=='last'
    assert p['grippers']['left']['open_q']>p['grippers']['left']['closed_q']
    assert p['grippers']['left']['contact_groups']==[['partner'],['tool']]
