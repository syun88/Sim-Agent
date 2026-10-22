import json
import numpy as np
import pytest
from backend.app.robots.actuators import apply_actuator_configuration
from backend.app.robots.urdf import URDFModel
from backend.app.robots.registry import asset_signature


def test_explicit_motor_metadata_changes_normalized_limits_and_invalidates_signature(tmp_path):
    source=tmp_path/'robot.urdf'
    source.write_text('''<robot name="motor"><link name="base"/><link name="arm"/>
      <joint name="turn" type="continuous"><parent link="base"/><child link="arm"/>
      <axis xyz="0 0 1"/><limit effort="1" velocity="2"/></joint></robot>''')
    before=source.read_bytes();signature=asset_signature(source)
    model=URDFModel(source)
    assert apply_actuator_configuration(model,source) is None
    metadata=source.with_name('actuators.json')
    metadata.write_text(json.dumps({'default':{'preset':'sts3215_c001_7_4v'}}))
    assert asset_signature(source)!=signature
    evidence=apply_actuator_configuration(model,source)
    joint=model.by_name['turn']
    assert joint['effort']==pytest.approx(1.91229675)
    assert joint['velocity']==pytest.approx(np.pi/(3.*.192))
    assert float(model.xml.find('joint/limit').get('effort'))==pytest.approx(joint['effort'])
    assert evidence['joints']['turn']['rated_effort_Nm']==pytest.approx(.4903325)
    assert source.read_bytes()==before
    metadata.write_text(json.dumps({'joints':{'typo':{'preset':'sts3215_c001_7_4v'}}}))
    with pytest.raises(ValueError,match='Unknown actuator joints'):apply_actuator_configuration(model,source)
