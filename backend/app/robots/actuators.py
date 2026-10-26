"""Explicit hardware metadata overrides URDF export placeholders, in SI units."""
import json
from pathlib import Path
import numpy as np


PRESETS={
    'xm430_w350':{
        'manufacturer':'ROBOTIS','model':'XM430-W350-T','voltage_V':12.0,
        'peak_effort_Nm':4.1,'no_load_velocity_rad_s':46.*np.pi/30.,
        'source_url':'https://emanual.robotis.com/docs/en/dxl/x/xm430-w350/',
        # The turtlebot3_lime description already carries this motor in its velocity field: the
        # arm joints declare velocity="4.8" and 46 rev/min is 4.82 rad/s, while the wheels
        # declare 8.063 rad/s, which is 77 rev/min and belongs to the XM430-W210 the description
        # names for them in a comment. Arm mass 0.683 kg and reach 0.353 m match OpenManipulator-X
        # (0.70 kg, 380 mm), whose official actuator is this one.
        'limit_semantics':'Peak stall torque and no-load speed are separate limits; not a measured torque-speed curve.'},
    'xm430_w210':{
        'manufacturer':'ROBOTIS','model':'XM430-W210-T','voltage_V':12.0,
        'peak_effort_Nm':3.0,'no_load_velocity_rad_s':8.063,
        'source_url':'https://emanual.robotis.com/docs/en/dxl/x/xm430-w210/',
        # Both numbers come from the official turtlebot3_lime description, which names
        # this motor in a comment right above the limit it sets (bigwheel_orin xacro,
        # wheel_left/right joints). 8.063 rad/s is 77.0 rpm, the no-load figure at 12 V.
        'limit_semantics':'Peak stall torque and no-load speed are separate limits; not a measured torque-speed curve.'},
    'sts3215_c001_7_4v':{
        'manufacturer':'FEETECH','model':'STS3215-C001','voltage_V':7.4,
        'peak_effort_Nm':19.5*.0980665,'rated_effort_Nm':5.*.0980665,
        'no_load_velocity_rad_s':np.pi/(3.*.192),
        'source_url':'https://akizukidenshi.com/goodsaffix/STS3215-C001%20SEPC.pdf',
        'limit_semantics':'Peak stall torque and no-load speed are separate limits; not a measured torque-speed curve.'}}


def apply_actuator_configuration(model,source):
    path=Path(source).with_name('actuators.json')
    if not path.is_file():return None
    settings=json.loads(path.read_text())
    unknown=set(settings.get('joints',{}))-set(model.by_name)
    if unknown:raise ValueError(f'Unknown actuator joints: {sorted(unknown)}')
    applied={}
    for joint in model.joints:
        if joint['type'] not in ('continuous','revolute'):continue
        config=settings.get('joints',{}).get(joint['name'],settings.get('default'))
        if config is None:continue
        preset=config.get('preset')
        if preset not in PRESETS:raise ValueError(f'Unknown actuator preset: {preset}')
        motor=dict(PRESETS[preset])
        joint['effort']=float(motor['peak_effort_Nm']);joint['velocity']=float(motor['no_load_velocity_rad_s'])
        element=model.xml.find(f"joint[@name='{joint['name']}']/limit")
        if element is None:
            if joint['type']!='continuous':raise ValueError(f"Missing URDF limit for {joint['name']}")
            # A continuous joint has no position limits to preserve, but its
            # effort/velocity limits are exactly what an actuator preset
            # supplies (CAD exports commonly omit the element entirely).
            import xml.etree.ElementTree as ET
            element=ET.SubElement(model.xml.find(f"joint[@name='{joint['name']}']"),'limit')
        element.set('effort',str(joint['effort']));element.set('velocity',str(joint['velocity']))
        applied[joint['name']]={'preset':preset,**motor}
    return {'source':str(path),'joints':applied}
