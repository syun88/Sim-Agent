"""Explicit hardware metadata overrides URDF export placeholders, in SI units."""
import json
from pathlib import Path
import numpy as np


PRESETS={
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
        if element is None:raise ValueError(f"Missing URDF limit for {joint['name']}")
        element.set('effort',str(joint['effort']));element.set('velocity',str(joint['velocity']))
        applied[joint['name']]={'preset':preset,**motor}
    return {'source':str(path),'joints':applied}
