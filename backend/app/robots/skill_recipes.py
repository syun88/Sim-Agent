"""Verified, parameterised geometric experience; never cached motor playback."""
import json
from pathlib import Path
import time
import numpy as np


def _valid_hint(value,count):
    rotation=np.asarray(value['object_rotation'],float)
    return (rotation.shape==(3,3) and np.max(abs(rotation.T@rotation-np.eye(3)))<.01
        and np.linalg.det(rotation)>.99 and np.asarray(value['bias']).shape==(3,)
        and np.asarray(value['open_arm_q']).shape==(count,)
        and np.isfinite(np.r_[rotation.ravel(),value['bias'],value['open_arm_q']]).all())


def load_exchange_recipe(profile,donor,receiver):
    try:
        data=json.loads((Path(profile['directory'])/'handover_skills.json').read_text())
        if not isinstance(data,dict):return None
        if data.get('version')!=1 or data.get('configuration_sha256')!=profile['configuration_sha256']:return None
        value=data['recipes'][f'{donor}:{receiver}']
        count=len(profile['chains'][receiver]['joints'])
        rotation=np.asarray(value['donor_rotation'],float)
        if (float(value['verification_lift_m'])<.021 or float(value['hold_duration_s'])<.3
                or not np.isfinite([value['verification_lift_m'],value['hold_duration_s'],value['physically_verified_at']]).all()
                or rotation.shape!=(3,3) or np.max(abs(rotation.T@rotation-np.eye(3)))>.01
                or np.linalg.det(rotation)<.99 or np.asarray(value['center']).shape!=(3,)
                or not np.isfinite(np.r_[rotation.ravel(),value['center']]).all()
                or not _valid_hint(value['predicted_receiver'],count) or not _valid_hint(value['receiver'],count)):
            return None
        return value
    except (OSError,ValueError,KeyError,TypeError):return None


def record_exchange_recipe(profile,donor,receiver,value):
    path=Path(profile['directory'])/'handover_skills.json'
    try:
        if (float(value['verification_lift_m'])<.021 or float(value['hold_duration_s'])<.3
                or not np.isfinite([value['verification_lift_m'],value['hold_duration_s']]).all()):return False
        try:data=json.loads(path.read_text())
        except (OSError,ValueError):data={}
        if not isinstance(data,dict):data={}
        if data.get('configuration_sha256')!=profile['configuration_sha256'] or data.get('version')!=1:
            data={'version':1,'configuration_sha256':profile['configuration_sha256'],'recipes':{}}
        if not isinstance(data.get('recipes'),dict):data['recipes']={}
        data['recipes'][f'{donor}:{receiver}']={**value,'physically_verified_at':time.time()}
        temporary=path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(data,indent=2)+'\n');temporary.replace(path)
        return True
    except (OSError,ValueError,KeyError,TypeError):return False
