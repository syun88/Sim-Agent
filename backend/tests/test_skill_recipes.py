import json
import numpy as np
from backend.app.robots.skill_recipes import record_exchange_recipe,load_exchange_recipe


def test_geometric_recipe_is_configuration_bound_and_rejects_invalid_rotations(tmp_path):
    profile={'directory':str(tmp_path),'configuration_sha256':'shape-and-motors-a',
             'chains':{'left':{'joints':['a','b']}}}
    hint={'object_rotation':np.eye(3).tolist(),'bias':[.01,0.,0.],'open_arm_q':[.1,.2]}
    value={'center':[0.,0.,1.],'donor_rotation':np.eye(3).tolist(),
           'predicted_receiver':hint,'receiver':hint,
           'verification_lift_m':.029,'hold_duration_s':.3}
    assert not record_exchange_recipe(profile,'right','left',{**value,'verification_lift_m':.002})
    assert not record_exchange_recipe(profile,'right','left',{**value,'hold_duration_s':.05})
    assert record_exchange_recipe(profile,'right','left',value)
    loaded=load_exchange_recipe(profile,'right','left')
    assert loaded['receiver']['bias']==[.01,0.,0.] and loaded['physically_verified_at']>0.
    assert 'path' not in loaded
    assert load_exchange_recipe({**profile,'configuration_sha256':'changed'},'right','left') is None
    assert load_exchange_recipe(profile,'left','right') is None
    path=tmp_path/'handover_skills.json';data=json.loads(path.read_text())
    data['recipes']['right:left']['hold_duration_s']=.05
    path.write_text(json.dumps(data))
    assert load_exchange_recipe(profile,'right','left') is None
    path.write_text('[]')
    assert load_exchange_recipe(profile,'right','left') is None
    assert record_exchange_recipe(profile,'right','left',value)
    assert load_exchange_recipe(profile,'right','left') is not None
    data['recipes']['right:left']['hold_duration_s']=.3
    data['recipes']['right:left']['receiver']['object_rotation']=np.zeros((3,3)).tolist()
    path.write_text(json.dumps(data))
    assert load_exchange_recipe(profile,'right','left') is None
