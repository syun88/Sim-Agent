import os
from fastapi.testclient import TestClient
from backend.app.main import app


def test_model_selection_api():
    client = TestClient(app)
    # 1. Get models list
    res = client.get('/api/config/models')
    assert res.status_code == 200
    data = res.json()
    assert 'current' in data
    assert 'available' in data
    assert isinstance(data['available'], list)
    
    # 2. Switch model
    old_model = data['current']
    target_model = 'lfm2.5:latest'
    res = client.post('/api/config/model', json={'model': target_model})
    assert res.status_code == 200
    assert res.json()['selected_model'] == target_model
    assert os.getenv('OPENARM_OLLAMA_MODEL') == target_model
    
    # 3. Verify it is reflected
    res = client.get('/api/config/models')
    assert res.json()['current'] == target_model
    
    # 4. Restore original model
    client.post('/api/config/model', json={'model': old_model})
