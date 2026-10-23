from fastapi.testclient import TestClient

from backend.app.main import app


def test_scene_is_a_viewer_snapshot_and_backend_never_creates_a_simulator():
    with TestClient(app) as client:
        assert app.state.engine is None
        assert client.get("/api/scene").status_code == 503
        assert client.post("/api/genesis/register", json={"viewer_id": "one", "provider": "local_ollama"}).status_code == 200
        assert client.post("/api/genesis/register", json={"viewer_id": "two"}).status_code == 409
        snapshot = {"robot": {"type": "openarm_v2"}, "objects": [{"id": "blue_cube", "position": [.44, .2, 1.03]}]}
        assert client.post("/api/genesis/state", json={"viewer_id": "one", "world_state": snapshot}).status_code == 200
        assert client.get("/api/scene").json() == snapshot
        assert client.get("/api/health").json()["provider"] == "local_ollama"


def test_stop_removes_a_queued_command_instead_of_executing_it_later():
    with TestClient(app) as client:
        task = client.post("/api/tasks", json={"instruction": "右手で青を掴んで"}).json()["task_id"]
        client.post(f"/api/tasks/{task}/stop")
        assert client.get(f"/api/tasks/{task}").json()["status"] == "STOPPED"
        assert client.get("/api/genesis/next").json() == {"action": None}


def test_running_stop_is_visible_to_the_controller_and_late_success_cannot_override_it():
    with TestClient(app) as client:
        task = client.post("/api/tasks", json={"instruction": "右手で青を掴んで"}).json()["task_id"]
        assert client.get("/api/genesis/next").json()["task_id"] == task
        client.post(f"/api/tasks/{task}/stop")
        assert client.get(f"/api/genesis/control/{task}").json()["stop"]
        client.post("/api/genesis/result", json={"task_id": task, "success": True})
        record = client.get(f"/api/tasks/{task}").json()
        assert record["status"] == "STOPPED" and record["success"] is False


def test_reset_reports_queued_until_the_viewer_completes_it():
    with TestClient(app) as client:
        reset = client.post("/api/scene/reset?randomize=true").json()
        assert reset["status"] == "queued"
        assert client.get(f"/api/tasks/{reset['task_id']}").json()["status"] == "QUEUED_FOR_GENESIS_VIEWER"
        command = client.get("/api/genesis/next").json()
        assert command["action"] == "reset" and command["randomize"]
        client.post("/api/genesis/result", json={"task_id": command["task_id"], "success": True})
        assert client.get(f"/api/tasks/{reset['task_id']}").json()["status"] == "SUCCEEDED"


def test_a_second_process_cannot_consume_the_registered_viewers_command_or_fake_its_result():
    with TestClient(app) as client:
        client.post("/api/genesis/register", json={"viewer_id": "owner"})
        task = client.post("/api/tasks", json={"instruction": "青を掴んで"}).json()["task_id"]
        assert client.get("/api/genesis/next").status_code == 409
        assert client.get("/api/genesis/next?viewer_id=owner").json()["task_id"] == task
        assert client.post("/api/genesis/result", json={"task_id": task, "success": True}).status_code == 409


def test_random_reset_carries_a_replayable_seed_and_side_shuffle_through_the_queue():
    with TestClient(app) as client:
        requested=client.post('/api/scene/reset?randomize=true&seed=0&shuffle_sides=false').json()
        stored=client.get('/api/tasks/'+requested['task_id']).json()
        assert stored['randomization_request']=={'seed':0,'shuffle_sides':False}
        command=client.get('/api/genesis/next').json()
        assert requested['seed']==command['seed']==0
        assert command['shuffle_sides'] is False
        fresh=[client.post('/api/scene/reset?randomize=true').json()['seed'] for _ in range(2)]
        assert fresh[0]!=fresh[1]
        assert client.post('/api/scene/reset?randomize=true&seed=-1').status_code==422
        assert client.post('/api/tasks',json={'instruction':'青を掴んで','randomize':True,'seed':2**32}).status_code==422


def test_randomized_task_uses_the_requested_seed():
    with TestClient(app) as client:
        client.post('/api/tasks',json={'instruction':'青を掴んで','randomize':True,'seed':123,'shuffle_sides':False})
        command=client.get('/api/genesis/next').json()
        assert command['seed']==123 and command['shuffle_sides'] is False


def test_agent_diagnosis_and_failed_attempts_survive_broker_updates():
    with TestClient(app) as client:
        client.post('/api/genesis/register',json={'viewer_id':'owner'})
        task=client.post('/api/tasks',json={'instruction':'青を掴んで'}).json()['task_id']
        client.get('/api/genesis/next?viewer_id=owner')  # Isolated test queue only.
        trace=[{'kind':'reset_and_retry','explanation':'実測の失敗を記録して初期化します。'}]
        client.post('/api/genesis/state',json={'viewer_id':'owner','task_id':task,'agent_trace':trace})
        assert client.get('/api/tasks/'+task).json()['agent_trace']==trace
        attempts=[{'result':{'success':False}},{'result':{'success':True}}]
        client.post('/api/genesis/result',json={'viewer_id':'owner','task_id':task,'success':True,
            'agent_trace':trace,'agent_attempts':attempts,'diagnosis':{'category':'MOTION_BLOCKED'}})
        stored=client.get('/api/tasks/'+task).json()
        assert stored['success'] and stored['agent_attempts']==attempts
        assert stored['diagnosis']['category']=='MOTION_BLOCKED'
