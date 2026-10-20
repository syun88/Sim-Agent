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
