from fastapi.testclient import TestClient

from backend.app.main import app


def test_health_and_scene():
    with TestClient(app) as client:
        health = client.get("/api/health")
        assert health.status_code == 200
        assert health.json()["robot"] == "OpenArm v2"
        scene = client.get("/api/scene")
        assert scene.status_code == 200
        assert "red_cube" in [item["id"] for item in scene.json()["objects"]]


def test_config_does_not_expose_secrets():
    with TestClient(app) as client:
        response = client.get("/api/config/status").json()
        assert "password" not in str(response).lower()
