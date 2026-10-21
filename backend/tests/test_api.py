from fastapi.testclient import TestClient

from backend.app import simulation
from backend.app.main import app
from backend.app.planning.agent_planner import AgentPlanner
from backend.app.simulation import GPU_ENVIRONMENT_VARIABLE, prefer_gpu_from_environment


def test_health_and_scene():
    with TestClient(app) as client:
        health = client.get("/api/health")
        assert health.status_code == 200
        assert health.json()["robot"] == "OpenArm v2"


def test_config_does_not_expose_secrets():
    with TestClient(app) as client:
        response = client.get("/api/config/status").json()
        assert "password" not in str(response).lower()


def test_documented_gpu_setting_is_read(monkeypatch):
    """`GENESIS_USE_GPU` is documented in `.env.example`, so it must not be a setting with no effect."""
    monkeypatch.delenv(GPU_ENVIRONMENT_VARIABLE, raising=False)
    assert prefer_gpu_from_environment() is False
    monkeypatch.setenv(GPU_ENVIRONMENT_VARIABLE, "1")
    assert prefer_gpu_from_environment() is True
    monkeypatch.setenv(GPU_ENVIRONMENT_VARIABLE, "0")
    assert prefer_gpu_from_environment() is False


def test_factory_hands_the_gpu_setting_to_the_engine(monkeypatch):
    """The setting has to reach the engine, not just be parsed."""
    requested: list[tuple[bool, str]] = []

    class RecordingEngine:
        def __init__(self, prefer_gpu: bool = False, show_viewer: bool = False, scene_profile: str = "cubes"):
            requested.append((prefer_gpu, scene_profile))

    monkeypatch.setattr(simulation, "GenesisOpenArmV2Environment", RecordingEngine)
    monkeypatch.setenv(GPU_ENVIRONMENT_VARIABLE, "1")
    simulation.create_backend()
    assert requested == [(True, "cubes")]
    simulation.create_backend("wine")
    assert requested[-1] == (True, "wine")
