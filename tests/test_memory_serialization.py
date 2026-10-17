from pathlib import Path

from backend.app.memory.graph_memory import MemoryStore


def test_local_memory_roundtrip(tmp_path: Path):
    store = MemoryStore(tmp_path / "memory.json")
    assert store.available is False
    record = {"task_id": "t1", "execution_id": "e1", "instruction": "Move red cube", "success": True, "object_ids": ["red_cube"], "skills": ["approach_object"]}
    saved = store.save_execution(record)
    assert saved["task_id"] == "t1"
    assert store.retrieve("place_in_zone", "red_cube")[0]["execution_id"] == "e1"
