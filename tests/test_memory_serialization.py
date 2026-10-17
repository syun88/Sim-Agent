from pathlib import Path

import pytest

from backend.app.memory.graph_memory import MemoryStore


def test_local_memory_roundtrip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("NEO4J_URI", raising=False)
    monkeypatch.delenv("NEO4J_USERNAME", raising=False)
    monkeypatch.delenv("NEO4J_PASSWORD", raising=False)
    monkeypatch.delenv("NEO4J_DATABASE", raising=False)
    store = MemoryStore(tmp_path / "memory.json")
    assert store.available is False
    record = {"task_id": "t1", "execution_id": "e1", "instruction": "Move red cube", "success": True, "object_ids": ["red_cube"], "skills": ["approach_object"]}
    saved = store.save_execution(record)
    assert saved["task_id"] == "t1"
    assert store.retrieve("place_in_zone", "red_cube")[0]["execution_id"] == "e1"


def test_retrieved_memory_is_a_summary_not_a_whole_record(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A plan's `memory_used` must not carry the whole history inside it.

    Each saved record embeds the earlier records it was planned from, so storing whole records
    nests the entire history exponentially; the fallback file was measured at 4.2 GB with a single
    345 MB record before retrieval was reduced to a summary. This pins that the summary is flat.
    """
    monkeypatch.delenv("NEO4J_URI", raising=False)
    monkeypatch.delenv("NEO4J_USERNAME", raising=False)
    monkeypatch.delenv("NEO4J_PASSWORD", raising=False)
    monkeypatch.delenv("NEO4J_DATABASE", raising=False)
    store = MemoryStore(tmp_path / "memory.json")
    base = {"task_id": "t1", "execution_id": "e1", "instruction": "Move the red cube into the blue zone.", "success": True, "object_ids": ["red_cube"], "skills": ["approach_object"]}
    store.save_execution(base)
    for i in range(3):
        memory = store.retrieve("place_in_zone", "red_cube")
        record = {
            **base,
            "task_id": f"t{i + 2}",
            "execution_id": f"e{i + 2}",
            "attempts": [{"plan": {"memory_used": memory}}],
        }
        store.save_execution(record)
    saved = store.records[-1]
    embedded = saved["attempts"][0]["plan"]["memory_used"]
    assert embedded, "the last record should have retrieved something"
    for summary in embedded:
        assert "attempts" not in summary
        assert "memory_used" not in summary
    assert (tmp_path / "memory.json").stat().st_size < 20_000
