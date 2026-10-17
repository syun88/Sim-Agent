"""Run the verified pick-and-place tasks through the FastAPI task API from the default spawn pose.

Reports success, retries, horizontal/vertical error and grasp_mode for the left-arm red-cube task
and the right-arm blue-cube task.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from backend.app.main import app  # noqa: E402


def run_task(instruction: str) -> dict:
    with TestClient(app) as client:
        # Ensure we start from the default spawn pose.
        reset = client.post("/api/scene/reset?randomize=false")
        assert reset.status_code == 200, reset.text

        created = client.post("/api/tasks", json={"instruction": instruction})
        assert created.status_code == 200, created.text
        task_id = created.json()["task_id"]

        record = None
        for _ in range(600):
            response = client.get(f"/api/tasks/{task_id}")
            assert response.status_code == 200, response.text
            record = response.json()
            # The final record uses "success"; during execution current uses "status".
            if "success" in record:
                break
            time.sleep(0.1)
        assert record is not None, "task record not found"
        assert "success" in record, f"task did not finish: {record}"
        return record


def summarize(record: dict) -> dict:
    attempts = record.get("attempts", [])
    last_attempt = attempts[-1] if attempts else {}
    verification = last_attempt.get("verification", {})
    # Grasp mode is reported by the grasp_object skill in the final attempt.
    grasp_mode = None
    for skill in last_attempt.get("skills", []):
        if skill.get("skill") == "grasp_object":
            grasp_mode = skill.get("grasp_mode")
            break
    return {
        "success": record.get("success"),
        "retries": record.get("retry_count"),
        "horizontal_error_mm": verification.get("horizontal_position_error", 0) * 1000,
        "vertical_error_mm": verification.get("vertical_position_error", 0) * 1000,
        "grasp_mode": grasp_mode,
        "status": "SUCCEEDED" if record.get("success") else "FAILED",
    }


def main():
    print("Running left-arm red-cube task...")
    left = run_task("Move the red cube into the blue zone.")
    print("Running right-arm blue-cube task...")
    right = run_task("Move the blue cube into the green zone.")

    print("\nLeft-arm red-cube task:")
    print(summarize(left))
    print("\nRight-arm blue-cube task:")
    print(summarize(right))


if __name__ == "__main__":
    main()
