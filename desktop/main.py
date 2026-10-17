from __future__ import annotations

import atexit
import base64
import json
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import dearpygui.dearpygui as dpg
import httpx
import numpy as np
from PIL import Image
from websockets.sync.client import connect


ROOT = Path(__file__).resolve().parents[1]
BACKEND_URL = "http://127.0.0.1:8000"
WS_URL = "ws://127.0.0.1:8000/ws"
IMAGE_SIZE = (320, 240)


@dataclass
class AppState:
    latest: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    history: list[dict[str, Any]] = field(default_factory=list)
    memory: dict[str, Any] = field(default_factory=dict)
    health: dict[str, Any] = field(default_factory=dict)
    texture: list[float] = field(default_factory=list)
    connected_ws: bool = False
    backend_process: subprocess.Popen | None = None


state = AppState()


def ensure_backend() -> subprocess.Popen | None:
    try:
        response = httpx.get(f"{BACKEND_URL}/api/health", timeout=1.0)
        if response.status_code == 200:
            return None
    except Exception:
        pass
    process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "backend.app.main:app", "--host", "127.0.0.1", "--port", "8000"],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )
    for _ in range(80):
        try:
            if httpx.get(f"{BACKEND_URL}/api/health", timeout=0.5).status_code == 200:
                return process
        except Exception:
            time.sleep(0.1)
    process.terminate()
    raise RuntimeError("Unable to start embedded backend")


def stop_backend() -> None:
    if state.backend_process and state.backend_process.poll() is None:
        state.backend_process.terminate()
        try:
            state.backend_process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            state.backend_process.kill()


def ws_reader() -> None:
    while True:
        try:
            with connect(WS_URL, open_timeout=2) as websocket:
                state.connected_ws = True
                for raw in websocket:
                    message = json.loads(raw)
                    state.latest = message
                    if message.get("events"):
                        state.events = [*message["events"], *state.events][:200]
                    image = message.get("image")
                    if image:
                        state.texture = image_texture_data(image)
        except Exception:
            state.connected_ws = False
            time.sleep(0.6)


def poller() -> None:
    while True:
        try:
            state.health = httpx.get(f"{BACKEND_URL}/api/health", timeout=1.0).json()
            state.history = httpx.get(f"{BACKEND_URL}/api/executions", timeout=1.0).json()
            state.memory = httpx.get(f"{BACKEND_URL}/api/memory", timeout=1.0).json()
        except Exception:
            pass
        time.sleep(1.0)


def image_texture_data(data_url: str) -> list[float]:
    payload = base64.b64decode(data_url.split(",", 1)[1])
    image = Image.open(__import__("io").BytesIO(payload)).convert("RGBA")
    array = np.asarray(image, dtype=np.float32) / 255.0
    return array.reshape(-1).tolist()


def submit_task(force_failure: bool = False) -> None:
    instruction = dpg.get_value("instruction_text")
    def worker():
        try:
            response = httpx.post(
                f"{BACKEND_URL}/api/tasks",
                json={"instruction": instruction, "randomize": False, "force_failure": force_failure},
                timeout=2.0,
            )
            response.raise_for_status()
        except Exception as exc:
            state.events = [{"kind": "desktop_error", "detail": str(exc), "timestamp": time.time()}, *state.events]
    threading.Thread(target=worker, daemon=True).start()


def stop_task() -> None:
    task_id = (state.latest.get("current") or {}).get("task_id")
    if task_id:
        try:
            httpx.post(f"{BACKEND_URL}/api/tasks/{task_id}/stop", timeout=1.0)
        except Exception:
            pass


def reset_scene(randomize: bool) -> None:
    try:
        httpx.post(f"{BACKEND_URL}/api/scene/reset", params={"randomize": randomize}, timeout=1.0)
    except Exception:
        pass


def pretty(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


def update_ui() -> None:
    health = state.health
    current = state.latest.get("current") or {}
    agent_state = state.latest.get("agent_state") or health.get("agent", "unknown")
    dpg.set_value(
        "status_text",
        f"Robot: {health.get('robot', 'unknown')} | Sim: {health.get('simulation', 'unknown')} | "
        f"AI: {health.get('provider', 'unknown')} | Neo4j: {health.get('memory', 'unknown')} | "
        f"Agent: {agent_state} | WS: {'Connected' if state.connected_ws else 'Reconnecting'}",
    )
    if state.texture:
        dpg.set_value("sim_texture", state.texture)

    plan = next((event.get("plan") for event in reversed(state.events) if event.get("kind") == "plan_generated"), None)
    verification = next((event.get("verification") for event in reversed(state.events) if event.get("kind") == "verification_completed"), None)
    memory_event = next((event for event in reversed(state.events) if event.get("kind") == "memory_retrieved"), None)
    dpg.set_value(
        "plan_text",
        pretty({
            "current_task": current,
            "plan": plan,
            "memory_decision": plan.get("memory_decision") if plan else None,
        }) if plan else "No plan yet.",
    )
    dpg.set_value("verification_text", pretty(verification) if verification else "Awaiting verification.")
    retrieved = memory_event.get("memories", []) if memory_event else state.memory.get("history", [])[-3:]
    dpg.set_value(
        "memory_text",
        pretty({
            "backend": "Neo4j" if state.memory.get("available") else "Local fallback",
            "last_error": state.memory.get("last_error"),
            "retrieved": retrieved,
        }),
    )
    dpg.set_value("events_text", "\n".join(f"{event.get('kind')}: {pretty(event.get('detail') or event.get('verification') or event.get('plan') or '')}" for event in state.events[:40]))
    dpg.set_value(
        "history_text",
        pretty([
            {
                "instruction": item.get("instruction"),
                "success": item.get("success"),
                "duration_s": item.get("total_duration"),
                "retries": item.get("retry_count"),
                "failure": item.get("failure_category"),
            }
            for item in state.history[-12:]
        ]),
    )


def build_ui() -> None:
    dpg.create_context()
    dpg.create_viewport(title="OpenArm v2 — Embodied AI Agent", width=1420, height=900)

    with dpg.texture_registry():
        dpg.add_dynamic_texture(*IMAGE_SIZE, [0.02, 0.04, 0.08, 1.0] * (IMAGE_SIZE[0] * IMAGE_SIZE[1]), tag="sim_texture")

    with dpg.window(tag="Primary"):
        dpg.add_text("OpenArm v2 · MuJoCo · Think. Act. Learn.", tag="title_text")
        dpg.add_text("Starting backend...", tag="status_text")
        dpg.add_separator()
        with dpg.group(horizontal=True):
            with dpg.child_window(width=360, height=760, border=True):
                dpg.add_text("Natural Language Task")
                dpg.add_input_text(tag="instruction_text", default_value="Move the red cube into the blue zone.", width=-1)
                with dpg.group(horizontal=True):
                    dpg.add_button(label="Execute", callback=lambda: submit_task(False))
                    dpg.add_button(label="Stop", callback=lambda: stop_task())
                    dpg.add_button(label="Reset", callback=lambda: reset_scene(False))
                    dpg.add_button(label="Random", callback=lambda: reset_scene(True))
                dpg.add_button(label="Run recoverable fault demo", callback=lambda: submit_task(True), width=-1)
                dpg.add_separator()
                dpg.add_text("Agent Plan")
                dpg.add_input_text(tag="plan_text", multiline=True, readonly=True, default_value="No plan yet.", height=220, width=-1)
                dpg.add_text("Verification")
                dpg.add_input_text(tag="verification_text", multiline=True, readonly=True, default_value="Awaiting verification.", height=150, width=-1)

            with dpg.child_window(width=1010, height=760, border=True):
                with dpg.tab_bar():
                    with dpg.tab(label="Simulation"):
                        dpg.add_image("sim_texture", width=960, height=720)
                        dpg.add_text("Live OpenArm v2 render from MuJoCo. Contact-assisted weld grasp mode.")
                    with dpg.tab(label="Agent Events"):
                        dpg.add_input_text(tag="events_text", multiline=True, readonly=True, default_value="", height=-1, width=-1)
                    with dpg.tab(label="Graph Memory"):
                        dpg.add_input_text(tag="memory_text", multiline=True, readonly=True, default_value="", height=-1, width=-1)
                    with dpg.tab(label="History"):
                        dpg.add_input_text(tag="history_text", multiline=True, readonly=True, default_value="", height=-1, width=-1)

    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window("Primary", True)


def main() -> None:
    state.backend_process = ensure_backend()
    atexit.register(stop_backend)
    threading.Thread(target=ws_reader, daemon=True).start()
    threading.Thread(target=poller, daemon=True).start()
    build_ui()
    try:
        while dpg.is_dearpygui_running():
            update_ui()
            dpg.render_dearpygui_frame()
    finally:
        dpg.destroy_context()


if __name__ == "__main__":
    main()
