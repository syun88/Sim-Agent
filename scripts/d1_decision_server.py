"""Local typed-decision server for the monitor agent (d1-omni-600M).

A generation model answers by producing text, which then has to be parsed and can be wrong about its own
format. A decision model answers by reading a distribution over named options: zero output tokens, a
probability per option, and no parsing. That is what a monitor needs - "did this succeed", "which legal
recovery", "is the gripper holding" - so the planner can interrupt on a number instead of on prose.

Runs out of tree from the app: its own virtualenv under /data, one process, bound with MemoryMax. The app
talks to it over HTTP and treats it as optional, so a missing or slow server degrades to the deterministic
fallback rather than failing a task.

Usage:
    /data/d1-venv/bin/python scripts/d1_decision_server.py --port 11435
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from transformers import AutoModel

MODEL = "LiquidAI/d1-omni-600M"
_lock = threading.Lock()
_model = None
_loaded_at = 0.0


def model():
    global _model, _loaded_at
    if _model is None:
        started = time.time()
        _model = AutoModel.from_pretrained(MODEL, trust_remote_code=True, dtype=torch.float32).eval()
        _loaded_at = time.time() - started
    return _model


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send(self, code, payload):
        body = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send(200, {"status": "ok", "model": MODEL, "loaded": _model is not None,
                             "load_seconds": round(_loaded_at, 1)})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/decide":
            self._send(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length") or 0)
        try:
            request = json.loads(self.rfile.read(length) or b"{}")
        except ValueError:
            self._send(400, {"error": "invalid json"})
            return
        state, questions = request.get("state"), request.get("questions")
        if not isinstance(state, str) or not isinstance(questions, dict) or not questions:
            self._send(400, {"error": "state (str) and questions (dict) are required"})
            return
        started = time.time()
        try:
            with _lock:  # one forward pass at a time: the model is shared and the host is shared
                answer = model().system_one(state, questions)
        except Exception as error:  # a bad question must not take the server down
            self._send(500, {"error": f"{type(error).__name__}: {str(error)[:300]}"})
            return
        self._send(200, {**answer, "latency_seconds": round(time.time() - started, 3)})

    def log_message(self, *args):
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=11435)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--preload", action="store_true", help="load the weights before serving")
    args = parser.parse_args()
    if args.preload:
        model()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"d1 decision server on http://{args.host}:{args.port} (model {MODEL})", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
