"""Typed judgements from a local decision model, when one is running.

The monitor asks questions whose answers are a choice among named options - "should the robot resume",
"which legal recovery", "is the object held" - and a decision model answers them from a distribution with
zero output tokens, so there is no prose to parse and no way to invent a format. Optional by design: with
no server configured, or a server that is slow or wrong, every caller falls back to what it did before.
"""
from __future__ import annotations

import json
import os
import urllib.request


def base_url() -> str:
    return os.getenv("OPENARM_DECISION_URL", "").strip().rstrip("/")


def available() -> bool:
    return bool(base_url())


def ask(state: str, questions: dict, *, timeout: float = 8.) -> dict | None:
    """Ask the monitor for typed answers, or None if it cannot answer.

    None is the contract for "no judgement available" - never an exception and never a guess, because
    every caller has a deterministic path it must be able to keep using.
    """
    url = base_url()
    if not url or not questions or not state:
        return None
    body = json.dumps({"state": state, "questions": questions}, ensure_ascii=False).encode()
    request = urllib.request.Request(url + "/decide", data=body,
                                     headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read())
    except Exception:
        return None
    return payload if isinstance(payload, dict) and isinstance(payload.get("answers"), dict) else None
