"""Text-prompted segmentation as a local service (SAM 3.1).

The planner must not do arithmetic and the decision model cannot read spatial layout - measured 2026-10-27: d1
answered "how many cubes" correctly at 0.97 but did not respond to swapping their sides. Segmentation is the
honest fix for the spatial half: a text prompt returns the instance's mask and box, and relations follow from
geometry the caller can check (which box is further left, which mask is partly hidden).

Runs out of tree like the decision server, in the same virtualenv, and is optional: the app treats a missing
service as "no perception" rather than failing.

Usage:
    /data/d1-venv/bin/python scripts/sam_segment_server.py --port 11436
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from transformers import Sam3Model, Sam3Processor

MODEL = "unsloth/sam3.1"
_lock = threading.Lock()
_model = None
_processor = None
_loaded_at = 0.0


def loaded():
    global _model, _processor, _loaded_at
    if _model is None:
        started = time.time()
        _model = Sam3Model.from_pretrained(MODEL, dtype=torch.float32).eval()
        _processor = Sam3Processor.from_pretrained(MODEL)
        _loaded_at = time.time() - started
    return _model, _processor


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
        if self.path != "/segment":
            self._send(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length") or 0)
        try:
            request = json.loads(self.rfile.read(length) or b"{}")
        except ValueError:
            self._send(400, {"error": "invalid json"})
            return
        image = request.get("image")
        prompts = request.get("prompts") or []
        threshold = float(request.get("threshold") or .3)
        if not isinstance(image, str) or not prompts:
            self._send(400, {"error": "image (base64) and prompts (list of strings) are required"})
            return
        try:
            from PIL import Image
            picture = Image.open(io.BytesIO(base64.b64decode(image))).convert("RGB")
        except Exception as error:
            self._send(400, {"error": f"image decode failed: {type(error).__name__}"})
            return
        started = time.time()
        results = {}
        try:
            model, processor = loaded()
            import numpy as np
            for prompt in prompts:
                inputs = processor(images=picture, text=prompt, return_tensors="pt")
                with _lock, torch.no_grad():
                    output = model(**inputs)
                processed = processor.post_process_instance_segmentation(
                    output, threshold=threshold, target_sizes=[picture.size[::-1]])[0]
                boxes = processed.get("boxes")
                scores = processed.get("scores")
                masks = processed.get("masks")
                found = []
                for index in range(len(masks)):
                    box = [float(v) for v in boxes[index].tolist()] if boxes is not None else None
                    mask = masks[index]
                    area = int(mask.sum().item()) if hasattr(mask, "sum") else None
                    ys, xs = (np.nonzero(mask.cpu().numpy())[0:2] if hasattr(mask, "cpu") else ([], []))
                    centroid = ([float(xs.mean()), float(ys.mean())] if len(xs) else None)
                    found.append({"box": box, "area_px": area, "centroid": centroid,
                                  "score": float(scores[index]) if scores is not None else None})
                results[prompt] = found
        except Exception as error:  # a bad prompt must not take the service down
            self._send(500, {"error": f"{type(error).__name__}: {str(error)[:300]}"})
            return
        self._send(200, {"segments": results, "image_size": list(picture.size),
                         "latency_seconds": round(time.time() - started, 2)})

    def log_message(self, *args):
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=11436)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--preload", action="store_true")
    args = parser.parse_args()
    if args.preload:
        loaded()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"sam segment server on http://{args.host}:{args.port} (model {MODEL})", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
