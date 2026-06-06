#!/usr/bin/env python3
"""Fake plot daemon — a dependency-free reference implementation of the
remote-plotter REST contract (see specs/remote-plotter.md §6–§9).

It simulates plotting (no hardware, no pyaxidraw) so the Plottter network
transport + connection-aware dialog can be developed and tested with nothing
plugged in. It also serves as the executable reference the real Raspberry Pi
daemon should mirror.

Usage:
    python3 tests/fakes/fake_plot_daemon.py [--host H] [--port P]
                                        [--token TOK | --no-auth]
                                        [--speed FACTOR]

    --token TOK   require "Authorization: Bearer TOK" on control endpoints
                  (health/version stay open). If neither --token nor --no-auth
                  is given, a token is generated and printed (secure-by-default).
    --no-auth     run fully open (no token).
    --speed       wall-clock seconds the simulated plot takes per layer
                  (default 8). Lower = faster tests.

Endpoints (base path /api/v1):
    GET  /health                 open
    GET  /version                open
    POST /jobs                   {svg, settings} -> {job_id}; 409 if busy
    GET  /jobs/{id}              status/progress
    POST /jobs/{id}/control      {action: pause|resume|stop}
    POST /manual                 {command, settings}; 409 if a job is plotting
"""

from __future__ import annotations

import argparse
import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

API_BASE = "/api/v1"
API_VERSION = "1.0"
DEVICE_NAME = "iDraw H SE A2 (FAKE)"
FIRMWARE = "0.0.0-fake"

# ---------------------------------------------------------------------------
# Shared, thread-safe daemon state. One job at a time (spec §2).
# ---------------------------------------------------------------------------


class _State:
    def __init__(self, layer_seconds: float) -> None:
        self.lock = threading.RLock()
        self.layer_seconds = layer_seconds
        self.job: dict | None = None          # the single active/last job
        self._worker: threading.Thread | None = None
        self._job_counter = 0

    # -- helpers (call with lock held unless noted) --

    def is_busy(self) -> bool:
        return self.job is not None and self.job["state"] in ("plotting", "paused")

    def new_job(self, svg: str, settings: dict) -> dict:
        self._job_counter += 1
        job_id = f"job_{self._job_counter:04d}"
        total_layers = max(1, _count_layers(svg))
        per_layer_pause = bool(settings.get("per_layer_pause", False))
        job = {
            "job_id": job_id,
            "state": "plotting",
            "percent": 0.0,
            "layer": {"index": 1, "total": total_layers, "name": f"Layer 1"},
            "paused_reason": None,
            "elapsed_s": 0,
            "eta_s": int(self.layer_seconds * total_layers),
            "error": None,
            # private control flags
            "_pause": False,
            "_resume": False,
            "_stop": False,
            "_per_layer_pause": per_layer_pause,
            "_started": time.monotonic(),
        }
        self.job = job
        self._worker = threading.Thread(target=_run_job, args=(self, job_id), daemon=True)
        self._worker.start()
        return job


def _count_layers(svg: str) -> int:
    """Best-effort layer count from an SVG (Inkscape layer groups)."""
    n = len(re.findall(r'inkscape:groupmode\s*=\s*"layer"', svg))
    return n if n > 0 else 1


def _public_view(job: dict) -> dict:
    """Strip private (_-prefixed) keys for the API response."""
    return {k: v for k, v in job.items() if not k.startswith("_")}


# ---------------------------------------------------------------------------
# Simulated plot worker
# ---------------------------------------------------------------------------


def _run_job(state: "_State", job_id: str) -> None:
    """Advance a simulated plot, honouring pause/resume/stop and per-layer pause."""
    step = 0.1  # seconds per tick
    while True:
        with state.lock:
            job = state.job
            if job is None or job["job_id"] != job_id:
                return
            total = job["layer"]["total"]
            per_layer = job["_per_layer_pause"]

            if job["_stop"]:
                job["state"] = "stopped"
                job["paused_reason"] = None
                return

            if job["state"] == "paused":
                if job["_resume"]:
                    job["_resume"] = False
                    job["state"] = "plotting"
                    job["paused_reason"] = None
                else:
                    pass  # stay paused; fall through to sleep
            else:
                if job["_pause"]:
                    job["_pause"] = False
                    job["state"] = "paused"
                    job["paused_reason"] = "user"
                else:
                    # advance progress
                    total_seconds = state.layer_seconds * total
                    job["elapsed_s"] = int(time.monotonic() - job["_started"])
                    per_tick = (100.0 / total_seconds) * step
                    new_pct = min(100.0, job["percent"] + per_tick)

                    # detect crossing into a new layer
                    layer_span = 100.0 / total
                    cur_layer = min(total, int(new_pct // layer_span) + 1)
                    crossed = cur_layer > job["layer"]["index"]

                    job["percent"] = new_pct
                    job["eta_s"] = max(0, int(total_seconds - job["elapsed_s"]))

                    if new_pct >= 100.0:
                        job["state"] = "done"
                        job["percent"] = 100.0
                        job["eta_s"] = 0
                        return

                    if crossed:
                        # advance layer; optionally pause for a pen swap
                        job["layer"]["index"] = cur_layer
                        job["layer"]["name"] = f"Layer {cur_layer}"
                        if per_layer:
                            job["state"] = "paused"
                            job["paused_reason"] = "between_layers"
        time.sleep(step)


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------


class _Handler(BaseHTTPRequestHandler):
    # injected by main()
    state: "_State"
    token: str | None

    def log_message(self, fmt, *args):  # quieter logs
        print(f"[fake-daemon] {self.address_string()} {fmt % args}")

    # -- small response helpers --

    def _send(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0) or 0)
        if not length:
            return {}
        try:
            return json.loads(self.rfile.read(length).decode() or "{}")
        except json.JSONDecodeError:
            return {}

    def _authed(self) -> bool:
        if not self.token:
            return True
        header = self.headers.get("Authorization", "")
        return header == f"Bearer {self.token}"

    # -- routing --

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0].rstrip("/")
        if path == f"{API_BASE}/health":
            return self._health()
        if path == f"{API_BASE}/version":
            return self._send(200, {
                "daemon": "fake_plot_daemon",
                "pyaxidraw": None,
                "api_version": API_VERSION,
            })
        m = re.fullmatch(rf"{re.escape(API_BASE)}/jobs/([^/]+)", path)
        if m:
            return self._job_status(m.group(1))
        self._send(404, {"error": "not found"})

    def do_POST(self) -> None:
        path = self.path.split("?", 1)[0].rstrip("/")
        if path == f"{API_BASE}/jobs":
            return self._create_job()
        m = re.fullmatch(rf"{re.escape(API_BASE)}/jobs/([^/]+)/control", path)
        if m:
            return self._control(m.group(1))
        if path == f"{API_BASE}/manual":
            return self._manual()
        self._send(404, {"error": "not found"})

    # -- endpoint implementations --

    def _health(self) -> None:
        with self.state.lock:
            job = self.state.job
            if job and job["state"] in ("plotting", "paused"):
                dev_state = job["state"]
                current = job["job_id"]
            else:
                dev_state = "idle"
                current = None
        self._send(200, {
            "ok": True,
            "device_connected": True,
            "device": DEVICE_NAME,
            "firmware": FIRMWARE,
            "state": dev_state,
            "current_job": current,
            "auth_required": bool(self.token),
            "api_version": API_VERSION,
        })

    def _create_job(self) -> None:
        if not self._authed():
            return self._send(401, {"error": "missing/invalid token"})
        body = self._body()
        svg = body.get("svg")
        settings = body.get("settings", {})
        if not isinstance(svg, str) or not svg:
            return self._send(400, {"error": "missing 'svg'"})
        with self.state.lock:
            if self.state.is_busy():
                return self._send(409, {"error": "plotter busy", "current_job": self.state.job["job_id"]})
            job = self.state.new_job(svg, settings)
            self._send(200, {"job_id": job["job_id"]})

    def _job_status(self, job_id: str) -> None:
        if not self._authed():
            return self._send(401, {"error": "missing/invalid token"})
        with self.state.lock:
            job = self.state.job
            if job is None or job["job_id"] != job_id:
                return self._send(404, {"error": "no such job"})
            self._send(200, _public_view(job))

    def _control(self, job_id: str) -> None:
        if not self._authed():
            return self._send(401, {"error": "missing/invalid token"})
        action = self._body().get("action")
        if action not in ("pause", "resume", "stop"):
            return self._send(400, {"error": "action must be pause|resume|stop"})
        with self.state.lock:
            job = self.state.job
            if job is None or job["job_id"] != job_id:
                return self._send(404, {"error": "no such job"})
            if action == "pause":
                job["_pause"] = True
            elif action == "resume":
                job["_resume"] = True
            elif action == "stop":
                job["_stop"] = True
            self._send(200, {"ok": True, "action": action})

    def _manual(self) -> None:
        if not self._authed():
            return self._send(401, {"error": "missing/invalid token"})
        body = self._body()
        command = body.get("command")
        valid = {"raise_pen", "lower_pen", "disable_xy", "enable_xy", "walk_home"}
        if command not in valid:
            return self._send(400, {"error": f"command must be one of {sorted(valid)}"})
        with self.state.lock:
            if self.state.is_busy():
                return self._send(409, {"error": "plotter busy"})
        # A real daemon would call run_manual_command here; the fake just acks.
        self._send(200, {"ok": True, "command": command})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--token", default=None)
    ap.add_argument("--no-auth", action="store_true")
    ap.add_argument("--speed", type=float, default=8.0, help="simulated seconds per layer")
    args = ap.parse_args()

    if args.no_auth:
        token = None
    elif args.token:
        token = args.token
    else:
        # secure-by-default: generate and print a token
        token = f"fake-{int(time.time()) % 100000:05d}"

    _Handler.state = _State(layer_seconds=args.speed)
    _Handler.token = token

    server = ThreadingHTTPServer((args.host, args.port), _Handler)
    print(f"[fake-daemon] listening on http://{args.host}:{args.port}{API_BASE}")
    print(f"[fake-daemon] auth: {'OPEN (no token)' if not token else 'token = ' + token}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[fake-daemon] shutting down")
        server.shutdown()


if __name__ == "__main__":
    main()
