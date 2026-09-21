"""
demo_server.py
================
A dependency-free (Python standard library only) HTTP server exposing the
exact same scoring logic as app/main.py (they share app/scoring.py).

Why this exists: the production API (app/main.py) is written against
FastAPI/uvicorn, which is the right tool for a real deployment (GitHub
Codespaces, Docker, etc. -- see README "Deployment"). But this sandbox has
no network access, so `pip install fastapi uvicorn` cannot run here. This
demo server lets the full scoring pipeline be exercised and verified
end-to-end (same model artifacts, same feature engineering, same output
schema) using ONLY modules already available, so the "deployed app"
deliverable is demonstrably working, not just written.

Run:   python app/demo_server.py [port]
Then:  curl -X POST http://localhost:8000/score -H "Content-Type: application/json" -d '{...}'
       curl http://localhost:8000/health
"""
from __future__ import annotations

import json
import mimetypes
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.scoring import score_policy  # noqa: E402

FRONTEND_INDEX = ROOT / "frontend" / "index.html"
OUTPUTS_DIR = ROOT / "outputs"

REQUIRED_FIELDS = [
    "Area", "VehPower", "VehAge", "DrivAge", "BonusMalus",
    "Density", "Region", "VehGas", "VehBrand", "Exposure",
]


class Handler(BaseHTTPRequestHandler):
    def _send_json(self, payload: dict, status: int = 200):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path, content_type: str | None = None, status: int = 200):
        body = path.read_bytes()
        ctype = content_type or mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):  # quieter logs
        sys.stderr.write("[demo_server] " + (fmt % args) + "\n")

    def do_HEAD(self):
        # Mirrors do_GET's routing but without a body, for spec-correct
        # HEAD support (the frontend also HEADs /outputs/*.png to probe
        # plot availability before rendering the evidence gallery).
        if self.path in ("/", "/index.html") and FRONTEND_INDEX.exists():
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            return
        if self.path in ("/health", "/api"):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            return
        if self.path.startswith("/outputs/"):
            rel = self.path[len("/outputs/"):]
            target = (OUTPUTS_DIR / rel).resolve()
            if OUTPUTS_DIR.resolve() in target.parents and target.exists():
                self.send_response(200)
                self.send_header("Content-Type", mimetypes.guess_type(str(target))[0] or "application/octet-stream")
                self.end_headers()
                return
        self.send_response(404)
        self.end_headers()

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            if FRONTEND_INDEX.exists():
                return self._send_file(FRONTEND_INDEX, "text/html; charset=utf-8")
            return self._send_json({"status": "ok", "note": "frontend/index.html not found"})

        if self.path == "/health":
            return self._send_json({"status": "ok"})

        if self.path == "/api":
            return self._send_json({
                "service": "Bivariate Survival & Claim Frequency Risk Scoring API (stdlib demo)",
                "endpoints": ["/health", "/score (POST)", "/score/batch (POST)"],
                "note": "Equivalent to app/main.py (FastAPI); see README for the production server.",
            })

        if self.path.startswith("/outputs/"):
            rel = self.path[len("/outputs/"):]
            target = (OUTPUTS_DIR / rel).resolve()
            if OUTPUTS_DIR.resolve() in target.parents and target.exists() and target.is_file():
                return self._send_file(target)
            return self._send_json({"error": "not found"}, status=404)

        return self._send_json({"error": "not found"}, status=404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            return self._send_json({"error": "invalid JSON"}, status=400)

        try:
            if self.path == "/score":
                missing = [f for f in REQUIRED_FIELDS if f not in payload]
                if missing:
                    return self._send_json({"error": f"missing fields: {missing}"}, status=422)
                result = score_policy(payload)
                return self._send_json(result)

            elif self.path == "/score/batch":
                policies = payload.get("policies", [])
                results = []
                for p in policies:
                    missing = [f for f in REQUIRED_FIELDS if f not in p]
                    if missing:
                        return self._send_json({"error": f"missing fields: {missing}"}, status=422)
                    results.append(score_policy(p))
                return self._send_json({"results": results})
            else:
                self._send_json({"error": "not found"}, status=404)
        except FileNotFoundError as e:
            self._send_json({"error": str(e)}, status=503)
        except Exception as e:
            self._send_json({"error": f"scoring failed: {e}"}, status=400)


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    print(f"Demo risk-scoring server listening on http://0.0.0.0:{port}")
    print("Endpoints: GET /health, GET /, POST /score, POST /score/batch")
    server.serve_forever()


if __name__ == "__main__":
    main()
