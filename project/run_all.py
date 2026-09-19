"""
One-command launcher for the whole project.

    python run_all.py

What it does:
  1. Trains the pipeline (data_prep -> bivariate_analysis -> train_model)
     ONLY if models/ artifacts are missing, so re-runs after the first
     one are fast.
  2. Starts the FastAPI server (api.main:app) as a background process.
  3. Waits for it to report healthy.
  4. Opens ui/index.html in your default browser, already pointed at
     the local API.
  5. Keeps running in the foreground - press Ctrl+C to stop the API
     server and exit cleanly.
"""
from __future__ import annotations
import argparse
import os
import sys
import time
import signal
import socket
import subprocess
import webbrowser
import urllib.request
import urllib.error

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
UI_PATH = os.path.join(BASE_DIR, "ui", "index.html")
HOST = "127.0.0.1"

REQUIRED_ARTIFACTS = [
    "preprocessor.joblib", "qda_model.joblib", "gbm_model.joblib",
    "mvn_params.joblib", "feature_schema.json",
]


def artifacts_present() -> bool:
    if not all(os.path.exists(os.path.join(MODEL_DIR, f)) for f in REQUIRED_ARTIFACTS):
        return False
    # Files exist, but joblib artifacts can be pickled against a specific
    # scikit-learn version. If loading fails (e.g. a version mismatch
    # like a missing `_loss` module), treat it as "not present" so we
    # retrain fresh against whatever's actually installed here.
    try:
        import joblib
        for f in REQUIRED_ARTIFACTS:
            if f.endswith(".joblib"):
                joblib.load(os.path.join(MODEL_DIR, f))
        return True
    except Exception as e:
        print(f"[run_all] existing model artifacts failed to load ({e!r}) - "
              f"likely a scikit-learn version mismatch. Retraining fresh "
              f"against your installed packages instead.")
        return False


def run_step(script: str):
    print(f"[run_all] running {script} ...")
    result = subprocess.run([sys.executable, os.path.join("src", script)], cwd=BASE_DIR)
    if result.returncode != 0:
        print(f"[run_all] {script} failed (exit code {result.returncode}). Aborting.")
        sys.exit(result.returncode)


def port_in_use(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((host, port)) == 0


def wait_for_health(port: int, timeout: float = 60.0) -> bool:
    url = f"http://{HOST}:{port}/health"
    start = time.time()
    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen(url, timeout=1.5) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, ConnectionError, OSError):
            pass
        time.sleep(0.5)
    return False


def main():
    parser = argparse.ArgumentParser(description="Train (if needed), serve, and open the risk-scoring dashboard.")
    parser.add_argument("--port", type=int, default=8000,
                         help="Port for the API (default: 8000). Use this if 8000 is already taken.")
    parser.add_argument("--no-browser", action="store_true",
                         help="Don't automatically open the dashboard in a browser.")
    args = parser.parse_args()
    port = args.port

    # 1. Train only if artifacts are missing.
    if not artifacts_present():
        print("[run_all] model artifacts not found - running the pipeline once "
              "(this trains the models; takes a minute or two).")
        run_step("data_prep.py")
        run_step("bivariate_analysis.py")
        run_step("train_model.py")
    else:
        print("[run_all] found existing model artifacts, skipping training.")
        print("          (delete the models/ folder to force a retrain.)")

    # 2. Start the API, unless something's already listening on the port.
    proc = None
    if port_in_use(HOST, port):
        print(f"[run_all] something is already listening on {HOST}:{port} - "
              f"assuming the API is already running there.")
        print(f"          (pass a different --port if that's not the case, "
              f"e.g. `python run_all.py --port 8001`)")
    else:
        print(f"[run_all] starting API on http://{HOST}:{port} ...")
        proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "api.main:app",
             "--host", HOST, "--port", str(port)],
            cwd=BASE_DIR,
        )

    # 3. Wait for it to come up.
    print("[run_all] waiting for the API to report healthy ...")
    if not wait_for_health(port):
        print("[run_all] API did not become healthy in time. Check the "
              "terminal output above for errors.")
        if proc:
            proc.terminate()
        sys.exit(1)
    print("[run_all] API is up.")

    # 4. Open the dashboard, pointed at the actual port via a query param
    #    the page reads on load (falls back to localhost:8000 if absent).
    if args.no_browser:
        print(f"[run_all] skipping browser launch (--no-browser). Open "
              f"ui/index.html manually and set the API endpoint field to "
              f"http://{HOST}:{port}")
    else:
        ui_url = "file://" + UI_PATH.replace(os.sep, "/") + f"?api=http://{HOST}:{port}"
        print(f"[run_all] opening dashboard: {ui_url}")
        webbrowser.open(ui_url)
        if port != 8000:
            print(f"[run_all] note: the dashboard defaults its endpoint field to "
                  f"port 8000. If it doesn't auto-fill to {port}, just edit the "
                  f"'API endpoint' field in the page to http://{HOST}:{port}")

    if proc is None:
        print("[run_all] using an already-running API - nothing more to "
              "supervise here, exiting.")
        return

    print("[run_all] running. Press Ctrl+C to stop the API and exit.")
    try:
        proc.wait()
    except KeyboardInterrupt:
        print("\n[run_all] stopping API ...")
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        print("[run_all] stopped.")


if __name__ == "__main__":
    main()
