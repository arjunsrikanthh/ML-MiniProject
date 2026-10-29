"""Start or reopen the local drawing canvas without launching training."""
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib.request import urlopen
import webbrowser

ROOT = Path(__file__).resolve().parent
URL = "http://127.0.0.1:7862"


def ready():
    try:
        with urlopen(URL + "/config", timeout=2) as response:
            import json
            return json.load(response).get("title") == "Live drawing — CNN and XGBoost"
    except (OSError, ValueError):
        return False


if __name__ == "__main__":
    if not ready():
        runtime = ROOT / ".runtime"
        runtime.mkdir(exist_ok=True)
        env = {**os.environ, "GRADIO_ANALYTICS_ENABLED": "False", "TF_CPP_MIN_LOG_LEVEL": "2",
               "OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4"}
        with (runtime / "live-demo-7862.log").open("a") as log:
            proc = subprocess.Popen([sys.executable, str(ROOT / "src/live_demo.py")], cwd=ROOT,
                                    env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        for _ in range(180):
            if ready():
                break
            if proc.poll() is not None:
                raise SystemExit("Canvas could not start. See .runtime/live-demo-7862.log.")
            time.sleep(0.5)
        else:
            raise SystemExit("Canvas startup timed out. See .runtime/live-demo-7862.log.")
    if "--no-open" not in sys.argv:
        webbrowser.open(URL)
    print("Live drawing demo ready at " + URL)
