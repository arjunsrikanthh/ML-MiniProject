"""Start this project's local Jupyter server and open its demo notebook."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib.request import Request, urlopen
import webbrowser

ROOT = Path(__file__).resolve().parent
RUNTIME = ROOT / ".runtime" / "jupyter"
RUNTIME.mkdir(parents=True, exist_ok=True, mode=0o700)


def running_server():
    for path in RUNTIME.glob("jpserver-*.json"):
        try:
            info = json.loads(path.read_text())
            if info["port"] != 8890 or Path(info["root_dir"]) != ROOT:
                continue
            headers = {"Authorization": "token " + info["token"]} if info.get("token") else {}
            req = Request(info["url"] + "api", headers=headers)
            with urlopen(req, timeout=1):
                return info
        except (OSError, ValueError, KeyError):
            continue
    return None


if __name__ == "__main__":
    info = running_server()
    if info is None:
        env = {**os.environ, "JUPYTER_RUNTIME_DIR": str(RUNTIME),
               "MPLCONFIGDIR": str(ROOT / ".runtime" / "matplotlib"),
               "TF_CPP_MIN_LOG_LEVEL": "2", "OMP_NUM_THREADS": "4",
               "OPENBLAS_NUM_THREADS": "4", "VECLIB_MAXIMUM_THREADS": "4"}
        with (ROOT / ".runtime" / "jupyter.log").open("a") as log:
            proc = subprocess.Popen(
                [sys.executable, "-m", "jupyterlab", "--no-browser",
                 f"--ServerApp.root_dir={ROOT}", "--ServerApp.ip=127.0.0.1",
                 "--IdentityProvider.token=",
                 "--ServerApp.port=8890", "--ServerApp.port_retries=0",
                 "--ServerApp.default_url=/lab/tree/demonstration/demo.ipynb"],
                cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        for _ in range(150):
            info = running_server()
            if info:
                break
            if proc.poll() is not None:
                raise SystemExit("Jupyter could not start. See .runtime/jupyter.log.")
            time.sleep(0.2)
        if info is None:
            raise SystemExit("Jupyter did not become ready. See .runtime/jupyter.log.")
    url = info["url"] + "lab/tree/demonstration/demo.ipynb"
    if "--no-open" not in sys.argv:
        webbrowser.open(url)
        print("Demo opened at http://127.0.0.1:8890/lab/tree/demonstration/demo.ipynb")
    else:
        print("Demo server ready on port 8890.")
