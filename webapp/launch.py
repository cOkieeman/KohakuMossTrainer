"""Start or reuse the local workbench and open its browser page."""

import ctypes
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

if os.name == "nt":
    import msvcrt

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "internal" / "studio"
PORTS = range(8776, 8786)
HTTP = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def is_workbench(port):
    try:
        with HTTP.open(f"http://127.0.0.1:{port}/api/health", timeout=0.5) as response:
            data = json.load(response)
        return (
            data.get("app") == "KohakuMoss Studio"
            and Path(data.get("root", "")).resolve() == ROOT
        )
    except (OSError, ValueError, urllib.error.URLError):
        return False


def ensure_server():
    STATE.mkdir(parents=True, exist_ok=True)
    with (STATE / "launcher.lock").open("a+b") as lock:
        lock.seek(0)
        lock.write(b"0")
        lock.flush()
        lock.seek(0)
        if os.name == "nt":
            msvcrt.locking(lock.fileno(), msvcrt.LK_LOCK, 1)
        try:
            for port in PORTS:
                if is_workbench(port):
                    return f"http://127.0.0.1:{port}"
            if not (ROOT / "webapp" / "frontend" / "dist" / "index.html").is_file():
                raise RuntimeError(
                    "Frontend build is missing. Run npm run build in webapp/frontend."
                )
            for port in PORTS:
                with socket.socket() as probe:
                    try:
                        probe.bind(("127.0.0.1", port))
                    except OSError:
                        continue
                with (STATE / "server.log").open("ab", buffering=0) as log:
                    process = subprocess.Popen(
                        [
                            str(ROOT / ".venv" / "Scripts" / "python.exe"),
                            "-m",
                            "uvicorn",
                            "webapp.server.app:app",
                            "--host",
                            "127.0.0.1",
                            "--port",
                            str(port),
                        ],
                        cwd=ROOT,
                        stdin=subprocess.DEVNULL,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        creationflags=(
                            subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
                        ),
                        env={**os.environ, "PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1"},
                    )
                deadline = time.monotonic() + 25
                while process.poll() is None and time.monotonic() < deadline:
                    if is_workbench(port):
                        return f"http://127.0.0.1:{port}"
                    time.sleep(0.2)
                if process.poll() is None:
                    raise RuntimeError(
                        "Workbench startup timed out. Check internal/studio/server.log."
                    )
            raise RuntimeError(
                "No available workbench port. Check internal/studio/server.log."
            )
        finally:
            if os.name == "nt":
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)


def main():
    try:
        url = ensure_server()
        if "--no-browser" in sys.argv:
            print(url)
        else:
            os.startfile(url)
    except Exception as error:
        if os.name == "nt" and sys.stdout is None:
            ctypes.windll.user32.MessageBoxW(
                None, str(error), "KohakuMoss Studio", 0x10
            )
        else:
            raise


if __name__ == "__main__":
    main()
