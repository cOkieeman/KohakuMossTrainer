"""Snapshot-based job submission and local worker ownership."""

import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

import psutil

from .configuration import (
    ROOT,
    STATE_ROOT,
    TrainSettings,
    native_knobs,
    validate_paths,
    write_snapshot,
)
from .external import observer
from .native import get_operation, train_values, write_operation
from .processes import process_matches
from .store import ACTIVE, store


def reconcile():
    observer.discover()
    for job in store.jobs():
        if job.get("origin") == "external":
            continue
        if job["status"] not in ACTIVE or job["status"] == "queued":
            continue
        if (
            not job["worker_pid"]
            and time.time() - (job["started_at"] or job["created_at"]) < 15
        ):
            continue
        if not process_matches(job["worker_pid"], job["worker_created"]):
            store.update_job(
                job["id"],
                status="interrupted",
                finished_at=time.time(),
                error="The task worker exited before recording a result",
            )


def worker_environment():
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONUTF8": "1",
            "PYTHONUNBUFFERED": "1",
            "PYTHONFAULTHANDLER": "1",
            "NO_COLOR": "1",
            "KOHAKUMOSS_KERNEL_CACHE": str(ROOT / ".cache" / "kernels"),
            "TRITON_CACHE_DIR": str(ROOT / ".cache" / "triton"),
            "TORCHINDUCTOR_CACHE_DIR": str(ROOT / ".cache" / "torchinductor"),
        }
    )
    return environment


def start_job(project, kind):
    reconcile()
    settings = TrainSettings.model_validate(project["config"])
    if settings.native_overrides.get("RESUME") == "latest":
        base = Path(native_knobs(settings, project["id"], "manual-run")["OUTPUT_DIR"])
        checkpoints = [
            path.parent
            for path in base.rglob("state.json")
            if (path.parent / "trainable.safetensors").is_file()
        ]
        if not checkpoints:
            raise ValueError("No complete checkpoint was found for this project")
        settings.native_overrides["RESUME"] = str(
            max(checkpoints, key=lambda path: (path / "state.json").stat().st_mtime)
        )
    if kind not in ("check", "cache", "train", "doctor"):
        operation = get_operation(kind)
        if operation.linux and os.name == "nt":
            raise ValueError(
                "This native operation requires Linux; its configuration can be exported"
            )
    if (
        kind == "train"
        and os.name == "nt"
        and train_values(settings).get("PARALLEL") != "none"
    ):
        raise ValueError(
            "Native multi-GPU training requires Linux; export this configuration for Linux"
        )
    validate_paths(settings, kind)
    job_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    folder = STATE_ROOT / "jobs" / job_id
    output = ROOT / "runs" / "studio" / project["id"] / job_id
    if kind == "train":
        values = native_knobs(settings, project["id"], job_id)
        output = Path(values["OUTPUT_DIR"]).expanduser().resolve() / values["RUN_NAME"]
    job = {
        "id": job_id,
        "project_id": project["id"],
        "kind": kind,
        "status": "preparing",
        "config": settings.model_dump(),
        "project_name": project["name"],
        "folder": str(folder),
        "output_dir": str(output),
        "created_at": time.time(),
    }
    store.insert_job(job)
    try:
        if kind == "doctor":
            folder.mkdir(parents=True)
        else:
            write_snapshot(settings, project["id"], job_id)
        if kind not in ("check", "cache", "train", "doctor"):
            write_operation(settings, kind, folder, output, project["id"])
        store.ready_job(job_id)
        dispatch()
    except (OSError, ValueError, psutil.Error) as error:
        store.update_job(
            job_id, status="failed", error=str(error), finished_at=time.time()
        )
        raise
    return store.job(job_id)


def dispatch():
    while job := store.claim_job():
        launch_worker(job)


def launch_worker(job):
    job_id, folder = job["id"], STATE_ROOT / "jobs" / job["id"]
    try:
        with (folder / "worker.log").open("ab", buffering=0) as log:
            process = subprocess.Popen(
                [sys.executable, "-m", "webapp.server.worker", job_id],
                cwd=ROOT,
                env=worker_environment(),
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
        created = psutil.Process(process.pid).create_time()
        store.update_job(job_id, worker_pid=process.pid, worker_created=created)
    except (OSError, ValueError, psutil.Error) as error:
        store.update_job(
            job_id, status="failed", error=str(error), finished_at=time.time()
        )


def cancel_job(job):
    if job.get("origin") == "external":
        return observer.cancel(job)
    return store.request_cancel(job["id"])
