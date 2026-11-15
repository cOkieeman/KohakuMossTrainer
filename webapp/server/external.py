"""Discover native lifecycle records and existing on-disk training runs."""

import hashlib
import json
import os
import shutil
import threading
import time
from pathlib import Path

import psutil

from kohakumoss.logging.run_record import registry_directory

from .configuration import ROOT, STATE_ROOT, ProjectInput, TrainSettings
from .native import TRAIN_SCRIPT, render_script, settings_from_knobs
from .processes import process_matches, stop_tree
from .store import ACTIVE, store


def canonical(path):
    return os.path.normcase(str(Path(path).expanduser().resolve()))


def read_record(path):
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError("Invalid run record")
    return value


def live_process(value, previous=None):
    """Return a matching process identity for a currently running manifest."""
    try:
        process = psutil.Process(int(value["pid"]))
        created = process.create_time()
        if created > float(value["started_at"]) + 0.1:
            return None
        if previous and previous.get("worker_created"):
            return (
                process
                if process_matches(process.pid, previous["worker_created"])
                else None
            )
        return (
            process
            if process.is_running() and process.status() != psutil.STATUS_ZOMBIE
            else None
        )
    except (psutil.Error, KeyError, TypeError, ValueError):
        return None


class ExternalObserver:
    """Idempotent run discovery into a durable Studio store."""

    def __init__(self, database=store, registry=None, runs=None, state=None):
        self.store = database
        self.registry = Path(registry) if registry else registry_directory()
        self.runs = Path(runs) if runs else ROOT / "runs"
        self.state = Path(state) if state else STATE_ROOT
        self.lock = threading.Lock()
        self.checked = 0.0
        self.seen = {}

    def discover(self, force=False):
        with self.lock:
            if not force and time.monotonic() - self.checked < 2:
                return
            self.checked = time.monotonic()
            jobs = self.store.jobs()
            managed = {
                canonical(job["output_dir"])
                for job in jobs
                if job.get("origin") != "external"
            }
            for path in self.registry.glob("*.json"):
                self.inspect(path, managed)
            for path in self.runs.rglob("config.json"):
                output = canonical(path.parent)
                if output in managed or (path.parent / "run-state.json").exists():
                    continue
                if (self.state / "jobs" / path.parent.name).is_dir():
                    continue
                self.inspect(path, managed, legacy=True)

    def inspect(self, path, managed, legacy=False):
        try:
            stamp = path.stat().st_mtime_ns
            if self.seen.get(str(path)) == stamp:
                return
            value = read_record(path)
            if legacy:
                if "STEPS" not in value or "OUTPUT_DIR" not in value:
                    return
                value = dict(
                    run_id="legacy-"
                    + hashlib.sha256(canonical(path.parent).encode()).hexdigest()[:24],
                    output_dir=str(path.parent),
                    config=value,
                    status="imported",
                    kind="train",
                    label=path.parent.name,
                    log_path=str(path.parent / "log.txt"),
                )
            elif canonical(value["repository"]) != canonical(ROOT):
                return
            if value.get("studio_job_id"):
                return
            self.register(value, stamp / 1e9)
            if value.get("status") != "running":
                self.seen[str(path)] = stamp
        except (OSError, ValueError, TypeError, KeyError):
            return

    def project(self, value, previous, settings):
        projects = self.store.projects()
        project_id = value.get("project_id") or (previous or {}).get("project_id")
        for project in projects:
            if project["id"] == project_id:
                return project
        output = Path(value["output_dir"]).resolve()
        for project in projects:
            base = ROOT / "runs" / "studio" / project["id"]
            custom = project["config"].get("native_overrides", {}).get("OUTPUT_DIR")
            if output.is_relative_to(base) or (
                custom and canonical(output.parent) == canonical(custom)
            ):
                return project
        return self.store.create_project(
            ProjectInput(
                name=("External / " + str(value.get("label") or output.name))[:120],
                config=TrainSettings.model_validate(settings),
            )
        )

    def register(self, value, modified):
        job_id = "external-" + str(value["run_id"])
        if not all(character.isalnum() or character == "-" for character in job_id):
            raise ValueError("Invalid run identifier")
        previous = self.store.job(job_id)
        if (
            previous
            and previous["status"] not in ACTIVE
            and value["status"] == "running"
        ):
            return
        settings = settings_from_knobs(value["config"])
        project = self.project(value, previous, settings)
        folder = self.state / "jobs" / job_id
        folder.mkdir(parents=True, exist_ok=True)
        snapshot = folder / "train.py"
        if not snapshot.exists():
            snapshot.write_text(
                render_script(TRAIN_SCRIPT, value["config"]), encoding="utf-8"
            )
        status = value["status"]
        process = live_process(value, previous) if status == "running" else None
        cancel = (previous or {}).get("cancel_requested", 0)
        finished = value.get("finished_at")
        error = value.get("error")
        if status == "running":
            status = "stopping" if cancel else "running"
            if process is None:
                status = "cancelled" if cancel else "interrupted"
                finished = time.time()
                error = (
                    None
                    if cancel
                    else "The external process exited without a final run record"
                )
        elif cancel and status == "failed":
            status, error = "cancelled", None
        log_path = value.get("log_path", "")
        if status not in ACTIVE:
            for source, name in (
                (log_path, "console.log"),
                (
                    value.get("metrics_path")
                    or str(Path(value["output_dir"]) / "metrics.jsonl"),
                    "metrics.jsonl",
                ),
            ):
                target = folder / name
                if source and Path(source).is_file() and not target.exists():
                    shutil.copy2(source, target)
            if (folder / "console.log").is_file():
                log_path = str(folder / "console.log")
        self.store.upsert_external(
            dict(
                id=job_id,
                project_id=project["id"],
                project_name=project["name"],
                kind=value.get("kind", "train"),
                status=status,
                phase=value.get("phase", "train"),
                config=settings,
                folder=str(folder),
                output_dir=str(Path(value["output_dir"]).resolve()),
                origin="external",
                run_label=str(value.get("label", "")),
                log_path=log_path,
                can_retry=int(value.get("kind", "train") == "train"),
                worker_pid=value.get("pid"),
                worker_created=process.create_time() if process else None,
                cancel_requested=cancel,
                error=error,
                created_at=value.get("started_at") or modified,
                started_at=value.get("started_at"),
                finished_at=finished,
            )
        )

    def cancel(self, job):
        if job["status"] not in ACTIVE:
            return job
        if not process_matches(job["worker_pid"], job["worker_created"]):
            raise ValueError("The external process identity no longer matches")
        record = read_record(self.registry / f"{job['id'][len('external-'):]}.json")
        if (
            canonical(record["repository"]) != canonical(ROOT)
            or record["pid"] != job["worker_pid"]
            or canonical(record["output_dir"]) != canonical(job["output_dir"])
        ):
            raise ValueError("The external process ownership could not be verified")
        self.store.request_cancel(job["id"])
        stop_tree(psutil.Process(job["worker_pid"]))
        self.discover(force=True)
        return self.store.job(job["id"])


observer = ExternalObserver()
