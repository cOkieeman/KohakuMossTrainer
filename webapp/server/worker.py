"""Execute one recorded task and update its durable status."""

import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import psutil

from .configuration import ROOT, TrainSettings
from .inspection import inspect_dataset
from .jobs import worker_environment
from .processes import stop_tree
from .store import store


def stop_child(process):
    if process.poll() is not None:
        return
    stop_tree(psutil.Process(process.pid))
    process.wait()


def phase_commands(job):
    scripts = ROOT / ".venv" / "Scripts"
    kogine = str(scripts / "kogine.exe") if os.name == "nt" else "kogine"
    folder = Path(job["folder"])
    operation_path = folder / "operation.json"
    if operation_path.is_file():
        operation = json.loads(operation_path.read_text(encoding="utf-8"))
        if operation["driver"] == "python":
            return [
                (job["kind"], [sys.executable, operation["script"], *operation["args"]])
            ]
        return [
            (
                job["kind"],
                [
                    kogine,
                    "run",
                    operation["script"],
                    "--config",
                    str(folder / "operation.py"),
                ],
            )
        ]
    if job["kind"] == "doctor":
        doctor = (
            str(scripts / "kohakumoss-doctor.exe")
            if os.name == "nt"
            else "kohakumoss-doctor"
        )
        return [
            (
                "doctor",
                [
                    doctor,
                    "--json",
                    str(ROOT / "internal" / "deployment" / "doctor.json"),
                ],
            )
        ]
    commands = [
        (
            "check",
            [
                kogine,
                "run",
                "scripts/data/inspect_data.py",
                "--config",
                str(folder / "data.py"),
            ],
        )
    ]
    settings = TrainSettings.model_validate(job["config"])
    for knob, phase, script in (
        ("TEXT_CACHE", "text_cache", "scripts/data/cache_text.py"),
        ("LATENT_CACHE", "latent_cache", "scripts/data/cache_latents.py"),
    ):
        if job["kind"] == "cache" or (
            job["kind"] == "train"
            and settings.native_overrides.get(knob, settings.cache_mode) == "offline"
        ):
            commands.append(
                (phase, [kogine, "run", script, "--config", str(folder / "data.py")])
            )
    if job["kind"] == "train":
        checker = (
            str(scripts / "kohakumoss-config.exe")
            if os.name == "nt"
            else "kohakumoss-config"
        )
        commands.insert(0, ("validate", [checker, "check", str(folder / "train.py")]))
        commands.append(
            (
                "train",
                [
                    kogine,
                    "run",
                    "scripts/train/moss.py",
                    "--config",
                    str(folder / "train.py"),
                ],
            )
        )
    return commands


def main(job_id):
    job = store.job(job_id)
    if job is None:
        raise ValueError("Unknown task")
    process = psutil.Process()
    store.update_job(
        job_id,
        worker_pid=process.pid,
        worker_created=process.create_time(),
        started_at=time.time(),
    )
    cancelled = False
    code = 0
    child = None
    log_path = Path(job["folder"]) / "console.log"
    environment = worker_environment()
    environment.update(
        KOHAKUMOSS_STUDIO_JOB_ID=job_id,
        KOHAKUMOSS_STUDIO_PROJECT_ID=job["project_id"],
    )
    try:
        with log_path.open("ab", buffering=0) as log:
            for phase, command in phase_commands(job):
                if store.job(job_id)["cancel_requested"]:
                    cancelled = True
                    break
                store.update_job(job_id, status="running", phase=phase)
                log.write(f"\n--- {phase} ---\n".encode())
                child = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    env=environment,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                )
                store.update_job(job_id, child_pid=child.pid)
                while child.poll() is None:
                    if store.job(job_id)["cancel_requested"]:
                        cancelled = True
                        stop_child(child)
                        break
                    time.sleep(0.4)
                code = child.wait()
                store.update_job(job_id, child_pid=None)
                if cancelled or code:
                    break
                if phase == "check":
                    settings = TrainSettings.model_validate(job["config"])
                    analysis = inspect_dataset(
                        settings, lambda: bool(store.job(job_id)["cancel_requested"])
                    )
                    (Path(job["folder"]) / "analysis.json").write_text(
                        json.dumps(analysis, ensure_ascii=True), encoding="utf-8"
                    )
                    store.set_analysis(job["project_id"], analysis, job["config"])
                    if not sum(item["images"] for item in analysis["datasets"]):
                        raise ValueError("Dataset contains no readable training images")
                    if (
                        analysis["regularization"] is not None
                        and not analysis["regularization"]["images"]
                    ):
                        raise ValueError(
                            "Regularization dataset contains no readable images"
                        )
        status = "cancelled" if cancelled else ("completed" if code == 0 else "failed")
        error = (
            f"The {store.job(job_id)['phase']} command exited with code {code}"
            if code and not cancelled
            else None
        )
        store.update_job(
            job_id,
            status=status,
            exit_code=code,
            error=error,
            child_pid=None,
            finished_at=time.time(),
        )
    except InterruptedError:
        if child is not None and child.poll() is None:
            stop_child(child)
        store.update_job(
            job_id, status="cancelled", child_pid=None, finished_at=time.time()
        )
    except Exception as error:
        if child is not None and child.poll() is None:
            stop_child(child)
        with log_path.open("a", encoding="utf-8") as log:
            traceback.print_exc(file=log)
        store.update_job(
            job_id,
            status="failed",
            error=str(error),
            child_pid=None,
            finished_at=time.time(),
        )


if __name__ == "__main__":
    main(sys.argv[1])
