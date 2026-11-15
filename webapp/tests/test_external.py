"""Model-free lifecycle, discovery, persistence and process ownership checks."""

import json
import os
import subprocess
import sys
import time

import psutil
import pytest

from kohakumoss.logging.run_record import RunRecord
from webapp.server import native_api
from webapp.server.configuration import ROOT, ProjectInput
from webapp.server.external import ExternalObserver, live_process
from webapp.server.inspection import artifact_path, artifacts, log_chunk, metric_rows
from webapp.server.store import Store


@pytest.fixture
def discovery(tmp_path, monkeypatch):
    registry = tmp_path / "registry"
    monkeypatch.setenv("KOHAKUMOSS_RUN_REGISTRY", str(registry))
    monkeypatch.delenv("KOHAKUMOSS_STUDIO_JOB_ID", raising=False)
    database = Store(tmp_path / "studio.sqlite3")
    return ExternalObserver(database, registry, tmp_path / "runs", tmp_path / "state")


def test_external_lifecycle_logs_metrics_and_restart(discovery, tmp_path):
    output = tmp_path / "outside-runs" / "training"
    config = dict(STEPS=4, OUTPUT_DIR=str(output.parent), RUN_NAME=output.name)
    with RunRecord(output, config, label="external training") as record:
        (output / "log.txt").write_text("real process log\n", encoding="utf-8")
        (output / "metrics.jsonl").write_text('{"step":1,"loss":0.4}\n')
        discovery.discover(force=True)
        job = discovery.store.jobs()[0]
        assert job["origin"] == "external" and job["status"] == "running"
        assert job["config"]["steps"] == 4 and job["can_retry"] == 1
        assert log_chunk(job, 0)["text"].splitlines() == ["real process log"]
        assert {item["name"] for item in artifacts(job)["files"]} >= {
            "log.txt",
            "metrics.jsonl",
        }
        with pytest.raises(FileNotFoundError):
            artifact_path(job, "../outside.txt")
        restarted = ExternalObserver(
            discovery.store, discovery.registry, discovery.runs, discovery.state
        )
        restarted.discover(force=True)
        assert len(discovery.store.jobs()) == 1
        assert discovery.store.jobs()[0]["status"] == "running"
        record.update(phase="train")
    discovery.discover(force=True)
    assert discovery.store.jobs()[0]["status"] == "completed"
    assert discovery.store.jobs()[0]["finished_at"] is not None
    previous = discovery.store.jobs()[0]
    with RunRecord(output, config):
        (output / "log.txt").write_text("second run\n")
        (output / "metrics.jsonl").write_text('{"step":2,"loss":0.3}\n')
        discovery.discover(force=True)
        assert len(discovery.store.jobs()) == 2
        assert log_chunk(previous, 0)["text"].splitlines() == ["real process log"]
        assert metric_rows(previous) == [{"step": 1, "loss": 0.4}]


def test_managed_dedup_failure_and_no_benchmark_retry(discovery, tmp_path, monkeypatch):
    monkeypatch.setenv("KOHAKUMOSS_STUDIO_JOB_ID", "owned-studio-job")
    with RunRecord(tmp_path / "managed", dict(STEPS=3)):
        discovery.discover(force=True)
        assert not discovery.store.jobs()
    monkeypatch.delenv("KOHAKUMOSS_STUDIO_JOB_ID")
    with pytest.raises(RuntimeError):
        with RunRecord(
            tmp_path / "failed", dict(STEPS=3), kind="text_encoder_benchmark"
        ):
            raise RuntimeError("integration failed")
    discovery.discover(force=True)
    job = discovery.store.jobs()[0]
    assert job["status"] == "failed" and not job["can_retry"]
    assert "integration failed" in job["error"]


def test_legacy_import_and_managed_exclusion(discovery):
    for name in ("old-native", "managed"):
        output = discovery.runs / name
        output.mkdir(parents=True)
        (output / "config.json").write_text(
            json.dumps(dict(STEPS=20, OUTPUT_DIR=str(discovery.runs)))
        )
    (discovery.state / "jobs" / "managed").mkdir(parents=True)
    discovery.discover(force=True)
    jobs = discovery.store.jobs()
    assert len(jobs) == 1 and jobs[0]["status"] == "imported"
    assert jobs[0]["started_at"] is None and jobs[0]["config"]["steps"] == 20
    discovery.discover(force=True)
    assert len(discovery.store.jobs()) == 1


def test_pid_reuse_and_terminated_run(discovery, tmp_path):
    process = psutil.Process()
    assert (
        live_process(dict(pid=process.pid, started_at=process.create_time() - 10))
        is None
    )
    with RunRecord(tmp_path / "lost", dict(STEPS=1)) as record:
        record.value["started_at"] = process.create_time() - 10
        record.publish()
        discovery.discover(force=True)
        assert discovery.store.jobs()[0]["status"] == "interrupted"


def test_external_cancellation_preserves_other_process(discovery, tmp_path):
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    command = [sys.executable, "-c", "import time; time.sleep(30)"]
    child = subprocess.Popen(command, cwd=tmp_path, creationflags=flags)
    other = subprocess.Popen(command, cwd=ROOT, creationflags=flags)
    try:
        with RunRecord(tmp_path / "cancel", dict(STEPS=1)) as record:
            record.update(pid=child.pid, started_at=time.time())
            discovery.discover(force=True)
            job = discovery.store.jobs()[0]
            result = discovery.cancel(job)
            assert result["status"] == "cancelled"
            assert child.poll() is not None and other.poll() is None
    finally:
        for process in (child, other):
            if process.poll() is None:
                process.terminate()
            process.wait(10)


def test_external_running_blocks_gpu_queue(discovery, tmp_path):
    project = discovery.store.create_project(ProjectInput())
    with RunRecord(tmp_path / "gpu", dict(STEPS=1)):
        discovery.discover(force=True)
        discovery.store.insert_job(
            dict(
                id="queued",
                project_id=project["id"],
                project_name=project["name"],
                kind="train",
                status="queued",
                config=project["config"],
                folder="",
                output_dir="",
                created_at=time.time(),
            )
        )
        assert discovery.store.claim_job() is None
    discovery.discover(force=True)
    assert discovery.store.claim_job()["id"] == "queued"


def test_external_retry_allocates_new_run_name(discovery, tmp_path, monkeypatch):
    with RunRecord(tmp_path / "retry", dict(STEPS=4, RUN_NAME="old-name")):
        pass
    discovery.discover(force=True)
    job = discovery.store.jobs()[0]
    monkeypatch.setattr(native_api, "store", discovery.store)
    monkeypatch.setattr(native_api, "start_job", lambda project, kind: project)
    retried = native_api.retry(job["id"])
    assert "RUN_NAME" not in retried["config"]["native_overrides"]
    assert (
        discovery.store.job(job["id"])["config"]["native_overrides"]["RUN_NAME"]
        == "old-name"
    )


def test_native_keyboard_stop_records_cancellation(discovery, tmp_path):
    with pytest.raises(KeyboardInterrupt):
        with RunRecord(tmp_path / "keyboard-stop", dict(STEPS=4)):
            raise KeyboardInterrupt()
    discovery.discover(force=True)
    job = discovery.store.jobs()[0]
    assert job["status"] == "cancelled" and job["error"] is None
