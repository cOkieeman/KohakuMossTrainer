"""Model-free resume ancestry, image indexing and download route checks."""

import json
import os
import shutil
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from webapp.server import gallery_api, native_api
from webapp.server.configuration import ROOT, ProjectInput, TrainSettings
from webapp.server.inspection import artifact_path
from webapp.server.store import Store
from webapp.server.training_history import (
    inherited_artifacts,
    linked_runs,
    sample_points,
)


def saved_run(database, project, root, name, created, resume=""):
    output, folder = root / name, root / "snapshots" / name
    output.mkdir(parents=True, exist_ok=True)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "console.log").write_text('{"plan":{"batches":98}}\n')
    config = TrainSettings(native_overrides={"RESUME": str(resume)} if resume else {})
    database.insert_job(
        dict(
            id=name,
            project_id=project["id"],
            project_name=project["name"],
            kind="train",
            status="completed",
            phase="train",
            config=config.model_dump(),
            folder=str(folder),
            output_dir=str(output),
            created_at=created,
            started_at=created,
        )
    )
    return database.job(name)


def checkpoint(job, step):
    folder = Path(job["output_dir"]) / "checkpoints" / f"step_{step:08d}"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "state.json").write_text(json.dumps(dict(step=step)))
    (folder / "trainable.safetensors").touch()
    (folder / "optim.pt").touch()
    export = folder / "export"
    export.mkdir(exist_ok=True)
    (export / f"{job['id']}.safetensors").touch()
    return folder


def sample(job, step):
    folder = Path(job["output_dir"]) / "samples" / f"step_{step:08d}"
    folder.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / "docs/assets/teaser-style-lora.webp", folder / "000.png")
    (folder / "settings.json").write_text(
        json.dumps(
            dict(
                step=step,
                seeds=[42],
                steps=25,
                cfg=3,
                sizes=[1344, 768],
                prompts=["An illustrated street."],
            )
        )
    )
    return folder


@pytest.fixture
def history(tmp_path):
    database = Store(tmp_path / "studio.sqlite3")
    project = database.create_project(ProjectInput())
    first = saved_run(database, project, tmp_path, "first", 100)
    initial = checkpoint(first, 294)
    sample(first, 294)
    middle = saved_run(database, project, tmp_path, "middle", 200, initial)
    checkpoint(middle, 588)
    latest = checkpoint(middle, 882)
    sample(middle, 588)
    sample(middle, 882)
    current = saved_run(database, project, tmp_path, "current", 300, latest)
    return database, project, first, middle, current


def test_resume_displays_ancestor_weights_checkpoints_and_samples(history):
    database, _, first, middle, current = history
    result = inherited_artifacts(current, database)
    assert [row["step"] for row in result["checkpoints"]] == [882, 588, 294]
    assert {row["job_id"] for row in result["weights"]} == {first["id"], middle["id"]}
    for row in result["weights"]:
        assert artifact_path(database.job(row["job_id"]), row["path"]).is_file()
    points, runs = sample_points(current, database)
    assert [row["step"] for row in points] == [882, 588, 294]
    assert [row["epoch"] for row in points] == [9, 6, 3]
    assert all(row["inherited"] for row in points)
    assert len(runs) == 3


def test_rewind_excludes_samples_and_weights_after_resume_checkpoint(history):
    database, project, _, middle, current = history
    source = Path(middle["output_dir"]) / "checkpoints/step_00000588"
    branch = saved_run(
        database, project, Path(current["output_dir"]).parent, "branch", 400, source
    )
    result = inherited_artifacts(branch, database)
    assert [row["step"] for row in result["checkpoints"]] == [588, 294]
    points, _ = sample_points(branch, database)
    assert [row["step"] for row in points] == [588, 294]


def test_init_and_unrelated_project_do_not_join_resume_history(history):
    database, project, _, middle, current = history
    root = Path(current["output_dir"]).parent
    fresh = saved_run(database, project, root, "fresh", 500)
    fresh["config"]["native_overrides"] = {
        "INIT_FROM": str(Path(middle["output_dir"]) / "checkpoints/step_00000882")
    }
    assert len(linked_runs(fresh, database)) == 1
    another = database.create_project(ProjectInput())
    external = saved_run(
        database,
        another,
        root,
        "another-project",
        600,
        Path(middle["output_dir"]) / "checkpoints/step_00000882",
    )
    assert len(linked_runs(external, database)) == 1


def test_parent_lookup_survives_more_than_200_records(history):
    database, project, _, _, current = history
    for number in range(201):
        database.insert_job(
            dict(
                id=f"noise-{number}",
                project_id=project["id"],
                project_name=project["name"],
                kind="check",
                status="completed",
                config=project["config"],
                folder="",
                output_dir="",
                created_at=1000 + number,
            )
        )
    assert len(database.jobs(project["id"])) == 200
    assert len(linked_runs(current, database)) == 3


def test_partial_sample_metadata_retries_and_unknown_epoch_stays_unknown(history):
    database, _, _, _, current = history
    folder = sample(current, 1176)
    (folder / "settings.json").write_text('{"step":')
    points, _ = sample_points(current, database)
    assert all(point["step"] != 1176 for point in points)
    (folder / "settings.json").write_text('{"step":1176}')
    (Path(current["folder"]) / "console.log").write_text(
        "Native training log without an inspection report"
    )
    points, _ = sample_points(current, database)
    assert points[0]["step"] == 1176 and points[0]["epoch"] is None


def test_latest_resume_links_same_output_predecessor(history):
    database, project, _, middle, current = history
    folder = Path(middle["output_dir"]) / "checkpoints/step_00000882"
    os.utime(folder / "state.json", (250, 250))
    shared = saved_run(
        database, project, Path(current["output_dir"]).parent, "shared", 300, "latest"
    )
    shared["output_dir"] = middle["output_dir"]
    assert linked_runs(shared, database)[1][0]["id"] == middle["id"]


def test_gallery_pagination_inline_images_and_cpu_thumbnails(history, monkeypatch):
    database, _, first, _, current = history
    monkeypatch.setattr(gallery_api, "store", database)
    application = FastAPI()
    application.include_router(gallery_api.router)
    client = TestClient(application)
    page = client.get(f"/api/jobs/{current['id']}/gallery?limit=2").json()
    assert [point["step"] for point in page["points"]] == [882, 588]
    assert page["total"] == 3 and page["next_offset"] == 2
    assert (
        client.get(f"/api/jobs/{current['id']}/gallery?offset=2&limit=2").json()[
            "points"
        ][0]["step"]
        == 294
    )
    relative = "samples/step_00000294/000.png"
    response = client.get(f"/api/jobs/{first['id']}/image/{relative}")
    assert response.status_code == 200 and response.headers[
        "content-disposition"
    ].startswith("inline")
    thumbnail = client.get(f"/api/jobs/{first['id']}/image/{relative}?thumbnail=true")
    assert (
        thumbnail.status_code == 200
        and thumbnail.headers["content-type"] == "image/jpeg"
    )
    assert thumbnail.headers["etag"]
    assert (
        client.get(
            f"/api/jobs/{first['id']}/image/checkpoints/step_00000294/state.json"
        ).status_code
        == 400
    )
    assert (
        client.get(
            f"/api/jobs/{first['id']}/image/samples/%2e%2e/%2e%2e/%2e%2e/outside.png"
        ).status_code
        == 404
    )


def test_gallery_history_crosses_page_boundary_and_refreshes_when_changed(
    history, monkeypatch
):
    database, _, _, _, current = history
    for number in range(1, 42):
        sample(current, 882 + number * 98)
    monkeypatch.setattr(gallery_api, "store", database)
    application = FastAPI()
    application.include_router(gallery_api.router)
    client = TestClient(application)
    url = f"/api/jobs/{current['id']}/gallery?limit=40"
    page = client.get(url).json()
    assert len(page["points"]) == 40 and len(page["history"]) == 44
    assert page["history"][-1]["step"] == 294
    assert page["history"][40]["id"] not in {row["id"] for row in page["points"]}
    revision = page["history_revision"]
    unchanged = client.get(f"{url}&history_revision={revision}").json()
    assert unchanged["history"] is None
    assert unchanged["history_revision"] == revision
    sample(current, 882 + 42 * 98)
    updated = client.get(f"{url}&history_revision={revision}").json()
    assert len(updated["history"]) == 45
    assert updated["history_revision"] != revision


def test_inherited_checkpoint_can_configure_resume_without_starting_training(
    history, monkeypatch
):
    database, project, first, _, current = history
    monkeypatch.setattr(native_api, "store", database)
    checkpoint_path = str(Path(first["output_dir"]) / "checkpoints/step_00000294")
    before = database.job(current["id"])["config"]
    result = native_api.resume(
        project["id"], native_api.ResumeRequest(checkpoint=checkpoint_path)
    )
    assert result["project"]["config"]["native_overrides"]["RESUME"] == checkpoint_path
    assert database.job(current["id"])["config"] == before
    assert len(database.training_jobs(project["id"])) == 3
