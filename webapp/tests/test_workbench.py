"""Model-free regressions for workbench state, snapshots and process ownership."""

import ast
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import psutil
import pytest
from fastapi.testclient import TestClient

import webapp.server.app as api_module
from kohakumoss_data import DataConfig
from kohakumoss_data.caption.template import build_template
from kohakumoss_data.dataset import MossDataset
from kohakumoss_data.resolution.batching import BATCHINGS, BatchLimits
from webapp.server.configuration import (
    DATA_FIELDS,
    ROOT,
    ProjectInput,
    TrainSettings,
    native_knobs,
    render_train,
    validate_paths,
)
from webapp.server.inspection import artifact_path, inspect_dataset, log_chunk
from webapp.server.jobs import process_matches
from webapp.server.store import BusyJob, Store
from webapp.server.worker import stop_child


def test_snapshot_mappings_and_literal_safety():
    settings = TrainSettings(dataset_path="", sample_prompt="a quote ' and a newline\n")
    knobs = native_knobs(settings, "project", "job")
    assert knobs["DATASETS"][0]["path"] == ""
    assert knobs["SAMPLE_PROMPTS"] == [settings.sample_prompt]
    ast.parse(render_train(settings, "project", "job"))
    assert native_knobs(settings.model_copy(update={"algorithm": "dora"}), "p", "j")[
        "DORA"
    ]
    lokr = native_knobs(
        settings.model_copy(update={"algorithm": "lokr", "factor": 4}), "p", "j"
    )
    assert lokr["ALGO"] == "lokr" and lokr["ALGO_KWARGS"] == {"factor": 4}


def test_fp16_fixed_batch_maps_to_native_packer():
    settings = TrainSettings(precision="fp16", batching="count", batch_size=2)
    knobs = native_knobs(settings, "project", "job")
    assert knobs["PRECISION"] == "fp16" and knobs["FP16_FIX"] == "none"
    limits = BatchLimits(max_tokens=knobs["MAX_TOKENS"], batch_size=knobs["BATCH_SIZE"])
    batches = BATCHINGS[knobs["BATCHING"]](list(range(4)), [4200] * 4, limits)
    assert batches == [[0, 1], [2, 3]]


def test_old_project_loads_batch_defaults(tmp_path):
    database = Store(tmp_path / "studio.sqlite3")
    project = database.create_project(ProjectInput())
    old_config = project["config"]
    old_config.pop("batching")
    old_config.pop("batch_size")
    with database.connection() as db:
        db.execute(
            "UPDATE projects SET config=? WHERE id=?",
            (json.dumps(old_config), project["id"]),
        )
    loaded = database.project(project["id"])["config"]
    assert loaded["batching"] == "tokens" and loaded["batch_size"] == 1
    assert loaded["max_tokens"] == old_config["max_tokens"]


def test_regularization_native_mix_and_inspection(tmp_path):
    training = tmp_path / "train"
    regularization = tmp_path / "reg"
    training.mkdir()
    regularization.mkdir()
    for folder, caption in (
        (training, "An illustrated character."),
        (regularization, "forest, landscape"),
    ):
        shutil.copy2(ROOT / "docs/assets/teaser-style-lora.webp", folder / "image.webp")
        (folder / "image.txt").write_text(caption, encoding="utf-8")
    settings = TrainSettings(
        dataset_path=str(training),
        train_repeat=2,
        reg_enabled=True,
        reg_dataset_path=str(regularization),
        reg_repeat=2,
        reg_loss_weight=0.5,
    )
    entries = native_knobs(settings, "p", "j")["DATASETS"]
    dataset = MossDataset(
        DataConfig(
            datasets=entries,
            cache_dir=str(tmp_path / "cache"),
            latent_cache="off",
            text_cache="off",
        )
    )
    assert dataset.catalog.weight.tolist() == [2, 2]
    assert dataset.loss_weight.tolist() == [1, 0.5]
    assert dataset.caption(0, 0) == "An illustrated character."
    assert dataset.caption(1, 0) == "forest, landscape"
    analysis = inspect_dataset(settings)
    assert analysis["images"] == analysis["regularization"]["images"] == 1
    assert (
        analysis["effective_samples"]
        == analysis["regularization"]["effective_samples"]
        == 2
    )
    assert analysis["regularization"]["preview"][0]["caption"] == "forest, landscape"


def test_jsonl_inspection_uses_image_locators_for_repeated_rows(tmp_path):
    image = tmp_path / "reference.webp"
    shutil.copyfile(ROOT / "docs/assets/teaser-style-lora.webp", image)
    manifest = tmp_path / "data.jsonl"
    manifest.write_text(
        "\n".join(
            json.dumps({"image": image.name, "caption": caption})
            for caption in ("First caption.", "Second caption.")
        ),
        encoding="utf-8",
    )
    settings = TrainSettings(
        native_overrides={"DATASETS": [{"source": "jsonl", "path": str(manifest)}]}
    )
    report = inspect_dataset(settings)
    assert report["images"] == 2 and report["invalid_images"] == 0
    assert {item["path"] for item in report["preview"]} == {image.name}
    assert {item["caption"] for item in report["preview"]} == {
        "First caption.",
        "Second caption.",
    }


def test_regularization_paths_and_disabled_defaults(tmp_path):
    settings = TrainSettings(dataset_path=str(tmp_path), reg_enabled=True)
    with pytest.raises(ValueError, match="regularization directory"):
        validate_paths(settings, "check")
    settings.reg_enabled = False
    validate_paths(settings, "check")
    assert len(native_knobs(settings, "p", "j")["DATASETS"]) == 1


def test_regularization_change_invalidates_analysis(tmp_path):
    database = Store(tmp_path / "studio.sqlite3")
    project = database.create_project(ProjectInput())
    old = project["config"].copy()
    for key in DATA_FIELDS:
        if key.startswith("reg_") or key == "train_repeat":
            old.pop(key)
    database.set_analysis(project["id"], {"images": 1}, old)
    assert database.project(project["id"])["analysis"] == {"images": 1}
    changed = ProjectInput(
        config=TrainSettings(reg_enabled=True, reg_dataset_path=str(tmp_path))
    )
    database.save_project(project["id"], changed)
    database.set_analysis(project["id"], {"images": 2}, old)
    assert database.project(project["id"])["analysis"] is None


def test_regularization_thumbnail_root(tmp_path, monkeypatch):
    training, regularization = tmp_path / "train", tmp_path / "reg"
    training.mkdir()
    regularization.mkdir()
    shutil.copy2(
        ROOT / "docs/assets/teaser-style-lora.webp", regularization / "reg.webp"
    )
    config = TrainSettings(
        dataset_path=str(training),
        reg_enabled=True,
        reg_dataset_path=str(regularization),
    ).model_dump()
    monkeypatch.setattr(
        api_module, "get_project", lambda project_id: {"config": config}
    )
    client = TestClient(api_module.app, base_url="http://127.0.0.1")
    url = "/api/projects/test/thumbnail"
    assert (
        client.get(
            url, params={"relative": "reg.webp", "dataset": "regularization"}
        ).status_code
        == 200
    )
    assert client.get(url, params={"relative": "reg.webp"}).status_code == 404
    assert client.get(url, params={"relative": "../reg/reg.webp"}).status_code == 404


@pytest.mark.parametrize("caption_format", ["nl", "tags"])
def test_native_caption_template(caption_format):
    knobs = native_knobs(TrainSettings(caption_format=caption_format), "p", "j")
    build_template(knobs["CAPTION"])


def test_atomic_gpu_admission(tmp_path):
    database = Store(tmp_path / "studio.sqlite3")
    project = database.create_project(ProjectInput())

    def submit(number):
        job = dict(
            id=str(number),
            project_id=project["id"],
            kind="train",
            status="queued",
            config=project["config"],
            project_name=project["name"],
            folder="",
            output_dir="",
            created_at=time.time(),
        )
        try:
            database.insert_job(job)
            return True
        except BusyJob:
            return False

    with ThreadPoolExecutor(max_workers=4) as pool:
        assert sum(pool.map(submit, range(4))) == 4
    with ThreadPoolExecutor(max_workers=4) as pool:
        claimed = list(pool.map(lambda _: database.claim_job(), range(4)))
    assert sum(item is not None for item in claimed) == 1
    assert sum(item["status"] == "queued" for item in database.jobs()) == 3
    active = next(item for item in claimed if item is not None)
    database.update_job(active["id"], status="completed")
    assert database.claim_job() is not None


def test_analysis_does_not_replace_changed_dataset(tmp_path):
    database = Store(tmp_path / "studio.sqlite3")
    project = database.create_project(ProjectInput())
    changed = ProjectInput(config=TrainSettings(dataset_path=str(tmp_path)))
    database.save_project(project["id"], changed)
    database.set_analysis(project["id"], {"images": 7}, project["config"])
    assert database.project(project["id"])["analysis"] is None
    database.set_analysis(project["id"], {"images": 7}, changed.config.model_dump())
    assert database.project(project["id"])["analysis"] == {"images": 7}


def test_log_utf8_boundary_and_terminal_drain(tmp_path):
    path = tmp_path / "console.log"
    encoded = "caption \u4e2d\u6587".encode()
    path.write_bytes(encoded[:-1])
    job = {"folder": str(tmp_path), "status": "running"}
    first = log_chunk(job, 0)
    assert "\ufffd" not in first["text"]
    path.write_bytes(encoded)
    second = log_chunk(job, first["offset"])
    assert first["text"] + second["text"] == encoded.decode()
    path.write_bytes(b"a" * 300_000)
    job["status"] = "completed"
    chunk = log_chunk(job, 0)
    assert chunk["more"] and chunk["offset"] == 256_000
    assert not log_chunk(job, chunk["offset"])["more"]


def test_pid_ownership_and_confined_artifacts(tmp_path):
    process = psutil.Process(os.getpid())
    assert process_matches(process.pid, process.create_time())
    assert not process_matches(process.pid, process.create_time() - 10)
    assert not process_matches(None, None)
    outside = tmp_path / "outside.txt"
    outside.write_text("test")
    output = tmp_path / "output"
    output.mkdir()
    with pytest.raises(FileNotFoundError):
        artifact_path({"output_dir": str(output)}, "../outside.txt")


def test_inspection_cancellation(tmp_path):
    (tmp_path / "image.png").write_bytes(b"invalid image")
    with pytest.raises(InterruptedError):
        inspect_dataset(TrainSettings(dataset_path=str(tmp_path)), lambda: True)


def test_native_sidecars_and_kohya_layout(tmp_path):
    folder = tmp_path / "character"
    folder.mkdir()
    shutil.copyfile(ROOT / "docs/assets/teaser-style-lora.webp", folder / "sample.webp")
    (folder / "sample.txt").write_text("old caption", encoding="utf-8")
    (folder / "sample.json").write_text(
        '{"caption_nl":"current caption"}', encoding="utf-8"
    )
    settings = TrainSettings(dataset_path=str(tmp_path), source="kohya")
    result = inspect_dataset(settings)
    assert result["images"] == 1 and result["captions"] == 1
    assert result["preview"][0]["caption"] == "current caption"


def test_stop_only_owned_child():
    command = [sys.executable, "-c", "import time; time.sleep(30)"]
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    child = subprocess.Popen(command, creationflags=flags)
    unrelated = subprocess.Popen(command, creationflags=flags)
    try:
        stop_child(child)
        assert child.poll() is not None
        assert unrelated.poll() is None
    finally:
        if child.poll() is None:
            child.kill()
        unrelated.terminate()
        unrelated.wait(10)
