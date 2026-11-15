"""Model-free coverage of native schemas, queued tasks and configuration imports."""

import ast
import json
import time

import pytest

from kohakumoss.cli.config.validate import load_knobs
from webapp.server.configuration import (
    ProjectInput,
    TrainSettings,
    cache_encoders,
    native_knobs,
    render_train,
)
from webapp.server.inspection import artifact_path, artifacts
from webapp.server.native import (
    TRAIN_SCRIPT,
    import_native,
    operation_catalog,
    render_script,
    schema,
    tool_values,
    write_operation,
)
from webapp.server.store import Store
from webapp.server.worker import phase_commands


def test_all_native_training_fields_are_exposed():
    metadata = schema(TRAIN_SCRIPT)
    assert len(metadata["fields"]) >= 168
    assert {field["name"] for field in metadata["fields"]} == set(metadata["defaults"])
    assert "main" not in metadata["defaults"]
    assert "DTYPES" not in schema("scripts/export/text_quant.py")["defaults"]


def test_all_kogine_operations_have_renderable_schemas(tmp_path):
    settings = TrainSettings(sample_prompt="A forest.")
    operations = operation_catalog()
    assert len(operations) >= 60
    for operation in operations.values():
        if operation.driver == "kogine":
            metadata = schema(operation.script)
            assert {field["name"] for field in metadata["fields"]} == set(
                metadata["defaults"]
            ), operation.id
            ast.parse(
                render_script(
                    operation.script, tool_values(settings, operation.id, tmp_path)
                )
            )


def test_operation_inheritance_and_command_snapshot(tmp_path):
    settings = TrainSettings(
        steps=4000, sample_steps=7, sample_prompt="A forest.", resolution=768
    )
    sample = tool_values(settings, "sample", tmp_path)
    assert sample["STEPS"] == 7 and sample["PROMPT"] == "A forest."
    assert sample["FP8_RECIPE"] == "row"
    assert tool_values(settings, "fp8_quant", tmp_path)["PRECISION"] == "mxfp8"
    assert tool_values(settings, "text_quant", tmp_path)["PRECISION"] == "fp8_compute"
    assert tool_values(settings, "heal_train", tmp_path)["OUT"] == str(tmp_path)
    assert tool_values(settings, "probe_activations", tmp_path)["OUT"].endswith(".json")
    assert sample["SIZES"] == [[768, 768]]
    cached = tool_values(settings, "cache_text", tmp_path)
    assert (
        "project-cache"
        in tool_values(settings, "cache_text", tmp_path, "project-cache")["CACHE_DIR"]
    )
    assert cached["TEXT_ENCODER"] == "kohakumoss.encoders.text_encoder"
    assert cached["TEXT_ENCODER_KWARGS"]["path"] == settings.text_encoder_path.replace(
        "\\", "/"
    )
    assert cached["TARGET_AREA"] == 768 * 768
    write_operation(settings, "sample", tmp_path, tmp_path / "outputs")
    job = {"folder": str(tmp_path), "kind": "sample", "config": settings.model_dump()}
    phase, command = phase_commands(job)[0]
    assert phase == "sample" and command[-1] == str(tmp_path / "operation.py")
    with pytest.raises(ValueError, match="Unknown"):
        tool_values(
            settings.model_copy(
                update={"tool_options": {"sample": {"INVALID_KNOB": 1}}}
            ),
            "sample",
            tmp_path,
        )


def test_native_import_and_overrides(tmp_path):
    path = tmp_path / "imported.py"
    path.write_text(
        'LR = 0.002\nDIM = 16\nTRANSFORMER = "a model.safetensors"\nRESUME = "checkpoint"\n',
        encoding="utf-8",
    )
    imported = import_native(path)
    settings = TrainSettings.model_validate(imported["config"])
    assert settings.learning_rate == 0.002 and settings.rank == 16
    assert settings.transformer_path == "a model.safetensors"
    assert native_knobs(settings, "p", "j")["RESUME"] == "checkpoint"


def test_queue_cancellation_and_next_claim(tmp_path):
    database = Store(tmp_path / "queue.sqlite3")
    project = database.create_project(ProjectInput())
    for index, kind in enumerate(("train", "sample", "check")):
        database.insert_job(
            dict(
                id=str(index),
                project_id=project["id"],
                kind=kind,
                status="queued",
                config=project["config"],
                project_name=project["name"],
                folder="",
                output_dir="",
                created_at=time.time() + index,
            )
        )
    assert database.claim_job()["id"] == "0"
    assert database.claim_job()["id"] == "2"
    assert database.claim_job() is None
    assert database.request_cancel("1")["status"] == "cancelled"
    database.ready_job("1")
    assert database.job("1")["status"] == "cancelled"
    database.update_job("0", status="completed")
    assert database.request_cancel("0")["status"] == "completed"


def test_project_copy_delete_restore(tmp_path):
    database = Store(tmp_path / "project.sqlite3")
    original = database.create_project(
        ProjectInput(
            config=TrainSettings(
                reg_enabled=True, native_overrides={"RESUME": "checkpoint"}
            )
        )
    )
    copy = database.copy_project(original["id"])
    assert copy["id"] != original["id"] and copy["config"] == original["config"]
    assert database.archive_project(original["id"])["archived"] == 1
    assert database.archive_project(original["id"], False)["archived"] == 0


def test_custom_tool_artifact_download_is_confined(tmp_path):
    folder = tmp_path / "job"
    folder.mkdir()
    output = tmp_path / "report.json"
    output.write_text('{"result": 1}', encoding="utf-8")
    (folder / "operation.json").write_text(
        json.dumps({"values": {"OUT": str(output)}}), encoding="utf-8"
    )
    job = {"folder": str(folder), "output_dir": str(folder / "outputs")}
    report = artifacts(job)["files"][0]
    assert artifact_path(job, report["path"]) == output
    with pytest.raises(FileNotFoundError):
        artifact_path(job, "__tool__/../report.json")


def test_cache_uses_effective_native_components_and_plans():
    settings = TrainSettings(
        native_overrides={
            "WEIGHTS": "/pipeline",
            "DEVICES": [1],
            "TEXT_DEVICE": "cpu",
            "TEXT_ENCODER": "custom-text.safetensors",
            "VAE": "custom-vae.safetensors",
            "TEXT_ENCODE": {"memory": "offload", "dtype": "fp16"},
            "VAE_ENCODE": {"memory": "resident"},
            "VAE_DTYPE": "fp32",
            "LATENT_DRAW": "sample",
        }
    )
    values = native_knobs(settings, "p", "j")
    encoders = cache_encoders(values)
    text, image = encoders["TEXT_ENCODER_KWARGS"], encoders["IMAGE_ENCODER_KWARGS"]
    assert text["path"] == "custom-text.safetensors"
    assert text["device"] == "cpu" and text["plan"] == values["TEXT_ENCODE"]
    assert image["path"] == "custom-vae.safetensors"
    assert image["device"] == "cuda:1" and image["dtype"] == "fp32"
    assert image["mode"] == "sample" and image["plan"] == values["VAE_ENCODE"]


def test_train_json_arrays_restore_native_tuple_types(tmp_path):
    path = tmp_path / "train.py"
    path.write_text(
        render_train(
            TrainSettings(
                native_overrides={
                    "DEVICES": [0],
                    "BETAS": [0.9, 0.99],
                    "SAMPLE_SIZE": [1024, 1024],
                }
            ),
            "p",
            "j",
        ),
        encoding="utf-8",
    )
    values = load_knobs(path)
    assert values["DEVICES"] == (0,)
    assert values["BETAS"] == (0.9, 0.99)
    assert values["SAMPLE_SIZE"] == (1024, 1024)


def test_train_only_caches_enabled_offline_components(tmp_path):
    settings = TrainSettings(
        native_overrides={"TEXT_CACHE": "off", "LATENT_CACHE": "offline"}
    )
    phases = [
        phase
        for phase, _ in phase_commands(
            {"folder": str(tmp_path), "kind": "train", "config": settings.model_dump()}
        )
    ]
    assert "text_cache" not in phases and "latent_cache" in phases
