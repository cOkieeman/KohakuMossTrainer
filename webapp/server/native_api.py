"""Project management and native workflow endpoints."""

import ast
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

from .configuration import ROOT, STATE_ROOT, ProjectInput, TrainSettings, render_train
from .jobs import start_job
from .native import (
    BINDINGS,
    TRAIN_SCRIPT,
    get_operation,
    import_native,
    operation_catalog,
    render_script,
    required_inputs,
    schema,
    tool_values,
    train_values,
    validate_snapshot,
)
from .store import store

router = APIRouter(prefix="/api")


class ImportRequest(BaseModel):
    path: str = ""
    text: str = ""


class ResumeRequest(BaseModel):
    checkpoint: str
    fresh: bool = False


def project_for(project_id):
    project = store.project(project_id)
    if project is None:
        raise HTTPException(404, "Project not found")
    return project


@router.get("/native/operations")
def operations():
    return {
        "operations": [asdict(item) for item in operation_catalog().values()],
        "platform": sys.platform,
        "bindings": BINDINGS,
    }


@router.post("/native/resolve/{kind}")
def resolve(kind: str, settings: TrainSettings, project_id: str = "preview"):
    if kind == "train":
        return {**schema(TRAIN_SCRIPT), "values": train_values(settings, project_id)}
    operation = get_operation(kind)
    if operation.driver == "python":
        return {
            "script": operation.script,
            "fields": [],
            "defaults": {},
            "values": {},
            "doc": ast.get_docstring(
                ast.parse((ROOT / operation.script).read_text(encoding="utf-8"))
            )
            or "",
        }
    values = tool_values(settings, kind, ROOT / "out" / "studio" / kind, project_id)
    if "OUT" in values and not settings.tool_options.get(kind, {}).get("OUT"):
        values["OUT"] = ""
    return {**schema(operation.script), "values": values}


@router.post("/native/validate/{kind}")
def validate(kind: str, settings: TrainSettings, project_id: str = "preview"):
    if kind != "train" and get_operation(kind).driver == "python":
        return {"errors": [], "valid": True, "scope": "argument-list"}
    STATE_ROOT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="validate-", dir=STATE_ROOT) as temporary:
        path = Path(temporary) / "config.py"
        values = (
            train_values(settings)
            if kind == "train"
            else tool_values(settings, kind, Path(temporary) / "output", project_id)
        )
        text = (
            render_train(settings, "preview", "manual-run")
            if kind == "train"
            else render_script(
                get_operation(kind).script,
                values,
            )
        )
        path.write_text(text, encoding="utf-8")
        errors = validate_snapshot(path)
        errors.extend(required_inputs(kind, values))
    if (
        kind == "train"
        and os.name == "nt"
        and train_values(settings).get("PARALLEL") != "none"
    ):
        errors.append("Native multi-GPU execution requires Linux")
    return {"errors": errors, "valid": not errors, "scope": "configuration"}


@router.get("/native/help/{kind}")
def operation_help(kind: str):
    operation = get_operation(kind)
    if operation.driver != "python":
        return {"text": schema(operation.script)["doc"]}
    result = subprocess.run(
        [sys.executable, str(ROOT / operation.script), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    return {"text": result.stdout + result.stderr}


@router.get("/projects/{project_id}/studio.json")
def export_project(project_id: str):
    project = project_for(project_id)
    return JSONResponse(
        {"name": project["name"], "config": project["config"]},
        headers={
            "Content-Disposition": f'attachment; filename="studio-{project_id}.json"'
        },
    )


@router.get("/projects/{project_id}/native/{kind}.py")
def export_operation(project_id: str, kind: str):
    project = project_for(project_id)
    settings = TrainSettings.model_validate(project["config"])
    operation = get_operation(kind)
    if operation.driver == "python":
        command = [
            str(ROOT / ".venv/Scripts/python.exe"),
            str(ROOT / operation.script),
            *settings.tool_args.get(kind, []),
        ]
        text = (
            '"""Run the saved native tool arguments."""\n\n'
            "import subprocess\n\n"
            f"subprocess.run({ascii(command)}, cwd={ascii(str(ROOT))}, check=True)\n"
        )
        return Response(
            text,
            media_type="text/plain",
            headers={"Content-Disposition": f'attachment; filename="{kind}.py"'},
        )
    text = render_script(
        operation.script,
        tool_values(settings, kind, ROOT / "out" / "studio" / kind, project_id),
    )
    return Response(
        text,
        media_type="text/plain",
        headers={
            "Content-Disposition": f'attachment; filename="{kind}-{project_id}.py"'
        },
    )


@router.post("/projects/import", status_code=201)
def import_project(value: ImportRequest):
    if value.path:
        path = Path(value.path).expanduser().resolve()
        if not path.is_file():
            raise HTTPException(404, "Configuration file not found")
        try:
            imported = import_native(path)
        except (OSError, SyntaxError, ImportError, TypeError, SystemExit) as error:
            raise HTTPException(400, f"Cannot import configuration: {error}") from error
    else:
        imported = json.loads(value.text)
    if "config" not in imported:
        imported = {
            "name": "Imported project",
            "config": {"native_overrides": imported},
        }
    return store.create_project(ProjectInput.model_validate(imported))


@router.post("/projects/{project_id}/copy", status_code=201)
def copy_project(project_id: str):
    project_for(project_id)
    return store.copy_project(project_id)


@router.delete("/projects/{project_id}")
def archive_project(project_id: str):
    project_for(project_id)
    return store.archive_project(project_id)


@router.post("/projects/{project_id}/restore")
def restore_project(project_id: str):
    project_for(project_id)
    return store.archive_project(project_id, False)


@router.post("/projects/{project_id}/resume")
def resume(project_id: str, value: ResumeRequest):
    project = project_for(project_id)
    overrides = {**project["config"]["native_overrides"]}
    overrides["INIT_FROM" if value.fresh else "RESUME"] = value.checkpoint
    overrides.pop("RESUME" if value.fresh else "INIT_FROM", None)
    project["config"]["native_overrides"] = overrides
    saved = store.save_project(
        project_id,
        ProjectInput(
            name=project["name"], config=TrainSettings.model_validate(project["config"])
        ),
    )
    return {"project": saved}


@router.post("/jobs/{job_id}/retry", status_code=201)
def retry(job_id: str):
    job = store.job(job_id)
    if job is None:
        raise HTTPException(404, "Task not found")
    if job["status"] in ("queued", "starting", "running", "stopping", "preparing"):
        raise HTTPException(409, "The task is still active")
    if not job.get("can_retry", 1):
        raise HTTPException(400, "This external task cannot be resubmitted as training")
    if job.get("origin") == "external":
        job["config"]["native_overrides"].pop("RUN_NAME", None)
    project = {
        "id": job["project_id"],
        "name": job["project_name"],
        "config": job["config"],
    }
    return start_job(project, job["kind"])
