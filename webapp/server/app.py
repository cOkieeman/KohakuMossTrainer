"""Loopback API and static frontend for KohakuMoss Studio."""

import asyncio
import importlib.metadata
import io
import json
import os
import subprocess
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from PIL import Image, ImageOps
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .configuration import (
    MODEL_ROOT,
    ROOT,
    JobInput,
    ProjectInput,
    TrainSettings,
    render_train,
)
from .gallery_api import router as gallery_router
from .inspection import (
    artifact_path,
    browse,
    job_log_path,
    log_chunk,
    metric_rows,
    preview_bytes,
)
from .jobs import cancel_job, dispatch, reconcile, start_job
from .native import BINDINGS
from .native_api import router as native_router
from .store import BusyJob, store
from .training_history import inherited_artifacts

DIST = ROOT / "webapp" / "frontend" / "dist"
GPU_CACHE = {"time": 0, "value": None}


@asynccontextmanager
async def lifespan(application):
    reconcile()
    if not store.projects():
        store.create_project(ProjectInput(name="\u65b0\u8bad\u7ec3\u9879\u76ee"))

    async def service_queue():
        while True:
            await asyncio.to_thread(reconcile)
            await asyncio.to_thread(dispatch)
            await asyncio.sleep(1)

    task = asyncio.create_task(service_queue())
    try:
        yield
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


app = FastAPI(title="KohakuMoss Studio", version="0.1.0", lifespan=lifespan)
app.include_router(native_router)
app.include_router(gallery_router)
app.add_middleware(
    TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "[::1]"]
)


@app.middleware("http")
async def local_origin(request: Request, call_next):
    origin = request.headers.get("origin")
    if request.method not in ("GET", "HEAD", "OPTIONS") and origin:
        parsed = urlparse(origin)
        if parsed.hostname not in (
            "127.0.0.1",
            "localhost",
            "::1",
        ) or parsed.netloc != request.headers.get("host"):
            return JSONResponse(
                {"detail": "Requests must come from the local workbench"},
                status_code=403,
            )
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; "
        "script-src 'self'; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'"
    )
    return response


@app.exception_handler(ValueError)
async def value_error(request, error):
    return JSONResponse({"detail": str(error)}, status_code=400)


def get_project(project_id):
    value = store.project(project_id)
    if value is None:
        raise HTTPException(404, "Project not found")
    return value


def get_job(job_id):
    value = store.job(job_id)
    if value is None:
        raise HTTPException(404, "Task not found")
    return value


def gpu_state():
    if time.monotonic() - GPU_CACHE["time"] < 3:
        return GPU_CACHE["value"]
    value = None
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=4,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        if result.returncode == 0 and result.stdout.strip():
            name, total, used, utilization = result.stdout.splitlines()[0].split(",")
            value = {
                "name": name.strip(),
                "total_mib": float(total),
                "used_mib": float(used),
                "utilization": float(utilization),
            }
    except (OSError, ValueError, subprocess.TimeoutExpired):
        value = None
    GPU_CACHE.update(time=time.monotonic(), value=value)
    return value


def environment():
    doctor_path = ROOT / "internal" / "deployment" / "doctor.json"
    doctor = {}
    if doctor_path.is_file():
        try:
            doctor = json.loads(doctor_path.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            doctor = {}
    defaults = TrainSettings()
    assets = []
    transformers = sorted((MODEL_ROOT / "diffusion_models").glob("*.safetensors"))
    paths = [("transformer_path", path) for path in transformers]
    if not paths:
        paths.append(("transformer_path", Path(defaults.transformer_path)))
    for key, directory in (("text_encoder_path", "text_encoders"), ("vae_path", "vae")):
        files = sorted((MODEL_ROOT / directory).glob("*.safetensors"))
        paths.extend((key, path) for path in files or [Path(getattr(defaults, key))])
    for key, path in paths:
        assets.append(
            {
                "key": key,
                "path": str(path),
                "name": path.name,
                "exists": path.is_file(),
                "size": path.stat().st_size if path.is_file() else 0,
            }
        )
    packages = {}
    for name in ("kohakumoss", "torch", "triton-windows", "lycoris-lora"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "root": str(ROOT),
        "python": sys.version.split()[0],
        "packages": packages,
        "gpu": gpu_state(),
        "assets": assets,
        "doctor": doctor,
        "checked_at": doctor_path.stat().st_mtime if doctor_path.is_file() else None,
        "ready": bool(doctor)
        and not doctor.get("missing")
        and not doctor.get("backend_failures")
        and all(item["exists"] for item in assets),
    }


@app.get("/api/health")
def health():
    return {"app": "KohakuMoss Studio", "status": "ok", "root": str(ROOT)}


@app.get("/api/bootstrap")
def bootstrap():
    reconcile()
    return {
        "projects": store.projects(),
        "jobs": store.jobs(),
        "environment": environment(),
        "bindings": BINDINGS,
    }


@app.get("/api/environment")
def get_environment():
    return environment()


@app.get("/api/projects")
def projects():
    reconcile()
    return store.projects()


@app.post("/api/projects", status_code=201)
def create_project(value: ProjectInput):
    return store.create_project(value)


@app.put("/api/projects/{project_id}")
def update_project(project_id: str, value: ProjectInput):
    get_project(project_id)
    return store.save_project(project_id, value)


@app.get("/api/projects/{project_id}/config")
def project_config(project_id: str):
    project = get_project(project_id)
    text = render_train(
        TrainSettings.model_validate(project["config"]), project_id, "manual-run"
    )
    return Response(
        text,
        media_type="text/plain",
        headers={
            "Content-Disposition": f'attachment; filename="kohakumoss-{project_id}.py"'
        },
    )


@app.get("/api/browse")
def browse_path(path: str = "", mode: str = "directory"):
    try:
        return browse(path, mode)
    except PermissionError as error:
        raise HTTPException(403, "This directory cannot be read") from error


@app.get("/api/models")
def models():
    rows = []
    for path in sorted((MODEL_ROOT / "diffusion_models").glob("*.safetensors")):
        rows.append({"path": str(path), "name": path.name, "size": path.stat().st_size})
    return rows


@app.get("/api/jobs")
def jobs(project_id: str | None = None):
    reconcile()
    return store.jobs(project_id)


@app.post("/api/projects/{project_id}/jobs", status_code=201)
def create_job(project_id: str, value: JobInput):
    try:
        return start_job(get_project(project_id), value.kind)
    except BusyJob as error:
        raise HTTPException(409, str(error)) from error


@app.get("/api/jobs/{job_id}")
def job_detail(job_id: str):
    reconcile()
    job = get_job(job_id)
    return {
        "job": job,
        "metrics": metric_rows(job),
        "artifacts": inherited_artifacts(job, store),
    }


@app.post("/api/jobs/{job_id}/cancel")
def stop_job(job_id: str):
    return cancel_job(get_job(job_id))


@app.get("/api/jobs/{job_id}/log")
def job_log(job_id: str, offset: int = -1):
    return log_chunk(get_job(job_id), offset)


@app.get("/api/jobs/{job_id}/console.log")
def download_log(job_id: str):
    path = job_log_path(get_job(job_id))
    if not path.is_file():
        raise HTTPException(404, "Log has not been written")
    return FileResponse(path, media_type="text/plain", filename=f"{job_id}.log")


@app.get("/api/jobs/{job_id}/config")
def job_config(job_id: str):
    folder = Path(get_job(job_id)["folder"])
    path = folder / (
        "operation.py" if (folder / "operation.py").is_file() else "train.py"
    )
    if not path.is_file():
        raise HTTPException(404, "This task has no training configuration")
    return FileResponse(path, media_type="text/plain", filename=f"{job_id}.py")


@app.get("/api/jobs/{job_id}/artifacts/{relative:path}")
def download_artifact(job_id: str, relative: str):
    try:
        path = artifact_path(get_job(job_id), relative)
    except FileNotFoundError as error:
        raise HTTPException(404, str(error)) from error
    return FileResponse(path, filename=path.name)


@app.post("/api/jobs/{job_id}/open-folder")
def open_output_folder(job_id: str):
    path = Path(get_job(job_id)["output_dir"])
    if not path.is_dir():
        path = Path(get_job(job_id)["folder"])
    os.startfile(path)
    return {"opened": str(path)}


@app.get("/api/projects/{project_id}/thumbnail")
def thumbnail(
    project_id: str, relative: str = Query(min_length=1), dataset: str = "train"
):
    project = get_project(project_id)
    index = {"train": 0, "regularization": 1}.get(dataset)
    if index is None:
        try:
            index = int(dataset)
        except ValueError as error:
            raise HTTPException(422, "Invalid dataset index") from error
    try:
        content = preview_bytes(project, index, relative)
        with Image.open(io.BytesIO(content)) as image:
            preview = ImageOps.exif_transpose(image).convert("RGB")
            preview.thumbnail((360, 360))
            buffer = io.BytesIO()
            preview.save(buffer, format="JPEG", quality=82)
    except FileNotFoundError as error:
        raise HTTPException(404, str(error)) from error
    except (OSError, ValueError) as error:
        raise HTTPException(400, "Image cannot be decoded") from error
    return Response(buffer.getvalue(), media_type="image/jpeg")


@app.get("/{relative:path}")
def frontend(relative: str):
    if relative.startswith("api/"):
        raise HTTPException(404)
    target = (DIST / relative).resolve()
    if not target.is_relative_to(DIST.resolve()):
        raise HTTPException(404)
    if relative and target.is_file():
        return FileResponse(target)
    index = DIST / "index.html"
    if not index.is_file():
        raise HTTPException(
            503, "Build the frontend with npm run build in webapp/frontend"
        )
    return FileResponse(index)
