"""CPU-only sample indexes, thumbnails and inline image responses."""

import hashlib
import io
import json

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, Response
from PIL import Image, ImageOps

from .inspection import IMAGE_EXTENSIONS, artifact_path
from .store import store
from .training_history import sample_points

router = APIRouter(prefix="/api")


def job_for(job_id):
    job = store.job(job_id)
    if job is None:
        raise HTTPException(404, "Task not found")
    return job


@router.get("/jobs/{job_id}/gallery")
def gallery(
    job_id: str,
    offset: int = Query(0, ge=0),
    limit: int = Query(40, ge=1, le=100),
    history_revision: str = "",
):
    points, runs = sample_points(job_for(job_id), store)
    end = min(len(points), offset + limit)
    revision = hashlib.sha256(json.dumps(points, sort_keys=True).encode()).hexdigest()
    return dict(
        points=points[offset:end],
        history=points if history_revision != revision else None,
        history_revision=revision,
        total=len(points),
        next_offset=end if end < len(points) else None,
        runs=runs,
    )


@router.get("/jobs/{job_id}/image/{relative:path}")
def inline_image(job_id: str, relative: str, thumbnail: bool = False):
    try:
        path = artifact_path(job_for(job_id), relative)
    except FileNotFoundError as error:
        raise HTTPException(404, str(error)) from error
    if path.suffix.lower() not in IMAGE_EXTENSIONS:
        raise HTTPException(400, "Choose an image file")
    if not thumbnail:
        return FileResponse(path, filename=path.name, content_disposition_type="inline")
    try:
        with Image.open(path) as image:
            preview = ImageOps.exif_transpose(image).convert("RGB")
            preview.thumbnail((360, 280))
            output = io.BytesIO()
            preview.save(output, format="JPEG", quality=82)
        content = output.getvalue()
    except (OSError, ValueError, Image.DecompressionBombError) as error:
        raise HTTPException(
            409, "The image is still being written or cannot be read"
        ) from error
    etag = hashlib.sha256(content).hexdigest()[:24]
    return Response(
        content,
        media_type="image/jpeg",
        headers={"ETag": f'"{etag}"', "Cache-Control": "private, max-age=3600"},
    )
