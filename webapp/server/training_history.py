"""Checkpoint and sample provenance across a training run's resume ancestry."""

import json
import re
from pathlib import Path

from .external import canonical
from .inspection import artifacts, job_log_path


def read_json(path):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def step_number(path):
    match = re.search(r"(?:^|[/\\])step_(\d+)(?:[/\\]|$)", str(path))
    return int(match[1]) if match else None


def resume_path(job):
    value = job["config"].get("native_overrides", {}).get("RESUME")
    if not value:
        return None
    if value != "latest":
        return Path(value).expanduser().resolve()
    started = job.get("started_at") or job["created_at"]
    candidates = [
        path.parent
        for path in (Path(job["output_dir"]) / "checkpoints").glob("step_*/state.json")
        if path.stat().st_mtime <= started
        and (path.parent / "trainable.safetensors").is_file()
    ]
    return max(candidates, key=lambda path: step_number(path) or 0, default=None)


def linked_runs(job, database):
    """Nearest run first, with each ancestor cut at the checkpoint actually resumed."""
    candidates = database.training_jobs(job["project_id"])
    result, visited = [], set()
    current, cutoff = job, None
    while current and current["id"] not in visited:
        visited.add(current["id"])
        result.append((current, cutoff))
        checkpoint = resume_path(current)
        if checkpoint is None:
            break
        step = read_json(checkpoint / "state.json").get("step", step_number(checkpoint))
        if not isinstance(step, int) or step < 0:
            break
        cutoff = step if cutoff is None else min(cutoff, step)
        parents = [
            candidate
            for candidate in candidates
            if candidate["id"] not in visited
            and candidate["created_at"] <= current["created_at"]
            and canonical(checkpoint.parent.parent)
            == canonical(candidate["output_dir"])
        ]
        current = parents[0] if parents else None
    return result


def epoch_length(job):
    """Optimizer steps per data epoch from the saved native inspection report."""
    path = Path(job["folder"]) / "console.log"
    if not path.is_file():
        path = job_log_path(job)
    try:
        with path.open("rb") as stream:
            text = stream.read(240_000).decode("utf-8", errors="replace")
    except OSError:
        return None
    match = re.search(r'"batches"\s*:\s*(\d+)', text)
    accumulation = (
        job["config"]
        .get("native_overrides", {})
        .get("GRAD_ACC", job["config"].get("grad_acc", 1))
    )
    if (
        not match
        or int(match[1]) <= 0
        or not isinstance(accumulation, (int, float))
        or accumulation <= 0
    ):
        return None
    return int(match[1]) / accumulation


def inherited_artifacts(job, database):
    """Current files with ancestor checkpoints, weights and sample provenance."""
    result = artifacts(job)
    if job["kind"] != "train":
        return result
    result["weights"], result["checkpoints"], result["samples"] = [], [], []
    seen = {"weights": set(), "checkpoints": set(), "samples": set()}
    for source, cutoff in linked_runs(job, database):
        source_artifacts = artifacts(source)
        for kind in seen:
            for value in source_artifacts[kind]:
                step = step_number(value["path"])
                if cutoff is not None and (step is None or step > cutoff):
                    continue
                absolute = (
                    Path(value["path"])
                    if kind == "checkpoints"
                    else Path(source["output_dir"]) / value["path"]
                )
                key = canonical(absolute)
                if key in seen[kind]:
                    continue
                seen[kind].add(key)
                result[kind].append(
                    {
                        **value,
                        "job_id": source["id"],
                        "step": step,
                        "source_started_at": source.get("started_at")
                        or source["created_at"],
                        "inherited": source["id"] != job["id"],
                    }
                )
    result["checkpoints"].sort(key=lambda row: row.get("step") or 0, reverse=True)
    return result


def sample_points(job, database):
    """All completed native sample points in the selected resume branch."""
    points, seen = [], set()
    runs = linked_runs(job, database)
    for source, cutoff in runs:
        length = epoch_length(source)
        for folder in (Path(source["output_dir"]) / "samples").glob("step_*"):
            metadata = folder / "settings.json"
            settings = read_json(metadata)
            if metadata.exists() and not settings:
                continue
            step = settings.get("step", step_number(folder))
            if (
                not isinstance(step, int)
                or step < 0
                or (cutoff is not None and step > cutoff)
            ):
                continue
            images = []
            for path in sorted(folder.glob("*.png")):
                if not path.stem.isdigit() or canonical(path) in seen:
                    continue
                stat = path.stat()
                if not stat.st_size:
                    continue
                seen.add(canonical(path))
                images.append(
                    dict(
                        name=path.name,
                        path=path.relative_to(source["output_dir"]).as_posix(),
                        size=stat.st_size,
                        modified=stat.st_mtime,
                        job_id=source["id"],
                    )
                )
            if not images:
                continue
            points.append(
                dict(
                    id=f"{source['id']}:{folder.name}",
                    step=step,
                    epoch=round(step / length, 3) if length else None,
                    job_id=source["id"],
                    source_started_at=source.get("started_at") or source["created_at"],
                    inherited=source["id"] != job["id"],
                    modified=max(image["modified"] for image in images),
                    images=images,
                    settings={
                        key: settings[key]
                        for key in ("prompts", "seeds", "steps", "cfg", "sizes")
                        if key in settings
                    },
                )
            )
    points.sort(
        key=lambda point: (point["step"], point["source_started_at"]), reverse=True
    )
    return points, [
        dict(id=source["id"], started_at=source.get("started_at"), cutoff=cutoff)
        for source, cutoff in runs
    ]
