"""Read-only dataset summaries, local paths and training artifacts."""

import codecs
import io
import json
import math
import re
import time
from pathlib import Path

import numpy as np
from PIL import Image

from kohakumoss_data.caption.template import build_template
from kohakumoss_data.dataset import build_source
from kohakumoss_data.sources.base import IMAGE_EXTS

from .configuration import TrainSettings, dataset_entries, native_knobs

IMAGE_EXTENSIONS = set(IMAGE_EXTS)


def inspect_dataset(settings, cancelled=lambda: False):
    entries = dataset_entries(settings)
    if not entries:
        raise ValueError("DATASETS must contain a dataset")
    reports = [
        inspect_entry(
            entry,
            settings.seed,
            cancelled,
            native_knobs(settings, "inspection", "inspection")["CAPTION"],
        )
        for entry in entries
    ]
    for index, report in enumerate(reports):
        report["index"] = index
    main = reports[0].copy()
    main["datasets"] = reports
    main["regularization"] = (
        reports[1]
        if settings.reg_enabled and "DATASETS" not in settings.native_overrides
        else None
    )
    return main


def source_records(source):
    if hasattr(source, "walk"):
        for item in source.walk():
            try:
                for row in item.probe():
                    yield row.locator, row, None
            except (OSError, ValueError, TypeError) as error:
                yield item.key, None, error
    else:
        for row in source.scan():
            yield row.locator, row, None


def inspect_entry(entry, seed, cancelled, caption_format):
    source = build_source(entry)
    template = build_template(entry.get("caption", caption_format))
    rng = np.random.default_rng(seed)
    images, issues, sizes = [], [], {}
    captions = invalid = missing = 0
    effective = 0.0
    for locator, row, probe_error in source_records(source):
        if cancelled():
            raise InterruptedError("Task cancelled")
        try:
            if probe_error:
                raise probe_error
            with Image.open(io.BytesIO(source.open(locator))) as image:
                width, height = image.size
                image.verify()
            text = template.render(row, rng).strip()
        except (OSError, ValueError, TypeError, Image.DecompressionBombError) as error:
            invalid += 1
            if len(issues) < 50:
                issues.append(
                    {"file": locator, "kind": "invalid", "message": str(error)}
                )
            continue
        effective += row.weight * entry.get("weight", 1)
        if text:
            captions += 1
        else:
            missing += 1
            if len(issues) < 50:
                issues.append(
                    {
                        "file": locator,
                        "kind": "caption",
                        "message": "Caption is missing or empty",
                    }
                )
        ratio = str(round(width / height, 2))
        sizes[ratio] = sizes.get(ratio, 0) + 1
        if len(images) < 24:
            images.append(
                {"path": locator, "width": width, "height": height, "caption": text}
            )
    return {
        "checked_at": time.time(),
        "root": str(entry.get("path", "")),
        "images": captions + missing,
        "captions": captions,
        "missing_captions": missing,
        "invalid_images": invalid,
        "aspect_ratios": sizes,
        "issues": issues,
        "preview": images,
        "effective_samples": effective,
    }


def preview_bytes(project, index, relative):
    settings = TrainSettings.model_validate(project["config"])
    entries = dataset_entries(settings)
    if index < 0 or index >= len(entries):
        raise FileNotFoundError("Dataset not found")
    entry = entries[index]
    if entry["source"] in ("folder", "kohya"):
        root = Path(entry["path"]).resolve()
        target = (root / relative).resolve()
        if not target.is_relative_to(root) or not target.is_file():
            raise FileNotFoundError("Image not found")
    else:
        analysis = project.get("analysis") or {}
        reports = analysis.get("datasets", [])
        if index >= len(reports) or relative not in {
            item["path"] for item in reports[index]["preview"]
        }:
            raise FileNotFoundError("Image is not in the inspected preview")
    return build_source(entry).open(relative)


def browse(path: str, mode: str):
    if not path:
        drives = [Path(f"{letter}:/") for letter in "CDEFGHIJKLMNOPQRSTUVWXYZ"]
        return {
            "path": "",
            "parent": "",
            "entries": [
                {"name": str(drive), "path": str(drive), "directory": True}
                for drive in drives
                if drive.exists()
            ],
        }
    folder = Path(path).expanduser().resolve()
    if folder.is_file():
        folder = folder.parent
    if not folder.is_dir():
        raise ValueError("Directory does not exist")
    rows = []
    for entry in folder.iterdir():
        if entry.name.startswith("."):
            continue
        try:
            directory = entry.is_dir()
            if not directory and (
                mode == "directory"
                or (mode == "file" and entry.suffix != ".safetensors")
                or (mode == "config" and entry.suffix.lower() not in (".py", ".json"))
            ):
                continue
            rows.append(
                {"name": entry.name, "path": str(entry), "directory": directory}
            )
        except OSError:
            continue
    rows.sort(key=lambda row: (not row["directory"], row["name"].casefold()))
    return {
        "path": str(folder),
        "parent": str(folder.parent) if folder.parent != folder else "",
        "entries": rows,
    }


def tool_output(job):
    metadata = Path(job["folder"]) / "operation.json"
    if not metadata.is_file():
        return None
    values = json.loads(metadata.read_text(encoding="utf-8"))["values"]
    out = values.get("OUT") or values.get("OUTPUT_DIR")
    if not isinstance(out, str) or not out:
        return None
    path = Path(out).expanduser().resolve()
    if values.get("FORMAT") in ("diffusers", "hf") and not path.suffix:
        return path, False
    return path, bool(path.suffix) and not path.is_dir()


def artifact_path(job, relative: str):
    root = Path(job["output_dir"]).resolve()
    if relative.startswith("__tool__/"):
        output = tool_output(job)
        if output is None:
            raise FileNotFoundError("Artifact not found")
        path, single = output
        root = path.parent if single else path
        relative = relative[len("__tool__/") :]
        if single and relative != path.name:
            raise FileNotFoundError("Artifact not found")
    target = (root / relative).resolve()
    if not target.is_relative_to(root) or not target.is_file():
        raise FileNotFoundError("Artifact not found")
    return target


def artifacts(job):
    root = Path(job["output_dir"]).resolve()
    candidates = (
        [
            (path, path.relative_to(root).as_posix())
            for path in root.rglob("*")
            if path.is_file()
        ]
        if root.is_dir()
        else []
    )
    output = tool_output(job)
    if output:
        path, single = output
        if single and path.is_file() and not path.is_relative_to(root):
            candidates.append((path, "__tool__/" + path.name))
        elif (
            not single
            and path.is_dir()
            and path != root
            and not path.is_relative_to(root)
        ):
            candidates.extend(
                (item, "__tool__/" + item.relative_to(path).as_posix())
                for item in path.rglob("*")
                if item.is_file()
            )
    weights, samples, files, checkpoints = [], [], [], []
    for path, relative in candidates:
        stat = path.stat()
        record = {
            "name": path.name,
            "path": relative,
            "size": stat.st_size,
            "modified": stat.st_mtime,
        }
        if path.suffix == ".safetensors":
            weights.append(record)
        elif path.suffix.lower() in IMAGE_EXTENSIONS:
            samples.append(record)
        else:
            files.append(record)
        if (
            path.name == "state.json"
            and (path.parent / "trainable.safetensors").is_file()
        ):
            checkpoints.append(
                {
                    "name": path.parent.name,
                    "path": str(path.parent),
                    "modified": stat.st_mtime,
                }
            )
    for rows in (weights, samples, files, checkpoints):
        rows.sort(key=lambda row: row["modified"], reverse=True)
    return {
        "weights": weights,
        "samples": samples[:100],
        "files": files[:200],
        "checkpoints": checkpoints,
    }


def metric_rows(job):
    path = Path(job["output_dir"]) / "metrics.jsonl"
    snapshot = Path(job["folder"]) / "metrics.jsonl"
    if job.get("origin") == "external" and snapshot.is_file():
        path = snapshot
    if not path.is_file():
        return []
    with path.open("rb") as stream:
        size = path.stat().st_size
        start = max(0, size - 2_000_000)
        stream.seek(start)
        if start:
            stream.readline()
        lines = stream.read().decode("utf-8", errors="replace").splitlines()
    rows = []
    for line in lines:
        try:
            row = json.loads(line)
        except ValueError:
            continue
        rows.append(
            {
                key: value
                for key, value in row.items()
                if isinstance(value, (int, float)) and math.isfinite(value)
            }
        )
    if len(rows) > 500:
        rows = rows[:: math.ceil(len(rows) / 499)] + rows[-1:]
    return rows


def job_log_path(job):
    return (
        Path(job["log_path"])
        if job.get("log_path")
        else Path(job["folder"]) / "console.log"
    )


def log_chunk(job, offset: int):
    path = job_log_path(job)
    if not path.is_file():
        return {"text": "", "offset": 0, "reset": offset > 0, "more": False}
    size = path.stat().st_size
    reset = offset > size
    start = max(0, size - 120_000) if offset < 0 else (0 if reset else offset)
    with path.open("rb") as stream:
        stream.seek(start)
        data = stream.read(256_000)
        next_offset = stream.tell()
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    final = (
        job["status"] not in ("queued", "running", "stopping") and next_offset >= size
    )
    text = decoder.decode(data, final=final)
    next_offset -= len(decoder.getstate()[0])
    text = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", text)
    return {
        "text": text,
        "offset": next_offset,
        "reset": reset,
        "more": next_offset < size and not decoder.getstate()[0],
    }
