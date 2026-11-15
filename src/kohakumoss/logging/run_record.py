"""Persistent process and lifecycle metadata for local training runs."""

import json
import os
import shutil
import time
import traceback
import uuid
from pathlib import Path

from kohakumoss.logging.log import get_logger

ROOT = Path(__file__).resolve().parents[3]
log = get_logger(__name__)


def registry_directory() -> Path:
    return Path(
        os.environ.get(
            "KOHAKUMOSS_RUN_REGISTRY", ROOT / "internal/studio/external-runs"
        )
    )


class RunRecord:
    """Atomic run metadata, with a repository-local discovery record."""

    def __init__(
        self,
        output,
        config,
        kind="train",
        label="",
        project_id="",
        log_path=None,
        enabled=True,
    ):
        self.enabled = enabled
        output = Path(output).expanduser().resolve()
        key = uuid.uuid4().hex
        self.paths = (output / "run-state.json", registry_directory() / f"{key}.json")
        self.value = dict(
            version=1,
            run_id=key,
            repository=str(ROOT),
            output_dir=str(output),
            config=config,
            kind=kind,
            label=label,
            pid=os.getpid(),
            started_at=time.time(),
            finished_at=None,
            status="running",
            phase=kind,
            error=None,
            studio_job_id=os.environ.get("KOHAKUMOSS_STUDIO_JOB_ID", ""),
            project_id=project_id or os.environ.get("KOHAKUMOSS_STUDIO_PROJECT_ID", ""),
            log_path=(
                str(Path(log_path).resolve()) if log_path else str(output / "log.txt")
            ),
            metrics_path=str(output / "metrics.jsonl"),
        )

    def __enter__(self):
        self.publish()
        return self

    def update(self, **values):
        self.value.update(values)
        self.publish()

    def publish(self):
        if not self.enabled:
            return
        text = json.dumps(self.value, default=str, ensure_ascii=True)
        for path in self.paths:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
                temporary.write_text(text, encoding="utf-8")
                temporary.replace(path)
            except OSError as error:
                log.warning(f"run metadata could not be written: {error}")

    def __exit__(self, error_type, error, tb):
        status = self.value["status"]
        if error_type is not None:
            cancelled = isinstance(error, KeyboardInterrupt) or (
                isinstance(error, SystemExit) and error.code in (130, 143)
            )
            status = "cancelled" if cancelled else "failed"
            self.value["error"] = (
                None
                if cancelled
                else "".join(traceback.format_exception(error_type, error, tb))
            )
        elif status == "running":
            status = "completed"
        if self.enabled:
            archive = self.paths[1].with_suffix("")
            for key, name in (
                ("log_path", "console.log"),
                ("metrics_path", "metrics.jsonl"),
            ):
                try:
                    source = Path(self.value[key])
                    if source.is_file():
                        archive.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(source, archive / name)
                        self.value[key] = str(archive / name)
                except OSError as snapshot_error:
                    log.warning(f"run log could not be archived: {snapshot_error}")
        self.update(status=status, finished_at=time.time())
        return False
