"""SQLite project settings and durable task state."""

import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from .configuration import DATA_FIELDS, STATE_ROOT, ProjectInput, TrainSettings

ACTIVE = ("preparing", "queued", "starting", "running", "stopping")


class BusyJob(ValueError):
    pass


class Store:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as db:
            db.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS projects (
                    id TEXT PRIMARY KEY, name TEXT NOT NULL, config TEXT NOT NULL,
                    analysis TEXT, created_at REAL NOT NULL, updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY, project_id TEXT NOT NULL, kind TEXT NOT NULL,
                    status TEXT NOT NULL, phase TEXT NOT NULL DEFAULT '',
                    config TEXT NOT NULL, project_name TEXT NOT NULL,
                    folder TEXT NOT NULL, output_dir TEXT NOT NULL,
                    worker_pid INTEGER, worker_created REAL, child_pid INTEGER,
                    cancel_requested INTEGER NOT NULL DEFAULT 0,
                    error TEXT, exit_code INTEGER,
                    created_at REAL NOT NULL, started_at REAL, finished_at REAL
                );
            """)
            columns = {row[1] for row in db.execute("PRAGMA table_info(projects)")}
            if "archived" not in columns:
                db.execute(
                    "ALTER TABLE projects ADD COLUMN archived INTEGER NOT NULL DEFAULT 0"
                )
            job_columns = {row[1] for row in db.execute("PRAGMA table_info(jobs)")}
            for name, definition in (
                ("origin", "TEXT NOT NULL DEFAULT 'studio'"),
                ("log_path", "TEXT NOT NULL DEFAULT ''"),
                ("can_retry", "INTEGER NOT NULL DEFAULT 1"),
                ("run_label", "TEXT NOT NULL DEFAULT ''"),
            ):
                if name not in job_columns:
                    db.execute(f"ALTER TABLE jobs ADD COLUMN {name} {definition}")

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        db.row_factory = sqlite3.Row
        try:
            yield db
        finally:
            db.close()

    @staticmethod
    def unpack(row):
        if row is None:
            return None
        value = dict(row)
        for key in ("config", "analysis"):
            if key in value and value[key] is not None:
                value[key] = json.loads(value[key])
        if "config" in value:
            value["config"] = TrainSettings.model_validate(value["config"]).model_dump()
        return value

    def projects(self):
        with self.connection() as db:
            return [
                self.unpack(row)
                for row in db.execute("SELECT * FROM projects ORDER BY created_at DESC")
            ]

    def project(self, project_id):
        with self.connection() as db:
            return self.unpack(
                db.execute(
                    "SELECT * FROM projects WHERE id=?", (project_id,)
                ).fetchone()
            )

    def create_project(self, value: ProjectInput):
        project_id, now = uuid.uuid4().hex[:12], time.time()
        with self.connection() as db:
            db.execute(
                "INSERT INTO projects(id,name,config,created_at,updated_at) VALUES(?,?,?,?,?)",
                (project_id, value.name, value.config.model_dump_json(), now, now),
            )
        return self.project(project_id)

    def save_project(self, project_id, value: ProjectInput):
        previous = self.project(project_id)
        if previous is None:
            return None
        changed = any(
            previous["config"][key] != getattr(value.config, key) for key in DATA_FIELDS
        )
        analysis = None if changed else json.dumps(previous["analysis"])
        with self.connection() as db:
            db.execute(
                "UPDATE projects SET name=?,config=?,analysis=?,updated_at=? WHERE id=?",
                (
                    value.name,
                    value.config.model_dump_json(),
                    analysis,
                    time.time(),
                    project_id,
                ),
            )
        return self.project(project_id)

    def set_analysis(self, project_id, result, expected_config):
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT config FROM projects WHERE id=?", (project_id,)
            ).fetchone()
            current = (
                TrainSettings.model_validate_json(row["config"]).model_dump()
                if row
                else None
            )
            expected = TrainSettings.model_validate(expected_config).model_dump()
            if current and all(current[key] == expected[key] for key in DATA_FIELDS):
                db.execute(
                    "UPDATE projects SET analysis=? WHERE id=?",
                    (json.dumps(result), project_id),
                )
            db.execute("COMMIT")

    def insert_job(self, job):
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            keys = list(job)
            db.execute(
                f"INSERT INTO jobs({','.join(keys)}) VALUES({','.join('?' for _ in keys)})",
                tuple(
                    json.dumps(job[key]) if key == "config" else job[key]
                    for key in keys
                ),
            )
            db.execute("COMMIT")

    def claim_job(self):
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            busy = db.execute(
                "SELECT 1 FROM jobs WHERE status IN ('starting','running','stopping') AND kind!='check' LIMIT 1"
            ).fetchone()
            query = "SELECT id FROM jobs WHERE status='queued'"
            if busy:
                query += " AND kind='check'"
            row = db.execute(query + " ORDER BY created_at LIMIT 1").fetchone()
            if row:
                db.execute(
                    "UPDATE jobs SET status='starting',started_at=? WHERE id=?",
                    (time.time(), row["id"]),
                )
            db.execute("COMMIT")
        return self.job(row["id"]) if row else None

    def request_cancel(self, job_id):
        with self.connection() as db:
            db.execute(
                "UPDATE jobs SET status='cancelled',cancel_requested=1,finished_at=? WHERE id=? AND status IN ('queued','preparing')",
                (time.time(), job_id),
            )
            db.execute(
                "UPDATE jobs SET status='stopping',cancel_requested=1 WHERE id=? AND status IN ('starting','running','stopping')",
                (job_id,),
            )
        return self.job(job_id)

    def ready_job(self, job_id):
        with self.connection() as db:
            db.execute(
                "UPDATE jobs SET status='queued' WHERE id=? AND status='preparing' AND cancel_requested=0",
                (job_id,),
            )

    def archive_project(self, project_id, archived=True):
        with self.connection() as db:
            db.execute(
                "UPDATE projects SET archived=?,updated_at=? WHERE id=?",
                (int(archived), time.time(), project_id),
            )
        return self.project(project_id)

    def copy_project(self, project_id):
        project = self.project(project_id)
        if project is None:
            return None
        return self.create_project(
            ProjectInput(
                name=project["name"] + " copy",
                config=TrainSettings.model_validate(project["config"]),
            )
        )

    def job(self, job_id):
        with self.connection() as db:
            return self.unpack(
                db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
            )

    def jobs(self, project_id=None):
        query = "SELECT * FROM jobs"
        args = ()
        if project_id:
            query += " WHERE project_id=?"
            args = (project_id,)
        query += " ORDER BY created_at DESC LIMIT 200"
        with self.connection() as db:
            return [self.unpack(row) for row in db.execute(query, args)]

    def update_job(self, job_id, **values):
        allowed = {
            "status",
            "phase",
            "worker_pid",
            "worker_created",
            "child_pid",
            "cancel_requested",
            "error",
            "exit_code",
            "started_at",
            "finished_at",
        }
        if not set(values) <= allowed:
            raise ValueError("Unsupported task field")
        with self.connection() as db:
            db.execute(
                "UPDATE jobs SET "
                + ",".join(f"{key}=?" for key in values)
                + " WHERE id=?",
                (*values.values(), job_id),
            )

    def upsert_external(self, job):
        with self.connection() as db:
            keys = list(job)
            values = [
                json.dumps(job[key]) if key == "config" else job[key] for key in keys
            ]
            updates = ",".join(
                f"{key}=excluded.{key}"
                for key in keys
                if key not in ("id", "cancel_requested", "status")
            )
            updates += (
                ",cancel_requested=MAX(jobs.cancel_requested,excluded.cancel_requested)"
            )
            updates += ",status=CASE WHEN jobs.cancel_requested=1 AND excluded.status='running' THEN 'stopping' ELSE excluded.status END"
            db.execute(
                f"INSERT INTO jobs({','.join(keys)}) VALUES({','.join('?' for _ in keys)}) "
                f"ON CONFLICT(id) DO UPDATE SET {updates}",
                values,
            )


store = Store(STATE_ROOT / "studio.sqlite3")
