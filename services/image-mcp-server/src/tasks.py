from __future__ import annotations

import json
import os
import sqlite3
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class TaskNotFound(Exception):
    pass


class TaskConflict(Exception):
    pass


@dataclass
class TaskRecord:
    task_id: str
    project_id: str
    idempotency_key: str
    engine: str
    status: str  # queued, running, completed, failed, canceled
    created_at: str
    updated_at: str
    duration_seconds: float = 0.0
    model: str | None = None
    prompt_preview: str | None = None
    request_params: dict[str, Any] = field(default_factory=dict)
    artifact_id: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> TaskRecord:
        params_str = row["request_params"] or "{}"
        try:
            params = json.loads(params_str)
        except Exception:
            params = {}

        return cls(
            task_id=row["task_id"],
            project_id=row["project_id"],
            idempotency_key=row["idempotency_key"],
            engine=row["engine"],
            status=row["status"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            duration_seconds=float(row["duration_seconds"] or 0.0),
            model=row["model"],
            prompt_preview=row["prompt_preview"],
            request_params=params,
            artifact_id=row["artifact_id"],
            error=row["error"],
        )


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS tasks (
    task_id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    idempotency_key TEXT NOT NULL,
    engine TEXT NOT NULL,
    model TEXT,
    prompt_preview TEXT,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    duration_seconds REAL DEFAULT 0.0,
    request_params TEXT NOT NULL,
    artifact_id TEXT,
    error TEXT,
    UNIQUE(project_id, idempotency_key)
);
CREATE INDEX IF NOT EXISTS idx_tasks_project ON tasks(project_id);
CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
CREATE INDEX IF NOT EXISTS idx_tasks_engine ON tasks(engine);
CREATE INDEX IF NOT EXISTS idx_tasks_created ON tasks(created_at DESC);
"""


class TaskStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

    @classmethod
    def from_environment(cls) -> TaskStore:
        root = Path(os.environ.get("IMAGE_ARTIFACT_ROOT", "/data/projects"))
        return cls(db_path=root / "tasks.db")

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            str(self.db_path),
            timeout=30.0,
            check_same_thread=False,
        )
        conn.row_factory = sqlite3.Row
        return conn

    def ensure_ready(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = self._get_conn()
        try:
            conn.isolation_level = None
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            conn.executescript(SCHEMA_SQL)
        finally:
            conn.close()

    def create(
        self,
        project_id: str,
        idempotency_key: str,
        engine: str,
        request_params: dict[str, Any],
        model: str | None = None,
    ) -> TaskRecord:
        from .project_paths import validate_project_id
        validate_project_id(project_id)
        self.ensure_ready()
        now = datetime.now(timezone.utc).isoformat()
        prompt = request_params.get("prompt", "")
        prompt_preview = prompt[:120].strip() if isinstance(prompt, str) else ""

        # 检查是否已有幂等任务
        with self._get_conn() as conn:
            cur = conn.execute(
                "SELECT * FROM tasks WHERE project_id = ? AND idempotency_key = ?",
                (project_id, idempotency_key),
            )
            existing = cur.fetchone()
            if existing:
                return TaskRecord.from_row(existing)

            task_id = f"imgtask-{uuid.uuid4().hex[:16]}"
            params_json = json.dumps(request_params, ensure_ascii=False)
            try:
                conn.execute(
                    """
                    INSERT INTO tasks (
                        task_id, project_id, idempotency_key, engine, model,
                        prompt_preview, status, created_at, updated_at,
                        duration_seconds, request_params, artifact_id, error
                    ) VALUES (?, ?, ?, ?, ?, ?, 'queued', ?, ?, 0.0, ?, NULL, NULL)
                    """,
                    (
                        task_id,
                        project_id,
                        idempotency_key,
                        engine,
                        model or request_params.get("model"),
                        prompt_preview,
                        now,
                        now,
                        params_json,
                    ),
                )
                conn.commit()
            except sqlite3.IntegrityError:
                # 并发冲突时重新查询
                cur = conn.execute(
                    "SELECT * FROM tasks WHERE project_id = ? AND idempotency_key = ?",
                    (project_id, idempotency_key),
                )
                row = cur.fetchone()
                if row:
                    return TaskRecord.from_row(row)
                raise

        return self.get(task_id)

    def get(self, task_id: str) -> TaskRecord:
        self.ensure_ready()
        with self._get_conn() as conn:
            cur = conn.execute("SELECT * FROM tasks WHERE task_id = ?", (task_id,))
            row = cur.fetchone()
            if not row:
                raise TaskNotFound(f"任务未找到: {task_id}")
            return TaskRecord.from_row(row)

    def update_status(
        self,
        task_id: str,
        status: str,
        artifact_id: str | None = None,
        error: str | None = None,
        duration_seconds: float | None = None,
    ) -> TaskRecord:
        self.ensure_ready()
        now = datetime.now(timezone.utc).isoformat()
        with self._get_conn() as conn:
            cur = conn.execute("SELECT * FROM tasks WHERE task_id = ?", (task_id,))
            row = cur.fetchone()
            if not row:
                raise TaskNotFound(f"任务未找到: {task_id}")

            current_art = artifact_id if artifact_id is not None else row["artifact_id"]
            current_err = error if error is not None else row["error"]
            current_dur = duration_seconds if duration_seconds is not None else row["duration_seconds"]

            conn.execute(
                """
                UPDATE tasks
                SET status = ?, updated_at = ?, artifact_id = ?, error = ?, duration_seconds = ?
                WHERE task_id = ?
                """,
                (status, now, current_art, current_err, current_dur, task_id),
            )
            conn.commit()

        return self.get(task_id)

    def list_tasks(
        self,
        project_id: str | None = None,
        engine: str | None = None,
        status: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[TaskRecord], int]:
        self.ensure_ready()
        conditions: list[str] = []
        params: list[Any] = []

        if project_id:
            conditions.append("project_id = ?")
            params.append(project_id)
        if engine:
            conditions.append("engine = ?")
            params.append(engine)
        if status:
            conditions.append("status = ?")
            params.append(status)

        where_clause = ("WHERE " + " AND ".join(conditions)) if conditions else ""

        with self._get_conn() as conn:
            count_cur = conn.execute(f"SELECT COUNT(*) FROM tasks {where_clause}", tuple(params))
            total = count_cur.fetchone()[0]

            query = f"""
                SELECT * FROM tasks
                {where_clause}
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
            """
            params.extend([limit, offset])
            cur = conn.execute(query, tuple(params))
            records = [TaskRecord.from_row(r) for r in cur.fetchall()]

        return records, total

    def get_stats(self) -> dict[str, Any]:
        self.ensure_ready()
        with self._get_conn() as conn:
            cur = conn.execute(
                """
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN status = 'queued' THEN 1 ELSE 0 END) as queued,
                    SUM(CASE WHEN status = 'running' THEN 1 ELSE 0 END) as running,
                    SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed,
                    SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed,
                    SUM(CASE WHEN status = 'canceled' THEN 1 ELSE 0 END) as canceled,
                    AVG(CASE WHEN status = 'completed' THEN duration_seconds ELSE NULL END) as avg_duration
                FROM tasks
                """
            )
            row = cur.fetchone()
            return {
                "total": row["total"] or 0,
                "queued": row["queued"] or 0,
                "running": row["running"] or 0,
                "completed": row["completed"] or 0,
                "failed": row["failed"] or 0,
                "canceled": row["canceled"] or 0,
                "avg_duration": round(row["avg_duration"] or 0.0, 2),
            }

    def recover_hanging_tasks(self) -> int:
        """服务重启时将残留的 running 任务标为 failed，防止永远卡死。"""
        self.ensure_ready()
        now = datetime.now(timezone.utc).isoformat()
        with self._get_conn() as conn:
            cur = conn.execute(
                """
                UPDATE tasks
                SET status = 'failed',
                    error = '服务重启，未完成任务已自动中断',
                    updated_at = ?
                WHERE status = 'running'
                """,
                (now,),
            )
            count = cur.rowcount
            conn.commit()
            return count

    def get_queued_tasks(self) -> list[TaskRecord]:
        """获取所有待消费的排队任务。"""
        self.ensure_ready()
        with self._get_conn() as conn:
            cur = conn.execute(
                "SELECT * FROM tasks WHERE status = 'queued' ORDER BY created_at ASC"
            )
            return [TaskRecord.from_row(r) for r in cur.fetchall()]
