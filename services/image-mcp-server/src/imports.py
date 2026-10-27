"""MCP image imports into project-scoped native artifacts (design chapter 8).

Mirrors the Video import semantics: a single-call import for small payloads
and a durable prepare/chunk/status/commit flow for larger files. A claimed
central ContentRef is provenance only and never grants read access.
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import io
import json
import os
import re
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from PIL import Image

from .artifacts import ArtifactStore
from .project_paths import validate_project_id

MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_SINGLE_BYTES = 3 * 1024 * 1024
MAX_CHUNK_BYTES = 512 * 1024
MAX_PIXELS = 50_000_000
STALE_SECONDS = 24 * 3600

IMPORT_ID = re.compile(r"imp_[0-9a-f]{32}")
SHA256 = re.compile(r"[0-9a-f]{64}")
UUID7 = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}")
FORMATS = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}
PIL_FORMATS = {"PNG": "image/png", "JPEG": "image/jpeg", "WEBP": "image/webp"}

LIMITS = {
    "formats": sorted(set(FORMATS.values())),
    "max_image_bytes": MAX_IMAGE_BYTES,
    "max_single_call_bytes": MAX_SINGLE_BYTES,
    "max_chunk_bytes": MAX_CHUNK_BYTES,
    "max_pixels": MAX_PIXELS,
    "stale_import_seconds": STALE_SECONDS,
    "source_url": False,
}


def require_filename(filename) -> str:
    if (not isinstance(filename, str) or not 1 <= len(filename) <= 200
            or "/" in filename or "\\" in filename or filename in (".", "..")
            or any(ord(c) < 32 for c in filename)):
        raise ValueError("invalid_import_filename")
    if Path(filename).suffix.lower() not in FORMATS:
        raise ValueError("unsupported_import_format")
    return filename


def require_project(project_id) -> str:
    try:
        validate_project_id(project_id)
    except ValueError:
        raise ValueError("invalid_project_id") from None
    return project_id


def validate_image(data: bytes, filename: str) -> dict:
    """Fully decode the bytes; the extension must match the detected format."""
    expected = FORMATS[Path(filename).suffix.lower()]
    try:
        with Image.open(io.BytesIO(data)) as image:
            detected = PIL_FORMATS.get(image.format or "")
            width, height = image.size
            if detected is None:
                raise ValueError("unsupported_import_format")
            if width * height > MAX_PIXELS:
                raise ValueError("import_pixel_budget_exceeded")
            image.load()
    except ValueError:
        raise
    except (OSError, Image.DecompressionBombError, SyntaxError):
        raise ValueError("import_decode_failed") from None
    if detected != expected:
        raise ValueError("import_format_mismatch")
    return {"media_type": detected, "width": width, "height": height}


def _decode(content_base64, limit: int) -> bytes:
    if not isinstance(content_base64, str) or not 0 < len(content_base64) <= 4 * ((limit + 2) // 3):
        raise ValueError("import_content_too_large_or_empty")
    try:
        content = base64.b64decode(content_base64, validate=True)
    except (ValueError, binascii.Error):
        raise ValueError("invalid_import_base64") from None
    if not content or len(content) > limit:
        raise ValueError("import_content_too_large_or_empty")
    return content


def _require_sha(value, code: str) -> str:
    if not isinstance(value, str) or not SHA256.fullmatch(value.lower()):
        raise ValueError(code)
    return value.lower()


def _record(artifact, media: dict, artifacts: ArtifactStore) -> dict:
    return {"artifact": artifact.to_dict(), "media": media,
            "download_path": artifacts.download_path(artifact.artifact_id)}


class ImportStore:
    def __init__(self, artifacts: ArtifactStore, root: Path | None = None):
        self.artifacts = artifacts
        self.root = root or (artifacts.root_dir / ".image-imports")

    @contextmanager
    def _transaction(self, *, write=True):
        self.root.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(self.root / "imports.sqlite", timeout=30)
        db.row_factory = sqlite3.Row
        try:
            db.execute("PRAGMA synchronous=FULL")
            db.execute("""CREATE TABLE IF NOT EXISTS imports (
                id TEXT PRIMARY KEY, project TEXT NOT NULL, idem TEXT NOT NULL,
                metadata TEXT NOT NULL, offset INTEGER NOT NULL DEFAULT 0,
                artifact_id TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'uploading',
                media TEXT, created_at REAL NOT NULL, UNIQUE(project, idem))""")
            db.execute("BEGIN IMMEDIATE" if write else "BEGIN")
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def _part(self, import_id: str) -> Path:
        return self.root / (import_id + ".part")

    def _read(self, db, project_id, import_id):
        project_id = require_project(project_id)
        if not isinstance(import_id, str) or not IMPORT_ID.fullmatch(import_id):
            raise ValueError("invalid_import_id")
        row = db.execute("SELECT * FROM imports WHERE id=? AND project=?", (import_id, project_id)).fetchone()
        if row is None:
            raise ValueError("import_not_found")
        return row

    def _public(self, row) -> dict:
        metadata = json.loads(row["metadata"])
        result = {"import_id": row["id"], "project_id": row["project"], "status": row["status"],
                  "offset": row["offset"], "chunk_max_bytes": MAX_CHUNK_BYTES, **metadata}
        if row["status"] == "committed":
            artifact = self.artifacts.get(row["artifact_id"], row["project"])
            result.update(_record(artifact, json.loads(row["media"]), self.artifacts))
        return result

    def _expire(self, db, now: float) -> None:
        stale = db.execute("SELECT id FROM imports WHERE status='uploading' AND created_at<?",
                           (now - STALE_SECONDS,)).fetchall()
        for row in stale:
            self._part(row["id"]).unlink(missing_ok=True)
            db.execute("DELETE FROM imports WHERE id=?", (row["id"],))

    def import_single(self, *, project_id, filename, expected_sha256, content_base64) -> dict:
        project_id = require_project(project_id)
        filename = require_filename(filename)
        expected = _require_sha(expected_sha256, "invalid_import_sha256")
        content = _decode(content_base64, MAX_SINGLE_BYTES)
        if hashlib.sha256(content).hexdigest() != expected:
            raise ValueError("import_sha256_mismatch")
        media = validate_image(content, filename)
        # Same project, name and bytes always map to one artifact, so a
        # retried call after an unknown response cannot create a duplicate.
        identity = hashlib.sha256(f"{project_id}\0{filename}\0{expected}".encode()).hexdigest()[:16]
        artifact = self.artifacts.create_from_bytes(
            project_id=project_id, filename=filename, data=content, media_type=media["media_type"],
            metadata={"origin": "user_import", "operation": "image.import", **media},
            artifact_id="art-" + identity)
        return {"status": "completed", "project_id": project_id, **_record(artifact, media, self.artifacts)}

    def prepare(self, *, project_id, idempotency_key, filename, size, sha256, purpose,
                source_content_ref=None) -> dict:
        project_id = require_project(project_id)
        filename = require_filename(filename)
        if not isinstance(idempotency_key, str) or not 1 <= len(idempotency_key.strip()) <= 128:
            raise ValueError("invalid_import_idempotency_key")
        if type(size) is not int or not 0 < size <= MAX_IMAGE_BYTES:
            raise ValueError("import_size_must_be_1_byte_to_20_MiB")
        sha256 = _require_sha(sha256, "invalid_import_sha256")
        if not isinstance(purpose, str) or not 1 <= len(purpose.strip()) <= 1000:
            raise ValueError("import_purpose_required")
        if source_content_ref is not None:
            if (not isinstance(source_content_ref, dict)
                    or set(source_content_ref) != {"store_id", "artifact_id", "version_id"}
                    or any(not isinstance(v, str) or not UUID7.fullmatch(v) for v in source_content_ref.values())):
                raise ValueError("fixed_content_ref_required")
        metadata = {"filename": filename, "size": size, "sha256": sha256,
                    "media_type": FORMATS[Path(filename).suffix.lower()], "purpose": purpose.strip(),
                    "source_content_ref": source_content_ref,
                    "source_verification": "caller_supplied" if source_content_ref else "local_bytes"}
        encoded = json.dumps(metadata, sort_keys=True, separators=(",", ":"))
        now = time.time()
        with self._transaction() as db:
            self._expire(db, now)
            old = db.execute("SELECT * FROM imports WHERE project=? AND idem=?",
                             (project_id, idempotency_key)).fetchone()
            if old:
                if old["metadata"] != encoded:
                    raise ValueError("import_idempotency_conflict")
                return self._public(old)
            identity = "imp_" + uuid.uuid4().hex
            db.execute("INSERT INTO imports(id,project,idem,metadata,artifact_id,created_at) VALUES (?,?,?,?,?,?)",
                       (identity, project_id, idempotency_key, encoded, "art-" + uuid.uuid4().hex[:16], now))
            return self._public(self._read(db, project_id, identity))

    def status(self, *, project_id, import_id) -> dict:
        with self._transaction(write=False) as db:
            return self._public(self._read(db, project_id, import_id))

    def chunk(self, *, project_id, import_id, offset, content_base64, sha256) -> dict:
        if type(offset) is not int or offset < 0:
            raise ValueError("invalid_import_offset")
        content = _decode(content_base64, MAX_CHUNK_BYTES)
        if hashlib.sha256(content).hexdigest() != _require_sha(sha256, "invalid_import_chunk_sha256"):
            raise ValueError("import_chunk_size_or_hash_mismatch")
        with self._transaction() as db:
            row = self._read(db, project_id, import_id)
            if row["status"] == "committed":
                return self._public(row)
            meta = json.loads(row["metadata"])
            if offset + len(content) > meta["size"]:
                raise ValueError("import_chunk_exceeds_size")
            path = self._part(import_id)
            if offset < row["offset"]:
                if offset + len(content) > row["offset"]:
                    raise ValueError("import_chunk_overlap")
                with path.open("rb") as stream:
                    stream.seek(offset)
                    if stream.read(len(content)) != content:
                        raise ValueError("import_repeated_chunk_mismatch")
                return self._public(row)
            if offset != row["offset"] or row["status"] != "uploading":
                raise ValueError("import_offset_or_status_conflict")
            if offset and (not path.is_file() or path.stat().st_size < offset):
                raise ValueError("import_committed_bytes_missing")
            with path.open("r+b" if path.exists() else "w+b") as stream:
                # Drop any uncommitted tail left by a crash before writing.
                stream.truncate(offset)
                stream.seek(offset)
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            db.execute("UPDATE imports SET offset=? WHERE id=?", (offset + len(content), import_id))
            return self._public(self._read(db, project_id, import_id))

    def commit(self, *, project_id, import_id) -> dict:
        with self._transaction() as db:
            row = self._read(db, project_id, import_id)
            if row["status"] == "committed":
                return self._public(row)
            meta = json.loads(row["metadata"])
            if row["offset"] != meta["size"]:
                raise ValueError("import_incomplete")
            path = self._part(import_id)
            data = path.read_bytes() if path.is_file() else b""
            if len(data) != meta["size"] or hashlib.sha256(data).hexdigest() != meta["sha256"]:
                raise ValueError("import_final_size_or_hash_mismatch")
            media = validate_image(data, meta["filename"])
            # create_from_bytes is idempotent for the preassigned identity, so a
            # crash between publishing and the status update recovers cleanly.
            self.artifacts.create_from_bytes(
                project_id=project_id, filename=meta["filename"], data=data, media_type=media["media_type"],
                metadata={"origin": "user_import", "operation": "image.import_commit", "import_id": import_id,
                          "purpose": meta["purpose"], "source_content_ref": meta["source_content_ref"], **media},
                artifact_id=row["artifact_id"])
            db.execute("UPDATE imports SET status='committed',media=? WHERE id=?", (json.dumps(media), import_id))
            path.unlink(missing_ok=True)
            return self._public(self._read(db, project_id, import_id))
