from __future__ import annotations

import hashlib
import json
import os
import shutil
import urllib.parse
import urllib.request
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class ArtifactNotFound(Exception):
    pass


class ArtifactError(Exception):
    pass


@dataclass
class ArtifactRecord:
    artifact_id: str
    project_id: str
    filename: str
    media_type: str
    size_bytes: int
    sha256: str
    created_at: str
    source_url: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ArtifactRecord:
        return cls(**data)


class ArtifactStore:
    def __init__(self, root_dir: Path, allowed_origins: list[str] | None = None) -> None:
        self.root_dir = root_dir
        self.allowed_origins = allowed_origins or []

    @classmethod
    def from_environment(cls) -> ArtifactStore:
        root = Path(os.environ.get("IMAGE_ARTIFACT_ROOT", "/data/projects"))
        origins = [x.strip() for x in os.environ.get("IMAGE_ASSET_IMPORT_ORIGINS", "").split(",") if x.strip()]
        return cls(root_dir=root, allowed_origins=origins)

    def ensure_ready(self) -> None:
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def _project_dir(self, project_id: str) -> Path:
        from .project_paths import project_path
        try:
            p = project_path(self.root_dir, project_id)
        except ValueError as exc:
            raise ArtifactError(str(exc)) from exc
        p.mkdir(parents=True, exist_ok=True)
        return p

    def _meta_path(self, project_id: str, artifact_id: str) -> Path:
        return self._project_dir(project_id) / f"{artifact_id}.json"

    def create_from_bytes(
        self,
        project_id: str,
        filename: str,
        data: bytes,
        media_type: str = "image/png",
        metadata: dict[str, Any] | None = None,
    ) -> ArtifactRecord:
        self.ensure_ready()
        artifact_id = f"art-{uuid.uuid4().hex[:16]}"
        sha = hashlib.sha256(data).hexdigest()
        now = datetime.now(timezone.utc).isoformat()

        content_file = self._project_dir(project_id) / f"{artifact_id}-{filename}"
        content_file.write_bytes(data)

        rec = ArtifactRecord(
            artifact_id=artifact_id,
            project_id=project_id,
            filename=filename,
            media_type=media_type,
            size_bytes=len(data),
            sha256=sha,
            created_at=now,
            metadata=metadata or {},
        )

        meta_file = self._meta_path(project_id, artifact_id)
        meta_file.write_text(json.dumps(rec.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        return rec

    def import_from_url(
        self,
        project_id: str,
        source_url: str,
        filename: str,
        expected_sha256: str,
    ) -> ArtifactRecord:
        self.ensure_ready()
        parsed = urllib.parse.urlparse(source_url)
        if parsed.scheme != "https":
            raise ArtifactError("仅支持 HTTPS URL 导入")

        origin = f"{parsed.scheme}://{parsed.netloc}"
        if self.allowed_origins and origin not in self.allowed_origins:
            raise ArtifactError(f"域名不在导入白名单内: {origin}")

        artifact_id = f"art-{uuid.uuid4().hex[:16]}"
        tmp_target = self._project_dir(project_id) / f".tmp-{artifact_id}"

        hasher = hashlib.sha256()
        size = 0
        try:
            req = urllib.request.Request(source_url, headers={"User-Agent": "VerdantFlare-Image-MCP"})
            with urllib.request.urlopen(req, timeout=60) as resp, open(tmp_target, "wb") as out_f:
                while chunk := resp.read(65536):
                    size += len(chunk)
                    if size > 100 * 1024 * 1024:
                        raise ArtifactError("导入素材文件大小超出 100MB 上限")
                    hasher.update(chunk)
                    out_f.write(chunk)
        except Exception as exc:
            if tmp_target.exists():
                tmp_target.unlink(missing_ok=True)
            raise ArtifactError(f"素材导入下载失败: {exc}") from exc

        actual_sha = hasher.hexdigest()
        if expected_sha256 and actual_sha.lower() != expected_sha256.lower():
            tmp_target.unlink(missing_ok=True)
            raise ArtifactError(f"SHA-256 校验失败: 期望 {expected_sha256}, 实际 {actual_sha}")

        final_content = self._project_dir(project_id) / f"{artifact_id}-{filename}"
        shutil.move(str(tmp_target), str(final_content))

        now = datetime.now(timezone.utc).isoformat()
        rec = ArtifactRecord(
            artifact_id=artifact_id,
            project_id=project_id,
            filename=filename,
            media_type="image/png",
            size_bytes=size,
            sha256=actual_sha,
            created_at=now,
            source_url=source_url,
        )

        meta_file = self._meta_path(project_id, artifact_id)
        meta_file.write_text(json.dumps(rec.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        return rec

    def get(self, artifact_id: str, project_id: str | None = None) -> ArtifactRecord:
        if project_id:
            meta_file = self._meta_path(project_id, artifact_id)
            if meta_file.is_file():
                return ArtifactRecord.from_dict(json.loads(meta_file.read_text(encoding="utf-8")))
            raise ArtifactNotFound(f"Artifact 未找到: {artifact_id} (项目: {project_id})")

        for meta_file in self.root_dir.rglob(f"artifacts/{artifact_id}.json"):
            if meta_file.is_file() and meta_file.resolve().is_relative_to(self.root_dir.resolve()):
                return ArtifactRecord.from_dict(json.loads(meta_file.read_text(encoding="utf-8")))

        raise ArtifactNotFound(f"Artifact 未找到: {artifact_id}")

    def content_path(self, record: ArtifactRecord) -> Path:
        return self._project_dir(record.project_id) / f"{record.artifact_id}-{record.filename}"

    def download_path(self, artifact_id: str) -> str:
        return f"/image/artifacts/{artifact_id}/content"
