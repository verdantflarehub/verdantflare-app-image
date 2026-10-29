from __future__ import annotations

import concurrent.futures
import hashlib
import io
import json
import os
import shutil
import threading
import urllib.parse
import urllib.request
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from PIL import Image, ImageOps
except ImportError:  # pragma: no cover
    Image = None  # type: ignore[assignment]
    ImageOps = None  # type: ignore[assignment]

ALLOWED_THUMBNAIL_EDGES = (320, 640)
THUMBNAIL_CACHE_VERSION = "v1"


def etag_matches(if_none_match: str | None, etag: str) -> bool:
    if not if_none_match or not etag:
        return False
    target = etag.strip()
    if target.startswith("W/"):
        target = target[2:].strip()
    for token in if_none_match.split(","):
        cand = token.strip()
        if cand == "*":
            return True
        if cand.startswith("W/"):
            cand = cand[2:].strip()
        if cand == target:
            return True
    return False


def detect_image_format(header: bytes) -> str:
    if len(header) >= 4 and header[:4] == b"\x89PNG":
        return "PNG"
    if len(header) >= 3 and header[:3] == b"\xff\xd8\xff":
        return "JPEG"
    if len(header) >= 12 and header[:4] == b"RIFF" and header[8:12] == b"WEBP":
        return "WebP"
    if len(header) >= 4 and header[:4] == b"GIF8":
        return "GIF"
    if len(header) >= 12 and header[4:12] in (b"ftypavif", b"ftypavis"):
        return "AVIF"
    return ""


def inspect_image_bytes(data: bytes) -> dict[str, Any]:
    specs: dict[str, Any] = {}
    fmt = detect_image_format(data[:16])
    if fmt:
        specs["format"] = fmt
    if Image is not None and data:
        try:
            with Image.open(io.BytesIO(data)) as img:
                w, h = int(img.width or 0), int(img.height or 0)
                if w > 0 and h > 0:
                    specs["width"] = w
                    specs["height"] = h
                if not specs.get("format") and img.format:
                    pil_fmt = str(img.format).upper()
                    specs["format"] = "WebP" if pil_fmt == "WEBP" else pil_fmt
        except Exception:
            pass
    return specs


def inspect_image_file(path: Path) -> dict[str, Any]:
    specs: dict[str, Any] = {}
    try:
        with path.open("rb") as stream:
            header = stream.read(16)
    except OSError:
        return specs
    fmt = detect_image_format(header)
    if fmt:
        specs["format"] = fmt
    if Image is not None:
        try:
            with Image.open(path) as img:
                w, h = int(img.width or 0), int(img.height or 0)
                if w > 0 and h > 0:
                    specs["width"] = w
                    specs["height"] = h
                if not specs.get("format") and img.format:
                    pil_fmt = str(img.format).upper()
                    specs["format"] = "WebP" if pil_fmt == "WEBP" else pil_fmt
        except Exception:
            pass
    return specs


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
    def __init__(
        self,
        root_dir: Path,
        allowed_origins: list[str] | None = None,
        prewarm_on_create: bool = False,
    ) -> None:
        self.root_dir = root_dir
        self.allowed_origins = allowed_origins or []
        self.prewarm_on_create = prewarm_on_create
        self._index_lock = threading.RLock()
        self._project_index: dict[str, str] = {}
        self._index_built = False
        self._record_cache: dict[tuple[str, str], tuple[int, ArtifactRecord]] = {}
        self._thumb_locks_guard = threading.Lock()
        self._thumb_locks: dict[str, threading.Lock] = {}
        self._thumb_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="img-thumb"
        )
        self.rglob_count = 0
        self.thumbnail_generate_count = 0

    @classmethod
    def from_environment(cls) -> ArtifactStore:
        root = Path(os.environ.get("IMAGE_ARTIFACT_ROOT", "/data/projects"))
        origins = [x.strip() for x in os.environ.get("IMAGE_ASSET_IMPORT_ORIGINS", "").split(",") if x.strip()]
        prewarm = os.environ.get("IMAGE_THUMBNAIL_PREWARM", "1").strip() not in ("0", "false", "no")
        return cls(root_dir=root, allowed_origins=origins, prewarm_on_create=prewarm)

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
        artifact_id: str | None = None,
    ) -> ArtifactRecord:
        self.ensure_ready()
        artifact_id = artifact_id or f"art-{uuid.uuid4().hex[:16]}"
        import re
        if not re.fullmatch(r"art-[0-9a-f]{16}", artifact_id):
            raise ArtifactError("invalid artifact identity")
        sha = hashlib.sha256(data).hexdigest()
        if self._meta_path(project_id, artifact_id).exists():
            existing = self.get(artifact_id, project_id)
            if existing.sha256 != sha or existing.filename != filename:
                raise ArtifactError("artifact_identity_conflict")
            if hashlib.sha256(self.content_path(existing).read_bytes()).hexdigest() != sha:
                raise ArtifactError("artifact_integrity_failed")
            return existing
        now = datetime.now(timezone.utc).isoformat()

        content_file = self._project_dir(project_id) / f"{artifact_id}-{filename}"
        if content_file.exists() and hashlib.sha256(content_file.read_bytes()).hexdigest() != sha:
            raise ArtifactError("artifact_identity_conflict")
        temporary = content_file.with_name(content_file.name + ".part")
        with temporary.open("wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, content_file)

        merged_meta = dict(metadata or {})
        for k, v in inspect_image_bytes(data).items():
            merged_meta.setdefault(k, v)

        rec = ArtifactRecord(
            artifact_id=artifact_id,
            project_id=project_id,
            filename=filename,
            media_type=media_type,
            size_bytes=len(data),
            sha256=sha,
            created_at=now,
            metadata=merged_meta,
        )

        meta_file = self._meta_path(project_id, artifact_id)
        temporary = meta_file.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(rec.to_dict(), stream, ensure_ascii=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, meta_file)
        with self._index_lock:
            self._project_index[artifact_id] = project_id
            try:
                self._record_cache[(project_id, artifact_id)] = (meta_file.stat().st_mtime_ns, rec)
            except OSError:
                pass
        if self.prewarm_on_create and rec.metadata.get("width") and rec.metadata.get("height"):
            self.prewarm_thumbnails_async(rec)
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
        inspected = inspect_image_file(final_content)
        rec = ArtifactRecord(
            artifact_id=artifact_id,
            project_id=project_id,
            filename=filename,
            media_type="image/png",
            size_bytes=size,
            sha256=actual_sha,
            created_at=now,
            source_url=source_url,
            metadata=inspected,
        )

        meta_file = self._meta_path(project_id, artifact_id)
        meta_file.write_text(json.dumps(rec.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        with self._index_lock:
            self._project_index[artifact_id] = project_id
            try:
                self._record_cache[(project_id, artifact_id)] = (meta_file.stat().st_mtime_ns, rec)
            except OSError:
                pass
        if self.prewarm_on_create and rec.metadata.get("width") and rec.metadata.get("height"):
            self.prewarm_thumbnails_async(rec)
        return rec

    def _load_meta_file(self, meta_file: Path) -> ArtifactRecord:
        stat = meta_file.stat()
        mtime_ns = stat.st_mtime_ns
        raw = json.loads(meta_file.read_text(encoding="utf-8"))
        rec = ArtifactRecord.from_dict(raw)
        cache_key = (rec.project_id, rec.artifact_id)
        with self._index_lock:
            self._project_index[rec.artifact_id] = rec.project_id
            cached = self._record_cache.get(cache_key)
            if cached is not None and cached[0] == mtime_ns:
                return cached[1]

        meta = rec.metadata
        needs_probe = not (
            isinstance(meta.get("width"), int)
            and meta.get("width", 0) > 0
            and isinstance(meta.get("height"), int)
            and meta.get("height", 0) > 0
            and isinstance(meta.get("format"), str)
            and bool(meta.get("format"))
        )
        if needs_probe and not meta.get("_spec_probed"):
            content_file = self.content_path(rec)
            if content_file.is_file():
                specs = inspect_image_file(content_file)
                changed = False
                for k, v in specs.items():
                    if meta.get(k) != v:
                        meta[k] = v
                        changed = True
                if rec.size_bytes <= 0:
                    try:
                        rec.size_bytes = content_file.stat().st_size
                        changed = True
                    except OSError:
                        pass
                if changed:
                    try:
                        tmp_meta = meta_file.with_suffix(".tmp")
                        with tmp_meta.open("w", encoding="utf-8") as stream:
                            json.dump(rec.to_dict(), stream, ensure_ascii=False)
                            stream.flush()
                            os.fsync(stream.fileno())
                        os.replace(tmp_meta, meta_file)
                        mtime_ns = meta_file.stat().st_mtime_ns
                    except OSError:
                        pass
                else:
                    meta["_spec_probed"] = True
        with self._index_lock:
            self._record_cache[cache_key] = (mtime_ns, rec)
        return rec

    def _build_index(self) -> None:
        if not self.root_dir.exists():
            return
        root_resolved = self.root_dir.resolve()
        self.rglob_count += 1
        for meta_file in self.root_dir.rglob("artifacts/art-*.json"):
            try:
                if not meta_file.is_file() or not meta_file.resolve().is_relative_to(root_resolved):
                    continue
                art_id = meta_file.stem
                rel_parts = meta_file.resolve().relative_to(root_resolved).parts
                if len(rel_parts) >= 3 and rel_parts[-2] == "artifacts":
                    proj_id = "/".join(rel_parts[:-2])
                    if proj_id:
                        self._project_index[art_id] = proj_id
            except OSError:
                continue
        self._index_built = True

    def get(self, artifact_id: str, project_id: str | None = None) -> ArtifactRecord:
        if project_id:
            meta_file = self._meta_path(project_id, artifact_id)
            if meta_file.is_file():
                with self._index_lock:
                    cached = self._record_cache.get((project_id, artifact_id))
                    try:
                        if cached is not None and cached[0] == meta_file.stat().st_mtime_ns:
                            return cached[1]
                    except OSError:
                        pass
                return self._load_meta_file(meta_file)
            raise ArtifactNotFound(f"Artifact 未找到: {artifact_id} (项目: {project_id})")

        with self._index_lock:
            indexed_project = self._project_index.get(artifact_id)
        if indexed_project:
            meta_file = self._meta_path(indexed_project, artifact_id)
            if meta_file.is_file():
                return self.get(artifact_id, indexed_project)

        with self._index_lock:
            if not self._index_built:
                self._build_index()
            indexed_project = self._project_index.get(artifact_id)
        if indexed_project:
            meta_file = self._meta_path(indexed_project, artifact_id)
            if meta_file.is_file():
                return self.get(artifact_id, indexed_project)

        self.rglob_count += 1
        for meta_file in self.root_dir.rglob(f"artifacts/{artifact_id}.json"):
            if meta_file.is_file() and meta_file.resolve().is_relative_to(self.root_dir.resolve()):
                return self._load_meta_file(meta_file)

        raise ArtifactNotFound(f"Artifact 未找到: {artifact_id}")

    def content_path(self, record: ArtifactRecord) -> Path:
        return self._project_dir(record.project_id) / f"{record.artifact_id}-{record.filename}"

    def download_path(self, artifact_id: str) -> str:
        return f"/image/artifacts/{artifact_id}/content"

    def thumbnail_path(self, artifact_id: str, edge: int = 320) -> str:
        return f"/image/artifacts/{artifact_id}/thumbnail?edge={edge}"

    @staticmethod
    def thumbnail_etag(sha256: str, edge: int = 320) -> str:
        return f'"{sha256}-{edge}-{THUMBNAIL_CACHE_VERSION}"'

    def _previews_dir(self) -> Path:
        p = self.root_dir / "_previews"
        p.mkdir(parents=True, exist_ok=True)
        return p

    def thumbnail_cache_path(self, sha256: str, edge: int = 320) -> Path:
        if edge not in ALLOWED_THUMBNAIL_EDGES:
            raise ValueError(f"unsupported thumbnail edge: {edge}")
        return self._previews_dir() / f"{sha256}_{edge}_{THUMBNAIL_CACHE_VERSION}.webp"

    def _get_thumb_lock(self, key: str) -> threading.Lock:
        with self._thumb_locks_guard:
            lock = self._thumb_locks.get(key)
            if lock is None:
                lock = threading.Lock()
                self._thumb_locks[key] = lock
            return lock

    def get_or_create_thumbnail(
        self,
        artifact_id: str,
        edge: int = 320,
        project_id: str | None = None,
    ) -> tuple[Path, str]:
        if edge not in ALLOWED_THUMBNAIL_EDGES:
            raise ValueError(f"unsupported thumbnail edge: {edge}")
        rec = self.get(artifact_id, project_id)
        target = self.thumbnail_cache_path(rec.sha256, edge)
        etag = self.thumbnail_etag(rec.sha256, edge)
        try:
            if target.is_file() and target.stat().st_size > 0:
                return target, etag
        except OSError:
            pass

        lock = self._get_thumb_lock(f"{rec.sha256}_{edge}")
        with lock:
            try:
                if target.is_file() and target.stat().st_size > 0:
                    return target, etag
            except OSError:
                pass

            if Image is None:
                raise ArtifactError("Pillow not available for thumbnail generation")

            src_path = self.content_path(rec)
            if not src_path.is_file():
                raise ArtifactNotFound(f"Artifact content missing: {artifact_id}")

            try:
                with Image.open(src_path) as img:
                    if getattr(img, "is_animated", False):
                        img.seek(0)
                    if ImageOps is not None:
                        try:
                            img = ImageOps.exif_transpose(img)
                        except Exception:
                            pass
                    w, h = int(img.width or 0), int(img.height or 0)
                    if w <= 0 or h <= 0:
                        raise ArtifactError("invalid image dimensions")
                    if w * h > 40_000_000:
                        raise ArtifactError("image pixel count exceeds limit")
                    has_alpha = img.mode in ("RGBA", "LA") or (
                        img.mode == "P" and "transparency" in img.info
                    )
                    work = img.convert("RGBA" if has_alpha else "RGB")
                    max_side = max(w, h)
                    if max_side > edge:
                        scale = edge / float(max_side)
                        new_w = max(1, int(round(w * scale)))
                        new_h = max(1, int(round(h * scale)))
                        resample = getattr(Image, "Resampling", Image).LANCZOS
                        work = work.resize((new_w, new_h), resample=resample)

                    out_buf = io.BytesIO()
                    work.save(out_buf, format="WEBP", quality=80, method=4)
                    webp_bytes = out_buf.getvalue()
                    if len(webp_bytes) > 512 * 1024:
                        out_buf = io.BytesIO()
                        work.save(out_buf, format="WEBP", quality=60, method=4)
                        webp_bytes = out_buf.getvalue()
            except ArtifactError:
                raise
            except Exception as exc:
                raise ArtifactError(f"failed to decode image for thumbnail: {exc}") from exc

            tmp_target = target.with_name(f".{target.name}.{uuid.uuid4().hex[:8]}.tmp")
            with tmp_target.open("wb") as stream:
                stream.write(webp_bytes)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(tmp_target, target)
            with self._thumb_locks_guard:
                self.thumbnail_generate_count += 1
            return target, etag

    def prewarm_thumbnails_sync(self, rec: ArtifactRecord) -> None:
        for edge in ALLOWED_THUMBNAIL_EDGES:
            try:
                self.get_or_create_thumbnail(rec.artifact_id, edge=edge, project_id=rec.project_id)
            except Exception:
                pass

    def prewarm_thumbnails_async(self, rec: ArtifactRecord) -> None:
        try:
            self._thumb_executor.submit(self.prewarm_thumbnails_sync, rec)
        except Exception:
            pass
