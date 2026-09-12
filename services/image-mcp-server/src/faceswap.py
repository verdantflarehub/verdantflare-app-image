from __future__ import annotations

import hashlib
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from PIL import Image

from .artifacts import ArtifactNotFound, ArtifactRecord, ArtifactStore


def resolve_source_face_path(
    artifacts: ArtifactStore,
    source_identity_artifact_id: str,
    project_id: str,
) -> Path:
    """解析基准角色卡物理路径（优先 ArtifactStore，兜底静态资产库路径）。"""
    # 1. 尝试从当前项目 Artifact 库读取
    try:
        source_rec = artifacts.get(source_identity_artifact_id, project_id)
        p = artifacts.content_path(source_rec)
        if p.is_file():
            return p
    except Exception:
        pass

    # 2. 尝试全局搜索 Artifact 库
    try:
        source_rec = artifacts.get(source_identity_artifact_id)
        p = artifacts.content_path(source_rec)
        if p.is_file():
            return p
    except Exception:
        pass

    # 3. 检查是否为直接文件路径
    candidate = Path(source_identity_artifact_id)
    if candidate.is_file():
        return candidate.resolve()

    # 4. 检查标准持久化资产路径与本地开发文档路径
    search_paths = [
        artifacts.root_dir / project_id / "assets" / source_identity_artifact_id,
        artifacts.root_dir / project_id / "assets" / f"{source_identity_artifact_id}.png",
        artifacts.root_dir / "assets" / source_identity_artifact_id,
        artifacts.root_dir / "assets" / f"{source_identity_artifact_id}.png",
        Path("/data/projects/creator/assets") / source_identity_artifact_id,
        Path("/data/projects/creator/assets") / f"{source_identity_artifact_id}.png",
        Path("docs/assets") / source_identity_artifact_id,
        Path("docs/assets") / f"{source_identity_artifact_id}.png",
    ]
    parents = Path(__file__).resolve().parents
    if len(parents) > 4:
        search_paths.extend([
            parents[4] / "docs" / "assets" / source_identity_artifact_id,
            parents[4] / "docs" / "assets" / f"{source_identity_artifact_id}.png",
        ])
    for sp in search_paths:
        if sp.is_file():
            return sp.resolve()

    raise FileNotFoundError(f"基准角色卡资产未找到: {source_identity_artifact_id}")


def resolve_target_image_path(
    artifacts: ArtifactStore,
    target_artifact_id: str,
    project_id: str | None = None,
) -> tuple[ArtifactRecord, Path]:
    """解析阶段一原始大片物理路径。"""
    if project_id:
        try:
            rec = artifacts.get(target_artifact_id, project_id)
            p = artifacts.content_path(rec)
            if p.is_file():
                return rec, p.resolve()
        except Exception:
            pass

    try:
        rec = artifacts.get(target_artifact_id)
        p = artifacts.content_path(rec)
        if p.is_file():
            return rec, p.resolve()
    except Exception as exc:
        raise ArtifactNotFound(f"目标大片 Artifact 未找到: {target_artifact_id}") from exc

    raise FileNotFoundError(f"目标大片文件不存在: {p}")


async def handle_faceswap(
    artifacts: ArtifactStore,
    target_artifact_id: str,
    source_identity_artifact_id: str,
    identity_strength: float = 0.95,
    restore_face: bool = True,
    restoration_fidelity: float = 0.85,
    project_id: str | None = None,
    target_face_index: int = 0,
) -> dict[str, Any]:
    """执行阶段二骨相置换与面容超分融合，内部向 image-face-fusion-api 发起零网络拷贝 RPC。"""
    # 1. 字段校验与范围规约
    strength = max(0.0, min(1.0, float(identity_strength)))
    fidelity = max(0.0, min(1.0, float(restoration_fidelity)))
    face_idx = max(0, int(target_face_index))

    # 2. 解析目标大片物理文件
    try:
        target_rec, target_path = resolve_target_image_path(
            artifacts=artifacts,
            target_artifact_id=target_artifact_id,
            project_id=project_id,
        )
    except Exception as exc:
        return {
            "status": "failed",
            "error": f"目标大片解析失败: {exc}",
        }

    resolved_project_id = project_id or target_rec.project_id

    # 3. 解析基准角色卡物理文件
    try:
        source_path = resolve_source_face_path(
            artifacts=artifacts,
            source_identity_artifact_id=source_identity_artifact_id,
            project_id=resolved_project_id,
        )
    except Exception as exc:
        return {
            "status": "failed",
            "error": f"基准角色卡解析失败: {exc}",
        }

    # 4. 规划零网络拷贝物理输出路径
    output_artifact_id = f"art-{uuid.uuid4().hex[:16]}"
    ext = Path(target_rec.filename).suffix or ".png"
    output_filename = f"fused-{target_rec.filename}"
    if not output_filename.endswith(ext):
        output_filename = f"{output_filename}{ext}"

    project_dir = artifacts._project_dir(resolved_project_id)
    output_physical_path = project_dir / f"{output_artifact_id}-{output_filename}"

    # 5. 构造微服务请求负载
    api_url = os.environ.get("IMAGE_FACE_FUSION_API_URL", "http://image-face-fusion-api:8000").rstrip("/")
    fuse_endpoint = f"{api_url}/v1/fuse"
    timeout_sec = float(os.environ.get("IMAGE_FACE_FUSION_TIMEOUT", "15.0"))

    payload = {
        "target_image_path": str(target_path),
        "source_face_path": str(source_path),
        "target_face_index": face_idx,
        "identity_strength": strength,
        "restore_face": bool(restore_face),
        "restoration_fidelity": fidelity,
        "output_path": str(output_physical_path.resolve()),
    }

    # 6. 发起内部 HTTP RPC
    try:
        async with httpx.AsyncClient(timeout=timeout_sec) as client:
            resp = await client.post(fuse_endpoint, json=payload)
    except httpx.TimeoutException:
        return {
            "status": "failed",
            "error": f"人脸融合微服务请求超时（超过 {timeout_sec} 秒）",
        }
    except Exception as exc:
        return {
            "status": "failed",
            "error": f"调用人脸融合微服务失败 ({fuse_endpoint}): {exc}",
        }

    if resp.status_code != 200:
        return {
            "status": "failed",
            "error": f"人脸融合微服务返回错误 ({resp.status_code}): {resp.text}",
        }

    try:
        fuse_data = resp.json()
    except Exception:
        return {
            "status": "failed",
            "error": f"解析人脸融合微服务响应失败: {resp.text}",
        }

    if fuse_data.get("status") != "success":
        return {
            "status": "failed",
            "error": fuse_data.get("detail") or fuse_data.get("error") or "人脸融合执行失败",
        }

    # 7. 产物物理完整性核验与 SHA-256 计算
    if not output_physical_path.is_file() or output_physical_path.stat().st_size == 0:
        return {
            "status": "failed",
            "error": "融合微服务未生成有效的产物文件或文件为空",
        }

    data_bytes = output_physical_path.read_bytes()
    sha256_hash = hashlib.sha256(data_bytes).hexdigest()

    # 获取图像尺寸
    width, height = 0, 0
    try:
        with Image.open(output_physical_path) as img:
            width, height = img.size
    except Exception:
        pass

    # 8. 登记不可变 ArtifactRecord 与元数据持久化
    now = datetime.now(timezone.utc).isoformat()
    record = ArtifactRecord(
        artifact_id=output_artifact_id,
        project_id=resolved_project_id,
        filename=output_filename,
        media_type="image/png",
        size_bytes=len(data_bytes),
        sha256=sha256_hash,
        created_at=now,
        metadata={
            "type": "face_fusion",
            "stage": 2,
            "target_artifact_id": target_artifact_id,
            "source_identity_artifact_id": source_identity_artifact_id,
            "identity_strength": strength,
            "restore_face": bool(restore_face),
            "restoration_fidelity": fidelity,
            "target_face_index": face_idx,
            "width": width,
            "height": height,
            "arcface_similarity": fuse_data.get("arcface_similarity", 0.0),
            "detected_faces": fuse_data.get("detected_faces", 1),
            "inference_time_ms": fuse_data.get("inference_time_ms", 0),
            "pipeline": fuse_data.get("pipeline", "retinaface+arcface512+inswapper128+codeformer"),
        },
    )

    meta_file = artifacts._meta_path(resolved_project_id, output_artifact_id)
    meta_file.write_text(json.dumps(record.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "status": "completed",
        "project_id": resolved_project_id,
        "artifact": record.to_dict(),
        "download_path": artifacts.download_path(record.artifact_id),
        "metrics": {
            "arcface_similarity": fuse_data.get("arcface_similarity", 0.0),
            "inference_time_ms": fuse_data.get("inference_time_ms", 0),
            "detected_faces": fuse_data.get("detected_faces", 1),
            "pipeline": fuse_data.get("pipeline", ""),
            "width": width,
            "height": height,
        },
    }
