from __future__ import annotations

import asyncio
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
from .gpu_lease import LeaseBusy, SchedulerLeaseClient, SchedulerLeaseGrant


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
    scheduler: SchedulerLeaseClient | None = None,
    http_client: httpx.AsyncClient | None = None,
) -> dict[str, Any]:
    """执行阶段二骨相置换与面容超分融合，内部向 image-face-fusion 发起零网络拷贝 RPC。"""
    # 1. 字段校验与范围规约
    strength = max(0.5, min(1.0, float(identity_strength)))
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
    output_filename = f"fused-{Path(target_rec.filename).stem}.png"

    project_dir = artifacts._project_dir(resolved_project_id)
    output_physical_path = project_dir / f"{output_artifact_id}-{output_filename}"

    # 5. 构造微服务请求负载
    api_url = os.environ.get("IMAGE_FACE_FUSION_API_URL", "http://image-face-fusion:8000").rstrip("/")
    fuse_endpoint = f"{api_url}/v1/fuse"
    timeout_sec = float(os.environ.get("IMAGE_FACE_FUSION_TIMEOUT", "150.0"))

    payload = {
        "target_image_path": str(target_path),
        "source_face_path": str(source_path),
        "target_face_index": face_idx,
        "identity_strength": strength,
        "restore_face": bool(restore_face),
        "restoration_fidelity": fidelity,
        "output_path": str(output_physical_path.resolve()),
    }

    # 6. 发起内部 HTTP RPC（仅在 CUDA 运行态下向 :5056 申请 face-fusion 共享切片并注入 6 元组）
    sched = scheduler or SchedulerLeaseClient.from_environment()
    if getattr(artifacts, "root_dir", None):
        sched.configure_journal(Path(artifacts.root_dir) / "gpu_pending_releases.json")
    grant: SchedulerLeaseGrant | None = None
    runtime_ident: dict[str, Any] = {}
    reclaim_proof: dict[str, Any] | None = None
    settled = False
    target_bound = False

    async def _run_rpc(client: httpx.AsyncClient) -> httpx.Response:
        nonlocal grant, runtime_ident, reclaim_proof, settled, target_bound
        if sched.enabled:
            cuda_active = False
            try:
                h_resp = await client.get(f"{api_url}/health", timeout=5.0)
                if h_resp.status_code == 200:
                    h_data = h_resp.json()
                    if isinstance(h_data, dict):
                        ep = str(h_data.get("execution_provider") or "").lower()
                        cuda_active = bool(h_data.get("cuda_available")) or ("cuda" in ep) or ("tensorrt" in ep)
                        if isinstance(h_data.get("runtime_identity"), dict):
                            runtime_ident = dict(h_data["runtime_identity"])
            except Exception:
                cuda_active = False

            if cuda_active:
                vis = runtime_ident.get("visible_gpu_uuids")
                grant = await asyncio.to_thread(
                    sched.acquire_for_queue,
                    consumer_kind="face-fusion",
                    task_id=output_artifact_id,
                    stage_id="fuse",
                    k8s_slots_required=0,
                    pod_uid=str(runtime_ident.get("pod_uid") or ""),
                    container_id=str(runtime_ident.get("container_id") or ""),
                    boot_id=str(runtime_ident.get("boot_id") or ""),
                    visible_gpu_uuids=vis if isinstance(vis, (list, tuple)) else None,
                )
                if grant is not None:
                    if not grant.is_target_bound and runtime_ident.get("pod_uid"):
                        await asyncio.to_thread(sched.bind_target, grant, runtime_identity=runtime_ident)
                    target_bound = grant.is_target_bound

        if grant is not None:
            r = await client.post(fuse_endpoint, json=payload, headers=grant.headers())
            raw_proof = r.headers.get("x-vf-reclaim-proof", "").strip()
            if raw_proof:
                try:
                    parsed = json.loads(raw_proof)
                    if isinstance(parsed, dict):
                        reclaim_proof = parsed
                except Exception:
                    reclaim_proof = None
            if reclaim_proof is None:
                tok = (
                    os.environ.get("IMAGE_RUNTIME_TOKEN", "").strip()
                    or sched.token
                )
                try:
                    rec_resp = await client.post(
                        f"{api_url}/internal/v1/gpu-guard/reclaim",
                        json={"lease_id": grant.lease_id, "fencing_token": grant.fencing_token},
                        headers={"Authorization": f"Bearer {tok}"} if tok else {},
                        timeout=5.0,
                    )
                    if rec_resp.status_code == 200:
                        rec_data = rec_resp.json()
                        if isinstance(rec_data, dict):
                            reclaim_proof = rec_data
                except Exception:
                    pass
            settled = bool(
                isinstance(reclaim_proof, dict)
                and reclaim_proof.get("reclaimed") is True
                and reclaim_proof.get("proof_source") == "receiver_guard"
                and reclaim_proof.get("token_sealed") is True
            )
            return r
        r = await client.post(fuse_endpoint, json=payload)
        settled = True
        return r

    try:
        if http_client is not None:
            resp = await _run_rpc(http_client)
        else:
            async with httpx.AsyncClient(timeout=timeout_sec) as client:
                resp = await _run_rpc(client)
    except LeaseBusy as exc:
        return {
            "status": "failed",
            "error": f"waiting_gpu_lease: {exc.holder or exc}",
        }
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
    finally:
        if grant is not None:
            try:
                if not target_bound:
                    rel_mode = "abort_unstarted"
                    rel_settled = False
                elif settled and reclaim_proof:
                    rel_mode = "warm_idle_settled"
                    rel_settled = True
                else:
                    rel_mode = ""
                    rel_settled = False
                await asyncio.to_thread(
                    sched.release,
                    grant,
                    worker_settled=rel_settled,
                    runtime_identity=runtime_ident if target_bound else None,
                    reclaim_mode=rel_mode,
                    reclaim_proof=reclaim_proof if rel_settled else None,
                )
            except Exception:
                pass

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
            "pipeline": fuse_data.get("pipeline", ""),
            "execution": fuse_data.get("execution", {}),
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
            "execution": fuse_data.get("execution", {}),
            "width": width,
            "height": height,
        },
    }
