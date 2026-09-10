from __future__ import annotations

import contextlib
import hmac
import json
import os
from pathlib import Path
from typing import Any

from mcp import types
from mcp.server.mcpserver import MCPServer
from mcp.server.transport_security import TransportSecuritySettings
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response
from starlette.routing import Mount, Route

from .artifacts import ArtifactError, ArtifactNotFound, ArtifactStore
from .dashboard import api_create_task, api_list_tasks, api_task_stats, dashboard_page
from .providers.codex import CodexProvider, CodexProviderError, DEFAULT_IMAGE_MODEL
from .providers.gemini import DEFAULT_GEMINI_MODEL, GeminiProvider, GeminiProviderError
from .queue import TaskQueueManager
from .tasks import TaskNotFound, TaskStore

artifacts = ArtifactStore.from_environment()
tasks = TaskStore.from_environment()
codex_provider = CodexProvider()
gemini_provider = GeminiProvider()
queue_manager = TaskQueueManager(
    tasks=tasks,
    artifacts=artifacts,
    codex_provider=codex_provider,
    gemini_provider=gemini_provider,
)

mcp = MCPServer("VerdantFlare Image")


def _result(value: dict[str, Any]) -> types.CallToolResult:
    return types.CallToolResult(
        content=[types.TextContent(type="text", text=json.dumps(value, ensure_ascii=False))],
        structuredContent=value,
    )


@mcp.tool(name="artifact.import")
def artifact_import(
    project_id: str, source_url: str, filename: str, expected_sha256: str
) -> types.CallToolResult:
    record = artifacts.import_from_url(
        project_id=project_id,
        source_url=source_url,
        filename=filename,
        expected_sha256=expected_sha256,
    )
    return _result(
        {
            "status": "completed",
            "project_id": project_id,
            "artifact": record.model_dump(),
            "download_path": artifacts.download_path(record.artifact_id),
        }
    )


@mcp.tool(name="image.generate")
async def image_generate(
    project_id: str,
    idempotency_key: str,
    prompt: str,
    engine: str = "codex",
    model: str = "",
    aspect_ratio: str = "16:9",
    resolution: str = "2k",
    quality: str = "high",
    background: str = "auto",
) -> types.CallToolResult:
    """根据文本提示词生成图像。

    遵循 GPT Image 2.5 官方提示词标准 (Scene, Subject, Details, Constraints)。
    - engine: 默认 "codex" (OpenAI Responses API 驱动)，亦可选 "gemini"
    - model:
        codex 引擎推荐: "gpt-image-2.5-sunburst" (画质首选，微观细节与皮肤毛孔) 或 "gpt-image-2.5-flare" (极速响应)
        gemini 引擎: "gemini-3.1-flash-image"
    - aspect_ratio: "16:9", "9:16", "1:1", "4:3", "3:4"
    - resolution: "2k" (默认), "4k" (对齐 3840x2160 / 2160x3840 官方尺寸约束)
    - quality: "auto", "low", "medium", "high", "xhigh", "max" (历史 "hd" 兼容映射为 "high")
    - background: "auto", "transparent" (纯净透明通道，输出 PNG), "opaque"
    """
    resolved_model = model or (
        DEFAULT_IMAGE_MODEL if engine.lower() == "codex" else DEFAULT_GEMINI_MODEL
    )
    params = {
        "action": "generate",
        "prompt": prompt,
        "model": resolved_model,
        "aspect_ratio": aspect_ratio,
        "resolution": resolution,
        "quality": quality,
        "background": background,
    }
    task = tasks.create(
        project_id=project_id,
        idempotency_key=idempotency_key,
        engine=engine,
        model=resolved_model,
        request_params=params,
    )

    if task.status == "queued":
        await queue_manager.enqueue(
            task_id=task.task_id,
            project_id=project_id,
            engine=engine,
            action="generate",
            params=params,
        )

    return _result(
        {
            "task_id": task.task_id,
            "status": task.status,
            "created_at": task.created_at,
        }
    )


@mcp.tool(name="image.edit")
async def image_edit(
    project_id: str,
    idempotency_key: str,
    source_artifact_id: str,
    prompt: str,
    reference_artifact_ids: list[str] | None = None,
    engine: str = "codex",
    model: str = "",
    size: str = "",
    quality: str = "auto",
    background: str = "auto",
) -> types.CallToolResult:
    """基于既有参考底图执行多模态指令编辑、骨相锁定换装或多图融合合成。

    遵循 GPT Image 2.5 官方编辑准则（明确指定“仅变更项”与“严格保留项”，为多参考图分配职责）。
    - source_artifact_id: 主参考底图 ID (如主体肖像底图)
    - reference_artifact_ids: 可选附加参考图 ID 列表 (如单品服装图、环境参考图)
    - prompt: 编辑指令，建议使用 "Change only X, preserve exact facial features, skin tone, and body pose"
    - model: "gpt-image-2.5-sunburst" (默认) 或 "gpt-image-2.5-flare"
    - background: "auto", "transparent" (扣除背景生成透明底), "opaque"
    - quality: "auto", "low", "medium", "high", "xhigh", "max"
    """
    resolved_model = model or (
        DEFAULT_IMAGE_MODEL if engine.lower() == "codex" else DEFAULT_GEMINI_MODEL
    )
    params = {
        "action": "edit",
        "source_artifact_id": source_artifact_id,
        "reference_artifact_ids": reference_artifact_ids or [],
        "prompt": prompt,
        "model": resolved_model,
        "size": size,
        "quality": quality,
        "background": background,
    }
    task = tasks.create(
        project_id=project_id,
        idempotency_key=idempotency_key,
        engine=engine,
        model=resolved_model,
        request_params=params,
    )

    if task.status == "queued":
        await queue_manager.enqueue(
            task_id=task.task_id,
            project_id=project_id,
            engine=engine,
            action="edit",
            params=params,
        )

    return _result(
        {
            "task_id": task.task_id,
            "status": task.status,
            "created_at": task.created_at,
        }
    )


@mcp.tool(name="image.inpaint")
async def image_inpaint(
    project_id: str,
    idempotency_key: str,
    source_artifact_id: str,
    mask_artifact_id: str,
    prompt: str,
    engine: str = "codex",
    model: str = "",
    background: str = "auto",
) -> types.CallToolResult:
    """基于遮罩（Mask）的精准局部重绘与无痕物体抹除。"""
    resolved_model = model or (
        DEFAULT_IMAGE_MODEL if engine.lower() == "codex" else DEFAULT_GEMINI_MODEL
    )
    params = {
        "action": "inpaint",
        "source_artifact_id": source_artifact_id,
        "mask_artifact_id": mask_artifact_id,
        "prompt": prompt,
        "model": resolved_model,
        "background": background,
    }
    task = tasks.create(
        project_id=project_id,
        idempotency_key=idempotency_key,
        engine=engine,
        model=resolved_model,
        request_params=params,
    )

    if task.status == "queued":
        await queue_manager.enqueue(
            task_id=task.task_id,
            project_id=project_id,
            engine=engine,
            action="inpaint",
            params=params,
        )

    return _result(
        {
            "task_id": task.task_id,
            "status": task.status,
            "created_at": task.created_at,
        }
    )


@mcp.tool(name="image.status")
def image_status(task_id: str) -> types.CallToolResult:
    task = tasks.get(task_id)
    return _result(
        {
            "task_id": task.task_id,
            "status": task.status,
            "created_at": task.created_at,
            "updated_at": task.updated_at,
            "duration_seconds": task.duration_seconds,
            "artifact_id": task.artifact_id,
            "error": task.error,
        }
    )


@mcp.tool(name="image.result")
def image_result(task_id: str) -> types.CallToolResult:
    task = tasks.get(task_id)
    if task.status != "completed" or not task.artifact_id:
        return _result(
            {
                "task_id": task.task_id,
                "status": task.status,
                "error": task.error or "任务尚未完成",
            }
        )

    artifact = artifacts.get(task.artifact_id, task.project_id)
    return _result(
        {
            "task_id": task.task_id,
            "artifact_id": artifact.artifact_id,
            "project_id": artifact.project_id,
            "filename": artifact.filename,
            "sha256": artifact.sha256,
            "size_bytes": artifact.size_bytes,
            "metadata": artifact.metadata,
            "duration_seconds": task.duration_seconds,
            "download_path": artifacts.download_path(artifact.artifact_id),
        }
    )


@mcp.tool(name="image.list")
def image_list(
    project_id: str | None = None,
    engine: str | None = None,
    status: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> types.CallToolResult:
    records, total = tasks.list_tasks(
        project_id=project_id,
        engine=engine,
        status=status,
        limit=limit,
        offset=offset,
    )
    return _result(
        {
            "tasks": [r.to_dict() for r in records],
            "total": total,
            "limit": limit,
            "offset": offset,
        }
    )


def transport_security_from_environment() -> TransportSecuritySettings:
    raw_hosts = [x.strip() for x in os.environ.get("IMAGE_MCP_ALLOWED_HOSTS", "").split(",") if x.strip()]
    hosts = set(raw_hosts)
    for h in raw_hosts:
        if not h.endswith(":*"):
            hosts.add(f"{h}:*")
    origins = [x.strip() for x in os.environ.get("IMAGE_MCP_ALLOWED_ORIGINS", "").split(",") if x.strip()]
    default_hosts = ["127.0.0.1:*", "localhost:*", "[::1]:*", "testserver", "testserver:*"]
    default_origins = ["http://127.0.0.1:*", "http://localhost:*", "http://[::1]:*", "http://testserver:*", "http://testserver"]
    return TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=list(hosts) if hosts else default_hosts,
        allowed_origins=origins if origins else default_origins,
    )


async def health(request: Request) -> JSONResponse:
    artifacts.ensure_ready()
    tasks.ensure_ready()
    return JSONResponse({"status": "ok", "service": "image-mcp-server", "version": "v0.2.0"})


async def artifact_content(request: Request) -> Response:
    try:
        artifact_id = request.path_params.get("artifact_id")
        rec = artifacts.get(artifact_id)
        path = artifacts.content_path(rec)
        return FileResponse(path, media_type=rec.media_type, filename=rec.filename)
    except (ValueError, ArtifactNotFound):
        return JSONResponse({"error": "artifact_not_found"}, status_code=404)


class MCPPathRewriteMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # 兼容直连 /image、/image/ 或 /，重写为 /mcp，使 streamable_http_app 能够无缝匹配
        path = request.url.path
        if path in ("/image", "/image/", "/", "/image/mcp", "/image/mcp/"):
            request.scope["path"] = "/mcp"
        return await call_next(request)


class BearerAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        # 放行健康检查、看板页面（前端自适应 Token 输入与存储）、API（内部鉴权）与产物下载
        if (
            path in ("/health", "/image/health")
            or "/dashboard" in path
            or path.startswith("/api/")
            or path.startswith("/image/api/")
            or "/artifacts/" in path
        ):
            return await call_next(request)

        token = os.environ.get("IMAGE_MCP_BEARER_TOKEN", "").strip()
        if token:
            auth_header = request.headers.get("authorization", "")
            query_token = request.query_params.get("token", "")
            valid_header = auth_header and hmac.compare_digest(auth_header, f"Bearer {token}")
            valid_query = query_token and hmac.compare_digest(query_token, token)
            if not (valid_header or valid_query):
                return JSONResponse({"error": "unauthorized"}, status_code=401)

        return await call_next(request)


@contextlib.asynccontextmanager
async def lifespan(app: Starlette):
    artifacts.ensure_ready()
    tasks.ensure_ready()
    app.state.tasks = tasks
    app.state.artifacts = artifacts
    app.state.queue_manager = queue_manager
    await queue_manager.start()
    try:
        if getattr(mcp.session_manager, "_has_started", False):
            mcp.session_manager._has_started = False
        async with mcp.session_manager.run():
            yield
    finally:
        await queue_manager.stop()


mcp_app = mcp.streamable_http_app(
    json_response=True,
    stateless_http=True,
    transport_security=transport_security_from_environment(),
)
app = Starlette(
    routes=[
        Route("/health", health, methods=["GET"]),
        Route("/image/health", health, methods=["GET"]),
        Route("/dashboard", dashboard_page, methods=["GET"]),
        Route("/dashboard/", dashboard_page, methods=["GET"]),
        Route("/image/dashboard", dashboard_page, methods=["GET"]),
        Route("/image/dashboard/", dashboard_page, methods=["GET"]),
        Route("/api/tasks", api_list_tasks, methods=["GET"]),
        Route("/image/api/tasks", api_list_tasks, methods=["GET"]),
        Route("/api/tasks", api_create_task, methods=["POST"]),
        Route("/image/api/tasks", api_create_task, methods=["POST"]),
        Route("/api/tasks/stats", api_task_stats, methods=["GET"]),
        Route("/image/api/tasks/stats", api_task_stats, methods=["GET"]),
        Route("/artifacts/{artifact_id}/content", artifact_content, methods=["GET"]),
        Route("/image/artifacts/{artifact_id}/content", artifact_content, methods=["GET"]),
        Mount("/mcp", app=mcp_app),
        Mount("/image", app=mcp_app),
        Mount("/", app=mcp_app),
    ],
    middleware=[
        Middleware(MCPPathRewriteMiddleware),
        Middleware(BearerAuthMiddleware),
    ],
    lifespan=lifespan,
)
