from __future__ import annotations

import contextlib
import hmac
import json
import os
import httpx
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
from .content_rating import ContentRatingError, ContentRatingStore, include_nsfw_from_headers
from .dashboard import (
    api_create_task,
    api_delete_task,
    api_list_tasks,
    api_task_content_rating,
    api_task_stats,
    api_update_task_title,
    api_upload_artifact,
    dashboard_page,
)
from .faceswap import handle_faceswap
from .providers.codex import CodexProvider, CodexProviderError, DEFAULT_IMAGE_MODEL
from .providers.gemini import DEFAULT_GEMINI_MODEL, GeminiProvider, GeminiProviderError
from .providers.moss import DEFAULT_MOSS_MODEL
from .queue import TaskQueueManager
from .registry import EtcdRegistry
from .tasks import TaskNotFound, TaskStore
from .channel_authority import CALLER, authorize, create_client as create_authority_client
from .providers.google_channel import GoogleChannel
from .providers.channel import ImageChannel
from .channel_discovery_client import ChannelDiscoveryClient

artifacts = ArtifactStore.from_environment()
tasks = TaskStore.from_environment()
ratings = ContentRatingStore(tasks.db_path, domain="image")
codex_provider = CodexProvider()
gemini_provider = GeminiProvider()
channel_client = httpx.Client(trust_env=False)
authority_client = create_authority_client()
google_url = os.environ.get('IMAGE_GOOGLE_CHANNEL_URL', '')
google_channel = GoogleChannel(google_url, os.environ.get('IMAGE_GOOGLE_CHANNEL_TOKEN', ''),
                               channel_client) if google_url else None
grok_url = os.environ.get('IMAGE_GROK_CHANNEL_URL', '')
grok_channel = ImageChannel(grok_url, os.environ.get('IMAGE_GROK_CHANNEL_TOKEN', ''),
                            channel_client) if grok_url else None
chatgpt_url = os.environ.get('IMAGE_CHATGPT_CHANNEL_URL', '')
chatgpt_channel = ImageChannel(chatgpt_url, os.environ.get('IMAGE_CHATGPT_CHANNEL_TOKEN', ''),
                               channel_client) if chatgpt_url else None
channel_discovery = ChannelDiscoveryClient.from_environment()
queue_manager = TaskQueueManager(
    tasks=tasks,
    artifacts=artifacts,
    codex_provider=codex_provider,
    gemini_provider=gemini_provider,
    google_channel=google_channel,
    grok_channel=grok_channel,
    chatgpt_channel=chatgpt_channel,
    channel_directory=channel_discovery.directory if channel_discovery else None,
    google_authorize=lambda project_id, authority: authorize(authority_client, project_id, authority),
)

mcp = MCPServer("VerdantFlare Image")

IMAGE_TOOLS_SCHEMA = [
    {
        "name": "image.create",
        "description": "Create an asynchronous image generation task using Codex (GPT Image 2.5), Gemini, Grok, or MOSS-Image.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "description": "Image description prompt"},
                "engine": {"type": "string", "enum": ["codex", "gemini", "grok", "moss-image"], "default": "codex"},
                "model": {
                    "type": "string",
                    "description": "Model identifier, e.g. gpt-image-2.5-sunburst, gemini-nano-banana-2.1, gemini-3-pro-image, gemini-3.1-flash-image, gemini-3.1-flash-lite-image",
                },
                "aspect_ratio": {"type": "string", "enum": ["16:9", "9:16", "1:1", "4:3", "3:4", "2:3", "3:2"], "default": "16:9"},
                "resolution": {"type": "string", "enum": ["1k", "2k", "4k"], "default": "2k"},
                "quality": {"type": "string", "description": "Engine-specific; omit for Grok"},
                "background": {"type": "string", "enum": ["auto", "transparent", "opaque"], "default": "auto"},
                "instructions": {"type": "string", "description": "Optional instructions for prompt interpreter"},
                "source_artifact_id": {"type": "string", "description": "Optional primary reference image artifact ID"},
                "reference_artifact_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional reference image artifact IDs (up to 6 for Gemini multimodal reference, 1 for MOSS-Image Qwen3.5-9B vision reference)",
                },
                "face_reference_artifact_id": {
                    "type": "string",
                    "description": "Optional identity/visual reference artifact ID for MOSS-Image Qwen3.5-9B vision conditioning",
                },
                "source_identity_artifact_id": {
                    "type": "string",
                    "description": "Alias for face_reference_artifact_id",
                },
                "identity_strength": {"type": "number", "default": 0.95, "description": "Face identity fusion strength [0.5, 1.0] (moss-image)"},
                "restore_face": {"type": "boolean", "default": True, "description": "Preserve face restoration metadata flag (moss-image)"},
                "restoration_fidelity": {"type": "number", "default": 0.85, "description": "Face restoration fidelity [0.0, 1.0] (moss-image)"},
                "target_face_index": {"type": "integer", "default": 0, "description": "Target face index sorted by largest area (moss-image)"},
                "negative_prompt": {"type": "string", "description": "Optional negative prompt (moss-image)"},
                "width": {"type": "integer", "description": "Pixel width, multiple of 16 (moss-image)"},
                "height": {"type": "integer", "description": "Pixel height, multiple of 16 (moss-image)"},
                "steps": {"type": "integer", "description": "Sampling steps (default 50 for moss-image)"},
                "cfg_scale": {"type": "number", "description": "Guidance scale (default 3.0 for moss-image)"},
                "sampler": {"type": "string", "description": "Sampler name (default euler for moss-image)"},
                "scheduler": {"type": "string", "description": "Scheduler name (default simple for moss-image)"},
                "shift": {"type": "number", "description": "ModelSamplingFlux/AuraFlow shift (default 5.0 for moss-image)"},
                "skip_single_layers": {"type": "string", "description": "SkipLayerGuidanceDiTSimple single_layers (default '22' for moss-image)"},
                "seed": {"type": "integer", "description": "Random seed (moss-image)"},
                "layout": {"type": "object", "description": "Experimental JSON layout (abs_centralized yxyx, <=32 items, moss-image)"},
                "project_id": {"type": "string", "description": "Project ID"},
                "idempotency_key": {"type": "string", "description": "Unique idempotency key"},
            },
            "required": ["prompt"],
        },
    },
    {
        "name": "image.edit",
        "description": "Edit an existing reference image with multimodal prompt directives.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "source_artifact_id": {"type": "string", "description": "Base image artifact ID"},
                "prompt": {"type": "string", "description": "Editing instructions"},
                "reference_artifact_ids": {"type": "array", "items": {"type": "string"}},
                "mask_artifact_id": {"type": "string"},
                "aspect_ratio": {"type": "string", "enum": ["16:9", "9:16", "1:1", "4:3", "3:4", "2:3", "3:2"]},
                "resolution": {"type": "string", "enum": ["1k", "2k", "4k"], "description": "Target resolution (1k/2k for Grok and MOSS-Image; 2k/4k for Codex)"},
                "engine": {"type": "string", "enum": ["codex", "gemini", "grok", "moss-image"], "default": "codex"},
                "model": {"type": "string"},
                "quality": {"type": "string", "description": "Engine-specific; omit for Grok"},
                "background": {"type": "string", "default": "auto"},
                "project_id": {"type": "string"},
                "idempotency_key": {"type": "string"},
            },
            "required": ["source_artifact_id", "prompt"],
        },
    },
    {
        "name": "image.inpaint",
        "description": "Inpaint a specified area using a mask artifact.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "source_artifact_id": {"type": "string", "description": "Base image artifact ID"},
                "mask_artifact_id": {"type": "string", "description": "Mask image artifact ID"},
                "prompt": {"type": "string", "description": "Inpainting instructions"},
                "reference_artifact_ids": {"type": "array", "items": {"type": "string"}},
                "engine": {"type": "string", "default": "codex"},
                "model": {"type": "string"},
                "background": {"type": "string", "default": "auto"},
                "project_id": {"type": "string"},
                "idempotency_key": {"type": "string"},
            },
            "required": ["source_artifact_id", "mask_artifact_id", "prompt"],
        },
    },
    {
        "name": "image.faceswap",
        "description": "Swap face identity onto a target image with optional enhancement.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "target_artifact_id": {"type": "string", "description": "Target image artifact ID"},
                "source_identity_artifact_id": {"type": "string", "description": "Source identity artifact ID"},
                "identity_strength": {"type": "number", "default": 0.95},
                "restore_face": {"type": "boolean", "default": True},
                "restoration_fidelity": {"type": "number", "default": 0.85},
                "project_id": {"type": "string"},
                "target_face_index": {"type": "integer", "default": 0},
            },
            "required": ["target_artifact_id", "source_identity_artifact_id"],
        },
    },
    {
        "name": "image.status",
        "description": "Query status and metadata of an image generation task.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "task_id": {"type": "string", "description": "Task ID"},
            },
            "required": ["task_id"],
        },
    },
    {
        "name": "image.result",
        "description": "Retrieve final generated image artifact and download path for a completed task.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "task_id": {"type": "string", "description": "Task ID"},
            },
            "required": ["task_id"],
        },
    },
]

for tool in IMAGE_TOOLS_SCHEMA:
    if tool['name'] == 'image.result':
        tool['inputSchema']['properties']['execution_proof_only'] = {'type': 'boolean', 'default': False}
    if tool['name'] in {'image.create', 'image.generate', 'image.edit', 'image.inpaint'}:
        tool['inputSchema']['properties'].update({
            'channel_account_id': {'type': 'string', 'description': 'Channel account for the selected engine; omitted uses explicit default'},
            'budget_limit': {'type': 'string', 'description': 'Maximum USD reservation, not an estimated charge'},
        })

IMAGE_TOOLS_SCHEMA.append(
    {
        "name": "image.rate_content",
        "description": "Record content rating (general or nsfw) for an image artifact or task version. Add-only; cannot clear an existing nsfw rating.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "version_id": {"type": "string", "description": "Target artifact_id or task_id"},
                "rating": {"type": "string", "enum": ["general", "nsfw"], "description": "Content rating"},
                "reason": {"type": "string", "description": "Reason for rating"},
                "rubric_version": {"type": "string", "default": "v1.0", "description": "Rubric version"},
                "project_id": {"type": "string", "description": "Optional project_id for cross-project isolation check"},
            },
            "required": ["version_id", "rating", "reason"],
        },
    }
)
IMAGE_TOOLS_SCHEMA.append({"name": "image.capabilities", "description": "Read image engines and configured adapter limits without generating media.",
                           "inputSchema": {"type": "object", "properties": {}}})
SERVICE_VERSION = "v0.2.37"
registry = EtcdRegistry(domain="image", tools=IMAGE_TOOLS_SCHEMA, version=SERVICE_VERSION)


def _channel_for_engine(engine: str):
    return {"gemini": queue_manager.google_channel,
            "grok": queue_manager.grok_channel,
            "codex": queue_manager.chatgpt_channel}.get(engine.lower())


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
            "artifact": record.to_dict(),
            "download_path": artifacts.download_path(record.artifact_id),
        }
    )


@mcp.tool(name="image.capabilities")
def image_capabilities() -> types.CallToolResult:
    return _result({**queue_manager.capabilities(), "service_version": SERVICE_VERSION})


async def api_capabilities(request: Request) -> JSONResponse:
    from .dashboard import check_auth
    if not check_auth(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    return JSONResponse({**queue_manager.capabilities(), "service_version": SERVICE_VERSION})


@mcp.tool(name="image.generate")
async def image_generate(
    project_id: str,
    idempotency_key: str,
    prompt: str = "",
    engine: str = "codex",
    model: str = "",
    aspect_ratio: str = "16:9",
    resolution: str = "2k",
    quality: str | None = None,
    background: str = "auto",
    instructions: str | None = None,
    channel_account_id: str | None = None,
    budget_limit: str | None = None,
    negative_prompt: str = "",
    width: int | None = None,
    height: int | None = None,
    steps: int | None = None,
    cfg_scale: float | None = None,
    guidance: float | None = None,
    sampler: str | None = None,
    scheduler: str | None = None,
    shift: float | None = None,
    skip_single_layers: str | None = None,
    seed: int | None = None,
    layout: dict[str, Any] | None = None,
    source_artifact_id: str | None = None,
    reference_artifact_ids: list[str] | None = None,
    face_reference_artifact_id: str | None = None,
    source_identity_artifact_id: str | None = None,
    identity_strength: float | None = None,
    restore_face: bool | None = None,
    restoration_fidelity: float | None = None,
    target_face_index: int | None = None,
    reference_images: list[str] | None = None,
    mask: str | None = None,
    prefer_4k: bool | None = None,
) -> types.CallToolResult:
    """根据文本提示词生成图像。

    遵循 GPT Image 2.5 官方提示词标准 (Scene, Subject, Details, Constraints)。
    - engine: 默认 "codex" (OpenAI Responses API 驱动)，亦可选 "gemini", "grok", "moss-image"
    - model:
        codex 引擎推荐: "gpt-image-2.5-sunburst" (画质首选，微观细节与皮肤毛孔) 或 "gpt-image-2.5-flare" (极速响应)
        gemini 引擎 (Google 渠道 Nano Banana 系列): 默认 "gemini-nano-banana-2.1" (Nano Banana 2.1，Flash 级高保真与多轮一致性)，可选 "gemini-3-pro-image" (Nano Banana Pro，旗舰光学摄影感，兼容 "-preview")、"gemini-3.1-flash-image" (Nano Banana 2，极速现实锚定，兼容 "-preview")、"gemini-3.1-flash-lite-image" (Nano Banana 2 Lite，轻量高性价比) 或 "gemini-2.5-flash-image" (Nano Banana，经典兼容底座)
        moss-image 引擎: "moss-image-sft-int8-convrot" (自托管 INT8-ConvRot 文字排版与版式生成)
    - aspect_ratio: "16:9", "9:16", "1:1", "4:3", "3:4"
    - resolution: "2k" (默认), "4k" (对齐 3840x2160 / 2160x3840 官方尺寸约束)
    - quality: "auto", "low", "medium", "high", "xhigh", "max" (历史 "hd" 兼容映射为 "high")
    - background: "auto", "transparent" (纯净透明通道，输出 PNG), "opaque"
    - instructions: 传递给上游主控中介模型的系统指令 (默认注入逐字直传规范)
    """
    resolved_model = model or (
        {"codex": DEFAULT_IMAGE_MODEL, "gemini": DEFAULT_GEMINI_MODEL,
         "grok": queue_manager.grok_provider.model, "moss-image": DEFAULT_MOSS_MODEL}.get(engine.lower(), "")
    )
    params: dict[str, Any] = {
        "action": "generate",
        "prompt": prompt,
        "model": resolved_model,
        "aspect_ratio": aspect_ratio,
        "resolution": resolution,
        "quality": quality,
        "background": background,
    }
    if instructions:
        params["instructions"] = str(instructions).strip()
    if channel_account_id is not None:
        params["channel_account_id"] = channel_account_id
    effective_budget = budget_limit if budget_limit is not None else (
        os.environ.get("IMAGE_DEFAULT_BUDGET_LIMIT", "1") if _channel_for_engine(engine) is not None else None
    )
    if effective_budget is not None:
        params["budget_limit"] = effective_budget
    if negative_prompt:
        params["negative_prompt"] = negative_prompt
    if width is not None:
        params["width"] = width
    if height is not None:
        params["height"] = height
    if steps is not None:
        params["steps"] = steps
    if cfg_scale is not None:
        params["cfg_scale"] = cfg_scale
    if guidance is not None:
        params["guidance"] = guidance
    if sampler is not None:
        params["sampler"] = sampler
    if scheduler is not None:
        params["scheduler"] = scheduler
    if shift is not None:
        params["shift"] = shift
    if skip_single_layers is not None:
        params["skip_single_layers"] = skip_single_layers
    if seed is not None:
        params["seed"] = seed
    if layout is not None:
        params["layout"] = layout
    if source_artifact_id:
        params["source_artifact_id"] = source_artifact_id
    if face_reference_artifact_id:
        params["face_reference_artifact_id"] = face_reference_artifact_id
    if source_identity_artifact_id:
        params["source_identity_artifact_id"] = source_identity_artifact_id
    if identity_strength is not None:
        params["identity_strength"] = identity_strength
    if restore_face is not None:
        params["restore_face"] = restore_face
    if restoration_fidelity is not None:
        params["restoration_fidelity"] = restoration_fidelity
    if target_face_index is not None:
        params["target_face_index"] = target_face_index
    if reference_artifact_ids is not None:
        params["reference_artifact_ids"] = list(reference_artifact_ids)
    if reference_images is not None:
        params["reference_images"] = reference_images
        if reference_images:
            merged_refs = list(params.get("reference_artifact_ids") or [])
            for item in reference_images:
                if item and item not in merged_refs:
                    merged_refs.append(item)
            params["reference_artifact_ids"] = merged_refs
    if mask is not None:
        params["mask"] = mask
    if prefer_4k is not None:
        params["prefer_4k"] = prefer_4k
    params = queue_manager.prepare(engine, project_id, params)
    task = tasks.create(
        project_id=project_id,
        idempotency_key=idempotency_key,
        engine=engine,
        model=params.get("model") or resolved_model,
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


@mcp.tool(name="image.create")
async def image_create(
    prompt: str = "",
    project_id: str = "",
    idempotency_key: str = "",
    engine: str = "codex",
    model: str = "",
    aspect_ratio: str = "16:9",
    resolution: str = "2k",
    quality: str | None = None,
    background: str = "auto",
    instructions: str | None = None,
    channel_account_id: str | None = None,
    budget_limit: str | None = None,
    negative_prompt: str = "",
    width: int | None = None,
    height: int | None = None,
    steps: int | None = None,
    cfg_scale: float | None = None,
    guidance: float | None = None,
    sampler: str | None = None,
    scheduler: str | None = None,
    shift: float | None = None,
    skip_single_layers: str | None = None,
    seed: int | None = None,
    layout: dict[str, Any] | None = None,
    source_artifact_id: str | None = None,
    reference_artifact_ids: list[str] | None = None,
    face_reference_artifact_id: str | None = None,
    source_identity_artifact_id: str | None = None,
    identity_strength: float | None = None,
    restore_face: bool | None = None,
    restoration_fidelity: float | None = None,
    target_face_index: int | None = None,
    reference_images: list[str] | None = None,
    mask: str | None = None,
    prefer_4k: bool | None = None,
) -> types.CallToolResult:
    """根据文本提示词创建图像生成异步任务（遵循 GPT Image 2.5 官方提示词标准）。"""
    import uuid
    if engine == "grok" and not idempotency_key:
        raise ValueError("idempotency_key_required")
    actual_idem = idempotency_key or f"idem_{uuid.uuid4().hex[:12]}"
    return await image_generate(
        project_id=project_id,
        idempotency_key=actual_idem,
        prompt=prompt,
        engine=engine,
        model=model,
        aspect_ratio=aspect_ratio,
        resolution=resolution,
        quality=quality,
        background=background,
        instructions=instructions,
        channel_account_id=channel_account_id,
        budget_limit=budget_limit,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        steps=steps,
        cfg_scale=cfg_scale,
        guidance=guidance,
        sampler=sampler,
        scheduler=scheduler,
        shift=shift,
        skip_single_layers=skip_single_layers,
        seed=seed,
        layout=layout,
        source_artifact_id=source_artifact_id,
        reference_artifact_ids=reference_artifact_ids,
        face_reference_artifact_id=face_reference_artifact_id,
        source_identity_artifact_id=source_identity_artifact_id,
        identity_strength=identity_strength,
        restore_face=restore_face,
        restoration_fidelity=restoration_fidelity,
        target_face_index=target_face_index,
        reference_images=reference_images,
        mask=mask,
        prefer_4k=prefer_4k,
    )


@mcp.tool(name="image.edit")
async def image_edit(
    project_id: str,
    idempotency_key: str,
    source_artifact_id: str,
    prompt: str,
    reference_artifact_ids: list[str] | None = None,
    mask_artifact_id: str | None = None,
    engine: str = "codex",
    model: str = "",
    size: str = "",
    aspect_ratio: str | None = None,
    resolution: str | None = None,
    quality: str | None = None,
    background: str = "auto",
    channel_account_id: str | None = None,
    budget_limit: str | None = None,
) -> types.CallToolResult:
    """基于既有参考底图执行多模态指令编辑、骨相锁定换装或多图融合合成。

    遵循 GPT Image 2.5 官方编辑准则（明确指定“仅变更项”与“严格保留项”，为多参考图分配职责）。
    - source_artifact_id: 主参考底图 ID (如主体肖像底图)
    - reference_artifact_ids: 可选附加参考图 ID 列表 (如单品服装图、环境参考图)
    - mask_artifact_id: 可选局部修改遮罩 ID
    - prompt: 编辑指令，建议使用 "Change only X, preserve exact facial features, skin tone, and body pose"
    - model: codex 默认 "gpt-image-2.5-sunburst" (或 "gpt-image-2.5-flare")；gemini 默认 "gemini-nano-banana-2.1" (Nano Banana 2.1)，可选 "gemini-3-pro-image" (Nano Banana Pro)、"gemini-3.1-flash-image" (Nano Banana 2)、"gemini-3.1-flash-lite-image" (Nano Banana 2 Lite)、"gemini-2.5-flash-image" (Nano Banana)
    - background: "auto", "transparent" (扣除背景生成透明底), "opaque"
    - quality: "auto", "low", "medium", "high", "xhigh", "max"
    """
    resolved_model = model or (
        {"codex": DEFAULT_IMAGE_MODEL, "gemini": DEFAULT_GEMINI_MODEL,
         "grok": queue_manager.grok_provider.model, "moss-image": DEFAULT_MOSS_MODEL}.get(engine.lower(), "")
    )
    params = {
        "action": "edit",
        "source_artifact_id": source_artifact_id,
        "reference_artifact_ids": reference_artifact_ids or [],
        "mask_artifact_id": mask_artifact_id,
        "prompt": prompt,
        "model": resolved_model,
        "size": size,
        "resolution": resolution,
        "quality": quality,
        "background": background,
    }
    if aspect_ratio is not None:
        params["aspect_ratio"] = aspect_ratio
    if channel_account_id is not None:
        params["channel_account_id"] = channel_account_id
    effective_budget = budget_limit if budget_limit is not None else (
        os.environ.get("IMAGE_DEFAULT_BUDGET_LIMIT", "1") if _channel_for_engine(engine) is not None else None
    )
    if effective_budget is not None:
        params["budget_limit"] = effective_budget
    params = queue_manager.prepare(engine, project_id, params)
    task = tasks.create(
        project_id=project_id,
        idempotency_key=idempotency_key,
        engine=engine,
        model=params.get("model") or resolved_model,
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
    reference_artifact_ids: list[str] | None = None,
    engine: str = "codex",
    model: str = "",
    background: str = "auto",
    channel_account_id: str | None = None,
    budget_limit: str | None = None,
) -> types.CallToolResult:
    """基于遮罩（Mask）的精准局部重绘与无痕物体抹除，支持可选服装/主体参考图融合。"""
    resolved_model = model or (
        {"codex": DEFAULT_IMAGE_MODEL, "gemini": DEFAULT_GEMINI_MODEL,
         "grok": queue_manager.grok_provider.model}.get(engine.lower(), "")
    )
    params = {
        "action": "inpaint",
        "source_artifact_id": source_artifact_id,
        "mask_artifact_id": mask_artifact_id,
        "reference_artifact_ids": reference_artifact_ids or [],
        "prompt": prompt,
        "model": resolved_model,
        "background": background,
    }
    if channel_account_id is not None:
        params["channel_account_id"] = channel_account_id
    effective_budget = budget_limit if budget_limit is not None else (
        os.environ.get("IMAGE_DEFAULT_BUDGET_LIMIT", "1") if _channel_for_engine(engine) is not None else None
    )
    if effective_budget is not None:
        params["budget_limit"] = effective_budget
    params = queue_manager.prepare(engine, project_id, params)
    task = tasks.create(
        project_id=project_id,
        idempotency_key=idempotency_key,
        engine=engine,
        model=params.get("model") or resolved_model,
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


@mcp.tool(name="image.faceswap")
async def image_faceswap(
    target_artifact_id: str,
    source_identity_artifact_id: str,
    identity_strength: float = 0.95,
    restore_face: bool = True,
    restoration_fidelity: float = 0.85,
    project_id: str | None = None,
    target_face_index: int = 0,
) -> types.CallToolResult:
    """图片换脸与可选人脸增强工具。

    将源身份应用到目标图片的指定人脸，保留实际处理参数与模型信息。
    - target_artifact_id: 阶段一原始大片 Artifact ID
    - source_identity_artifact_id: 源身份图片 Artifact ID 或资产路径
    - identity_strength: 身份强度 [0.5,1]，默认 0.95
    - restore_face: 是否开启面部超分修复，默认 True
    - restoration_fidelity: 超分修复保真度权重，默认 0.85
    - project_id: 可选项目 ID（若未指定则自动从 target_artifact 继承）
    - target_face_index: 按人脸面积降序排列的目标索引，默认 0
    """
    res = await handle_faceswap(
        artifacts=artifacts,
        target_artifact_id=target_artifact_id,
        source_identity_artifact_id=source_identity_artifact_id,
        identity_strength=identity_strength,
        restore_face=restore_face,
        restoration_fidelity=restoration_fidelity,
        project_id=project_id,
        target_face_index=target_face_index,
    )
    return _result(res)


def _active_ratings() -> ContentRatingStore:
    global ratings
    if ratings.db_path != tasks.db_path:
        ratings = ContentRatingStore(tasks.db_path, domain="image")
    return ratings


def _resolve_rating_target(version_id: str, project_id: str | None = None) -> tuple[str, str | None]:
    """Resolve (project_id, task_id) for an artifact_id or task_id, or raise TaskNotFound."""
    vid = (version_id or "").strip()
    if not vid:
        raise TaskNotFound("NOT_FOUND: version_id is required")
    matched_task = tasks.find_by_artifact_or_task(vid)
    if matched_task is not None:
        if project_id and matched_task.project_id != project_id:
            raise TaskNotFound(f"NOT_FOUND: {vid}")
        return matched_task.project_id, matched_task.task_id
    try:
        art = artifacts.get(vid, project_id)
        return art.project_id, None
    except ArtifactNotFound as exc:
        raise TaskNotFound(f"NOT_FOUND: {vid}") from exc


def _is_task_nsfw(task_id: str, artifact_id: str | None) -> bool:
    return _active_ratings().is_nsfw(artifact_id, task_id=task_id)


@mcp.tool(name="image.rate_content")
def image_rate_content(
    version_id: str,
    rating: str,
    reason: str | None = None,
    reasons: list[str] | None = None,
    rubric_version: str = "rubric v1.0",
    evaluator_model: str | None = None,
    project_id: str | None = None,
) -> types.CallToolResult:
    """为图片产物或任务版本记录内容分级（general 或 nsfw，只增不减）。"""
    resolved_project_id, resolved_task_id = _resolve_rating_target(version_id, project_id)
    caller = CALLER.get() or {}
    actor_id = caller.get("user_id") or "skill:k09"
    store = _active_ratings()
    res = store.rate_content(
        version_id=version_id,
        project_id=resolved_project_id,
        rating=rating,
        reasons=reasons,
        reason=reason,
        rubric_version=rubric_version,
        evaluator_model=evaluator_model,
        rated_by=actor_id,
        task_id=resolved_task_id,
    )
    if resolved_task_id:
        matched_task = tasks.find_by_artifact_or_task(version_id)
        if matched_task and matched_task.artifact_id and matched_task.artifact_id != version_id:
            store.rate_content(
                version_id=matched_task.artifact_id,
                project_id=resolved_project_id,
                rating=rating,
                reasons=reasons,
                reason=reason,
                rubric_version=rubric_version,
                evaluator_model=evaluator_model,
                rated_by=actor_id,
                task_id=resolved_task_id,
            )
    return _result(res)


@mcp.tool(name="image.status")
def image_status(task_id: str) -> types.CallToolResult:
    task = tasks.get(task_id)
    if _is_task_nsfw(task.task_id, task.artifact_id):
        raise TaskNotFound(f"NOT_FOUND: {task_id}")
    return _result(
        {
            "task_id": task.task_id,
            "status": task.status,
            "created_at": task.created_at,
            "updated_at": task.updated_at,
            "duration_seconds": task.duration_seconds,
            "artifact_id": task.artifact_id,
            "error": task.error,
            "phase": task.phase,
            "needs_attention": task.phase in {"submission_unknown", "collect_retry"},
        }
    )


@mcp.tool(name="image.result")
def image_result(task_id: str, execution_proof_only: bool = False) -> types.CallToolResult:
    task = tasks.get(task_id)
    if _is_task_nsfw(task.task_id, task.artifact_id):
        raise TaskNotFound(f"NOT_FOUND: {task_id}")
    if execution_proof_only:
        from .execution_proof import proof
        authorize(authority_client, task.project_id, require_write=False)
        return _result({'execution_proof': proof(task, artifacts)})
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
        include_nsfw=False,
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
    return JSONResponse({"status": "ok", "service": "image-mcp-server", "version": SERVICE_VERSION})


async def artifact_content(request: Request) -> Response:
    try:
        artifact_id = request.path_params.get("artifact_id")
        include_nsfw = include_nsfw_from_headers(request.headers)
        store = _active_ratings()
        if not include_nsfw and artifact_id:
            matched_task = tasks.find_by_artifact_or_task(artifact_id)
            if store.is_nsfw(artifact_id, task_id=matched_task.task_id if matched_task else None):
                return JSONResponse({"error": "artifact_not_found"}, status_code=404)
        rec = artifacts.get(artifact_id)
        path = artifacts.content_path(rec)
        return FileResponse(path, media_type=rec.media_type, filename=rec.filename)
    except (ValueError, ArtifactNotFound):
        return JSONResponse({"error": "artifact_not_found"}, status_code=404)


async def internal_clear_rating(request: Request) -> JSONResponse:
    version_id = (request.path_params.get("version_id") or "").strip()
    try:
        body = await request.json() if await request.body() else {}
    except Exception:
        body = {}
    project_id = body.get("project_id") or request.headers.get("X-Project-Id") or None
    note = body.get("note") or body.get("reason")
    actor_id = request.headers.get("X-User-Id", "").strip() or body.get("actor_user_id") or body.get("actor_id") or "human"
    store = _active_ratings()
    try:
        resolved_project_id, resolved_task_id = _resolve_rating_target(version_id, project_id)
        rec = store.clear_rating(
            version_id=version_id,
            project_id=resolved_project_id,
            actor_user_id=actor_id,
            note=note,
            task_id=resolved_task_id,
        )
        matched_task = tasks.find_by_artifact_or_task(version_id)
        if matched_task:
            for cid in {matched_task.task_id, matched_task.artifact_id}:
                if cid and cid != version_id:
                    store.clear_rating(
                        version_id=cid,
                        project_id=resolved_project_id,
                        actor_user_id=actor_id,
                        note=note,
                        task_id=resolved_task_id,
                    )
        return JSONResponse(rec)
    except TaskNotFound:
        return JSONResponse({"error": "not_found"}, status_code=404)
    except ContentRatingError as exc:
        return JSONResponse({"error": exc.error_code, "message": exc.message}, status_code=exc.status_code)


async def internal_mark_rating(request: Request) -> JSONResponse:
    version_id = (request.path_params.get("version_id") or "").strip()
    try:
        body = await request.json() if await request.body() else {}
    except Exception:
        body = {}
    project_id = body.get("project_id") or request.headers.get("X-Project-Id") or None
    reason = body.get("reason") or body.get("note")
    actor_id = request.headers.get("X-User-Id", "").strip() or body.get("actor_user_id") or body.get("actor_id") or "human"
    store = _active_ratings()
    try:
        resolved_project_id, resolved_task_id = _resolve_rating_target(version_id, project_id)
        rec = store.mark_rating(
            version_id=version_id,
            project_id=resolved_project_id,
            actor_user_id=actor_id,
            reason=reason,
            task_id=resolved_task_id,
        )
        matched_task = tasks.find_by_artifact_or_task(version_id)
        if matched_task:
            for cid in {matched_task.task_id, matched_task.artifact_id}:
                if cid and cid != version_id:
                    store.mark_rating(
                        version_id=cid,
                        project_id=resolved_project_id,
                        actor_user_id=actor_id,
                        reason=reason,
                        task_id=resolved_task_id,
                    )
        return JSONResponse(rec)
    except TaskNotFound:
        return JSONResponse({"error": "not_found"}, status_code=404)
    except ContentRatingError as exc:
        return JSONResponse({"error": exc.error_code, "message": exc.message}, status_code=exc.status_code)


class MCPPathRewriteMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # 兼容直连 /image、/image/ 或 /，重写为 /mcp，使 streamable_http_app 能够无缝匹配
        path = request.url.path
        if path in ("/image", "/image/", "/", "/image/mcp", "/image/mcp/"):
            request.scope["path"] = "/mcp"
        return await call_next(request)


class BearerAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Trust identity headers only when Studio's internal bearer is verified.
        token = os.environ.get('IMAGE_MCP_BEARER_TOKEN', '').strip()
        valid = bool(token) and hmac.compare_digest(request.headers.get('authorization', ''), 'Bearer ' + token)
        caller = {'user_id': request.headers.get('X-User-Id', ''),
                  'organization_id': request.headers.get('X-Organization-Id', '')}
        context = CALLER.set(caller if valid else None)
        try:
            return await self._dispatch(request, call_next)
        finally:
            CALLER.reset(context)

    async def _dispatch(self, request: Request, call_next):
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
    ratings.ensure_ready()
    app.state.tasks = tasks
    app.state.artifacts = artifacts
    app.state.ratings = ratings
    app.state.queue_manager = queue_manager
    if channel_discovery:
        channel_discovery.start()
    await queue_manager.start()
    await registry.start()
    try:
        if getattr(mcp.session_manager, "_has_started", False):
            mcp.session_manager._has_started = False
        async with mcp.session_manager.run():
            yield
    finally:
        if channel_discovery:
            await channel_discovery.stop()
        await registry.stop()
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
        Route("/api/capabilities", api_capabilities, methods=["GET"]),
        Route("/api/tasks", api_list_tasks, methods=["GET"]),
        Route("/image/api/tasks", api_list_tasks, methods=["GET"]),
        Route("/api/tasks", api_create_task, methods=["POST"]),
        Route("/image/api/tasks", api_create_task, methods=["POST"]),
        Route("/api/tasks/stats", api_task_stats, methods=["GET"]),
        Route("/image/api/tasks/stats", api_task_stats, methods=["GET"]),
        Route("/api/tasks/{task_id}/title", api_update_task_title, methods=["POST"]),
        Route("/image/api/tasks/{task_id}/title", api_update_task_title, methods=["POST"]),
        Route("/api/tasks/{task_id}/delete", api_delete_task, methods=["POST"]),
        Route("/image/api/tasks/{task_id}/delete", api_delete_task, methods=["POST"]),
        Route("/api/tasks/{task_id}", api_delete_task, methods=["DELETE"]),
        Route("/image/api/tasks/{task_id}", api_delete_task, methods=["DELETE"]),
        Route("/api/tasks/{task_id}/content-rating", api_task_content_rating, methods=["POST"]),
        Route("/image/api/tasks/{task_id}/content-rating", api_task_content_rating, methods=["POST"]),
        Route("/api/artifacts/upload", api_upload_artifact, methods=["POST"]),
        Route("/image/api/artifacts/upload", api_upload_artifact, methods=["POST"]),
        Route("/artifacts/{artifact_id}/content", artifact_content, methods=["GET"]),
        Route("/image/artifacts/{artifact_id}/content", artifact_content, methods=["GET"]),
        Route("/internal/content-ratings/{version_id}/clear", internal_clear_rating, methods=["POST"]),
        Route("/image/internal/content-ratings/{version_id}/clear", internal_clear_rating, methods=["POST"]),
        Route("/internal/content-ratings/{version_id}/mark", internal_mark_rating, methods=["POST"]),
        Route("/image/internal/content-ratings/{version_id}/mark", internal_mark_rating, methods=["POST"]),
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
app.state.tasks = tasks
app.state.artifacts = artifacts
app.state.ratings = ratings
app.state.queue_manager = queue_manager
