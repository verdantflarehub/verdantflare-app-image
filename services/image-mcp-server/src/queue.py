from __future__ import annotations

import asyncio
import os
import time
import hashlib
import json
import threading
import base64
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Coroutine

from .artifacts import ArtifactStore
from .gpu_lease import GpuCoordinator, LeaseBusy, SchedulerLeaseClient
from .providers.codex import CodexProvider, DEFAULT_IMAGE_MODEL
from .providers.gemini import (
    ADVERTISED_GEMINI_MODELS,
    DEFAULT_GEMINI_MODEL,
    GEMINI_MODEL_ALIASES,
    GEMINI_MODEL_CATALOG,
    GeminiProvider,
    SUPPORTED_GEMINI_MODELS,
)
from .providers.moss import DEFAULT_MOSS_MODEL, MossProvider
from .providers.comfyui import ComfyUICancelResult, ComfyUIProviderError, ComfyUIWorkflowProvider
from .tasks import TaskRecord, TaskStore
from .providers.grok import GrokImageProvider, GrokError
from .providers.google_channel import ChannelUnavailable


@dataclass
class TaskJob:
    task_id: str
    project_id: str
    engine: str
    action: str  # "generate", "edit", "inpaint"
    params: dict[str, Any]


def _image_storage_type(data: bytes) -> tuple[str, str]:
    """Return an extension and media type matching the returned image bytes."""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png", "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return ".jpg", "image/jpeg"
    if data.startswith(b"RIFF") and b"WEBP" in data[:12]:
        return ".webp", "image/webp"
    # Keep test/fallback provider bytes usable while preserving the historical
    # PNG contract for providers that return an undecorated payload.
    return ".png", "image/png"


class _DynamicConcurrencyGate:
    """Thread-safe dynamic concurrency gate (1..20, default 4) synced from ChatGPT channel."""

    def __init__(self, initial_limit: int = 4) -> None:
        self._cond = threading.Condition()
        self._limit = max(1, min(20, int(initial_limit)))
        self._active = 0

    @property
    def limit(self) -> int:
        with self._cond:
            return self._limit

    def update_limit(self, new_limit: Any) -> None:
        if isinstance(new_limit, bool) or not isinstance(new_limit, int):
            return
        clamped = max(1, min(20, new_limit))
        with self._cond:
            if clamped != self._limit:
                self._limit = clamped
                self._cond.notify_all()

    @contextmanager
    def hold(self):
        with self._cond:
            while self._active >= self._limit:
                self._cond.wait()
            self._active += 1
        try:
            yield
        finally:
            with self._cond:
                self._active -= 1
                self._cond.notify_all()


class TaskQueueManager:
    def __init__(
        self,
        tasks: TaskStore,
        artifacts: ArtifactStore,
        codex_provider: CodexProvider,
        gemini_provider: GeminiProvider,
        google_channel=None,
        google_authorize=None,
        grok_channel=None,
        chatgpt_channel=None,
        channel_directory=None,
        moss_provider: MossProvider | None = None,
        moss_gpu: GpuCoordinator | None = None,
        scheduler: SchedulerLeaseClient | None = None,
        comfyui_provider: ComfyUIWorkflowProvider | None = None,
    ) -> None:
        self.tasks = tasks
        self.artifacts = artifacts
        self.codex_provider = codex_provider
        self.gemini_provider = gemini_provider
        self.google_channel = google_channel
        self.chatgpt_channel = chatgpt_channel
        self.grok_channel = grok_channel
        self.google_authorize = google_authorize
        self.channel_directory = channel_directory
        self.google_lock = threading.Lock()
        self.chatgpt_gate = _DynamicConcurrencyGate(4)
        self._task_locks_guard = threading.Lock()
        self._task_locks: dict[str, threading.Lock] = {}
        self.grok_provider = GrokImageProvider()
        self.grok_sem = asyncio.Semaphore(max(1, int(os.getenv("XAI_IMAGE_MAX_CONCURRENCY", "1"))))
        self.grok_lock = threading.Lock()
        self.moss_provider = moss_provider or MossProvider()
        self.moss_gpu = moss_gpu or GpuCoordinator.from_environment()
        self.scheduler = scheduler or SchedulerLeaseClient.from_environment()
        if getattr(self.tasks, "db_path", None):
            from pathlib import Path
            self.scheduler.configure_journal(Path(self.tasks.db_path).parent / "gpu_pending_releases.json")
        self.moss_sem = asyncio.Semaphore(max(1, int(os.getenv("MOSS_IMAGE_MAX_CONCURRENCY", "1"))))
        self.comfyui_provider = comfyui_provider or ComfyUIWorkflowProvider()
        self.comfyui_sem = asyncio.Semaphore(max(1, int(os.getenv("COMFYUI_IMAGE_MAX_CONCURRENCY", "4"))))

        codex_limit = max(1, min(20, int(os.environ.get("CODEX_MAX_CONCURRENCY", "20"))))
        gemini_limit = int(os.environ.get("GEMINI_MAX_CONCURRENCY", "5"))
        worker_count = max(1, int(os.environ.get("IMAGE_QUEUE_WORKERS", "20")))

        self.codex_sem = asyncio.Semaphore(codex_limit)
        self.gemini_sem = asyncio.Semaphore(gemini_limit)
        self.worker_count = worker_count

        self._queue: asyncio.Queue[TaskJob] = asyncio.Queue()
        self._worker_tasks: list[asyncio.Task] = []
        self._running = False

    async def start(self) -> None:
        if self._running:
            return
        self._running = True

        # 1. 恢复异常退出的运行中任务
        recovered = self.tasks.recover_hanging_tasks()
        self.recover_grok()
        self.recover_comfyui()
        if recovered > 0:
            print(f"⚠️ [TaskQueue] 保留 {recovered} 个原调用结果未知的任务；未重新提交")

        # 2. 启动 Worker 协程池（移除 15s 后台定时轮询 _collect_grok / _collect_channels，改为手动统一清理）
        for i in range(self.worker_count):
            task = asyncio.create_task(self._worker_loop(i))
            self._worker_tasks.append(task)
        self._worker_tasks.append(asyncio.create_task(self._idle_moss()))

        # 3. 恢复历史未消费的 queued 任务
        pending = self.tasks.get_queued_tasks()
        for rec in pending:
            action = rec.request_params.get("action", "generate")
            await self.enqueue(rec.task_id, rec.project_id, rec.engine, action, rec.request_params)

        print(f"[TaskQueue] 队列调度器启动完毕 (Workers: {self.worker_count})")

    async def stop(self) -> None:
        self._running = False
        for task in self._worker_tasks:
            task.cancel()
        if self._worker_tasks:
            await asyncio.gather(*self._worker_tasks, return_exceptions=True)
        self._worker_tasks.clear()
        print("[TaskQueue] 队列调度器已停止")

    async def enqueue(
        self,
        task_id: str,
        project_id: str,
        engine: str,
        action: str,
        params: dict[str, Any],
    ) -> None:
        job = TaskJob(
            task_id=task_id,
            project_id=project_id,
            engine=engine,
            action=action,
            params=params,
        )
        await self._queue.put(job)

    def prepare(self, engine, project_id, params):
        if engine not in {"codex", "gemini", "grok", "moss-image", "comfyui"}:
            raise ValueError("unsupported_engine")
        params = dict(params)
        # Internal route/authority cannot be supplied in generation parameters.
        for internal in ('channel_runtime', 'channel_authority', 'comfyui_authority', 'comfyui_workflow'):
            if internal in params:
                raise ValueError('reserved_channel_parameter')
        if engine == 'moss-image':
            normalized = self.moss_provider.normalize(params)
            if normalized.get("moss_mode") == "edit":
                if self.google_authorize is None:
                    raise ValueError("project_authorization_not_configured")
                authority = self.google_authorize(project_id, None)
                for rid in [normalized["source_artifact_id"], *normalized["reference_artifact_ids"]]:
                    record = self.artifacts.get(rid, project_id)
                    from PIL import Image, ImageOps
                    with Image.open(self.artifacts.content_path(record)) as image:
                        image = ImageOps.exif_transpose(image)
                        image.load()
                        if ("A" in image.getbands() or "transparency" in image.info) and image.convert("RGBA").getchannel("A").getextrema()[0] < 255:
                            raise ValueError("INVALID_REFERENCE_IMAGE: composite transparent images before editing")
                        if rid == normalized["source_artifact_id"] and not any(
                            params.get(key) for key in ("width", "height", "size", "aspect_ratio", "resolution")
                        ):
                            params.update(width=image.width, height=image.height)
                normalized = self.moss_provider.normalize(params)
                normalized["channel_authority"] = authority
            return normalized
        if engine == 'comfyui':
            return self.comfyui_provider.normalize(
                params,
                project_id,
                self.artifacts,
                authorize_fn=self.google_authorize,
            )
        if engine == 'grok':
            params = self.grok_provider.normalize(params, project_id, self.artifacts, channel=self.grok_channel is not None)
        if engine == 'gemini':
            if params.get('action') == 'inpaint' or params.get('mask_artifact_id') is not None or params.get('mask') is not None:
                raise ValueError('unsupported_parameter')
            if params.get('model') not in (None, '', *SUPPORTED_GEMINI_MODELS):
                raise ValueError('capability_not_supported')
            if params.get('model') in GEMINI_MODEL_ALIASES:
                params['model'] = GEMINI_MODEL_ALIASES[params['model']]
        channel = {'gemini': self.google_channel, 'grok': self.grok_channel, 'codex': self.chatgpt_channel}.get(engine)
        if self.channel_directory is not None and channel is None:
            raise ValueError('channel_not_configured')
        if channel is not None:
            has_refs = bool(
                params.get('source_artifact_id')
                or params.get('reference_artifact_ids')
                or params.get('reference_images')
                or params.get('mask_artifact_id')
            )
            discovery_action = 'edit' if (engine == 'codex' and has_refs) else params.get('action')
            self._check_channel_discovery(engine, discovery_action, channel)
            if self.google_authorize is None:
                raise ValueError('project_authorization_not_configured')
            import uuid
            from decimal import Decimal, InvalidOperation
            if params.get('channel_account_id') is not None:
                params['channel_account_id'] = str(uuid.UUID(params['channel_account_id']))
            ceiling = params.get('budget_limit')
            try:
                if not isinstance(ceiling, str) or len(ceiling) > 32:
                    raise ValueError('invalid_budget_limit')
                amount = Decimal(ceiling)
                if not amount.is_finite() or amount <= 0 or amount > Decimal('1000000'):
                    raise ValueError('invalid_budget_limit')
            except InvalidOperation:
                raise ValueError('invalid_budget_limit') from None
            params['channel_authority'] = self.google_authorize(project_id, None)
            params['channel_runtime'] = {'gemini': 'google-channel-v1', 'grok': 'grok-channel-v1', 'codex': 'chatgpt-channel-v1'}[engine]
        elif params.get('channel_account_id') is not None or params.get('budget_limit') is not None:
            raise ValueError('channel_parameters_not_supported')
        if engine == "grok":
            return params
        params["quality"] = params.get("quality") or ("high" if params["action"] == "generate" else "auto")
        return params

    def _check_channel_discovery(self, engine, action, channel):
        if self.channel_directory is None:
            return
        name = {'gemini': 'google', 'codex': 'chatgpt', 'grok': 'grok'}[engine]
        capability = 'image.generate' if action == 'generate' else 'image.edit'
        origin = self.channel_directory.resolve(name, capability)
        if not origin or origin != channel.endpoint:
            raise ValueError('channel_discovery_unavailable')

    def capabilities(self):
        grok = self.grok_provider.capability()
        if self.grok_channel is not None:
            grok.update(enabled=True, configured=True, adapter_version='grok-channel-v1')
        engines = [
            {"engine": "codex", "model": DEFAULT_IMAGE_MODEL, "enabled": True},
            {
                "engine": "gemini",
                "model": DEFAULT_GEMINI_MODEL,
                "enabled": True,
                "supported_models": list(ADVERTISED_GEMINI_MODELS),
                "model_catalog": [dict(item) for item in GEMINI_MODEL_CATALOG],
            },
            grok]
        for engine in engines:
            name, channel = {
                'codex': ('chatgpt', self.chatgpt_channel),
                'gemini': ('google', self.google_channel),
                'grok': ('grok', self.grok_channel),
            }[engine['engine']]
            if self.channel_directory is not None and channel is None:
                engine['enabled'] = False
            if channel is not None:
                engine.update(channel=name, adapter_version=name + '-channel-v1',
                    account_selection='explicit_or_configured_default',
                    budget_limit_required=True, budget_currency='USD',
                    budget_limit_kind='reservation_ceiling',
                    project_write_authorization_required=True)
                if engine['engine'] == 'gemini' and callable(getattr(channel, 'capabilities', None)):
                    try:
                        remote = channel.capabilities()
                        if (isinstance(remote, dict)
                                and remote.get('schema_version') == 1
                                and remote.get('channel_id') == 'google'
                                and isinstance(remote.get('capabilities'), list)):
                            base_map = {item['model_id']: dict(item) for item in GEMINI_MODEL_CATALOG}
                            merged_by_model: dict[str, dict[str, Any]] = {}
                            seen_models = []
                            for cap in remote['capabilities']:
                                if not isinstance(cap, dict) or cap.get('type') != 'image':
                                    continue
                                mid = cap.get('model_id')
                                if mid not in SUPPORTED_GEMINI_MODELS:
                                    continue
                                if mid not in merged_by_model:
                                    seen_models.append(mid)
                                    base_entry = dict(base_map.get(mid, {
                                        'model_id': mid,
                                        'display_name': cap.get('display_name') or cap.get('card_title') or mid,
                                        'mark': str(mid)[:3].upper(),
                                        'biz': '图像生成',
                                    }))
                                    base_entry['operations'] = []
                                    merged_by_model[mid] = base_entry
                                entry = merged_by_model[mid]
                                op = cap.get('operation')
                                if isinstance(op, str) and op and op not in entry['operations']:
                                    entry['operations'].append(op)
                                if isinstance(cap.get('display_name'), str) and cap['display_name'].strip():
                                    entry['display_name'] = cap['display_name'].strip()
                                for opt_key in ('card_group', 'card_title', 'official_title', 'model_logo', 'model_logo_mime', 'pricing_line', 'release_meta', 'output_summary', 'execution_state'):
                                    if cap.get(opt_key) is not None:
                                        entry[opt_key] = cap[opt_key]
                                cap_input = cap.get('input_summary')
                                if isinstance(cap_input, str) and len(cap_input) > len(str(entry.get('input_summary') or '')):
                                    entry['input_summary'] = cap_input
                            if merged_by_model:
                                engine['model_catalog'] = [merged_by_model[m] for m in seen_models]
                                engine['supported_models'] = seen_models
                    except Exception:
                        pass
                if self.channel_directory is not None:
                    engine['enabled'] = self.channel_directory.resolve(name, 'image.generate') == channel.endpoint
        engines.append(self.moss_provider.capability())
        engines.append(self.comfyui_provider.capability())
        return {
            "default_engine": "codex",
            "engines": engines,
            "content_rating": {
                "supported": True,
                "tool": "image.rate_content",
                "rubric_version": "rubric v1.0",
                "add_only": True,
            },
        }

    def _get_semaphore(self, engine: str) -> asyncio.Semaphore:
        eng = engine.lower()
        if eng == "grok":
            return self.grok_sem
        if eng == "moss-image":
            return self.moss_sem
        if eng == "comfyui":
            return self.comfyui_sem
        if eng not in {"codex", "gemini"}:
            raise ValueError("unsupported_engine")
        return self.gemini_sem if eng == "gemini" else self.codex_sem

    async def _worker_loop(self, worker_id: int) -> None:
        while self._running:
            try:
                job = await self._queue.get()
            except asyncio.CancelledError:
                break

            sem = self._get_semaphore(job.engine)
            # 等待对应 AI 通道的并发槽位
            async with sem:
                start_time = time.time()
                if not self.tasks.claim_task(job.task_id):
                    self._queue.task_done()
                    continue
                loop = asyncio.get_running_loop()

                try:
                    if job.params.get('channel_runtime') in {'google-channel-v1', 'grok-channel-v1', 'chatgpt-channel-v1'}:
                        await loop.run_in_executor(None, self._execute_channel, job)
                        continue
                    elif job.engine == "grok":
                        await loop.run_in_executor(None, self._execute_grok, job)
                    elif job.engine == "moss-image":
                        await loop.run_in_executor(None, self._execute_moss, job)
                    elif job.engine == "comfyui":
                        await loop.run_in_executor(None, self._execute_comfyui, job)
                    elif job.action == "generate":
                        await loop.run_in_executor(None, self._execute_generate, job)
                    elif job.action in ("edit", "inpaint"):
                        await loop.run_in_executor(None, self._execute_edit, job)
                    else:
                        raise ValueError(f"未知的任务动作: {job.action}")

                    duration = round(time.time() - start_time, 2)
                    self.tasks.update_status(
                        job.task_id,
                        status="completed",
                        duration_seconds=duration,
                    )
                    print(f"[TaskQueue] 任务成功: {job.task_id} (action: {job.action}, engine: {job.engine}) 耗时: {duration}s", flush=True)
                except Exception as exc:
                    duration = round(time.time() - start_time, 2)
                    if job.params.get('channel_runtime') in {'google-channel-v1', 'grok-channel-v1', 'chatgpt-channel-v1'}:
                        record = self.tasks.get(job.task_id)
                        # This worker owns the claimed task. The submit fence is
                        # persisted before any POST, so a missing phase proves
                        # this attempt failed before sending. Never apply this
                        # inference to recovered tasks after a process restart.
                        if record.phase is None:
                            self.tasks.update_status(job.task_id, 'failed',
                                error='channel_preparation_failed', phase='failed',
                                duration_seconds=duration)
                        elif record.status == 'running':
                            self.tasks.update_status(job.task_id, 'running',
                                error='channel_recovery_pending',
                                phase='collecting' if record.phase == 'collecting' else 'submission_unknown')
                        continue
                    if job.engine == "grok":
                        code = exc.code if isinstance(exc, GrokError) else "collection_unavailable"
                        uncertain = not isinstance(exc, GrokError) or exc.uncertain
                        journal = self._grok_journal(job.task_id)
                        phase = "collect_retry" if journal.with_suffix(".json").exists() else "submission_unknown"
                        self.tasks.update_status(job.task_id, "running" if uncertain else "failed",
                                                 error=code, phase=phase if uncertain else "failed")
                        continue
                    if job.engine == "comfyui":
                        code = exc.code if isinstance(exc, ComfyUIProviderError) else "comfyui_execution_failed"
                        uncertain = isinstance(exc, ComfyUIProviderError) and exc.uncertain
                        journal = self._comfyui_journal(job.task_id)
                        cancel_marker = journal.with_suffix(".cancel")
                        cur_rec = self.tasks.get(job.task_id)
                        cancel_requested = cancel_marker.exists() or cur_rec.error == "cancel_requested" or cur_rec.phase == "canceling"
                        if not uncertain and cancel_requested and code in {"canceled_by_caller", "comfyui_execution_cancelled"}:
                            self.tasks.update_status(
                                job.task_id,
                                "canceled",
                                error="canceled_by_caller",
                                phase="canceled",
                                duration_seconds=duration,
                            )
                            continue
                        if uncertain and cancel_requested:
                            self.tasks.update_status(
                                job.task_id,
                                "running",
                                error="cancel_requested",
                                phase="canceling",
                                duration_seconds=None,
                            )
                            continue
                        phase = "collect_retry" if journal.exists() else "submission_unknown"
                        self.tasks.update_status(
                            job.task_id,
                            "running" if uncertain else "failed",
                            error=code,
                            phase=phase if uncertain else "failed",
                            duration_seconds=None if uncertain else duration,
                        )
                        continue
                    err_msg = str(exc)
                    print(f"[TaskQueue] 任务失败: {job.task_id} (action: {job.action}, engine: {job.engine}, model: {job.params.get('model')}) 耗时: {duration}s, 错误详情: {err_msg}", flush=True)
                    self.tasks.update_status(
                        job.task_id,
                        status="failed",
                        error=err_msg,
                        duration_seconds=duration,
                    )
                finally:
                    self._queue.task_done()

    @staticmethod
    def _format_channel_failure(result: dict[str, Any]) -> str:
        reason = str(result.get('reason_code') or '').strip()
        raw_err = result.get('error')
        detail = ''
        if isinstance(raw_err, dict):
            detail = str(raw_err.get('message') or raw_err.get('detail') or raw_err.get('code') or '').strip()
        elif isinstance(raw_err, str):
            detail = raw_err.strip()
        if reason and detail and detail != reason:
            return f"channel_call_failed: {reason}: {detail}"
        if reason:
            return f"channel_call_failed: {reason}"
        if detail:
            return f"channel_call_failed: {detail}"
        return 'channel_call_failed'

    def _execute_google(self, job, recovery=False):
        return self._execute_channel(job, recovery)

    def _task_lock(self, task_id: str) -> threading.Lock:
        with self._task_locks_guard:
            lock = self._task_locks.get(task_id)
            if lock is None:
                lock = threading.Lock()
                self._task_locks[task_id] = lock
            return lock

    def _execute_channel(self, job, recovery=False):
        # The worker and collector serialize collection per task; ChatGPT channel
        # tasks run concurrently up to the channel-configured concurrent_limit (1..20, default 4).
        runtime = job.params.get('channel_runtime')
        is_chatgpt = runtime == 'chatgpt-channel-v1'
        outer_ctx = self.chatgpt_gate.hold() if is_chatgpt else self.google_lock
        with outer_ctx, self._task_lock(job.task_id):
            record = self.tasks.get(job.task_id)
            if record.status != 'running':
                return
            is_grok = runtime == 'grok-channel-v1'
            channel_name, channel = {
                'google-channel-v1': ('google', self.google_channel),
                'grok-channel-v1': ('grok', self.grok_channel),
                'chatgpt-channel-v1': ('chatgpt', self.chatgpt_channel),
            }.get(runtime, ('unknown', None))
            if channel is None or self.google_authorize is None:
                raise ChannelUnavailable('channel_not_configured')
            actor = self.google_authorize(job.project_id, job.params.get('channel_authority'))
            if actor.get('organization_id') != job.params.get('channel_authority', {}).get('organization_id'):
                raise ChannelUnavailable('channel_authority_mismatch')
            organization = actor['organization_id']
            if recovery or record.phase is not None:
                result = channel.recover(organization, job.task_id, job.task_id)
            else:
                if runtime == 'google-channel-v1' and (
                    job.action == 'inpaint' or job.params.get('mask_artifact_id') or job.params.get('mask')
                ):
                    raise ValueError('unsupported_parameter')
                images = []
                ids = [job.params.get('source_artifact_id')]
                ids.extend(job.params.get('reference_artifact_ids') or [])
                ids.extend(job.params.get('reference_images') or [])
                for artifact_id in dict.fromkeys(item for item in ids if item):
                    artifact = self.artifacts.get(artifact_id, job.project_id)
                    data = self.artifacts.content_path(artifact).read_bytes()
                    if hashlib.sha256(data).hexdigest() != artifact.sha256:
                        raise ValueError('source_integrity_failed')
                    if is_grok and job.params.get('source_sha256') != artifact.sha256:
                        raise ValueError('source_integrity_failed')
                    images.append({'mime_type': artifact.media_type,
                                   'data': base64.b64encode(data).decode()})
                payload = {'prompt': job.params['prompt'], 'images': images}
                effective_action = 'edit' if (job.action in ('edit', 'inpaint') or images) else job.action
                if runtime == 'google-channel-v1' and job.params.get('aspect_ratio'):
                    payload['aspect_ratio'] = job.params['aspect_ratio']
                if runtime == 'google-channel-v1':
                    payload['resolution'] = (job.params.get('resolution') or '2k').lower()
                if is_grok:
                    effective_action = job.action
                    payload = {key: job.params[key] for key in ('prompt', 'action', 'resolution')}
                    if job.action == 'edit':
                        payload['image'] = images[0]
                    else:
                        payload['aspect_ratio'] = job.params['aspect_ratio']
                if is_chatgpt:
                    p = job.params
                    res_lower = (p.get('resolution') or '2k').lower()
                    sq_size = '1024x1024' if res_lower == '1k' else '2048x2048'
                    size = {'16:9': '2048x1152', '9:16': '1152x2048', '1:1': sq_size,
                            '4:3': '1792x1344', '3:4': '1344x1792',
                            '3:2': '1536x1024', '2:3': '1024x1536'}.get(p.get('aspect_ratio'), p.get('size') or '2048x1152')
                    effective_action = 'edit' if (job.action in ('edit', 'inpaint') or images or p.get('mask_artifact_id')) else job.action
                    payload = {'prompt': p['prompt'], 'action': effective_action, 'images': images,
                        'size': size, 'prefer_4k': (p.get('resolution') or '2k').lower() == '4k',
                        'quality': p.get('quality') or ('high' if effective_action == 'generate' else 'auto'),
                        'background': p.get('background') or 'auto'}
                    if p.get('instructions') is not None:
                        payload['instructions'] = p['instructions']
                    if p.get('mask_artifact_id'):
                        mask = self.artifacts.get(p['mask_artifact_id'], job.project_id)
                        data = self.artifacts.content_path(mask).read_bytes()
                        if hashlib.sha256(data).hexdigest() != mask.sha256:
                            raise ValueError('source_integrity_failed')
                        payload['mask'] = {'mime_type': mask.media_type, 'data': base64.b64encode(data).decode()}
                self._check_channel_discovery(job.engine, effective_action, channel)
                self.tasks.update_status(job.task_id, 'running', phase='submitting')
                result = channel.submit(organization, job.task_id, job.task_id,
                    job.params.get('model') or (DEFAULT_IMAGE_MODEL if is_chatgpt else DEFAULT_GEMINI_MODEL),
                    payload,
                    job.params.get('channel_account_id'), job.params.get('budget_limit'))
            if is_chatgpt and isinstance(result, dict) and 'concurrent_limit' in result:
                self.chatgpt_gate.update_limit(result.get('concurrent_limit'))
            if result['accepted'] is False or result.get('state') == 'failed':
                err = self._format_channel_failure(result)
                self.tasks.update_status(job.task_id, 'failed',
                    error=err, phase='failed')
            elif result.get('state') == 'completed' and result['accepted'] is True:
                self.tasks.update_status(job.task_id, 'running', phase='collecting')
                data, media_type = channel.result(organization, job.task_id, result['call_id'])
                extension = {'image/png': '.png', 'image/jpeg': '.jpg', 'image/webp': '.webp'}[media_type]
                artifact = self.artifacts.create_from_bytes(project_id=job.project_id,
                    artifact_id='art-' + hashlib.sha256(job.task_id.encode()).hexdigest()[:16],
                    filename=job.task_id + extension, data=data, media_type=media_type,
                    metadata={'engine': job.engine, 'channel': channel_name,
                              'channel_call_id': result['call_id'], 'task_id': job.task_id})
                self.tasks.update_status(job.task_id, 'completed', artifact_id=artifact.artifact_id,
                                         error='', phase='completed')
            else:
                self.tasks.update_status(job.task_id, 'running',
                    error='submission_unknown' if result['accepted'] is None else '',
                    phase='submission_unknown' if result['accepted'] is None else 'channel_running')

    async def _collect_channels(self):
        while self._running:
            await asyncio.sleep(15)
            records, _ = self.tasks.list_tasks(status='running', limit=10000, include_nsfw=True)
            for record in records:
                if record.request_params.get('channel_runtime') not in {'google-channel-v1', 'grok-channel-v1', 'chatgpt-channel-v1'}:
                    continue
                try:
                    await asyncio.to_thread(self._execute_channel, TaskJob(record.task_id,
                        record.project_id, record.engine, record.request_params['action'],
                        record.request_params), True)
                except Exception:
                    # Read/collection failure never re-enqueues a generation POST.
                    pass

    def _recover_channel_unknown(self, rec: TaskRecord) -> None:
        runtime = rec.request_params.get("channel_runtime")
        channel_name, channel = {
            "google-channel-v1": ("google", self.google_channel),
            "grok-channel-v1": ("grok", self.grok_channel),
            "chatgpt-channel-v1": ("chatgpt", self.chatgpt_channel),
        }.get(runtime, ("unknown", None))
        if channel is None:
            return
        org = (rec.request_params.get("channel_authority") or {}).get("organization_id")
        if not org and self.google_authorize is not None:
            actor = self.google_authorize(rec.project_id, rec.request_params.get("channel_authority"))
            org = actor.get("organization_id")
        if not org:
            return
        result = channel.recover(org, rec.task_id, rec.task_id)
        if result.get("accepted") is False or result.get("state") in ("failed", "rejected"):
            err = self._format_channel_failure(result)
            self.tasks.update_status(rec.task_id, "failed", error=err, phase="failed")
            return
        if result.get("state") == "completed" and result.get("accepted") is True:
            self.tasks.update_status(rec.task_id, "running", phase="collecting")
            data, media_type = channel.result(org, rec.task_id, result["call_id"])
            extension = {"image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp"}[media_type]
            artifact = self.artifacts.create_from_bytes(
                project_id=rec.project_id,
                artifact_id="art-" + hashlib.sha256(rec.task_id.encode()).hexdigest()[:16],
                filename=rec.task_id + extension,
                data=data,
                media_type=media_type,
                metadata={
                    "engine": rec.engine,
                    "channel": channel_name,
                    "channel_call_id": result["call_id"],
                    "task_id": rec.task_id,
                },
            )
            self.tasks.update_status(
                rec.task_id,
                "completed",
                artifact_id=artifact.artifact_id,
                error="",
                phase="completed",
            )

    def cleanup_tasks(self, *, clean_unknown: bool = True, clean_failed: bool = False) -> dict[str, Any]:
        recovered_ids: list[str] = []
        cleaned_unknown_ids: list[str] = []
        cleaned_failed_ids: list[str] = []

        if clean_unknown:
            running_records, _ = self.tasks.list_tasks(status="running", limit=10000, include_nsfw=True)
            unknown_records = [
                rec for rec in running_records
                if rec.phase in ("submission_unknown", "collect_retry", "canceling")
                or rec.error in ("submission_unknown", "cancel_requested")
            ]
            still_unknown: list[TaskRecord] = []
            for rec in unknown_records:
                runtime = rec.request_params.get("channel_runtime")
                if runtime in {"google-channel-v1", "grok-channel-v1", "chatgpt-channel-v1"}:
                    try:
                        self._recover_channel_unknown(rec)
                    except Exception:
                        pass
                elif rec.engine == "grok" and self._grok_journal(rec.task_id).with_suffix(".json").exists():
                    try:
                        self._execute_grok(
                            TaskJob(
                                rec.task_id,
                                rec.project_id,
                                rec.engine,
                                rec.request_params.get("action", "generate"),
                                rec.request_params,
                            )
                        )
                    except Exception:
                        pass
                elif rec.engine == "comfyui":
                    cj = self._comfyui_journal(rec.task_id)
                    if rec.phase != "canceling" and rec.error != "cancel_requested" and not cj.with_suffix(".cancel").exists():
                        try:
                            self._execute_comfyui(
                                TaskJob(
                                    rec.task_id,
                                    rec.project_id,
                                    rec.engine,
                                    rec.request_params.get("action", "generate"),
                                    rec.request_params,
                                )
                            )
                            refreshed_c = self.tasks.get(rec.task_id)
                            if refreshed_c.artifact_id and refreshed_c.status == "running":
                                self.tasks.update_status(rec.task_id, "completed", error="", phase="completed")
                        except Exception:
                            pass

                refreshed = self.tasks.get(rec.task_id)
                if refreshed.status == "completed":
                    recovered_ids.append(rec.task_id)
                else:
                    still_unknown.append(refreshed)

            by_channel_org: dict[tuple[str, str], list[str]] = {}
            for rec in still_unknown:
                runtime = rec.request_params.get("channel_runtime")
                if runtime in {"google-channel-v1", "grok-channel-v1", "chatgpt-channel-v1"}:
                    org = (rec.request_params.get("channel_authority") or {}).get("organization_id")
                    if org:
                        by_channel_org.setdefault((runtime, org), []).append(rec.task_id)

            for (runtime, org), op_ids in by_channel_org.items():
                channel = {
                    "google-channel-v1": self.google_channel,
                    "grok-channel-v1": self.grok_channel,
                    "chatgpt-channel-v1": self.chatgpt_channel,
                }.get(runtime)
                if channel is not None and hasattr(channel, "cleanup_unknown"):
                    try:
                        channel.cleanup_unknown(org, op_ids)
                    except Exception:
                        pass

            for rec in still_unknown:
                if rec.engine == "comfyui":
                    outcome = ComfyUICancelResult(confirmed_terminal=False, reason="cancel_failed")
                    try:
                        outcome = self.cancel_comfyui_task(rec)
                    except Exception:
                        outcome = ComfyUICancelResult(confirmed_terminal=False, reason="cancel_failed")
                    if not outcome.confirmed_terminal:
                        try:
                            self.tasks.update_status(
                                rec.task_id,
                                "running",
                                error="cancel_requested",
                                phase="canceling",
                            )
                        except Exception:
                            pass
                        continue
                    try:
                        cj = self._comfyui_journal(rec.task_id)
                        for p in (cj, cj.with_suffix(".tmp"), cj.with_suffix(".cancel")):
                            if p.exists():
                                p.unlink(missing_ok=True)
                    except Exception:
                        pass
                try:
                    journal = self._grok_journal(rec.task_id)
                    for p in (journal, journal.with_suffix(".json"), journal.with_suffix(".part"), journal.with_suffix(".tmp")):
                        if p.exists():
                            p.unlink(missing_ok=True)
                except Exception:
                    pass
                try:
                    self.tasks.delete_task(rec.task_id)
                    cleaned_unknown_ids.append(rec.task_id)
                except Exception:
                    pass

        if clean_failed:
            failed_records, _ = self.tasks.list_tasks(status="failed", limit=10000, include_nsfw=True)
            for rec in failed_records:
                try:
                    self.tasks.delete_task(rec.task_id)
                    cleaned_failed_ids.append(rec.task_id)
                except Exception:
                    pass

        return {
            "ok": True,
            "recovered_count": len(recovered_ids),
            "cleaned_unknown_count": len(cleaned_unknown_ids),
            "cleaned_failed_count": len(cleaned_failed_ids),
            "recovered_task_ids": recovered_ids,
            "cleaned_unknown_task_ids": cleaned_unknown_ids,
            "cleaned_failed_task_ids": cleaned_failed_ids,
        }

    def _grok_journal(self, task_id):
        root = self.tasks.db_path.parent / ".grok-results"
        root.mkdir(parents=True, exist_ok=True)
        return root / task_id

    def _execute_grok(self, job):
        with self.grok_lock:
            journal = self._grok_journal(job.task_id)
            manifest = journal.with_suffix(".json")
            if manifest.exists():
                info = json.loads(manifest.read_text(encoding="utf-8"))
                data = journal.read_bytes()
                if hashlib.sha256(data).hexdigest() != info["sha256"]:
                    raise GrokError("result_integrity_failed")
            else:
                data, info = self.grok_provider.generate(job.params, job.project_id, self.artifacts)
                info.update(sha256=hashlib.sha256(data).hexdigest(), task_id=job.task_id)
                temporary = journal.with_suffix(".part")
                temporary.write_bytes(data)
                os.replace(temporary, journal)
                temporary = manifest.with_suffix(".tmp")
                temporary.write_text(json.dumps(info), encoding="utf-8")
                os.replace(temporary, manifest)
            self.tasks.update_status(job.task_id, "running", phase="collecting")
            rec = self.artifacts.create_from_bytes(project_id=job.project_id,
                artifact_id="art-" + hashlib.sha256(job.task_id.encode()).hexdigest()[:16],
                filename="grok." + info["extension"], data=data,
                media_type=info["media_type"], metadata=info)
            self.tasks.update_status(job.task_id, "completed", artifact_id=rec.artifact_id,
                                     error="", phase="completed")

    def recover_grok(self):
        records, _ = self.tasks.list_tasks(engine="grok", status="running", limit=10000, include_nsfw=True)
        for rec in records:
            if rec.request_params.get('channel_runtime') == 'grok-channel-v1':
                continue
            if self._grok_journal(rec.task_id).with_suffix(".json").exists():
                try:
                    self._execute_grok(TaskJob(rec.task_id, rec.project_id, rec.engine,
                                              rec.request_params["action"], rec.request_params))
                except Exception:
                    self.tasks.update_status(rec.task_id, "running", error="collection_unavailable", phase="collect_retry")
            else:
                self.tasks.update_status(rec.task_id, "running", error=rec.error or "submission_unknown", phase="submission_unknown")

    def _comfyui_journal(self, task_id: str):
        root = self.tasks.db_path.parent / ".comfyui-tasks"
        root.mkdir(parents=True, exist_ok=True)
        return root / f"{task_id}.json"

    def _execute_comfyui(self, job: TaskJob) -> None:
        with self._task_lock(job.task_id):
            journal = self._comfyui_journal(job.task_id)
            raw_bytes, mime_type, remote_filename, meta = self.comfyui_provider.execute_or_recover(
                task_id=job.task_id,
                project_id=job.project_id,
                params=job.params,
                artifacts=self.artifacts,
                journal_path=journal,
                on_phase=lambda ph: self.tasks.update_status(job.task_id, "running", phase=ph),
            )
            extension, detected_media_type = _image_storage_type(raw_bytes)
            final_media_type = mime_type if mime_type in {"image/png", "image/jpeg", "image/webp"} else detected_media_type
            filename = f"comfyui-{job.task_id}{extension}"
            artifact_id = "art-" + hashlib.sha256(job.task_id.encode("utf-8")).hexdigest()[:16]
            record = self.artifacts.create_from_bytes(
                project_id=job.project_id,
                artifact_id=artifact_id,
                filename=filename,
                data=raw_bytes,
                media_type=final_media_type,
                metadata=meta,
            )
            self.tasks.update_status(
                job.task_id,
                "running",
                artifact_id=record.artifact_id,
                error="",
                phase="completed",
            )

    def authorize_task_caller(self, rec: TaskRecord, *, require_write: bool) -> dict[str, str]:
        if rec.engine == "comfyui":
            return self.comfyui_provider.authorize_task_caller(
                project_id=rec.project_id,
                params=rec.request_params,
                authorize_fn=self.google_authorize,
                require_write=require_write,
            )
        return {}

    def recover_comfyui(self) -> None:
        records, _ = self.tasks.list_tasks(engine="comfyui", status="running", limit=10000, include_nsfw=True)
        for rec in records:
            self.poll_comfyui_task(rec.task_id)

    def poll_comfyui_task(self, task_id: str) -> None:
        rec = self.tasks.get(task_id)
        if rec.engine != "comfyui" or rec.status != "running":
            return
        journal = self._comfyui_journal(rec.task_id)
        cancel_marker = journal.with_suffix(".cancel")
        cancel_requested = cancel_marker.exists() or rec.phase == "canceling" or rec.error == "cancel_requested"
        if cancel_requested:
            outcome = self.cancel_comfyui_task(rec)
            if outcome.confirmed_terminal:
                if outcome.terminal_status == "completed" and not rec.artifact_id:
                    pass
                elif outcome.terminal_status == "failed":
                    self.tasks.update_status(rec.task_id, "failed", error="comfyui_execution_failed", phase="failed")
                    return
                else:
                    self.tasks.update_status(rec.task_id, "canceled", error="canceled_by_caller", phase="canceled")
                    return
            else:
                self.tasks.update_status(rec.task_id, "running", error="cancel_requested", phase="canceling")
                return

        lock = self._task_lock(rec.task_id)
        if not lock.acquire(blocking=False):
            return
        try:
            if journal.exists():
                try:
                    raw_bytes, mime_type, _remote_filename, meta = self.comfyui_provider.execute_or_recover(
                        task_id=rec.task_id,
                        project_id=rec.project_id,
                        params=rec.request_params,
                        artifacts=self.artifacts,
                        journal_path=journal,
                        on_phase=lambda ph: self.tasks.update_status(rec.task_id, "running", phase=ph),
                    )
                    extension, detected_media_type = _image_storage_type(raw_bytes)
                    final_media_type = mime_type if mime_type in {"image/png", "image/jpeg", "image/webp"} else detected_media_type
                    filename = f"comfyui-{rec.task_id}{extension}"
                    artifact_id = "art-" + hashlib.sha256(rec.task_id.encode("utf-8")).hexdigest()[:16]
                    art_rec = self.artifacts.create_from_bytes(
                        project_id=rec.project_id,
                        artifact_id=artifact_id,
                        filename=filename,
                        data=raw_bytes,
                        media_type=final_media_type,
                        metadata=meta,
                    )
                    self.tasks.update_status(
                        rec.task_id,
                        "completed",
                        artifact_id=art_rec.artifact_id,
                        error="",
                        phase="completed",
                    )
                except ComfyUIProviderError as exc:
                    if exc.uncertain:
                        self.tasks.update_status(rec.task_id, "running", error=exc.code, phase="collect_retry")
                    elif cancel_marker.exists() and exc.code in {"canceled_by_caller", "comfyui_execution_cancelled"}:
                        self.tasks.update_status(rec.task_id, "canceled", error="canceled_by_caller", phase="canceled")
                    else:
                        self.tasks.update_status(rec.task_id, "failed", error=exc.code, phase="failed")
                except Exception:
                    self.tasks.update_status(rec.task_id, "running", error="comfyui_recovery_pending", phase="collect_retry")
            else:
                self.tasks.update_status(
                    rec.task_id,
                    "running",
                    error=rec.error or "submission_unknown",
                    phase="submission_unknown",
                )
        finally:
            lock.release()

    def cancel_comfyui_task(self, rec: TaskRecord) -> ComfyUICancelResult:
        if rec.engine != "comfyui":
            return ComfyUICancelResult(confirmed_terminal=False, reason="unsupported_engine")
        journal = self._comfyui_journal(rec.task_id)
        lock = self._task_lock(rec.task_id)
        acquired = lock.acquire(blocking=False)
        if not acquired:
            return self.comfyui_provider.cancel_remote_task(
                task_id=rec.task_id,
                project_id=rec.project_id,
                params=rec.request_params,
                journal_path=journal,
                artifacts=self.artifacts,
                worker_active=True,
            )
        try:
            fresh = self.tasks.get(rec.task_id)
            if fresh.status in {"completed", "failed", "canceled"}:
                return ComfyUICancelResult(
                    confirmed_terminal=True,
                    terminal_status=fresh.status,
                    reason="already_terminal",
                )
            return self.comfyui_provider.cancel_remote_task(
                task_id=fresh.task_id,
                project_id=fresh.project_id,
                params=fresh.request_params,
                journal_path=journal,
                artifacts=self.artifacts,
                worker_active=False,
            )
        finally:
            lock.release()

    async def _collect_grok(self):
        while self._running:
            await asyncio.sleep(15)
            # Only persisted results are retried; never repeat a POST.
            records, _ = self.tasks.list_tasks(engine="grok", status="running", limit=10000, include_nsfw=True)
            for rec in records:
                if rec.request_params.get('channel_runtime') == 'grok-channel-v1':
                    continue
                if rec.phase == "collect_retry" and self._grok_journal(rec.task_id).with_suffix(".json").exists():
                    try:
                        await asyncio.to_thread(self._execute_grok, TaskJob(rec.task_id, rec.project_id,
                            rec.engine, rec.request_params["action"], rec.request_params))
                    except Exception:
                        pass

    async def _idle_moss(self) -> None:
        while self._running:
            await asyncio.sleep(15)
            try:
                await asyncio.to_thread(self.moss_gpu.maybe_scale_down)
            except Exception:
                pass

    def _execute_moss(self, job: TaskJob) -> None:
        from contextlib import nullcontext

        scene_edit = job.params.get("moss_mode") == "edit"
        if scene_edit:
            if self.google_authorize is None:
                raise ValueError("project_authorization_not_configured")
            authority = job.params.get("channel_authority")
            if not isinstance(authority, dict):
                raise ValueError("verified_studio_identity_required")
            self.google_authorize(job.project_id, authority)

        self.tasks.update_status(job.task_id, "running", phase="acquiring_gpu")
        reps = None
        if hasattr(self.moss_gpu, "replicas") and callable(self.moss_gpu.replicas):
            try:
                raw_reps = self.moss_gpu.replicas()
                if isinstance(raw_reps, int) and not isinstance(raw_reps, bool):
                    reps = raw_reps
            except Exception:
                reps = None
        k8s_slots = 1 if reps == 0 else 0
        pre_ident: dict[str, Any] = {}
        if self.scheduler.enabled and reps != 0 and hasattr(self.moss_provider, "health_details"):
            hd = self.moss_provider.health_details()
            if isinstance(hd.get("runtime_identity"), dict):
                pre_ident = dict(hd["runtime_identity"])

        deadline = self.scheduler._monotonic() + self.scheduler.queue_seconds
        delay = 0.2
        attempt_idx = 0
        grant = None
        hold_ctx = nullcontext()
        while True:
            attempt_idx += 1
            pre_vis = pre_ident.get("visible_gpu_uuids")
            grant = self.scheduler.acquire_for_queue(
                consumer_kind="moss-image",
                task_id=job.task_id,
                stage_id="generate",
                attempt_id=f"att-{job.task_id[:8]}-{attempt_idx}",
                k8s_slots_required=k8s_slots,
                pod_uid=str(pre_ident.get("pod_uid") or ""),
                container_id=str(pre_ident.get("container_id") or ""),
                boot_id=str(pre_ident.get("boot_id") or ""),
                visible_gpu_uuids=pre_vis if isinstance(pre_vis, (list, tuple)) else None,
            )
            if self.moss_gpu.enabled:
                try:
                    hold_ctx = self.moss_gpu.try_hold().__enter__()
                except LeaseBusy:
                    if grant is not None and not grant.is_target_bound:
                        try:
                            self.scheduler.release(
                                grant,
                                worker_settled=False,
                                reclaim_mode="abort_unstarted",
                            )
                        except Exception:
                            pass
                        grant = None
                    elif grant is not None:
                        try:
                            self.scheduler.release(
                                grant,
                                worker_settled=False,
                                runtime_identity=pre_ident,
                            )
                        except Exception:
                            pass
                        grant = None
                    if self.scheduler._monotonic() >= deadline:
                        raise
                    self.scheduler._sleep(min(5.0, delay))
                    delay = min(5.0, delay * 1.5)
                    continue
            break

        settled = False
        target_bound = bool(grant is not None and grant.is_target_bound)
        post_ident: dict[str, Any] = dict(pre_ident)
        reclaim_proof: dict[str, Any] | None = None
        try:
            try:
                self.moss_gpu.ensure_worker_ready(self.moss_provider.health)
                if grant is not None and hasattr(self.moss_provider, "health_details"):
                    hd = self.moss_provider.health_details()
                    if isinstance(hd.get("runtime_identity"), dict):
                        post_ident = dict(hd["runtime_identity"])
                    need_bind = (
                        k8s_slots > 0
                        or not grant.is_target_bound
                        or (post_ident.get("pod_uid") and post_ident.get("pod_uid") != grant.pod_uid)
                    )
                    if need_bind:
                        self.scheduler.bind_target(grant, runtime_identity=post_ident)
                        target_bound = True
                self.tasks.update_status(job.task_id, "running", phase="generating")
                primary_ref_id = (
                    job.params.get("face_reference_artifact_id")
                    or job.params.get("source_identity_artifact_id")
                    or job.params.get("source_artifact_id")
                )
                raw_ref_ids = [
                    primary_ref_id,
                    *(job.params.get("reference_artifact_ids") or []),
                    *(job.params.get("reference_images") or []),
                ]
                ref_ids = list(
                    dict.fromkeys(
                        str(rid).strip()
                        for rid in raw_ref_ids
                        if rid and str(rid).strip()
                    )
                )
                gen_params = dict(job.params)
                if ref_ids:
                    ref_b64_list: list[str] = []
                    for rid in ref_ids:
                        ref_record = self.artifacts.get(rid, job.project_id)
                        ref_bytes = self.artifacts.content_path(ref_record).read_bytes()
                        ref_b64_list.append(base64.b64encode(ref_bytes).decode("ascii"))
                    gen_params["reference_images_base64"] = ref_b64_list

                if grant is not None:
                    img_bytes, meta = self.moss_provider.generate(gen_params, extra_headers=grant.headers())
                else:
                    img_bytes, meta = self.moss_provider.generate(gen_params)
                reclaim_proof = getattr(self.moss_provider, "last_reclaim_proof", None)
                if grant is not None and reclaim_proof is None and hasattr(self.moss_provider, "reclaim_guard"):
                    reclaim_proof = self.moss_provider.reclaim_guard(
                        lease_id=grant.lease_id,
                        fencing_token=grant.fencing_token,
                        runtime_token=self.scheduler.token,
                    )
                settled = bool(
                    grant is None
                    or (
                        isinstance(reclaim_proof, dict)
                        and reclaim_proof.get("reclaimed") is True
                        and reclaim_proof.get("proof_source") == "receiver_guard"
                        and reclaim_proof.get("token_sealed") is True
                    )
                )
                meta["task_id"] = job.task_id
                if ref_ids and scene_edit:
                    meta.update(
                        {
                            "source_artifact_id": ref_ids[0],
                            "reference_artifact_ids": ref_ids[1:],
                            "conditioning_mode": "moss_edit_reference_latents",
                            "moss_mode": "edit",
                        }
                    )
                elif ref_ids:
                    first_ref = ref_ids[0]
                    meta.update(
                        {
                            "source_identity_artifact_id": first_ref,
                            "face_reference_artifact_id": first_ref,
                            "reference_artifact_ids": ref_ids,
                            "conditioning_mode": "qwen3.5_9b_vision",
                        }
                    )
                extension, media_type = _image_storage_type(img_bytes)
                filename = f"moss-{job.task_id}{extension}"
                artifact_id = "art-" + hashlib.sha256(job.task_id.encode()).hexdigest()[:16]
                record = self.artifacts.create_from_bytes(
                    project_id=job.project_id,
                    artifact_id=artifact_id,
                    filename=filename,
                    data=img_bytes,
                    media_type=media_type,
                    metadata=meta,
                )
                self.tasks.update_status(
                    job.task_id,
                    "running",
                    artifact_id=record.artifact_id,
                    phase="completed",
                )
            finally:
                if self.moss_gpu.enabled:
                    hold_ctx.__exit__(None, None, None)
        finally:
            if grant is not None:
                if not target_bound:
                    rel_mode = "abort_unstarted"
                    rel_settled = False
                elif settled and reclaim_proof:
                    rel_mode = "warm_idle_settled"
                    rel_settled = True
                else:
                    rel_mode = ""
                    rel_settled = False
                try:
                    self.scheduler.release(
                        grant,
                        worker_settled=rel_settled,
                        runtime_identity=post_ident if target_bound else None,
                        reclaim_mode=rel_mode,
                        reclaim_proof=reclaim_proof if rel_settled else None,
                    )
                except Exception:
                    pass

    def _execute_generate(self, job: TaskJob) -> None:
        p = job.params
        prompt = p.get("prompt", "")
        engine = job.engine.lower()
        model = p.get("model", "")
        aspect_ratio = p.get("aspect_ratio", "16:9")
        resolution = p.get("resolution", "2k")
        quality = p.get("quality", "high")
        background = p.get("background", "auto")
        instructions = p.get("instructions")

        prefer_4k = resolution.lower() == "4k"
        size_map = {
            "16:9": "2048x1152",
            "9:16": "1152x2048",
            "1:1": "1024x1024",
            "4:3": "1792x1344",
            "3:4": "1344x1792",
            "3:2": "1536x1024",
            "2:3": "1024x1536",
        }
        size = size_map.get(aspect_ratio, "2048x1152")

        source_id = p.get("source_artifact_id")
        ref_ids = list(p.get("reference_artifact_ids") or p.get("reference_images") or [])
        all_ref_ids = [rid for rid in dict.fromkeys([source_id, *ref_ids]) if rid]

        source_bytes_list: list[bytes] = []
        for rid in all_ref_ids:
            rec = self.artifacts.get(rid, job.project_id)
            source_bytes_list.append(self.artifacts.content_path(rec).read_bytes())

        if engine == "gemini":
            img_bytes = self.gemini_provider.generate(
                prompt=prompt,
                model=model or DEFAULT_GEMINI_MODEL,
                source_images=source_bytes_list or None,
                aspect_ratio=aspect_ratio,
            )
        elif source_bytes_list:
            img_bytes = self.codex_provider.edit(
                prompt=prompt,
                source_bytes=source_bytes_list if len(source_bytes_list) > 1 else source_bytes_list[0],
                size=size,
                prefer_4k=prefer_4k,
                quality=quality,
                background=background,
                model=model or DEFAULT_IMAGE_MODEL,
                instructions=instructions,
            )
        else:
            img_bytes = self.codex_provider.generate(
                prompt=prompt,
                size=size,
                prefer_4k=prefer_4k,
                quality=quality,
                background=background,
                model=model or DEFAULT_IMAGE_MODEL,
                instructions=instructions,
            )

        extension, media_type = _image_storage_type(img_bytes)
        filename = f"gen-{job.task_id}{extension}"
        meta: dict[str, Any] = {
            "engine": engine,
            "model": model or (DEFAULT_GEMINI_MODEL if engine == "gemini" else DEFAULT_IMAGE_MODEL),
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "quality": quality,
            "background": background,
            "instructions": instructions or "verbatim",
        }
        if engine != "gemini":
            meta["resolution"] = resolution
        try:
            import io
            from PIL import Image
            with Image.open(io.BytesIO(img_bytes)) as pil_img:
                meta["width"], meta["height"] = pil_img.size
        except Exception:
            pass
        if source_id:
            meta["source_artifact_id"] = source_id
        if ref_ids:
            meta["reference_artifact_ids"] = ref_ids
        record = self.artifacts.create_from_bytes(
            project_id=job.project_id,
            filename=filename,
            data=img_bytes,
            media_type=media_type,
            metadata=meta,
        )
        self.tasks.update_status(job.task_id, "running", artifact_id=record.artifact_id)

    def _execute_edit(self, job: TaskJob) -> None:
        p = job.params
        prompt = p.get("prompt", "")
        engine = job.engine.lower()
        source_id = p.get("source_artifact_id", "")
        ref_ids = p.get("reference_artifact_ids", [])
        mask_id = p.get("mask_artifact_id")
        model = p.get("model", "")
        aspect_ratio = p.get("aspect_ratio")
        resolution = p.get("resolution", "2k")
        prefer_4k = (resolution or "2k").lower() == "4k"
        if aspect_ratio:
            size_map = {
                "16:9": "2048x1152",
                "9:16": "1152x2048",
                "1:1": "1024x1024",
                "4:3": "1792x1344",
                "3:4": "1344x1792",
                "3:2": "1536x1024",
                "2:3": "1024x1536",
            }
            size = size_map.get(aspect_ratio, "2048x1152")
            if prefer_4k and hasattr(self.codex_provider, "_resolve_target_size"):
                size = self.codex_provider._resolve_target_size(size, prefer_4k=True)
        else:
            size = p.get("size", "2048x1152")
            if prefer_4k and hasattr(self.codex_provider, "_resolve_target_size"):
                size = self.codex_provider._resolve_target_size(size, prefer_4k=True)
        quality = p.get("quality", "auto")
        background = p.get("background", "auto")

        source_rec = self.artifacts.get(source_id, job.project_id)
        source_bytes = self.artifacts.content_path(source_rec).read_bytes()

        source_bytes_list: list[bytes] = [source_bytes]
        if ref_ids and isinstance(ref_ids, list):
            for rid in ref_ids:
                if rid and rid != source_id:
                    rec = self.artifacts.get(rid, job.project_id)
                    source_bytes_list.append(self.artifacts.content_path(rec).read_bytes())

        mask_bytes = None
        if mask_id:
            mask_rec = self.artifacts.get(mask_id, job.project_id)
            mask_bytes = self.artifacts.content_path(mask_rec).read_bytes()

        if engine == "gemini":
            if mask_bytes is not None:
                raise ValueError("unsupported_parameter")
            img_bytes = self.gemini_provider.generate(
                prompt=prompt,
                model=model or DEFAULT_GEMINI_MODEL,
                source_images=source_bytes_list,
                aspect_ratio=aspect_ratio,
            )
        else:
            img_bytes = self.codex_provider.edit(
                prompt=prompt,
                source_bytes=source_bytes_list if len(source_bytes_list) > 1 else source_bytes,
                mask_bytes=mask_bytes,
                size=size or "2048x1152",
                prefer_4k=prefer_4k,
                quality=quality,
                background=background,
                model=model or DEFAULT_IMAGE_MODEL,
            )

        extension, media_type = _image_storage_type(img_bytes)
        filename = f"edit-{job.task_id}{extension}"
        record = self.artifacts.create_from_bytes(
            project_id=job.project_id,
            filename=filename,
            data=img_bytes,
            media_type=media_type,
            metadata={
                "engine": engine,
                "model": model or (DEFAULT_GEMINI_MODEL if engine == "gemini" else DEFAULT_IMAGE_MODEL),
                "source_artifact_id": source_id,
                "reference_artifact_ids": ref_ids,
                "mask_artifact_id": mask_id,
                "prompt": prompt,
                "size": size,
                "quality": quality,
                "background": background,
            },
        )
        self.tasks.update_status(job.task_id, "running", artifact_id=record.artifact_id)
