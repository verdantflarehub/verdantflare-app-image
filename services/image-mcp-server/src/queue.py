from __future__ import annotations

import asyncio
import os
import time
import hashlib
import json
import threading
import base64
from dataclasses import dataclass
from typing import Any, Callable, Coroutine

from .artifacts import ArtifactStore
from .gpu_lease import GpuCoordinator
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
        self.grok_provider = GrokImageProvider()
        self.grok_sem = asyncio.Semaphore(max(1, int(os.getenv("XAI_IMAGE_MAX_CONCURRENCY", "1"))))
        self.grok_lock = threading.Lock()
        self.moss_provider = moss_provider or MossProvider()
        self.moss_gpu = moss_gpu or GpuCoordinator.from_environment()
        self.moss_sem = asyncio.Semaphore(max(1, int(os.getenv("MOSS_IMAGE_MAX_CONCURRENCY", "1"))))

        codex_limit = int(os.environ.get("CODEX_MAX_CONCURRENCY", "2"))
        gemini_limit = int(os.environ.get("GEMINI_MAX_CONCURRENCY", "5"))
        worker_count = int(os.environ.get("IMAGE_QUEUE_WORKERS", "8"))

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
        if recovered > 0:
            print(f"⚠️ [TaskQueue] 保留 {recovered} 个原调用结果未知的任务；未重新提交")

        # 2. 启动 Worker 协程池
        for i in range(self.worker_count):
            task = asyncio.create_task(self._worker_loop(i))
            self._worker_tasks.append(task)
        self._worker_tasks.append(asyncio.create_task(self._collect_grok()))
        self._worker_tasks.append(asyncio.create_task(self._collect_channels()))
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
        if engine not in {"codex", "gemini", "grok", "moss-image"}:
            raise ValueError("unsupported_engine")
        params = dict(params)
        # Internal route/authority cannot be supplied in generation parameters.
        for internal in ('channel_runtime', 'channel_authority'):
            if internal in params:
                raise ValueError('reserved_channel_parameter')
        if engine == 'moss-image':
            return self.moss_provider.normalize(params)
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

    def _execute_google(self, job, recovery=False):
        return self._execute_channel(job, recovery)

    def _execute_channel(self, job, recovery=False):
        # The worker and collector serialize collection; only the claimed worker
        # may submit. A crash anywhere after claiming leaves GET-only recovery.
        with self.google_lock:
            record = self.tasks.get(job.task_id)
            if record.status != 'running':
                return
            runtime = job.params.get('channel_runtime')
            is_grok = runtime == 'grok-channel-v1'
            is_chatgpt = runtime == 'chatgpt-channel-v1'
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
                if is_grok:
                    effective_action = job.action
                    payload = {key: job.params[key] for key in ('prompt', 'action', 'resolution')}
                    if job.action == 'edit':
                        payload['image'] = images[0]
                    else:
                        payload['aspect_ratio'] = job.params['aspect_ratio']
                if is_chatgpt:
                    p = job.params
                    size = {'16:9': '2048x1152', '9:16': '1152x2048', '1:1': '1024x1024',
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
            if result['accepted'] is False or result.get('state') == 'failed':
                reason = result.get('reason_code') or result.get('error') or ''
                err = f"channel_call_failed: {reason}" if reason else 'channel_call_failed'
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
        with self.moss_gpu.hold():
            self.tasks.update_status(job.task_id, "running", phase="acquiring_gpu")
            self.moss_gpu.ensure_worker_ready(self.moss_provider.health)
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

            img_bytes, meta = self.moss_provider.generate(gen_params)
            meta["task_id"] = job.task_id
            if ref_ids:
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
