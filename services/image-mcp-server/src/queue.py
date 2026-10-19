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
from .providers.codex import CodexProvider, DEFAULT_IMAGE_MODEL
from .providers.gemini import GeminiProvider, DEFAULT_GEMINI_MODEL
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

        # 3. 恢复历史未消费的 queued 任务
        pending = self.tasks.get_queued_tasks()
        for rec in pending:
            action = rec.request_params.get("action", "generate")
            await self.enqueue(rec.task_id, rec.project_id, rec.engine, action, rec.request_params)

        print(f"✅ [TaskQueue] 队列调度器启动完毕 (Workers: {self.worker_count})")

    async def stop(self) -> None:
        self._running = False
        for task in self._worker_tasks:
            task.cancel()
        if self._worker_tasks:
            await asyncio.gather(*self._worker_tasks, return_exceptions=True)
        self._worker_tasks.clear()
        print("🛑 [TaskQueue] 队列调度器已停止")

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
        if engine not in {"codex", "gemini", "grok"}:
            raise ValueError("unsupported_engine")
        params = dict(params)
        # Internal route/authority cannot be supplied in generation parameters.
        for internal in ('channel_runtime', 'channel_authority'):
            if internal in params:
                raise ValueError('reserved_channel_parameter')
        if engine == 'grok':
            params = self.grok_provider.normalize(params, project_id, self.artifacts, channel=self.grok_channel is not None)
        channel = {'gemini': self.google_channel, 'grok': self.grok_channel, 'codex': self.chatgpt_channel}.get(engine)
        if self.channel_directory is not None and channel is None:
            raise ValueError('channel_not_configured')
        if channel is not None:
            self._check_channel_discovery(engine, params.get('action'), channel)
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
            if engine == 'gemini' and params.get('model') not in (None, '', DEFAULT_GEMINI_MODEL, 'gemini-3.1-flash-image-preview'):
                raise ValueError('capability_not_supported')
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
            {"engine": "gemini", "model": DEFAULT_GEMINI_MODEL, "enabled": True},
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
                if self.channel_directory is not None:
                    engine['enabled'] = self.channel_directory.resolve(name, 'image.generate') == channel.endpoint
        return {"default_engine": "codex", "engines": engines}

    def _get_semaphore(self, engine: str) -> asyncio.Semaphore:
        if engine.lower() == "grok":
            return self.grok_sem
        if engine.lower() not in {"codex", "gemini"}:
            raise ValueError("unsupported_engine")
        return self.gemini_sem if engine.lower() == "gemini" else self.codex_sem

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
                    print(f"✅ [TaskQueue] 任务成功: {job.task_id} (action: {job.action}, engine: {job.engine}) 耗时: {duration}s", flush=True)
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
                    print(f"❌ [TaskQueue] 任务失败: {job.task_id} (action: {job.action}, engine: {job.engine}, model: {job.params.get('model')}) 耗时: {duration}s, 错误详情: {err_msg}", flush=True)
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
                images = []
                ids = [job.params.get('source_artifact_id')]
                ids.extend(job.params.get('reference_artifact_ids') or [])
                if not is_chatgpt:
                    ids.append(job.params.get('mask_artifact_id'))
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
                if is_grok:
                    payload = {key: job.params[key] for key in ('prompt', 'action', 'resolution')}
                    if job.action == 'edit':
                        payload['image'] = images[0]
                    else:
                        payload['aspect_ratio'] = job.params['aspect_ratio']
                if is_chatgpt:
                    p = job.params
                    size = {'16:9': '2048x1152', '9:16': '1152x2048', '1:1': '1024x1024',
                            '4:3': '1792x1344', '3:4': '1344x1792'}.get(p.get('aspect_ratio'), p.get('size') or '2048x1152')
                    payload = {'prompt': p['prompt'], 'action': 'edit' if job.action == 'inpaint' else job.action, 'images': images,
                        'size': size, 'prefer_4k': (p.get('resolution') or '2k').lower() == '4k',
                        'quality': p.get('quality') or ('high' if job.action == 'generate' else 'auto'),
                        'background': p.get('background') or 'auto'}
                    if p.get('instructions') is not None:
                        payload['instructions'] = p['instructions']
                    if p.get('mask_artifact_id'):
                        mask = self.artifacts.get(p['mask_artifact_id'], job.project_id)
                        data = self.artifacts.content_path(mask).read_bytes()
                        if hashlib.sha256(data).hexdigest() != mask.sha256:
                            raise ValueError('source_integrity_failed')
                        payload['mask'] = {'mime_type': mask.media_type, 'data': base64.b64encode(data).decode()}
                self._check_channel_discovery(job.engine, job.action, channel)
                self.tasks.update_status(job.task_id, 'running', phase='submitting')
                result = channel.submit(organization, job.task_id, job.task_id,
                    job.params.get('model') or (DEFAULT_IMAGE_MODEL if is_chatgpt else DEFAULT_GEMINI_MODEL),
                    payload,
                    job.params.get('channel_account_id'), job.params.get('budget_limit'))
            if result['accepted'] is False or result.get('state') == 'failed':
                self.tasks.update_status(job.task_id, 'failed',
                    error='channel_call_failed', phase='failed')
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
            records, _ = self.tasks.list_tasks(status='running', limit=10000)
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
        records, _ = self.tasks.list_tasks(engine="grok", status="running", limit=10000)
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
            records, _ = self.tasks.list_tasks(engine="grok", status="running", limit=10000)
            for rec in records:
                if rec.request_params.get('channel_runtime') == 'grok-channel-v1':
                    continue
                if rec.phase == "collect_retry" and self._grok_journal(rec.task_id).with_suffix(".json").exists():
                    try:
                        await asyncio.to_thread(self._execute_grok, TaskJob(rec.task_id, rec.project_id,
                            rec.engine, rec.request_params["action"], rec.request_params))
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
        }
        size = size_map.get(aspect_ratio, "2048x1152")

        if engine == "gemini":
            img_bytes = self.gemini_provider.generate(
                prompt=prompt,
                model=model or DEFAULT_GEMINI_MODEL,
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
        record = self.artifacts.create_from_bytes(
            project_id=job.project_id,
            filename=filename,
            data=img_bytes,
            media_type=media_type,
            metadata={
                "engine": engine,
                "model": model or (DEFAULT_GEMINI_MODEL if engine == "gemini" else DEFAULT_IMAGE_MODEL),
                "prompt": prompt,
                "aspect_ratio": aspect_ratio,
                "resolution": resolution,
                "quality": quality,
                "background": background,
                "instructions": instructions or "verbatim",
            },
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
        prefer_4k = resolution.lower() == "4k"
        if aspect_ratio:
            size_map = {
                "16:9": "2048x1152",
                "9:16": "1152x2048",
                "1:1": "1024x1024",
                "4:3": "1792x1344",
                "3:4": "1344x1792",
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
            img_bytes = self.gemini_provider.generate(
                prompt=prompt,
                model=model or DEFAULT_GEMINI_MODEL,
                source_images=source_bytes_list,
                mask_bytes=mask_bytes,
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
