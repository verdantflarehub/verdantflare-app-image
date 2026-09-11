from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Coroutine

from .artifacts import ArtifactStore
from .providers.codex import CodexProvider, DEFAULT_IMAGE_MODEL
from .providers.gemini import GeminiProvider, DEFAULT_GEMINI_MODEL
from .tasks import TaskRecord, TaskStore


@dataclass
class TaskJob:
    task_id: str
    project_id: str
    engine: str
    action: str  # "generate", "edit", "inpaint"
    params: dict[str, Any]


class TaskQueueManager:
    def __init__(
        self,
        tasks: TaskStore,
        artifacts: ArtifactStore,
        codex_provider: CodexProvider,
        gemini_provider: GeminiProvider,
    ) -> None:
        self.tasks = tasks
        self.artifacts = artifacts
        self.codex_provider = codex_provider
        self.gemini_provider = gemini_provider

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
        if recovered > 0:
            print(f"⚠️ [TaskQueue] 已自动恢复并标记 {recovered} 个中断的任务为 failed")

        # 2. 启动 Worker 协程池
        for i in range(self.worker_count):
            task = asyncio.create_task(self._worker_loop(i))
            self._worker_tasks.append(task)

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

    def _get_semaphore(self, engine: str) -> asyncio.Semaphore:
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
                self.tasks.update_status(job.task_id, "running")
                loop = asyncio.get_running_loop()

                try:
                    if job.action == "generate":
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

        filename = f"gen-{job.task_id}.png"
        record = self.artifacts.create_from_bytes(
            project_id=job.project_id,
            filename=filename,
            data=img_bytes,
            media_type="image/png",
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
                source_image_bytes=source_bytes,
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

        filename = f"edit-{job.task_id}.png"
        record = self.artifacts.create_from_bytes(
            project_id=job.project_id,
            filename=filename,
            data=img_bytes,
            media_type="image/png",
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

