from __future__ import annotations

import base64
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
from PIL import Image

from src.artifacts import ArtifactStore
from src.providers.moss import (
    DEFAULT_MOSS_MODEL,
    MossProvider,
    MossProviderError,
)
from src.queue import TaskJob, TaskQueueManager
from src.tasks import TaskStore


def _make_png_bytes(width: int = 64, height: int = 64) -> bytes:
    buf = io.BytesIO()
    img = Image.new("RGB", (width, height), color=(20, 40, 80))
    img.save(buf, format="PNG")
    return buf.getvalue()


class TestMossProviderValidation(unittest.TestCase):
    def setUp(self) -> None:
        self.provider = MossProvider(base_url="http://image-moss-api:8000")

    def test_capability_declares_self_hosted_contract(self) -> None:
        caps = self.provider.capability()
        self.assertTrue(caps["enabled"])
        self.assertEqual(caps["provider"], "self-hosted")
        self.assertFalse(caps["requires_channel_account"])
        self.assertFalse(caps["budget_limit_required"])
        self.assertEqual(caps["model"], DEFAULT_MOSS_MODEL)
        self.assertEqual(caps["limits"]["max_layout_items"], 32)
        self.assertEqual(caps["limits"]["default_steps"], 50)
        self.assertEqual(caps["limits"]["default_cfg_scale"], 3.0)

    def test_normalize_defaults_and_aspect_ratio_mapping(self) -> None:
        prepared = self.provider.normalize(
            {
                "action": "generate",
                "prompt": "中文海报标题：山海寻踪",
                "aspect_ratio": "1:1",
                "resolution": "1k",
            }
        )
        self.assertEqual(prepared["width"], 1024)
        self.assertEqual(prepared["height"], 1024)
        self.assertEqual(prepared["steps"], 50)
        self.assertEqual(prepared["cfg_scale"], 3.0)
        self.assertEqual(prepared["sampler"], "euler")
        self.assertEqual(prepared["scheduler"], "simple")
        self.assertEqual(prepared["shift"], 5.0)
        self.assertEqual(prepared["skip_single_layers"], "22")
        self.assertEqual(prepared["model"], DEFAULT_MOSS_MODEL)

    def test_normalize_rejects_non_generate_actions(self) -> None:
        for action in ("edit", "inpaint", "faceswap"):
            with self.subTest(action=action):
                with self.assertRaises(ValueError):
                    self.provider.normalize({"action": action, "prompt": "test"})

    def test_normalize_rejects_unsupported_parameters(self) -> None:
        rejected_cases = [
            {"channel_account_id": "acc-1"},
            {"budget_limit": "5"},
            {"resolution": "4k"},
        ]
        for extra in rejected_cases:
            with self.subTest(extra=extra):
                with self.assertRaises(ValueError):
                    self.provider.normalize({"action": "generate", "prompt": "海报", **extra})

    def test_normalize_validates_dimensions(self) -> None:
        with self.assertRaises(ValueError):
            self.provider.normalize({"prompt": "test", "width": 1000, "height": 1024})
        with self.assertRaises(ValueError):
            self.provider.normalize({"prompt": "test", "width": 256, "height": 256})
        with self.assertRaises(ValueError):
            self.provider.normalize({"prompt": "test", "width": 2560, "height": 2048})

    def test_normalize_validates_layout_json(self) -> None:
        valid_layout = {
            "meta": "科技峰会海报",
            "items": [
                {
                    "id": 1,
                    "desc": "主标题：智启未来",
                    "box_2d": [100, 120, 900, 320],
                }
            ],
        }
        prepared = self.provider.normalize(
            {"prompt": "科技峰会海报", "width": 1024, "height": 1024, "layout": valid_layout}
        )
        self.assertEqual(prepared["layout"]["meta"], "科技峰会海报")

        # Out-of-bounds box_2d (> 1000)
        bad_bbox_layout = {
            "meta": "科技峰会海报",
            "items": [
                {
                    "id": 1,
                    "desc": "越界",
                    "box_2d": [100, 120, 1200, 320],
                }
            ],
        }
        with self.assertRaises(ValueError):
            self.provider.normalize(
                {"prompt": "科技峰会海报", "width": 1024, "height": 1024, "layout": bad_bbox_layout}
            )

        # >32 items rejected
        too_many_items = {
            "meta": "过多元素",
            "items": [
                {"id": i + 1, "desc": f"item-{i}", "box_2d": [10, 10, 100, 100]}
                for i in range(33)
            ],
        }
        with self.assertRaises(ValueError):
            self.provider.normalize(
                {"prompt": "过多元素", "width": 1024, "height": 1024, "layout": too_many_items}
            )


class TestMossProviderExecution(unittest.TestCase):
    def test_generate_posts_to_moss_api_and_decodes_png(self) -> None:
        png_bytes = _make_png_bytes(128, 128)
        b64_png = base64.b64encode(png_bytes).decode("ascii")

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/health":
                return httpx.Response(200, json={"status": "healthy"})
            if request.url.path == "/v1/generate" and request.method == "POST":
                return httpx.Response(
                    200,
                    json={
                        "status": "ok",
                        "image_base64": b64_png,
                        "mime_type": "image/png",
                        "width": 1024,
                        "height": 1024,
                        "resolved_seed": 42,
                        "steps": 50,
                        "cfg_scale": 3.0,
                        "sampler": "euler",
                        "scheduler": "simple",
                        "shift": 5.0,
                        "prompt_mode": "text",
                        "elapsed_ms": 46630,
                    },
                )
            return httpx.Response(404)

        client = httpx.Client(transport=httpx.MockTransport(handler))
        provider = MossProvider(base_url="http://image-moss-api:8000", client=client)
        try:
            self.assertTrue(provider.health())
            params = provider.normalize({"prompt": "山海寻踪", "width": 1024, "height": 1024, "seed": 42})
            out_bytes, meta = provider.generate(params)
            self.assertEqual(out_bytes, png_bytes)
            self.assertEqual(meta["resolved_seed"], 42)
            self.assertEqual(meta["elapsed_ms"], 46630)
            self.assertEqual(meta["engine"], "moss-image")
        finally:
            client.close()

    def test_queue_executes_moss_task_and_recovers_on_restart(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = ArtifactStore(root / "artifacts")
            tasks = TaskStore(root / "tasks.db")
            artifacts.ensure_ready()
            tasks.ensure_ready()

            png_bytes = _make_png_bytes(128, 128)
            moss_provider = MossProvider(base_url="http://image-moss-api:8000")
            moss_gpu = MagicMock()
            moss_gpu.hold.return_value.__enter__ = MagicMock(return_value=None)
            moss_gpu.hold.return_value.__exit__ = MagicMock(return_value=False)

            qm = TaskQueueManager(
                tasks=tasks,
                artifacts=artifacts,
                codex_provider=MagicMock(),
                gemini_provider=MagicMock(),
                moss_provider=moss_provider,
                moss_gpu=moss_gpu,
            )

            params = qm.prepare(
                "moss-image",
                "proj-moss",
                {"action": "generate", "prompt": "山海寻踪", "width": 1024, "height": 1024},
            )
            task = tasks.create(
                project_id="proj-moss",
                idempotency_key="idem-moss-1",
                engine="moss-image",
                model=DEFAULT_MOSS_MODEL,
                request_params=params,
            )
            # Simulate a hanging running moss-image task before artifact upload
            tasks.update_status(task.task_id, "running")
            tasks.recover_hanging_tasks()
            recovered = tasks.get(task.task_id)
            self.assertEqual(recovered.status, "queued")

            with patch.object(
                moss_provider,
                "generate",
                return_value=(
                    png_bytes,
                    {
                        "engine": "moss-image",
                        "model": DEFAULT_MOSS_MODEL,
                        "width": 1024,
                        "height": 1024,
                        "resolved_seed": 20261024,
                        "steps": 50,
                        "elapsed_ms": 46630,
                    },
                ),
            ):
                qm._execute_moss(
                    TaskJob(
                        task_id=task.task_id,
                        project_id="proj-moss",
                        engine="moss-image",
                        action="generate",
                        params=params,
                    )
                )

            completed = tasks.get(task.task_id)
            self.assertIsNotNone(completed.artifact_id)
            art = artifacts.get(completed.artifact_id, "proj-moss")
            self.assertEqual(art.metadata["engine"], "moss-image")
            self.assertEqual(art.metadata["resolved_seed"], 20261024)
            self.assertEqual(art.metadata["elapsed_ms"], 46630)
            moss_gpu.ensure_worker_ready.assert_called_once()

    def test_moss_two_stage_face_reference_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = ArtifactStore(root / "artifacts")
            tasks = TaskStore(root / "tasks.db")
            artifacts.ensure_ready()
            tasks.ensure_ready()

            png_bytes = _make_png_bytes(128, 128)
            fused_bytes = _make_png_bytes(128, 128)
            ref_rec = artifacts.create_from_bytes(
                project_id="proj-moss",
                artifact_id="art-01e71e31ecaf4f0d",
                filename="master-v20.png",
                data=png_bytes,
                media_type="image/png",
            )
            fused_rec = artifacts.create_from_bytes(
                project_id="proj-moss",
                artifact_id="art-aaaaaaaaaaaaaaaa",
                filename="fused.png",
                data=fused_bytes,
                media_type="image/png",
            )

            moss_provider = MossProvider(base_url="http://image-moss-api:8000")
            moss_gpu = MagicMock()
            moss_gpu.hold.return_value.__enter__ = MagicMock(return_value=None)
            moss_gpu.hold.return_value.__exit__ = MagicMock(return_value=False)

            qm = TaskQueueManager(
                tasks=tasks,
                artifacts=artifacts,
                codex_provider=MagicMock(),
                gemini_provider=MagicMock(),
                moss_provider=moss_provider,
                moss_gpu=moss_gpu,
            )

            params = qm.prepare(
                "moss-image",
                "proj-moss",
                {
                    "action": "generate",
                    "prompt": "晴子度假泳池写真",
                    "aspect_ratio": "9:16",
                    "resolution": "1k",
                    "face_reference_artifact_id": ref_rec.artifact_id,
                },
            )
            self.assertEqual(params["source_identity_artifact_id"], "art-01e71e31ecaf4f0d")
            self.assertEqual(params["reference_artifact_ids"], ["art-01e71e31ecaf4f0d"])

            task = tasks.create(
                project_id="proj-moss",
                idempotency_key="idem-moss-face-1",
                engine="moss-image",
                model=DEFAULT_MOSS_MODEL,
                request_params=params,
            )

            async def fake_faceswap(**kwargs):
                return {
                    "status": "completed",
                    "artifact": fused_rec.to_dict(),
                    "metrics": {"arcface_similarity": 0.912},
                }

            with patch.object(
                moss_provider,
                "generate",
                return_value=(
                    png_bytes,
                    {
                        "engine": "moss-image",
                        "model": DEFAULT_MOSS_MODEL,
                        "width": 768,
                        "height": 1344,
                        "resolved_seed": 888,
                        "steps": 50,
                        "elapsed_ms": 42000,
                    },
                ),
            ), patch("src.faceswap.handle_faceswap", side_effect=fake_faceswap) as mock_fs:
                qm._execute_moss(
                    TaskJob(
                        task_id=task.task_id,
                        project_id="proj-moss",
                        engine="moss-image",
                        action="generate",
                        params=params,
                    )
                )
                mock_fs.assert_called_once()

            completed = tasks.get(task.task_id)
            self.assertIsNotNone(completed.artifact_id)
            art = artifacts.get(completed.artifact_id, "proj-moss")
            self.assertEqual(art.metadata["source_identity_artifact_id"], "art-01e71e31ecaf4f0d")
            self.assertEqual(art.metadata["face_reference_artifact_id"], "art-01e71e31ecaf4f0d")
            self.assertEqual(art.metadata["reference_artifact_ids"], ["art-01e71e31ecaf4f0d"])
            self.assertIn("stage1_artifact_id", art.metadata)


if __name__ == "__main__":
    unittest.main()

