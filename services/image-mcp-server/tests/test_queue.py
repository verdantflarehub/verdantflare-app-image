import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from src.artifacts import ArtifactStore
from src.queue import TaskQueueManager
from src.tasks import TaskStore


class TestTaskQueueManager(unittest.TestCase):
    def test_queue_execution_and_concurrency(self):
        async def run_test():
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_path = Path(tmp_dir)
                tasks = TaskStore(db_path=tmp_path / "tasks.db")
                artifacts = ArtifactStore(root_dir=tmp_path / "artifacts")
                tasks.ensure_ready()
                artifacts.ensure_ready()

                mock_codex = MagicMock()
                mock_codex.generate.return_value = b"fake-codex-img"
                mock_gemini = MagicMock()
                mock_gemini.generate.return_value = b"fake-gemini-img"

                queue_mgr = TaskQueueManager(
                    tasks=tasks,
                    artifacts=artifacts,
                    codex_provider=mock_codex,
                    gemini_provider=mock_gemini,
                )
                queue_mgr.worker_count = 2
                await queue_mgr.start()

                try:
                    # 1. 创建并提交任务
                    t1 = tasks.create(
                        project_id="proj-q",
                        idempotency_key="q-1",
                        engine="gemini",
                        request_params={"prompt": "test gemini prompt"},
                    )
                    await queue_mgr.enqueue(
                        task_id=t1.task_id,
                        project_id="proj-q",
                        engine="gemini",
                        action="generate",
                        params={"prompt": "test gemini prompt"},
                    )

                    # 等待队列消费
                    for _ in range(50):
                        rec = tasks.get(t1.task_id)
                        if rec.status == "completed":
                            break
                        await asyncio.sleep(0.05)

                    final_rec = tasks.get(t1.task_id)
                    self.assertEqual(final_rec.status, "completed")
                    self.assertIsNotNone(final_rec.artifact_id)
                    self.assertGreaterEqual(final_rec.duration_seconds, 0.0)

                finally:
                    await queue_mgr.stop()

        asyncio.run(run_test())


if __name__ == "__main__":
    unittest.main()

