import tempfile
import unittest
from pathlib import Path
from src.tasks import TaskNotFound, TaskStore


class TestTasks(unittest.TestCase):
    def test_sqlite_task_lifecycle_and_idempotency(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "tasks.db"
            store = TaskStore(db_path=db_path)
            store.ensure_ready()

            # 1. 创建任务
            t1 = store.create(
                project_id="proj-a",
                idempotency_key="unit-1/attempt-1",
                engine="gemini",
                request_params={"prompt": "cyberpunk city", "model": "gemini-3.1-flash-image"},
                model="gemini-3.1-flash-image",
            )
            self.assertTrue(t1.task_id.startswith("imgtask-"))
            self.assertEqual(t1.status, "queued")
            self.assertEqual(t1.model, "gemini-3.1-flash-image")
            self.assertEqual(t1.prompt_preview, "cyberpunk city")

            # 2. 幂等查重测试
            t2 = store.create(
                project_id="proj-a",
                idempotency_key="unit-1/attempt-1",
                engine="gemini",
                request_params={"prompt": "cyberpunk city"},
            )
            self.assertEqual(t2.task_id, t1.task_id)

            # 3. 更新为运行中，记录耗时和结果
            store.update_status(t1.task_id, "running")
            running = store.get(t1.task_id)
            self.assertEqual(running.status, "running")

            updated = store.update_status(
                t1.task_id,
                status="completed",
                artifact_id="art-999",
                duration_seconds=12.34,
            )
            self.assertEqual(updated.status, "completed")
            self.assertEqual(updated.artifact_id, "art-999")
            self.assertAlmostEqual(updated.duration_seconds, 12.34, places=2)

            # 4. 创建第二个任务并测试列表与统计
            t3 = store.create(
                project_id="proj-b",
                idempotency_key="unit-2/attempt-1",
                engine="codex",
                request_params={"prompt": "anime character"},
                model="gpt-image-2.5-sunburst",
            )
            store.update_status(t3.task_id, "failed", error="upstream timeout", duration_seconds=5.0)

            # 5. 测试统计指标
            stats = store.get_stats()
            self.assertEqual(stats["total"], 2)
            self.assertEqual(stats["completed"], 1)
            self.assertEqual(stats["failed"], 1)
            self.assertEqual(stats["queued"], 0)
            self.assertAlmostEqual(stats["avg_duration"], 12.34, places=2)

            # 6. 测试列表过滤与分页
            records, total = store.list_tasks(status="completed")
            self.assertEqual(total, 1)
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].task_id, t1.task_id)

            records_proj, total_proj = store.list_tasks(project_id="proj-b")
            self.assertEqual(total_proj, 1)
            self.assertEqual(records_proj[0].engine, "codex")

            # 7. 测试不存在抛出
            with self.assertRaises(TaskNotFound):
                store.get("imgtask-non-existent")

    def test_recover_hanging_tasks(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "tasks.db"
            store = TaskStore(db_path=db_path)
            store.ensure_ready()

            t = store.create(
                project_id="proj-hang",
                idempotency_key="hang-1",
                engine="gemini",
                request_params={"prompt": "test hang"},
            )
            store.update_status(t.task_id, "running")

            # 模拟服务重启恢复
            recovered_count = store.recover_hanging_tasks()
            self.assertEqual(recovered_count, 1)

            recovered_task = store.get(t.task_id)
            self.assertEqual(recovered_task.status, "failed")
            self.assertIn("服务重启", recovered_task.error)


class TestCodexProvider(unittest.TestCase):
    def test_codex_spec_normalization_and_4k_resolution(self):
        from src.providers.codex import (
            CodexProvider,
            normalize_background,
            normalize_quality,
        )

        # 1. 质量档位规范化测试
        self.assertEqual(normalize_quality("auto"), "auto")
        self.assertEqual(normalize_quality("high"), "high")
        self.assertEqual(normalize_quality("xhigh"), "xhigh")
        self.assertEqual(normalize_quality("max"), "max")
        self.assertEqual(normalize_quality("medium"), "medium")
        self.assertEqual(normalize_quality("low"), "low")
        self.assertEqual(normalize_quality("hd"), "high")  # 兼容映射
        self.assertEqual(normalize_quality("standard"), "medium")  # 兼容映射
        self.assertEqual(normalize_quality(None), "auto")

        # 2. 背景模式规范化测试
        self.assertEqual(normalize_background("transparent"), "transparent")
        self.assertEqual(normalize_background("opaque"), "opaque")
        self.assertEqual(normalize_background("auto"), "auto")
        self.assertEqual(normalize_background(None), "auto")

        # 3. 4K 官方规格解析测试（单边必须 <= 3840，必须为 16 的整倍数）
        provider = CodexProvider(api_key="test-key")
        test_sizes = [
            ("2048x1152", "3840x2160"),
            ("1152x2048", "2160x3840"),
            ("1024x1024", "2048x2048"),
            ("1792x1344", "2880x2160"),
            ("1344x1792", "2160x2880"),
            ("1536x1024", "3072x2048"),
            ("1024x1536", "2048x3072"),
        ]
        for base, expected in test_sizes:
            res = provider._resolve_target_size(base, prefer_4k=True)
            self.assertEqual(res, expected)
            w, h = map(int, res.split("x"))
            self.assertLessEqual(w, 3840)
            self.assertLessEqual(h, 3840)
            self.assertEqual(w % 16, 0)
            self.assertEqual(h % 16, 0)


if __name__ == "__main__":
    unittest.main()

