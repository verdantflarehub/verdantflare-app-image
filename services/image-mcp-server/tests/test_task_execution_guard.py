import concurrent.futures
import tempfile
import unittest
from pathlib import Path

from src.tasks import TaskConflict, TaskStore


class TaskExecutionGuardTests(unittest.TestCase):
    def test_each_engine_replays_only_identical_input(self):
        with tempfile.TemporaryDirectory() as directory:
            store = TaskStore(Path(directory) / "tasks.db")
            for engine in ("codex", "gemini", "grok"):
                with self.subTest(engine=engine):
                    params = {"prompt": "sample", "model": "explicit-model"}
                    first = store.create("test-project", engine, engine, params)
                    replay = store.create("test-project", engine, engine, dict(params))
                    self.assertEqual(first.task_id, replay.task_id)
                    for changes in ({"prompt": "different"}, {"model": "different"},
                                    {"quality": "higher"}):
                        with self.assertRaises(TaskConflict):
                            store.create("test-project", engine, engine, params | changes)
                    with self.assertRaises(TaskConflict):
                        store.create("test-project", engine, "other-engine", params)
                    with self.assertRaises(TaskConflict):
                        store.create("test-project", engine, engine, params,
                                     model="different-model")

    def test_concurrent_workers_and_restart_never_reclaim_task(self):
        with tempfile.TemporaryDirectory() as directory:
            store = TaskStore(Path(directory) / "tasks.db")
            for engine in ("codex", "gemini", "grok"):
                with self.subTest(engine=engine):
                    task = store.create("test-project", engine, engine, {"prompt": "sample"})
                    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
                        claims = list(pool.map(lambda _: store.claim_task(task.task_id), range(24)))
                    self.assertEqual(sum(claims), 1)
                    self.assertFalse(TaskStore(store.db_path).claim_task(task.task_id))
                    store.update_status(task.task_id, "completed")
                    self.assertFalse(store.claim_task(task.task_id))


if __name__ == "__main__":
    unittest.main()
