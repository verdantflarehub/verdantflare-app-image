import concurrent.futures
import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import Mock, patch

from src.artifacts import ArtifactStore
from src.queue import TaskJob, TaskQueueManager
from src.tasks import SCHEMA_SQL, TaskStore


class TaskTimingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.store = TaskStore(self.root / 'tasks.db')

    def at(self, seconds):
        clock = Mock(wraps=datetime)
        clock.now.return_value = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=seconds)
        return patch('src.tasks.datetime', clock)

    def create(self, key='operation', engine='gemini', params=None):
        with self.at(0):
            return self.store.create('project-test', key, engine, params or {'prompt': 'test'})

    def test_all_channel_completion_paths_exclude_queue_wait(self):
        for engine, name in [('gemini', 'google'), ('codex', 'chatgpt'), ('grok', 'grok')]:
            with self.subTest(channel=name):
                channel = Mock()
                channel.submit.return_value = {'accepted': True, 'state': 'completed', 'call_id': 'call'}
                channel.result.return_value = (b'transport-validated-image', 'image/png')
                params = {'action': 'generate', 'prompt': 'test', 'resolution': '1k', 'aspect_ratio': '1:1',
                          'channel_runtime': name + '-channel-v1',
                          'channel_authority': {'organization_id': 'org'}}
                task = self.create(name, engine, params)
                with self.at(10):
                    self.assertTrue(self.store.claim_task(task.task_id))
                queue = TaskQueueManager(self.store, ArtifactStore(self.root / 'artifacts'), Mock(), Mock(),
                    google_authorize=Mock(return_value={'organization_id': 'org'}),
                    **{name + '_channel': channel})
                with self.at(50):
                    queue._execute_channel(TaskJob(task.task_id, task.project_id, engine, 'generate', params))
                record = self.store.get(task.task_id)
                self.assertEqual(record.status, 'completed')
                self.assertEqual(record.duration_seconds, 40)
                self.assertEqual(queue.artifacts.get(record.artifact_id, task.project_id).metadata['task_id'], task.task_id)
                channel.submit.assert_called_once()

    def test_restart_keeps_original_start_and_recovery_only_reads(self):
        params = {'action': 'generate', 'prompt': 'test', 'channel_runtime': 'google-channel-v1',
                  'channel_authority': {'organization_id': 'org'}}
        task = self.create(params=params)
        with self.at(10):
            self.store.claim_task(task.task_id)
        with self.at(20):
            self.store.update_status(task.task_id, 'running', phase='submission_unknown')
        self.store = TaskStore(self.store.db_path)
        with self.at(30):
            self.store.recover_hanging_tasks()
            self.assertFalse(self.store.claim_task(task.task_id))
        channel = Mock()
        channel.recover.return_value = {'accepted': True, 'state': 'completed', 'call_id': 'original-call'}
        channel.result.return_value = (b'transport-validated-image', 'image/png')
        queue = TaskQueueManager(self.store, ArtifactStore(self.root / 'artifacts'), Mock(), Mock(),
            google_channel=channel, google_authorize=Mock(return_value={'organization_id': 'org'}))
        job = TaskJob(task.task_id, task.project_id, task.engine, 'generate', params)
        with self.at(50):
            queue._execute_channel(job, recovery=True)
        self.assertEqual(self.store.get(task.task_id).duration_seconds, 40)
        with self.at(80):
            queue._execute_channel(job, recovery=True)
            self.store.update_status(task.task_id, 'completed', phase='completed')
        self.assertEqual(self.store.get(task.task_id).duration_seconds, 40)
        channel.submit.assert_not_called()
        channel.recover.assert_called_once_with('org', task.task_id, task.task_id)

    def test_terminal_outcomes_and_explicit_measurements(self):
        for status in ('completed', 'failed', 'canceled'):
            task = self.create(status)
            with self.at(10):
                self.store.claim_task(task.task_id)
            with self.at(25):
                result = self.store.update_status(task.task_id, status)
            self.assertEqual(result.duration_seconds, 15)
        task = self.create('explicit')
        with self.at(10):
            self.store.claim_task(task.task_id)
        with self.at(25):
            self.store.update_status(task.task_id, 'completed', duration_seconds=12.34)
        with self.at(50):
            self.assertEqual(self.store.update_status(task.task_id, 'completed').duration_seconds, 12.34)

    def test_legacy_and_invalid_starts_remain_unknown_and_do_not_bias_average(self):
        for i, start in enumerate((None, 'broken', '2026-01-01T00:00:10', '2026-01-02T00:00:00+00:00')):
            task = self.create(str(i))
            with self.store._get_conn() as conn:
                conn.execute("UPDATE tasks SET status='running', execution_started_at=? WHERE task_id=?", (start, task.task_id))
            with self.at(50):
                result = self.store.update_status(task.task_id, 'completed')
            self.assertEqual(result.duration_seconds, 0)
        self.assertEqual(self.store.get_stats()['avg_duration'], 0)
        measured = self.create('measured')
        self.store.update_status(measured.task_id, 'completed', duration_seconds=20)
        self.assertEqual(self.store.get_stats()['avg_duration'], 20)
        self.assertEqual(self.store.get_stats()['completed'], 5)

    def test_legacy_grok_claim_has_same_start_semantics(self):
        task = self.create(engine='grok')
        with self.at(10):
            self.assertTrue(self.store.claim_grok(task.task_id))
        with self.at(30):
            self.assertFalse(self.store.claim_grok(task.task_id))
        with self.at(50):
            self.assertEqual(self.store.update_status(task.task_id, 'completed').duration_seconds, 40)

    def test_concurrent_terminal_updates_freeze_first_observation(self):
        task = self.create()
        with self.at(10):
            self.store.claim_task(task.task_id)
        clock = Mock(wraps=datetime)
        clock.now.side_effect = [datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=s)
                                for s in range(50, 66)]
        with patch('src.tasks.datetime', clock), concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(lambda _: self.store.update_status(task.task_id, 'completed'), range(16)))
        self.assertTrue(all(record.duration_seconds == 40 for record in results))
        self.assertEqual(self.store.get(task.task_id).duration_seconds, 40)

    def test_concurrent_migration_preserves_legacy_task(self):
        with sqlite3.connect(self.store.db_path) as conn:
            conn.executescript(SCHEMA_SQL)
            conn.execute("INSERT INTO tasks (task_id, project_id, idempotency_key, engine, status, created_at, updated_at, request_params, artifact_id) VALUES ('legacy', 'project-test', 'legacy', 'gemini', 'completed', 'old', 'old', '{}', 'original-artifact')")
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(lambda _: TaskStore(self.store.db_path).ensure_ready(), range(16)))
        with self.at(50):
            record = self.store.update_status('legacy', 'completed')
        self.assertEqual(record.artifact_id, 'original-artifact')
        self.assertEqual(record.duration_seconds, 0)
        self.assertNotIn('execution_started_at', record.to_dict())


if __name__ == '__main__':
    unittest.main()
