import asyncio
import base64
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, Mock

import httpx
from PIL import Image

from src.artifacts import ArtifactStore
from src.providers.grok import GrokError, GrokImageProvider
from src.queue import TaskJob, TaskQueueManager
from src.tasks import TaskStore, TaskConflict


class GrokImageTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.artifacts = ArtifactStore(self.root)
        self.tasks = TaskStore(self.root / 'tasks.db')
        self.env = patch.dict('os.environ', {'XAI_BASE_URL': 'https://grok.example/v1',
            'XAI_API_KEY': 'private-fixture', 'XAI_IMAGE_ENABLED': 'true'})
        self.env.start()
        stream = io.BytesIO()
        Image.new('RGB', (32, 18), 'green').save(stream, 'PNG')
        self.png = stream.getvalue()

    def tearDown(self):
        self.env.stop()
        self.tmp.cleanup()

    def provider(self, handler):
        return GrokImageProvider(httpx.Client(transport=httpx.MockTransport(handler)))

    def params(self, provider, **updates):
        value = dict(action='generate', prompt='Exact prompt\n保持原文', resolution='2k', aspect_ratio='16:9')
        value.update(updates)
        return provider.normalize(value, 'demo', self.artifacts)

    def test_json_generation_and_real_dimensions(self):
        def respond(r):
            self.assertEqual(r.url.path, '/v1/images/generations')
            p = json.loads(r.content)
            self.assertEqual(p['prompt'], 'Exact prompt\n保持原文')
            self.assertEqual(p['response_format'], 'b64_json')
            self.assertNotIn('quality', p)
            return httpx.Response(200, json={'data':[{'b64_json':base64.b64encode(self.png).decode()}]})
        provider = self.provider(respond)
        data, info = provider.generate(self.params(provider), 'demo', self.artifacts)
        self.assertEqual(data, self.png)
        self.assertEqual((info['width'], info['height']), (32, 18))

    def test_edit_is_json_and_checks_source_integrity(self):
        source = self.artifacts.create_from_bytes('demo', 'source.png', self.png)
        def respond(r):
            p = json.loads(r.content)
            self.assertEqual(r.url.path, '/v1/images/edits')
            self.assertTrue(p['image']['url'].startswith('data:image/png;base64,'))
            self.assertNotIn('aspect_ratio', p)
            return httpx.Response(200, json={'data':[{'b64_json':base64.b64encode(self.png).decode()}]})
        provider = self.provider(respond)
        params = self.params(provider, action='edit', source_artifact_id=source.artifact_id)
        provider.generate(params, 'demo', self.artifacts)
        self.artifacts.content_path(source).write_bytes(b'changed')
        with self.assertRaisesRegex(GrokError, 'source_integrity_failed'):
            provider.generate(params, 'demo', self.artifacts)

    def test_reject_unsupported_and_foreign_project(self):
        provider = self.provider(lambda r: self.fail('Must not submit'))
        for update in ({'quality':'high'}, {'resolution':'4k'}, {'action':'inpaint'}, {'background':'transparent'}, {'aspect_ratio':'21:9'}):
            with self.subTest(update=update), self.assertRaises(GrokError):
                self.params(provider, **update)
        source = self.artifacts.create_from_bytes('other', 'source.png', self.png)
        with self.assertRaises(Exception):
            self.params(provider, action='edit', source_artifact_id=source.artifact_id)

    def test_portrait_aspect_ratios_supported(self):
        provider = self.provider(lambda r: None)
        for ratio in ('2:3', '3:2'):
            p = self.params(provider, aspect_ratio=ratio)
            self.assertEqual(p['aspect_ratio'], ratio)

    def test_timeout_is_unknown_without_retry_or_key_leak(self):
        seen = []
        def respond(r):
            seen.append(r)
            raise httpx.ReadTimeout('private-fixture', request=r)
        provider = self.provider(respond)
        with self.assertRaises(GrokError) as failure:
            provider.generate(self.params(provider), 'demo', self.artifacts)
        self.assertTrue(failure.exception.uncertain)
        self.assertNotIn('private-fixture', str(failure.exception))
        self.assertEqual(len(seen), 1)

    def test_idempotency_and_atomic_claim(self):
        provider = self.provider(lambda r: None)
        p = self.params(provider)
        first = self.tasks.create('demo', 'one', 'grok', p)
        self.assertEqual(first.task_id, self.tasks.create('demo', 'one', 'grok', p).task_id)
        with self.assertRaises(TaskConflict):
            self.tasks.create('demo', 'one', 'grok', {**p, 'prompt':'different'})
        self.assertTrue(self.tasks.claim_grok(first.task_id))
        self.assertFalse(self.tasks.claim_grok(first.task_id))
        self.tasks.recover_hanging_tasks()
        self.assertEqual(self.tasks.get(first.task_id).status, 'running')

    def test_recover_collected_bytes_without_second_post(self):
        q = TaskQueueManager(self.tasks, self.artifacts, Mock(), Mock())
        q.grok_provider = self.provider(lambda r: httpx.Response(200, json={'data':[{'b64_json':base64.b64encode(self.png).decode()}]}))
        p = self.params(q.grok_provider)
        task = self.tasks.create('demo', 'one', 'grok', p)
        self.tasks.claim_grok(task.task_id)
        job = TaskJob(task.task_id, 'demo', 'grok', 'generate', p)
        q._execute_grok(job)
        artifact_id = self.tasks.get(task.task_id).artifact_id
        self.tasks.update_status(task.task_id, 'running', phase='collect_retry')
        q.grok_provider.generate = Mock(side_effect=AssertionError('Duplicate charge'))
        q.recover_grok()
        self.assertEqual(self.tasks.get(task.task_id).artifact_id, artifact_id)
        self.assertEqual(self.tasks.get(task.task_id).status, 'completed')

    def test_restart_unknown_does_not_generate(self):
        q = TaskQueueManager(self.tasks, self.artifacts, Mock(), Mock())
        task = self.tasks.create('demo', 'one', 'grok', self.params(q.grok_provider))
        self.tasks.claim_grok(task.task_id)
        q.grok_provider.generate = Mock(side_effect=AssertionError('Duplicate charge'))
        q.recover_grok()
        self.assertEqual(self.tasks.get(task.task_id).phase, 'submission_unknown')
