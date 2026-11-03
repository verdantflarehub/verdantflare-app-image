"""Dashboard editing must preserve the mask through the actual channel boundary."""
import base64
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, Mock, patch

from PIL import Image
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

from src.artifacts import ArtifactStore
from src.dashboard import api_create_task
from src.queue import TaskJob, TaskQueueManager
from src.tasks import TaskStore


class DashboardEditInputsTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        self.tasks = TaskStore(root / 'tasks.db')
        self.artifacts = ArtifactStore(root / 'artifacts')
        self.channel = Mock()
        self.queue = TaskQueueManager(self.tasks, self.artifacts, Mock(), Mock(),
            google_authorize=Mock(return_value={'organization_id': 'org', 'user_id': 'user'}),
            chatgpt_channel=self.channel)
        self.queue.enqueue = AsyncMock()
        app = Starlette(routes=[Route('/api/tasks', api_create_task, methods=['POST'])])
        app.state.tasks, app.state.queue_manager = self.tasks, self.queue
        self.client = TestClient(app)
        self.addCleanup(self.client.close)
        env = patch.dict(os.environ, {'IMAGE_MCP_BEARER_TOKEN': 'test-only'})
        env.start()
        self.addCleanup(env.stop)
        self.raw, self.ids = [], []
        for i, color in enumerate(('red', 'green', 'blue', 'white')):
            buf = io.BytesIO()
            Image.new('RGBA', (16, 16), color).save(buf, format='PNG')
            raw = buf.getvalue()
            self.raw.append(raw)
            self.ids.append(self.artifacts.create_from_bytes('project-test', f'{i}.png', raw).artifact_id)
        self.body = {'project_id': 'project-test', 'idempotency_key': 'operation',
            'prompt': 'Preserve the scene outside the edit region', 'engine': 'codex',
            'action': 'edit', 'source_artifact_id': self.ids[0],
            'reference_artifact_ids': self.ids[1:3], 'mask_artifact_id': self.ids[3],
            'budget_limit': '1'}

    def post(self, **changes):
        return self.client.post('/api/tasks', json=self.body | changes,
                                headers={'Authorization': 'Bearer test-only'})

    def test_mask_and_order_reach_channel_bytes(self):
        response = self.post()
        self.assertEqual(response.status_code, 201)
        record = self.tasks.get(response.json()['task_id'])
        self.assertEqual(record.request_params.get('mask_artifact_id'), self.ids[3])
        self.assertEqual(record.request_params['reference_artifact_ids'], self.ids[1:3])
        self.channel.submit.return_value = {'accepted': True, 'state': 'completed', 'call_id': 'call'}
        self.channel.result.return_value = (self.raw[0], 'image/png')
        self.tasks.claim_task(record.task_id)
        self.queue._execute_channel(TaskJob(record.task_id, record.project_id, record.engine,
                                           'edit', record.request_params))
        payload = self.channel.submit.call_args.args[4]
        self.assertEqual(base64.b64decode(payload['mask']['data']), self.raw[3])
        self.assertEqual([base64.b64decode(x['data']) for x in payload['images']], self.raw[:3])
        self.assertEqual(self.tasks.get(record.task_id).status, 'completed')

    def test_unsupported_mask_rejected_before_task_creation(self):
        create = Mock(wraps=self.tasks.create)
        self.tasks.create = create
        response = self.post(engine='gemini')
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()['error'], 'unsupported_parameter')
        create.assert_not_called()
        self.queue.enqueue.assert_not_awaited()
        self.channel.submit.assert_not_called()

    def test_invalid_mask_id_rejected_before_task_creation(self):
        create = Mock(wraps=self.tasks.create)
        self.tasks.create = create
        for value in ('', '   ', 7, [], {}, True):
            with self.subTest(value=value):
                response = self.post(mask_artifact_id=value)
                self.assertEqual(response.status_code, 400)
                self.assertEqual(response.json()['error'], 'invalid_mask_artifact_id')
        create.assert_not_called()
        self.queue.enqueue.assert_not_awaited()

    def test_null_and_omitted_mask_remain_unmasked(self):
        for i, null in enumerate((True, False)):
            with self.subTest(null=null):
                if not null:
                    self.body.pop('mask_artifact_id')
                response = self.post(**({'mask_artifact_id': None} if null else {}),
                                     idempotency_key=f'unmasked-{i}')
                self.assertEqual(response.status_code, 201)
                record = self.tasks.get(response.json()['task_id'])
                self.assertFalse(record.request_params.get('mask_artifact_id'))


if __name__ == '__main__':
    unittest.main()
