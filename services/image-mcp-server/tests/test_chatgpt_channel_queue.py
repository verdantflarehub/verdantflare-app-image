import base64
import asyncio
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from PIL import Image
from src.artifacts import ArtifactStore
from src.providers.channel import ChannelUnavailable
from src.providers.codex import DEFAULT_IMAGE_MODEL
from src.queue import TaskJob, TaskQueueManager
from src.tasks import TaskStore


class ChatGPTChannelQueueTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        self.tasks = TaskStore(root / 'tasks.db')
        self.artifacts = ArtifactStore(root / 'artifacts')
        self.channel = Mock()
        self.authorize = Mock(return_value={'organization_id': 'org', 'user_id': 'user'})
        self.legacy = Mock()
        self.queue = TaskQueueManager(self.tasks, self.artifacts, self.legacy, Mock(),
            google_authorize=self.authorize, chatgpt_channel=self.channel)
        self.channel.submit.return_value = {'accepted': True, 'state': 'completed', 'call_id': 'call'}
        self.channel.recover.return_value = self.channel.submit.return_value
        output = io.BytesIO()
        Image.new('RGB', (2, 3), 'red').save(output, format='PNG')
        self.image = output.getvalue()
        self.channel.result.return_value = (self.image, 'image/png')

    def job(self, **changes):
        params = self.queue.prepare('codex', 'project-test', {
            'action': 'generate', 'prompt': 'Original prompt', 'budget_limit': '1', **changes})
        record = self.tasks.create('project-test', 'operation', 'codex', params)
        self.tasks.claim_task(record.task_id)
        return TaskJob(record.task_id, 'project-test', 'codex', params['action'], params)

    def test_generation_retains_model_size_quality_and_instructions(self):
        job = self.job(aspect_ratio='9:16', resolution='4k', quality='hd',
            background='transparent', instructions='Keep wording')
        self.assertEqual(job.params['channel_runtime'], 'chatgpt-channel-v1')
        self.queue._execute_channel(job)
        args = self.channel.submit.call_args.args
        self.assertEqual(args[:4], ('org', job.task_id, job.task_id, DEFAULT_IMAGE_MODEL))
        self.assertEqual(args[4], {'prompt': 'Original prompt', 'action': 'generate', 'images': [],
            'size': '1152x2048', 'prefer_4k': True, 'quality': 'hd',
            'background': 'transparent', 'instructions': 'Keep wording'})
        self.legacy.generate.assert_not_called()
        record = self.tasks.get(job.task_id)
        self.assertEqual(record.status, 'completed')
        self.assertEqual(self.artifacts.get(record.artifact_id, 'project-test').metadata['channel'], 'chatgpt')

    def test_edit_keeps_mask_separate_and_reference_order(self):
        sources = [self.artifacts.create_from_bytes('project-test', name + '.png', self.image)
                   for name in ('source', 'reference', 'mask')]
        job = self.job(action='edit', source_artifact_id=sources[0].artifact_id,
            reference_artifact_ids=[sources[1].artifact_id], mask_artifact_id=sources[2].artifact_id,
            size='1536x1024', resolution='4k', model='gpt-image-2')
        self.queue._execute_channel(job)
        args = self.channel.submit.call_args.args
        self.assertEqual(args[3], 'gpt-image-2')
        self.assertEqual(len(args[4]['images']), 2)
        self.assertEqual(base64.b64decode(args[4]['mask']['data']), self.image)
        self.assertEqual(args[4]['size'], '1536x1024')
        self.assertEqual(args[4]['quality'], 'auto')
        self.assertTrue(args[4]['prefer_4k'])
        self.legacy.edit.assert_not_called()

    def test_tampered_mask_is_rejected_before_sending(self):
        source = self.artifacts.create_from_bytes('project-test', 'source.png', self.image)
        mask = self.artifacts.create_from_bytes('project-test', 'mask.png', self.image)
        job = self.job(action='edit', source_artifact_id=source.artifact_id, mask_artifact_id=mask.artifact_id)
        self.artifacts.content_path(mask).write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'source_integrity_failed'):
            self.queue._execute_channel(job)
        self.channel.submit.assert_not_called()

    def test_inpaint_uses_masked_edit_without_rewriting_task(self):
        source = self.artifacts.create_from_bytes('project-test', 'source.png', self.image)
        mask = self.artifacts.create_from_bytes('project-test', 'mask.png', self.image)
        job = self.job(action='inpaint', source_artifact_id=source.artifact_id,
            mask_artifact_id=mask.artifact_id)
        self.queue._execute_channel(job)
        payload = self.channel.submit.call_args.args[4]
        self.assertEqual(payload['action'], 'edit')
        self.assertEqual(len(payload['images']), 1)
        self.assertEqual(base64.b64decode(payload['mask']['data']), self.image)
        self.assertEqual(self.tasks.get(job.task_id).request_params['action'], 'inpaint')
        self.assertEqual(self.tasks.get(job.task_id).status, 'completed')
        self.legacy.edit.assert_not_called()

    def test_restart_and_missing_channel_never_resubmit(self):
        job = self.job()
        self.channel.submit.return_value = {'accepted': None, 'state': 'acceptance_unknown'}
        self.queue._execute_channel(job)
        self.assertEqual(self.tasks.recover_hanging_tasks(), 0)
        self.channel.recover.side_effect = ChannelUnavailable('not_found')
        with self.assertRaises(ChannelUnavailable):
            self.queue._execute_channel(job, recovery=True)
        self.queue.chatgpt_channel = None
        with self.assertRaises(ChannelUnavailable):
            self.queue._execute_channel(job, recovery=True)
        self.channel.submit.assert_called_once()
        self.legacy.generate.assert_not_called()
        self.assertEqual(self.tasks.get(job.task_id).status, 'running')

    def test_completed_collection_is_idempotent(self):
        job = self.job()
        self.queue._execute_channel(job)
        first = self.tasks.get(job.task_id).artifact_id
        self.tasks.update_status(job.task_id, 'running', phase='collecting')
        self.queue._execute_channel(job, recovery=True)
        self.assertEqual(self.tasks.get(job.task_id).artifact_id, first)
        self.channel.submit.assert_called_once()

    def test_authority_is_verified_and_never_injected(self):
        with self.assertRaisesRegex(ValueError, 'reserved_channel_parameter'):
            self.job(channel_authority={'organization_id': 'foreign'})
        job = self.job()
        self.authorize.return_value = {'organization_id': 'foreign', 'user_id': 'user'}
        with self.assertRaises(ChannelUnavailable):
            self.queue._execute_channel(job)
        self.channel.submit.assert_not_called()

    def test_legacy_task_route_is_not_rewritten(self):
        self.queue.chatgpt_channel = None
        params = self.queue.prepare('codex', 'project-test', {'action': 'generate', 'prompt': 'test'})
        self.assertNotIn('channel_runtime', params)
        record = self.tasks.create('project-test', 'legacy', 'codex', params)
        self.tasks.claim_task(record.task_id)
        self.assertEqual(self.tasks.recover_hanging_tasks(), 1)
        self.assertEqual(self.tasks.get(record.task_id).status, 'running')
        self.assertEqual(self.tasks.get(record.task_id).phase, 'submission_unknown')
        self.assertFalse(self.tasks.claim_task(record.task_id))

    def test_capabilities_distinguish_channel_requirements_from_legacy(self):
        engines = {e['engine']: e for e in self.queue.capabilities()['engines']}
        self.assertEqual(engines['codex']['channel'], 'chatgpt')
        self.assertTrue(engines['codex']['budget_limit_required'])
        self.assertEqual(engines['codex']['budget_limit_kind'], 'reservation_ceiling')
        self.assertEqual(engines['codex']['budget_currency'], 'USD')
        self.assertNotIn('budget_limit_required', engines['gemini'])
        self.queue.chatgpt_channel = None
        legacy = self.queue.capabilities()['engines'][0]
        self.assertNotIn('channel', legacy)
        self.assertNotIn('budget_limit_required', legacy)


class ChannelWorkerFailureTests(unittest.IsolatedAsyncioTestCase):
    async def test_failures_preserve_submission_boundary(self):
        for failure in ('prepare', 'submit', 'collect'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                tasks = TaskStore(root / 'tasks.db')
                channel = Mock()
                authority = {'organization_id': 'org', 'user_id': 'user'}
                authorize = Mock(return_value=authority)
                queue = TaskQueueManager(tasks, ArtifactStore(root / 'artifacts'), Mock(), Mock(),
                    chatgpt_channel=channel, google_authorize=authorize)
                params = queue.prepare('codex', 'project-test', {
                    'action': 'generate', 'prompt': 'test', 'budget_limit': '1'})
                record = tasks.create('project-test', 'operation', 'codex', params)
                if failure == 'prepare':
                    authorize.side_effect = ValueError('project_access_denied')
                elif failure == 'submit':
                    channel.submit.side_effect = ChannelUnavailable('connection_lost')
                else:
                    channel.submit.return_value = {'accepted': True, 'state': 'completed', 'call_id': 'call'}
                    channel.result.side_effect = ChannelUnavailable('result_unavailable')
                queue._running = True
                worker = asyncio.create_task(queue._worker_loop(0))
                try:
                    await queue.enqueue(record.task_id, record.project_id, record.engine, 'generate', params)
                    await asyncio.wait_for(queue._queue.join(), timeout=5)
                finally:
                    queue._running = False
                    worker.cancel()
                    await asyncio.gather(worker, return_exceptions=True)
                result = tasks.get(record.task_id)
                if failure == 'prepare':
                    self.assertEqual((result.status, result.phase), ('failed', 'failed'))
                    self.assertEqual(result.error, 'channel_preparation_failed')
                    channel.submit.assert_not_called()
                else:
                    self.assertEqual(result.status, 'running')
                    self.assertEqual(result.phase, 'collecting' if failure == 'collect' else 'submission_unknown')
                    channel.recover.side_effect = ChannelUnavailable('not_available')
                    with self.assertRaises(ChannelUnavailable):
                        queue._execute_channel(TaskJob(record.task_id, record.project_id,
                            record.engine, 'generate', params), recovery=True)
                    channel.submit.assert_called_once()
