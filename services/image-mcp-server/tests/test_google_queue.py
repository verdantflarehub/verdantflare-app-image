import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from src.artifacts import ArtifactStore
from src.providers.google_channel import ChannelUnavailable
from src.queue import TaskJob, TaskQueueManager
from src.tasks import TaskStore


class GoogleQueueTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        self.store = TaskStore(root / 'tasks.db')
        self.channel = Mock()
        self.authorize = Mock(return_value={'organization_id': 'org'})
        self.queue = TaskQueueManager(self.store, ArtifactStore(root / 'artifacts'),
            Mock(), Mock(), self.channel, self.authorize)
        self.params = {'action': 'generate', 'prompt': 'test',
            'channel_runtime': 'google-channel-v1',
            'channel_authority': {'organization_id': 'org', 'user_id': 'user'},
            'budget_limit': '1'}
        self.record = self.store.create('project-test', 'operation', 'gemini', self.params)
        self.job = TaskJob(self.record.task_id, 'project-test', 'gemini', 'generate', self.params)
        self.store.claim_task(self.record.task_id)

    def test_restart_preserves_unknown_and_only_reads(self):
        self.channel.submit.return_value = {'accepted': None, 'state': 'acceptance_unknown'}
        self.queue._execute_google(self.job)
        self.assertEqual(self.store.recover_hanging_tasks(), 0)
        self.channel.recover.side_effect = ChannelUnavailable('not_found')
        with self.assertRaises(ChannelUnavailable):
            self.queue._execute_google(self.job, recovery=True)
        self.channel.submit.assert_called_once()
        self.channel.recover.assert_called_once_with('org', self.record.task_id, self.record.task_id)
        self.assertEqual(self.store.get(self.record.task_id).status, 'running')

    def test_crash_after_claim_before_submit_is_read_only(self):
        self.assertEqual(self.store.recover_hanging_tasks(), 0)
        self.channel.recover.side_effect = ChannelUnavailable('not_found')
        with self.assertRaises(ChannelUnavailable):
            self.queue._execute_google(self.job, recovery=True)
        self.channel.submit.assert_not_called()

    def test_completed_collection_reuses_artifact_after_crash(self):
        self.channel.submit.return_value = {'accepted': True, 'state': 'completed', 'call_id': 'call'}
        self.channel.recover.return_value = self.channel.submit.return_value
        self.channel.result.return_value = (b'validated-by-transport', 'image/png')
        self.queue._execute_google(self.job)
        first = self.store.get(self.record.task_id)
        self.store.update_status(first.task_id, 'running', phase='collecting')
        self.queue._execute_google(self.job, recovery=True)
        self.assertEqual(self.store.get(first.task_id).artifact_id, first.artifact_id)
        self.assertEqual(len(list(self.queue.artifacts.root_dir.rglob('*.png'))), 1)
        self.channel.submit.assert_called_once()

    def test_authority_mismatch_does_not_call_channel(self):
        self.authorize.return_value = {'organization_id': 'other'}
        with self.assertRaises(ChannelUnavailable):
            self.queue._execute_google(self.job)
        self.channel.submit.assert_not_called()
        self.channel.recover.assert_not_called()

    def test_rejected_call_is_failed(self):
        self.channel.submit.return_value = {'accepted': False, 'state': 'rejected'}
        self.queue._execute_google(self.job)
        self.assertEqual(self.store.get(self.record.task_id).status, 'failed')

    def test_prepare_uses_authorizer_and_rejects_supplied_authority(self):
        params = {'action': 'generate', 'prompt': 'test', 'budget_limit': '1'}
        prepared = self.queue.prepare('gemini', 'project-test', params)
        self.assertEqual(prepared['channel_runtime'], 'google-channel-v1')
        self.authorize.assert_called_once_with('project-test', None)
        with self.assertRaises(ValueError):
            self.queue.prepare('gemini', 'project-test', params | {'channel_authority': {'organization_id': 'other'}})
        for amount in (None, 1, '-1', 'NaN', 'Infinity', '0', '1e100'):
            with self.subTest(amount=amount), self.assertRaises(ValueError):
                self.queue.prepare('gemini', 'project-test', params | {'budget_limit': amount})

    def test_google_channel_forwards_aspect_ratio_and_reference_artifacts(self):
        ref = self.queue.artifacts.create_from_bytes(
            project_id='project-test',
            filename='ref.png',
            data=b'\x89PNG\r\n\x1a\nref-bytes',
            media_type='image/png',
        )
        params = {
            'action': 'generate',
            'prompt': 'vertical portrait with identity reference',
            'aspect_ratio': '9:16',
            'reference_artifact_ids': [ref.artifact_id],
            'channel_runtime': 'google-channel-v1',
            'channel_authority': {'organization_id': 'org', 'user_id': 'user'},
            'budget_limit': '1',
        }
        rec = self.store.create('project-test', 'op-ref-ar', 'gemini', params)
        self.store.claim_task(rec.task_id)
        self.channel.submit.return_value = {'accepted': True, 'state': 'completed', 'call_id': 'call-ref'}
        self.channel.result.return_value = (b'\x89PNG\r\n\x1a\nout', 'image/png')

        job = TaskJob(rec.task_id, 'project-test', 'gemini', 'generate', params)
        self.queue._execute_google(job)

        self.channel.submit.assert_called_once()
        submitted_payload = self.channel.submit.call_args.args[4]
        self.assertEqual(submitted_payload['aspect_ratio'], '9:16')
        self.assertEqual(len(submitted_payload['images']), 1)
        self.assertEqual(submitted_payload['images'][0]['mime_type'], 'image/png')

    def test_supported_gemini_models_allowed_and_unsupported_rejected(self):
        valid_models = {
            'gemini-3-pro-image-preview': 'gemini-3-pro-image-preview',
            'gemini-3.1-flash-image-preview': 'gemini-3.1-flash-image-preview',
            'gemini-3.1-flash-image': 'gemini-3.1-flash-image',
            'gemini-3-pro-image': 'gemini-3-pro-image-preview',
        }
        for m, expected in valid_models.items():
            with self.subTest(model=m):
                params = {'action': 'generate', 'prompt': 'test', 'budget_limit': '1', 'model': m}
                prepared = self.queue.prepare('gemini', 'project-test', params)
                self.assertEqual(prepared['model'], expected)

        invalid_models = ['imagen-4-ultra', 'imagen-4.0-ultra-generate-001', 'gemini-2.5-flash-image', 'gemini-unknown-model', 'gpt-4o']
        for m in invalid_models:
            with self.subTest(invalid_model=m), self.assertRaises(ValueError):
                self.queue.prepare('gemini', 'project-test', {'action': 'generate', 'prompt': 'test', 'budget_limit': '1', 'model': m})

    def test_capabilities_declares_supported_gemini_models(self):
        caps = self.queue.capabilities()
        gemini_cap = next(e for e in caps['engines'] if e.get('engine') == 'gemini')
        self.assertEqual(
            gemini_cap.get('supported_models'),
            ['gemini-3-pro-image-preview', 'gemini-3.1-flash-image-preview'],
        )


if __name__ == '__main__':
    unittest.main()
