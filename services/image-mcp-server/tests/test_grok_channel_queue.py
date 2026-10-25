import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from PIL import Image
from src.artifacts import ArtifactStore
from src.providers.channel import ChannelUnavailable
from src.queue import TaskJob, TaskQueueManager
from src.tasks import TaskStore


class GrokChannelQueueTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        env = patch.dict('os.environ', {'XAI_IMAGE_ENABLED':'true','XAI_API_KEY':'','XAI_IMAGE_MODEL':'grok-imagine-image-2.0'})
        env.start(); self.addCleanup(env.stop)
        self.tasks = TaskStore(root/'tasks.db'); self.artifacts = ArtifactStore(root/'artifacts')
        self.channel = Mock(); self.authorize = Mock(return_value={'organization_id':'org','user_id':'user'})
        self.queue = TaskQueueManager(self.tasks,self.artifacts,Mock(),Mock(),google_authorize=self.authorize,grok_channel=self.channel)
        self.queue.grok_provider.generate = Mock(side_effect=AssertionError('legacy route must not run'))
        data=io.BytesIO(); Image.new('RGB',(2,2)).save(data,format='PNG'); self.image=data.getvalue()

    def job(self, **changes):
        params=self.queue.prepare('grok','project-test',{'action':'generate','prompt':'test','budget_limit':'1',**changes})
        record=self.tasks.create('project-test','operation','grok',params)
        self.tasks.claim_task(record.task_id)
        return TaskJob(record.task_id,'project-test','grok',params['action'],params)

    def test_channel_generation_uses_verified_org_without_supplier_key(self):
        job=self.job(); self.assertEqual(job.params['channel_runtime'],'grok-channel-v1')
        self.assertNotIn('endpoint_profile_id',job.params)
        self.channel.submit.return_value={'accepted':True,'state':'completed','call_id':'call'}
        self.channel.result.return_value=(self.image,'image/png')
        self.queue._execute_channel(job)
        args=self.channel.submit.call_args.args
        self.assertEqual(args[0],'org'); self.assertEqual(args[3],'grok-imagine-image-2.0')
        self.assertEqual(args[4],{'prompt':'test','action':'generate','resolution':'2k','aspect_ratio':'16:9'})
        self.assertEqual(args[6],'1'); self.assertEqual(self.tasks.get(job.task_id).status,'completed')
        self.queue.grok_provider.generate.assert_not_called()
        self.assertTrue(self.queue.capabilities()['engines'][2]['configured'])

    def test_unknown_and_missing_remote_record_never_fall_back_to_direct(self):
        job=self.job(); self.channel.submit.return_value={'accepted':None,'state':'acceptance_unknown'}
        self.queue._execute_channel(job)
        self.assertEqual(self.tasks.recover_hanging_tasks(),0)
        self.channel.recover.side_effect=ChannelUnavailable('missing')
        with self.assertRaises(ChannelUnavailable): self.queue._execute_channel(job,True)
        self.channel.submit.assert_called_once(); self.queue.grok_provider.generate.assert_not_called()
        self.assertEqual(self.tasks.get(job.task_id).status,'running')

    def test_edit_sends_one_checked_reference_and_collection_reuses_artifact(self):
        source=self.artifacts.create_from_bytes('project-test','source.png',self.image)
        job=self.job(action='edit',source_artifact_id=source.artifact_id)
        result={'accepted':True,'state':'completed','call_id':'call'}
        self.channel.submit.return_value=result; self.channel.recover.return_value=result
        self.channel.result.return_value=(self.image,'image/png')
        self.queue._execute_channel(job)
        payload=self.channel.submit.call_args.args[4]
        self.assertEqual(payload['action'],'edit'); self.assertIn('image',payload); self.assertNotIn('aspect_ratio',payload)
        first=self.tasks.get(job.task_id).artifact_id
        self.tasks.update_status(job.task_id,'running',phase='collecting')
        self.queue._execute_channel(job,True)
        self.assertEqual(self.tasks.get(job.task_id).artifact_id,first); self.channel.submit.assert_called_once()

    def test_tampered_source_never_reaches_channel(self):
        source=self.artifacts.create_from_bytes('project-test','source.png',self.image)
        job=self.job(action='edit',source_artifact_id=source.artifact_id)
        self.artifacts.content_path(source).write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'source_integrity_failed'): self.queue._execute_channel(job)
        self.channel.submit.assert_not_called()

    def test_authority_cannot_be_injected_and_is_rechecked_before_submit(self):
        with self.assertRaisesRegex(ValueError,'reserved_channel_parameter'):
            self.job(channel_authority={'organization_id':'other'})
        job=self.job(); self.authorize.return_value={'organization_id':'other'}
        with self.assertRaises(ChannelUnavailable): self.queue._execute_channel(job)
        self.channel.submit.assert_not_called()

    def test_channel_rejection_records_reason_code_in_task_error(self):
        job=self.job()
        self.channel.submit.return_value={'accepted':False,'state':'rejected','reason_code':'provider_http_401'}
        self.queue._execute_channel(job)
        rec = self.tasks.get(job.task_id)
        self.assertEqual(rec.status, 'failed')
        self.assertEqual(rec.error, 'channel_call_failed: provider_http_401')
