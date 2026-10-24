import hashlib
import json
from types import SimpleNamespace
import unittest
from src.execution_proof import proof

class ProofTests(unittest.TestCase):
    def setUp(self):
        self.task = SimpleNamespace(status='completed',artifact_id='out',engine='codex',
            model='gpt-image-2.5-sunburst',task_id='task1',project_id='project1',request_params={
                'channel_runtime':'chatgpt-channel-v1','prompt':'角色肖像','action':'generate',
                'aspect_ratio':'9:16','channel_authority':{'user_id':'PRIVATE'},'channel_account_id':'PRIVATE'})
        self.store=SimpleNamespace(get=lambda artifact_id,project_id: SimpleNamespace(
            project_id=project_id,sha256='a'*64,size_bytes=100,media_type='image/png'))
    def test_native_prompt_and_actual_defaults_without_secrets(self):
        p=proof(self.task,self.store)
        self.assertEqual(p['prompt_sha256'],hashlib.sha256('角色肖像'.encode()).hexdigest())
        self.assertEqual(p['parameters'],dict(action='generate',size='1152x2048',prefer_4k=False,quality='high',background='auto'))
        self.assertNotIn('PRIVATE',json.dumps(p));self.assertNotIn('download',json.dumps(p))
    def test_rejects_failed_task_and_substituted_model(self):
        for attr,value in [('status','failed'),('model','other'),('engine','gemini')]:
            old=getattr(self.task,attr);setattr(self.task,attr,value)
            with self.assertRaises(ValueError):proof(self.task,self.store)
            setattr(self.task,attr,old)
    def test_edit_input_order_and_parameters(self):
        self.task.request_params.update(action='edit',source_artifact_id='one',reference_artifact_ids=['two'])
        seen=[]
        def get(artifact_id,project_id):
            seen.append(artifact_id)
            return SimpleNamespace(project_id=project_id,sha256='b'*64,size_bytes=10,media_type='image/jpeg')
        p=proof(self.task,SimpleNamespace(get=get))
        self.assertEqual(seen,['one','two','out']);self.assertEqual(p['parameters']['quality'],'auto')

class ProofBoundaryTests(unittest.TestCase):
    def test_result_proof_rechecks_read_permission(self):
        from unittest.mock import patch
        from src import server
        task=SimpleNamespace(task_id='task1',artifact_id='out',project_id='project1')
        with patch.object(server.tasks,'get',return_value=task), patch.object(server,'_is_task_nsfw',return_value=False), \
                patch.object(server,'authorize',side_effect=ValueError('denied')) as auth, \
                patch('src.execution_proof.proof') as build:
            with self.assertRaises(ValueError):server.image_result('task1',execution_proof_only=True)
            self.assertEqual(auth.call_args.kwargs,{'require_write':False});build.assert_not_called()
