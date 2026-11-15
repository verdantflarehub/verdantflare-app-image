import base64, json, tempfile, unittest, hashlib
from pathlib import Path
from unittest.mock import MagicMock, patch
import httpx
from test_moss_provider import _make_png_bytes
from src.providers.moss import MossProvider, MossProviderError
from src.queue import TaskQueueManager, TaskJob
from src.artifacts import ArtifactStore
from src.tasks import TaskStore, TaskConflict

class SceneEditTests(unittest.TestCase):

    def provider(self, client=None):
        with patch.dict('os.environ', {'MOSS_SCENE_EDIT_ENABLED': 'true'}):
            return MossProvider(client=client)

    def params(self):
        return dict(action='edit', moss_mode='edit', source_artifact_id='art-1111111111111111', reference_artifact_ids=['art-2222222222222222'], prompt='Preserve layout, age timber')

    def test_defaults_and_legacy(self):
        p = self.provider()
        n = p.normalize(self.params())
        self.assertEqual((n['action'], n['steps'], n['cfg_scale']), ('edit', 30, 4))
        self.assertNotIn('face_reference_artifact_id', n)
        old = p.normalize(dict(action='edit', source_artifact_id='art-1111111111111111', prompt='old'))
        self.assertEqual((old['action'], old['steps'], old['cfg_scale']), ('generate', 50, 3))

    def test_disabled_and_invalid_inputs(self):
        p = self.provider()
        p.scene_edit_enabled = False
        with self.assertRaises(ValueError):
            p.normalize(self.params())
        p.scene_edit_enabled = True
        for change in [{'action': 'generate'}, {'source_artifact_id': ''}, {'moss_mode': 'identity'}, {'reference_artifact_ids': ['art-1111111111111111']}, {'reference_artifact_ids': ['a', 'b']}, {'reference_artifact_ids': 'style'}, {'restore_face': False}, {'layout': {}}, {'mask_artifact_id': 'mask'}, {'source_identity_artifact_id': 'face'}, {'outfit_reference_artifact_id': 'outfit'}, {'reference_images_base64': ['raw']}]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                p.normalize(self.params() | change)

    def test_wire_and_no_fallback(self):
        captured = []

        def handle(req):
            captured.append(json.loads(req.content))
            return httpx.Response(200, json={'image_base64': base64.b64encode(_make_png_bytes()).decode(), 'prompt_mode': 'edit', 'execution': {'mode': 'edit', 'conditioning_mode': 'moss_edit_reference_latents', 'workflow_sha256': 'a' * 64, 'references': [{'input_sha256': hashlib.sha256(base64.b64decode(x)).hexdigest()} for x in captured[-1]['reference_images_base64']]}})
        with httpx.Client(transport=httpx.MockTransport(handle)) as client:
            p = self.provider(client)
            v = p.normalize(self.params())
            v['reference_images_base64'] = [base64.b64encode(x).decode() for x in [b'base', b'style']]
            (_, meta) = p.generate(v)
            self.assertTrue(captured[0]['edit'])
            self.assertEqual([base64.b64decode(x) for x in captured[0]['reference_images_base64']], [b'base', b'style'])
            self.assertEqual(captured[0]['reference_max_edge'], 1024)
            self.assertEqual(meta['conditioning_mode'], 'moss_edit_reference_latents')
        with httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, json={'prompt_mode': 'reference'}))) as client:
            with self.assertRaises(MossProviderError):
                self.provider(client).generate(v)

    def test_project_order_and_idempotency(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            art = ArtifactStore(root / 'artifacts')
            art.ensure_ready()
            tasks = TaskStore(root / 'tasks.db')
            tasks.ensure_ready()
            base = _make_png_bytes(512, 768)
            style = _make_png_bytes(768, 512)
            for (aid, data, proj) in [('art-1111111111111111', base, 'proj-scene'), ('art-2222222222222222', style, 'proj-scene'), ('art-3333333333333333', style, 'proj-other')]:
                art.create_from_bytes(project_id=proj, artifact_id=aid, filename=aid + '.png', data=data, media_type='image/png')
            auth = MagicMock(return_value={"user_id": "test-user", "organization_id": "test-org"})
            gpu = MagicMock()
            p = self.provider()
            q = TaskQueueManager(tasks, art, MagicMock(), MagicMock(), moss_provider=p, moss_gpu=gpu, google_authorize=auth)
            v = q.prepare('moss-image', 'proj-scene', self.params())
            auth.assert_called_with('proj-scene', None)
            self.assertEqual((v['width'], v['height']), (512, 768))
            with self.assertRaises(Exception):
                q.prepare('moss-image', 'proj-scene', self.params() | {'reference_artifact_ids': ['art-3333333333333333']})
            auth.side_effect = ValueError('denied')
            with self.assertRaisesRegex(ValueError, 'denied'):
                q.prepare('moss-image', 'proj-scene', self.params())
            auth.side_effect = None
            import io
            from PIL import Image
            raw = io.BytesIO()
            Image.new('RGBA', (512, 768), (1, 2, 3, 0)).save(raw, format='PNG')
            art.create_from_bytes(project_id='proj-scene', artifact_id='art-4444444444444444', filename='alpha.png', data=raw.getvalue(), media_type='image/png')
            with self.assertRaisesRegex(ValueError, 'transparent'):
                q.prepare('moss-image', 'proj-scene', self.params() | {'source_artifact_id': 'art-4444444444444444'})
            args = dict(project_id='proj-scene', idempotency_key='one', engine='moss-image', request_params=v, model=p.model)
            t = tasks.create(**args)
            self.assertEqual(t.task_id, tasks.create(**args).task_id)
            with self.assertRaises(TaskConflict):
                tasks.create(**args | {'request_params': v | {'prompt': 'changed'}})

            def generate(params):
                self.assertEqual([base64.b64decode(x) for x in params['reference_images_base64']], [base, style])
                return (base, {'prompt_mode': 'edit', 'execution': {'mode': 'edit'}})
            with patch.object(p, 'generate', side_effect=generate):
                q._execute_moss(TaskJob(t.task_id, 'proj-scene', 'moss-image', 'edit', v))
            out = art.get(tasks.get(t.task_id).artifact_id, 'proj-scene')
            self.assertEqual(out.metadata['source_artifact_id'], 'art-1111111111111111')
            self.assertNotIn('face_reference_artifact_id', out.metadata)
            self.assertEqual(auth.call_count, 5)

class SceneEditPublicContract(unittest.IsolatedAsyncioTestCase):

    async def test_tool_forwards_explicit_mode_and_legacy_omits_it(self):
        from unittest.mock import AsyncMock
        from src import server
        from types import SimpleNamespace
        schema = next((t for t in server.IMAGE_TOOLS_SCHEMA if t['name'] == 'image.edit'))['inputSchema']
        self.assertEqual(schema['properties']['moss_mode']['enum'], ['semantic', 'edit'])
        q = MagicMock()
        q.prepare.side_effect = lambda engine, project, params: params
        q.enqueue = AsyncMock()
        task = SimpleNamespace(task_id='test', status='queued', created_at='now')
        with patch.object(server, 'queue_manager', q), patch.object(server.tasks, 'create', return_value=task):
            await server.image_edit(project_id='proj-scene', idempotency_key='a', source_artifact_id='art-1111111111111111', prompt='edit timber', engine='moss-image', moss_mode='edit')
            self.assertEqual(q.prepare.call_args.args[2]['moss_mode'], 'edit')
            self.assertEqual(q.enqueue.call_args.kwargs['params']['moss_mode'], 'edit')
            await server.image_edit(project_id='proj-scene', idempotency_key='b', source_artifact_id='art-1111111111111111', prompt='legacy', engine='moss-image')
            self.assertNotIn('moss_mode', q.prepare.call_args.args[2])
            with self.assertRaises(ValueError):
                await server.image_edit(project_id='proj-scene', idempotency_key='c', source_artifact_id='art-1111111111111111', prompt='no fallback', engine='gemini', moss_mode='edit')

class SceneEditDetachedAuthorization(unittest.TestCase):
    def test_worker_reauthorizes_stored_actor_after_request_context_ends(self):
        from src.channel_authority import CALLER, authorize
        actor = {'user_id': 'test-user', 'organization_id': 'test-org'}
        requests = []
        permission = [200]

        def project(request):
            requests.append(dict(request.headers))
            return httpx.Response(permission[0], json={})

        with tempfile.TemporaryDirectory() as tmp, patch.dict('os.environ', {
            'MOSS_SCENE_EDIT_ENABLED': 'true',
            'IMAGE_PROJECT_SERVICE_URL': 'http://project',
            'IMAGE_PROJECT_SERVICE_TOKEN': 'test-internal',
        }), httpx.Client(transport=httpx.MockTransport(project)) as client:
            root = Path(tmp)
            artifacts = ArtifactStore(root / 'artifacts')
            artifacts.ensure_ready()
            tasks = TaskStore(root / 'tasks.db')
            tasks.ensure_ready()
            png = _make_png_bytes(512, 768)
            source = artifacts.create_from_bytes(project_id='proj-scene', artifact_id='art-1111111111111111',
                                                 filename='source.png', data=png, media_type='image/png')
            provider = MossProvider()
            gpu = MagicMock()
            queue = TaskQueueManager(tasks, artifacts, MagicMock(), MagicMock(), moss_provider=provider,
                                     moss_gpu=gpu, google_authorize=lambda pid, a: authorize(client, pid, a))
            token = CALLER.set(actor)
            try:
                prepared = queue.prepare('moss-image', 'proj-scene', {
                    'action': 'edit', 'moss_mode': 'edit', 'source_artifact_id': source.artifact_id,
                    'prompt': 'Keep the ship geometry, edit timber only',
                })
            finally:
                CALLER.reset(token)
            self.assertIsNone(CALLER.get())
            task = tasks.create(project_id='proj-scene', idempotency_key='detached', engine='moss-image',
                                model=provider.model, request_params=prepared)
            # Read the persisted request as a restarted/background worker would.
            job = TaskJob(task.task_id, 'proj-scene', 'moss-image', 'edit', tasks.get(task.task_id).request_params)
            with patch.object(provider, 'generate', return_value=(png, {'prompt_mode': 'edit'})) as generate:
                queue._execute_moss(job)
                generate.assert_called_once()
                self.assertEqual(requests[-1]['x-user-id'], actor['user_id'])
                permission[0] = 403
                with self.assertRaisesRegex(ValueError, 'project_access_denied'):
                    queue._execute_moss(job)
                self.assertEqual(generate.call_count, 1)
            # Caller-supplied serialized authority must never enter preparation.
            with self.assertRaisesRegex(ValueError, 'reserved_channel_parameter'):
                queue.prepare('moss-image', 'proj-scene', prepared)
