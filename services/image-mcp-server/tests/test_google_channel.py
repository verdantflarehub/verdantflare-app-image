import base64
import io
import json
import unittest

import httpx
from PIL import Image

from src.providers.google_channel import ChannelUnavailable, GoogleChannel


class GoogleChannelTests(unittest.TestCase):
    def adapter(self, handler):
        client = httpx.Client(transport=httpx.MockTransport(handler))
        self.addCleanup(client.close)
        return GoogleChannel('http://google:8001', 'x' * 32, client)

    def status(self, **changes):
        return dict(schema_version=1, task_id='task', call_id='call',
                    accepted=True, state='completed', **changes)

    def test_submit_timeout_and_recovery_404_never_resubmit(self):
        methods = []
        def handle(request):
            methods.append(request.method)
            self.assertEqual(request.headers['x-caller-id'], 'image-mcp')
            self.assertEqual(request.headers['x-organization-id'], 'org')
            if request.method == 'POST':
                self.assertEqual(json.loads(request.content)['capability'], 'image')
                raise httpx.ReadTimeout('secret upstream details')
            return httpx.Response(404, json={'error': 'not_found'})
        channel = self.adapter(handle)
        self.assertIsNone(channel.submit('org', 'task', 'op', 'model', {})['accepted'])
        with self.assertRaises(ChannelUnavailable):
            channel.recover('org', 'task', 'op')
        self.assertEqual(methods, ['POST', 'GET'])

    def test_malformed_acceptance_and_wrong_task_are_unknown(self):
        for changes in ({'accepted': 1}, {'task_id': 'other'}, {'accepted': 'false'}):
            value = self.status()
            value.update(changes)
            channel = self.adapter(lambda request: httpx.Response(200, json=value))
            self.assertIsNone(channel.submit('org', 'task', 'op', 'model', {})['accepted'])

    def test_valid_image_and_corruption(self):
        buffer = io.BytesIO()
        Image.new('RGB', (2, 2)).save(buffer, format='PNG')
        for data, valid in ((buffer.getvalue(), True), (b'\x89PNG\r\n\x1a\ninvalid', False)):
            value = self.status(image={'mime_type': 'image/png', 'data': base64.b64encode(data).decode()})
            channel = self.adapter(lambda request: httpx.Response(200, json=value))
            if valid:
                self.assertEqual(channel.result('org', 'task', 'call'), (data, 'image/png'))
            else:
                with self.assertRaises(ChannelUnavailable):
                    channel.result('org', 'task', 'call')

    def test_redirect_does_not_forward_authorization(self):
        seen = []
        def handle(request):
            seen.append(str(request.url))
            return httpx.Response(307, headers={'location': 'https://other.invalid/'})
        channel = self.adapter(handle)
        self.assertIsNone(channel.submit('org', 'task', 'op', 'model', {})['accepted'])
        self.assertEqual(seen, ['http://google:8001/v1/calls'])

    def test_capabilities_reads_mcp_channel_catalog(self):
        def handle(request):
            self.assertEqual(request.method, 'GET')
            self.assertEqual(request.url.path, '/v1/capabilities')
            self.assertEqual(request.headers['x-caller-id'], 'image-mcp')
            return httpx.Response(200, json={
                'schema_version': 1,
                'channel_id': 'google',
                'default_model_id': 'gemini-nano-banana-2.1',
                'capabilities': [
                    {
                        'capability_id': 'image.generate.nano_banana_2_1',
                        'type': 'image',
                        'model_id': 'gemini-nano-banana-2.1',
                        'display_name': 'Nano Banana 2.1',
                        'operation': 'image.generate',
                        'execution_state': 'enabled',
                    }
                ],
            })
        channel = self.adapter(handle)
        res = channel.capabilities('org')
        self.assertEqual(res['channel_id'], 'google')
        self.assertEqual(res['default_model_id'], 'gemini-nano-banana-2.1')
        self.assertEqual(res['capabilities'][0]['display_name'], 'Nano Banana 2.1')


if __name__ == '__main__':
    unittest.main()
