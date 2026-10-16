import json
import unittest
import uuid
from unittest.mock import patch

import httpx

from src.channel_authority import CALLER, authorize


class ChannelAuthorityTests(unittest.TestCase):
    def test_verified_actor_and_project_write_permission(self):
        def handle(request):
            self.assertEqual(json.loads(request.content), {'project_id': 'project', 'require_write': True})
            self.assertEqual(request.headers['x-organization-id'], 'org')
            self.assertEqual(uuid.UUID(request.headers['x-request-id']).version, 7)
            return httpx.Response(200, json={})
        with patch.dict('os.environ', {'IMAGE_PROJECT_SERVICE_URL': 'http://project', 'IMAGE_PROJECT_SERVICE_TOKEN': 'internal'}):
            with httpx.Client(transport=httpx.MockTransport(handle)) as client:
                context = CALLER.set({'user_id': 'user', 'organization_id': 'org'})
                try:
                    self.assertEqual(authorize(client, 'project'), CALLER.get())
                finally:
                    CALLER.reset(context)

    def test_no_identity_no_request(self):
        def unexpected(request):
            self.fail('unverified request reached Project')
        with httpx.Client(transport=httpx.MockTransport(unexpected)) as client:
            with self.assertRaisesRegex(ValueError, 'verified_studio_identity_required'):
                authorize(client, 'project')

    def test_denied_and_redirect_never_authorize(self):
        for status in (401, 403, 404, 307, 500):
            with self.subTest(status=status), patch.dict('os.environ', {
                'IMAGE_PROJECT_SERVICE_URL': 'http://project', 'IMAGE_PROJECT_SERVICE_TOKEN': 'internal'}):
                with httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(status))) as client:
                    with self.assertRaises(ValueError):
                        authorize(client, 'project', {'user_id': 'user', 'organization_id': 'org'})
