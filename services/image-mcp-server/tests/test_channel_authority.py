import json
import ssl
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

import httpx

from src.channel_authority import CALLER, authorize, create_client


class ChannelAuthorityTests(unittest.TestCase):
    def test_tls_client_keeps_certificate_and_hostname_verification(self):
        contexts = []
        factory = ssl.create_default_context

        def context():
            value = factory()
            contexts.append(value)
            return value

        with patch.dict('os.environ', {'IMAGE_PROJECT_CA_FILE': '', 'IMAGE_PROJECT_CA_PEM': '', 'HTTPS_PROXY': 'http://127.0.0.1:1'}):
            with patch('src.channel_authority.ssl.create_default_context', side_effect=context):
                with create_client() as client:
                    self.assertFalse(client.trust_env)
                    self.assertFalse(client.follow_redirects)
        self.assertTrue(contexts[0].check_hostname)
        self.assertEqual(contexts[0].verify_mode, ssl.CERT_REQUIRED)

    def test_invalid_ca_fails_without_unverified_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'ca.pem'
            for exists in (False, True):
                if exists:
                    path.write_text('not a certificate')
                with self.subTest(exists=exists), patch.dict('os.environ', {'IMAGE_PROJECT_CA_FILE': str(path)}):
                    with self.assertRaises(OSError):
                        create_client()

    def test_invalid_inline_ca_fails_closed(self):
        with patch.dict('os.environ', {'IMAGE_PROJECT_CA_FILE': '', 'IMAGE_PROJECT_CA_PEM': 'invalid PEM'}):
            with self.assertRaises(ssl.SSLError):
                create_client()

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
