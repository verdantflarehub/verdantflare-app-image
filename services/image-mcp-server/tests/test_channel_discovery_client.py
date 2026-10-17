import json
import unittest
import asyncio
from unittest.mock import patch

import httpx

from src.channel_discovery_client import ChannelDiscoveryClient, DiscoveryUnavailable, encode


class DiscoveryClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_configuration_is_optional_but_partial_configuration_is_rejected(self):
        with patch.dict('os.environ', {}, clear=True):
            self.assertIsNone(ChannelDiscoveryClient.from_environment())
        with patch.dict('os.environ', {'CHANNEL_DISCOVERY_SCOPE_ID': 'test'}, clear=True):
            with self.assertRaisesRegex(ValueError, 'invalid_discovery_configuration'):
                ChannelDiscoveryClient.from_environment()

    async def test_stop_cancels_pending_watch_and_closes_transport(self):
        entered = asyncio.Event()
        class Stream(httpx.AsyncByteStream):
            async def __aiter__(self):
                entered.set()
                await asyncio.Event().wait()
                yield b''
        def handler(request):
            if request.url.path.endswith('authenticate'):
                return httpx.Response(200, json={'token': 'synthetic-token'})
            if request.url.path.endswith('/range'):
                return httpx.Response(200, json={'header': {'revision': '10'}})
            return httpx.Response(200, stream=Stream())
        client = self.client(handler)
        client.start()
        await asyncio.wait_for(entered.wait(), 1)
        await asyncio.wait_for(client.stop(), 1)
        self.assertIsNone(client._task)
        self.assertTrue(client._client.is_closed)

    def client(self, handler):
        client = ChannelDiscoveryClient('http://etcd.test.svc.cluster.local:2379', 'test',
            'synthetic-reader', 'synthetic-password', {}, transport=httpx.MockTransport(handler))
        self.addAsyncCleanup(client.stop)
        return client

    def row(self, suffix='google/status', revision=10):
        return {'key': encode('/verdantflare/channels/v1/test/' + suffix),
                'value': encode(json.dumps({'state': 'disabled'})),
                'mod_revision': str(revision), 'lease': '42'}

    async def test_pagination_pins_revision_and_commits_only_complete_list(self):
        calls = []
        def handler(request):
            body = json.loads(request.content)
            if request.url.path.endswith('authenticate'):
                return httpx.Response(200, json={'token': 'synthetic-token'})
            self.assertEqual(request.headers['Authorization'], 'synthetic-token')
            calls.append(body)
            if len(calls) == 1:
                return httpx.Response(200, json={'header': {'revision': '10'},
                    'kvs': [self.row('google/instances/a')], 'more': True})
            self.assertEqual(client.directory.revision, 0)
            return httpx.Response(200, json={'header': {'revision': '12'}, 'kvs': [self.row()]})
        client = self.client(handler)
        await client.snapshot()
        self.assertEqual(calls[1]['revision'], '10')
        self.assertEqual(client.directory.revision, 10)

    async def test_failed_second_page_retains_snapshot(self):
        def handler(request):
            if request.url.path.endswith('authenticate'):
                return httpx.Response(200, json={'token': 'synthetic-token'})
            body = json.loads(request.content)
            if 'revision' in body:
                return httpx.Response(400, json={'error': 'compacted', 'code': 11})
            return httpx.Response(200, json={'header': {'revision': '12'},
                'kvs': [self.row()], 'more': True})
        client = self.client(handler)
        client.directory.replace([], 8)
        with self.assertRaises(DiscoveryUnavailable):
            await client.snapshot()
        self.assertEqual(client.directory.revision, 8)

    async def test_watch_creation_does_not_skip_historical_events(self):
        client = self.client(lambda request: httpx.Response(500))
        client.directory.replace([], 8)
        client._watch_result({'result': {'created': True, 'header': {'revision': '20'}}})
        self.assertEqual(client.directory.revision, 8)
        client._watch_result({'result': {'header': {'revision': '20'},
            'events': [{'kv': self.row(revision=10)}]}})
        self.assertEqual(client.directory.revision, 10)

    async def test_fragment_or_compaction_never_applies_partial_events(self):
        client = self.client(lambda request: httpx.Response(500))
        for extra in ({'fragment': True}, {'compact_revision': '15', 'canceled': True}):
            with self.assertRaises(DiscoveryUnavailable):
                client._watch_result({'result': {'events': [{'kv': self.row()}], **extra}})
            self.assertEqual(client.directory.revision, 0)

    async def test_auth_expiry_reauthenticates_but_permission_failure_does_not(self):
        auth, ranges = [], []
        def handler(request):
            if request.url.path.endswith('authenticate'):
                auth.append(1)
                return httpx.Response(200, json={'token': 'synthetic-token-' + str(len(auth))})
            self.assertIn('Authorization', request.headers)
            ranges.append(1)
            if len(ranges) == 1:
                return httpx.Response(401, json={'code': 16})
            return httpx.Response(403, json={'code': 7})
        client = self.client(handler)
        with self.assertRaises(DiscoveryUnavailable):
            await client.snapshot()
        self.assertEqual((len(auth), len(ranges)), (2, 2))

    async def test_stream_uses_snapshot_revision_plus_one_and_resyncs_on_eof(self):
        def handler(request):
            self.assertEqual(json.loads(request.content)['create_request']['start_revision'], '9')
            return httpx.Response(200, content=(json.dumps({'result': {
                'events': [{'kv': self.row()}], 'header': {'revision': '10'}}}) + '\n').encode())
        client = self.client(handler)
        client._token = 'synthetic-token'
        client.directory.replace([], 8)
        with self.assertRaisesRegex(DiscoveryUnavailable, 'disconnected'):
            await client.watch()
        self.assertEqual(client.directory.revision, 10)

    async def test_small_live_notification_is_applied_before_next_network_chunk(self):
        row = self.row()
        test = self
        class Stream(httpx.AsyncByteStream):
            async def __aiter__(self):
                yield (json.dumps({'result': {'events': [{'kv': row}]}}) + '\n').encode()
                test.assertEqual(client.directory.revision, 10)
                yield b'\n'
        client = self.client(lambda request: httpx.Response(200, stream=Stream()))
        client._token = 'synthetic-token'
        client.directory.replace([], 8)
        with self.assertRaisesRegex(DiscoveryUnavailable, 'disconnected'):
            await client.watch()
