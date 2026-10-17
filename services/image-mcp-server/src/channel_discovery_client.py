"""Authenticated bounded etcd List/Watch consumer, with no registry writes."""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
from urllib.parse import urlsplit

import httpx

from .channel_discovery import ChannelDirectory

MAX_FRAME = 2 * 1024 * 1024
MAX_KEYS = 10000
logger = logging.getLogger(__name__)


class DiscoveryUnavailable(RuntimeError):
    pass


def encode(value):
    return base64.b64encode(value.encode()).decode()


def decode(value):
    return base64.b64decode(value, validate=True).decode('utf-8')


class ChannelDiscoveryClient:
    @classmethod
    def from_environment(cls):
        endpoint = os.getenv('CHANNEL_DISCOVERY_ETCD_URL', '')
        scope = os.getenv('CHANNEL_DISCOVERY_SCOPE_ID', '')
        if not endpoint and not scope:
            return None
        origins = {name: f'http://{name}-channel.verdantflare-{name}.svc.cluster.local:8001'
                   for name in ('google', 'chatgpt', 'grok')}
        return cls(endpoint, scope, os.getenv('ETCD_USERNAME', ''),
                   os.getenv('ETCD_PASSWORD', ''), origins)

    def __init__(self, endpoint, scope, username, password, origins, *, transport=None):
        parsed = urlsplit(endpoint)
        if (parsed.scheme not in ('http', 'https') or not parsed.hostname
                or not parsed.hostname.endswith('.svc.cluster.local')
                or parsed.port not in (None, 2379) or parsed.username or parsed.password
                or parsed.path not in ('', '/') or parsed.query or parsed.fragment
                or not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}', scope)
                or not username or not password):
            raise ValueError('invalid_discovery_configuration')
        self.directory = ChannelDirectory(scope, origins)
        self._username, self._password = username, password
        self._token = None
        self._task = None
        self._client = httpx.AsyncClient(base_url=endpoint.rstrip('/'), timeout=10,
                                        trust_env=False, follow_redirects=False, transport=transport)

    async def _json(self, path, body, token=None):
        async with self._client.stream('POST', path, json=body,
                headers={'Authorization': token} if token else {}) as response:
            content = bytearray()
            async for chunk in response.aiter_bytes(chunk_size=65536):
                content.extend(chunk)
                if len(content) > MAX_FRAME:
                    raise DiscoveryUnavailable('discovery_response_too_large')
            value = json.loads(content)
            if not isinstance(value, dict):
                raise DiscoveryUnavailable('invalid_discovery_response')
            return response.status_code, value

    async def _authenticate(self):
        status, value = await self._json('/v3/auth/authenticate', {
            'name': self._username, 'password': self._password})
        token = value.get('token')
        if (status != 200 or not isinstance(token, str) or not token
                or len(token) > 16384 or any(c.isspace() for c in token)):
            raise DiscoveryUnavailable('discovery_authentication_failed')
        self._token = token

    async def _request(self, path, body):
        for attempt in range(2):
            if not self._token:
                await self._authenticate()
            status, value = await self._json(path, body, self._token)
            if status == 401 or value.get('code') in (16, '16'):
                self._token = None
                if attempt == 0:
                    continue
            if status != 200 or 'error' in value or 'code' in value:
                raise DiscoveryUnavailable('discovery_request_rejected')
            return value
        raise DiscoveryUnavailable('discovery_authentication_failed')

    @staticmethod
    def _kv(raw, deleted=False):
        result = {'key': decode(raw['key']), 'mod_revision': int(raw['mod_revision'])}
        if not deleted:
            result.update(lease=int(raw.get('lease', 0)), value=json.loads(decode(raw['value'])))
        return result

    async def snapshot(self):
        prefix = self.directory.prefix
        end = prefix[:-1] + chr(ord(prefix[-1]) + 1)
        start, revision, rows = prefix, None, []
        while True:
            body = {'key': encode(start), 'range_end': encode(end), 'limit': 256,
                    'sort_order': 'ASCEND', 'sort_target': 'KEY'}
            if revision is not None:
                body['revision'] = str(revision)
            value = await self._request('/v3/kv/range', body)
            if revision is None:
                revision = int(value['header']['revision'])
                if revision < 1:
                    raise DiscoveryUnavailable('invalid_snapshot_revision')
            page = [self._kv(row) for row in value.get('kvs', [])]
            previous = None
            for row in page:
                if (not start <= row['key'] < end
                        or previous is not None and row['key'] <= previous):
                    raise DiscoveryUnavailable('invalid_snapshot_order')
                previous = row['key']
            rows.extend(page)
            if len(rows) > MAX_KEYS:
                raise DiscoveryUnavailable('discovery_snapshot_too_large')
            if not value.get('more', False):
                self.directory.replace(rows, revision)
                return
            if not page:
                raise DiscoveryUnavailable('incomplete_discovery_snapshot')
            start = page[-1]['key'] + '\x00'

    def _watch_result(self, value):
        if 'error' in value:
            if value.get('code') in (16, '16'):
                self._token = None
            raise DiscoveryUnavailable('discovery_watch_rejected')
        result = value['result']
        if result.get('canceled') or int(result.get('compact_revision', 0)):
            raise DiscoveryUnavailable('discovery_watch_resync')
        if result.get('fragment'):
            # Fragmentation is not requested: never publish an incomplete revision.
            raise DiscoveryUnavailable('fragmented_discovery_watch')
        events = []
        for event in result.get('events', []):
            kind = event.get('type', 'PUT')
            events.append({'type': kind, 'kv': self._kv(event['kv'], kind == 'DELETE')})
        if events:
            revision = max(event['kv']['mod_revision'] for event in events)
            self.directory.apply(events, revision)
        elif not result.get('created'):
            self.directory.apply([], int(result['header']['revision']))

    async def watch(self):
        prefix = self.directory.prefix
        body = {'create_request': {'key': encode(prefix),
            'range_end': encode(prefix[:-1] + chr(ord(prefix[-1]) + 1)),
            'start_revision': str(self.directory.revision + 1), 'progress_notify': True}}
        async with self._client.stream('POST', '/v3/watch', json=body,
                headers={'Authorization': self._token}, timeout=httpx.Timeout(30, read=90)) as response:
            if response.status_code != 200:
                if response.status_code == 401:
                    self._token = None
                raise DiscoveryUnavailable('discovery_watch_rejected')
            buffer = bytearray()
            # Do not request a fixed chunk size: httpx would buffer small live
            # notifications until that size is reached, delaying status changes.
            async for chunk in response.aiter_bytes():
                buffer.extend(chunk)
                while b'\n' in buffer:
                    line, _, remaining = buffer.partition(b'\n')
                    if len(line) > MAX_FRAME:
                        raise DiscoveryUnavailable('discovery_frame_too_large')
                    buffer = bytearray(remaining)
                    if line.strip():
                        self._watch_result(json.loads(line))
                if len(buffer) > MAX_FRAME:
                    raise DiscoveryUnavailable('discovery_frame_too_large')
            # EOF, including a partial frame, always requires a full resync.
            raise DiscoveryUnavailable('discovery_watch_disconnected')

    async def run(self):
        delay = 1
        while True:
            try:
                await self.snapshot()
                await self.watch()
            except asyncio.CancelledError:
                raise
            except (DiscoveryUnavailable, httpx.HTTPError, ValueError, KeyError, TypeError):
                logger.warning('Channel discovery unavailable; retrying with a full snapshot')
                await asyncio.sleep(delay)
                delay = min(delay * 2, 30)

    def start(self):
        if self._task is None:
            self._task = asyncio.create_task(self.run())

    async def stop(self):
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        await self._client.aclose()
