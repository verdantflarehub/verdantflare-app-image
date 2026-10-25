"""Internal image channel transport. Callers persist intent before submit.

Organization comes from verified project authorization, never tool arguments.
Recovery only reads the original operation; a missing record is not permission
to submit again. This adapter does not own supplier credentials or retries.
"""
from __future__ import annotations

import base64
import io
from urllib.parse import quote, urlsplit

import httpx
from PIL import Image


class ChannelUnavailable(Exception):
    pass


class ImageChannel:
    def __init__(self, endpoint: str, token: str, client: httpx.Client):
        parsed = urlsplit(endpoint)
        if (parsed.scheme not in {'http', 'https'} or not parsed.hostname
                or parsed.username or parsed.password or parsed.query or parsed.fragment
                or parsed.path not in {'', '/'} or len(token) < 32):
            raise ValueError('invalid_channel_configuration')
        self.endpoint = endpoint.rstrip('/')
        self.token = token
        self.client = client

    def _request(self, method, path, organization, payload=None):
        if not organization or any(c in organization for c in '\r\n'):
            raise ValueError('verified_organization_required')
        try:
            with self.client.stream(method, self.endpoint + path, headers={
                'Authorization': 'Bearer ' + self.token,
                'X-Organization-ID': organization, 'X-Caller-ID': 'image-mcp',
            }, json=payload, timeout=200, follow_redirects=False) as response:
                # Do not trust arbitrary HTTP failures as proof of non-acceptance.
                if response.status_code != 200:
                    raise ChannelUnavailable('channel_response_unavailable')
                content = bytearray()
                for chunk in response.iter_bytes():
                    content.extend(chunk)
                    if len(content) > 48 * 1024 * 1024:
                        raise ChannelUnavailable('channel_response_too_large')
            import json
            value = json.loads(content)
            if not isinstance(value, dict):
                raise ValueError()
            return value
        except (httpx.HTTPError, ValueError):
            raise ChannelUnavailable('channel_response_unavailable') from None

    @staticmethod
    def _status(value, task_id):
        if (value.get('schema_version') != 1 or value.get('task_id') != task_id
                or not isinstance(value.get('call_id'), str) or not value['call_id']
                or 'accepted' not in value
                or not (value['accepted'] is None or type(value['accepted']) is bool)):
            raise ChannelUnavailable('invalid_channel_response')
        return value

    def submit(self, organization, task_id, operation_id, model_id, payload,
               account_id=None, budget_limit=None):
        try:
            return self._status(self._request('POST', '/v1/calls', organization, {
                'task_id': task_id, 'idempotency_key': operation_id,
                'model_id': model_id, 'capability': 'image', 'payload': payload,
                'account_id': account_id, 'budget_limit': budget_limit,
            }), task_id)
        except ChannelUnavailable:
            return {'task_id': task_id, 'call_id': None, 'accepted': None,
                    'state': 'acceptance_unknown'}

    def recover(self, organization, task_id, operation_id):
        return self._status(self._request('GET', '/v1/calls/by-idempotency/' +
                           quote(operation_id, safe=''), organization), task_id)

    def capabilities(self, organization='system-capabilities'):
        value = self._request('GET', '/v1/capabilities', organization)
        if (value.get('schema_version') != 1
                or not isinstance(value.get('channel_id'), str)
                or not isinstance(value.get('capabilities'), list)):
            raise ChannelUnavailable('invalid_channel_capabilities')
        return value

    def result(self, organization, task_id, call_id):
        value = self._status(self._request('GET', '/v1/calls/' +
                            quote(call_id, safe='') + '/result', organization), task_id)
        if value['call_id'] != call_id or value.get('state') != 'completed' or value['accepted'] is not True:
            raise ChannelUnavailable('result_not_available')
        try:
            output = value['image']
            data = base64.b64decode(output['data'], validate=True)
            with Image.open(io.BytesIO(data)) as decoded:
                if Image.MIME.get(decoded.format) != output['mime_type']:
                    raise ValueError()
                decoded.verify()
            return data, output['mime_type']
        except (KeyError, TypeError, ValueError, OSError, Image.DecompressionBombError):
            raise ChannelUnavailable('invalid_channel_image') from None
