"""Studio service identity plus Project write authorization for channel calls."""
from contextvars import ContextVar
import os
import secrets
import ssl
import time
import uuid

import httpx

CALLER = ContextVar('image_verified_caller', default=None)


def create_client():
    context = ssl.create_default_context()
    ca_file = os.environ.get('IMAGE_PROJECT_CA_FILE', '').strip()
    if ca_file:
        context.load_verify_locations(cafile=ca_file)
    ca_pem = os.environ.get('IMAGE_PROJECT_CA_PEM', '').strip()
    if ca_pem:
        context.load_verify_locations(cadata=ca_pem)
    return httpx.Client(verify=context, trust_env=False, follow_redirects=False)


def authorize(client, project_id, authority=None):
    actor = authority if authority is not None else CALLER.get()
    if not isinstance(actor, dict) or not all(actor.get(k) for k in ('user_id', 'organization_id')):
        raise ValueError('verified_studio_identity_required')
    url = os.environ.get('IMAGE_PROJECT_SERVICE_URL', '').rstrip('/')
    token = os.environ.get('IMAGE_PROJECT_SERVICE_TOKEN', '')
    if not url or not token:
        raise ValueError('project_authorization_not_configured')
    value = ((time.time_ns() // 1_000_000) << 80) | (7 << 76)
    value |= secrets.randbits(12) << 64 | (2 << 62) | secrets.randbits(62)
    try:
        response = client.post(url + '/internal/v1/project/open', headers={
            'Authorization': 'Bearer ' + token, 'X-User-Id': actor['user_id'],
            'X-Organization-Id': actor['organization_id'], 'X-Request-Id': str(uuid.UUID(int=value)),
        }, json={'project_id': project_id, 'require_write': True}, timeout=15, follow_redirects=False)
        if response.status_code in (401, 403, 404):
            raise ValueError('project_access_denied')
        response.raise_for_status()
    except httpx.HTTPError:
        raise ValueError('project_authorization_unavailable') from None
    return {key: actor[key] for key in ('user_id', 'organization_id')}
