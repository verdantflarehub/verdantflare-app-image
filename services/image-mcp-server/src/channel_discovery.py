"""Read-only channel directory state; transport supplies complete revisions."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from typing import Any


class ChannelDirectory:
    def __init__(self, scope: str, origins: dict[str, str]) -> None:
        if not scope or '/' in scope:
            raise ValueError('invalid_channel_scope')
        self.prefix = f'/verdantflare/channels/v1/{scope}/'
        self.origins = dict(origins)
        self._snapshot: tuple[dict, dict, int] = ({}, {}, 0)

    @property
    def revision(self):
        return self._snapshot[2]

    def replace(self, values: list[dict[str, Any]], revision: int) -> bool:
        """Caller must finish every pinned List page before calling this method."""
        if revision <= self.revision:
            return False
        candidate = {}
        for value in values:
            self._validate(value, revision)
            if value['key'] in candidate:
                raise ValueError('duplicate_channel_key')
            candidate[value['key']] = deepcopy(value)
        self._commit(candidate, revision)
        return True

    def apply(self, events: list[dict[str, Any]], revision: int) -> bool:
        """Apply one complete Watch revision, never a partial transaction."""
        if revision <= self.revision:
            return False
        candidate = deepcopy(self._snapshot[0])
        for event in events:
            value = event['kv']
            self._validate(value, revision, deleted=event.get('type') == 'DELETE')
            if value['mod_revision'] <= self.revision:
                raise ValueError('stale_channel_event_in_new_batch')
            if event.get('type', 'PUT') == 'PUT':
                candidate[value['key']] = deepcopy(value)
            elif event['type'] == 'DELETE':
                candidate.pop(value['key'], None)
            else:
                raise ValueError('invalid_channel_event')
        if len(candidate) > 10000:
            raise ValueError('channel_directory_too_large')
        self._commit(candidate, revision)
        return True

    def _validate(self, value: dict[str, Any], revision: int, deleted: bool = False) -> None:
        key = value.get('key')
        if not isinstance(key, str) or not key.startswith(self.prefix):
            raise ValueError('channel_key_outside_scope')
        mod = value.get('mod_revision')
        if type(mod) is not int or not 0 < mod <= revision:
            raise ValueError('invalid_channel_revision')
        if not deleted and (not isinstance(value.get('value'), dict)
                            or type(value.get('lease')) is not int):
            raise ValueError('invalid_channel_value')

    def _commit(self, candidate: dict[str, dict[str, Any]], revision: int) -> None:
        generations = dict(self._snapshot[1])
        for key, item in candidate.items():
            parts = key[len(self.prefix):].split('/')
            generation = item['value'].get('generation')
            if (len(parts) == 3 and parts[1] == 'instances' and parts[2]
                    and type(generation) is int and generation > 0):
                generations[parts[0]] = max(generation, generations.get(parts[0], 0))
        # Worker threads read one immutable snapshot reference.
        self._snapshot = candidate, generations, revision

    def resolve(self, channel: str, capability: str, now: datetime | None = None) -> str | None:
        """Return a verified origin for NEW calls only; recovery keeps its saved route."""
        now = now or datetime.now(timezone.utc)
        values, generations, _ = self._snapshot
        base = self.prefix + channel + '/'
        status = values.get(base + 'status')
        if not status or status['lease'] == 0:
            return None
        state = status['value']
        try:
            observed = datetime.fromisoformat(state['observed_at'].replace('Z', '+00:00'))
            age = (now - observed).total_seconds()
        except (KeyError, ValueError, TypeError, AttributeError):
            return None
        if not 0 <= age < 300 or state.get('state') != 'enabled':
            return None
        instances = []
        for key, item in values.items():
            suffix = key.removeprefix(base + 'instances/')
            if not key.startswith(base + 'instances/') or not suffix or '/' in suffix:
                continue
            data = item['value']
            if (item['lease'] != status['lease'] or data.get('schema_version') != 1
                    or data.get('service_identity') != channel
                    or data.get('generation') != generations.get(channel)
                    or data.get('execution_url') != self.origins.get(channel)
                    or channel not in self.origins):
                continue
            capabilities = data.get('capabilities')
            if not isinstance(capabilities, list):
                continue
            if any(isinstance(c, dict) and c.get('capability_id') == capability
                   and c.get('execution_state') == 'enabled' for c in capabilities):
                instances.append(data['execution_url'])
        return instances[0] if len(instances) == 1 else None
