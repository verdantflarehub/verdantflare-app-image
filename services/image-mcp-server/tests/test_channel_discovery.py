from copy import deepcopy
from datetime import datetime, timedelta, timezone
import unittest

from src.channel_discovery import ChannelDirectory


class ChannelDirectoryTest(unittest.TestCase):
    def setUp(self):
        self.now = datetime(2026, 10, 17, tzinfo=timezone.utc)
        self.origin = 'http://google-channel.test:8001'
        self.directory = ChannelDirectory('station-test', {'google': self.origin})
        prefix = self.directory.prefix + 'google/'
        self.rows = [
            {'key': prefix + 'status', 'lease': 42, 'mod_revision': 10,
             'value': {'state': 'enabled', 'revision': 1, 'observed_at': self.now.isoformat()}},
            {'key': prefix + 'instances/one', 'lease': 42, 'mod_revision': 10,
             'value': {'schema_version': 1, 'generation': 2, 'service_identity': 'google',
                       'execution_url': self.origin, 'capabilities': [
                           {'capability_id': 'image.generate', 'execution_state': 'enabled'}]}},
        ]

    def resolve(self, now=None):
        return self.directory.resolve('google', 'image.generate', now or self.now)

    def test_complete_snapshot_and_expiry(self):
        self.directory.replace(self.rows, 10)
        self.assertEqual(self.resolve(), self.origin)
        self.assertIsNone(self.resolve(self.now + timedelta(seconds=300)))
        self.assertIsNone(self.resolve(self.now - timedelta(seconds=1)))

    def test_bad_snapshot_does_not_replace_previous_state(self):
        self.directory.replace(self.rows, 10)
        broken = deepcopy(self.rows)
        broken[1]['mod_revision'] = 12
        with self.assertRaises(ValueError):
            self.directory.replace(broken, 11)
        self.assertEqual(self.directory.revision, 10)
        self.assertEqual(self.resolve(), self.origin)

    def test_status_and_instance_must_share_lease(self):
        self.rows[1]['lease'] = 43
        self.directory.replace(self.rows, 10)
        self.assertIsNone(self.resolve())

    def test_unverified_capability_and_untrusted_url_are_not_dispatchable(self):
        for field, value in [('capabilities', [{'capability_id': 'image.generate',
                                               'execution_state': 'adapter_connected_unverified'}]),
                             ('execution_url', 'https://external.test')]:
            with self.subTest(field=field):
                directory = ChannelDirectory('station-test', {'google': self.origin})
                rows = deepcopy(self.rows)
                rows[1]['value'][field] = value
                directory.replace(rows, 10)
                self.assertIsNone(directory.resolve('google', 'image.generate', self.now))

    def test_delete_and_old_revision_cannot_restore_channel(self):
        self.directory.replace(self.rows, 10)
        deleted = {'key': self.rows[1]['key'], 'mod_revision': 11}
        self.directory.apply([{'type': 'DELETE', 'kv': deleted}], 11)
        self.assertIsNone(self.resolve())
        self.assertFalse(self.directory.replace(self.rows, 10))
        self.assertIsNone(self.resolve())

    def test_newer_revision_with_old_generation_stays_unavailable(self):
        self.directory.replace(self.rows, 10)
        older = deepcopy(self.rows)
        older[1]['value']['generation'] = 1
        self.directory.replace(older, 12)
        self.assertIsNone(self.resolve())

    def test_invalid_event_batch_is_atomic(self):
        self.directory.replace(self.rows, 10)
        events = [
            {'type': 'DELETE', 'kv': {'key': self.rows[1]['key'], 'mod_revision': 11}},
            {'type': 'DELETE', 'kv': {'key': '/other-scope/key', 'mod_revision': 11}},
        ]
        with self.assertRaises(ValueError):
            self.directory.apply(events, 11)
        self.assertEqual(self.resolve(), self.origin)
        self.assertEqual(self.directory.revision, 10)

    def test_mixed_stale_and_new_events_cannot_overwrite_current_state(self):
        self.directory.replace(self.rows, 10)
        stale = deepcopy(self.rows[0])
        stale['value']['state'] = 'disabled'
        newer = deepcopy(self.rows[1])
        newer['mod_revision'] = 12
        with self.assertRaisesRegex(ValueError, 'stale_channel_event'):
            self.directory.apply([{'kv': stale}, {'kv': newer}], 12)
        self.assertEqual(self.directory.revision, 10)
        self.assertEqual(self.resolve(), self.origin)

    def test_draining_and_deleted_status_block_new_calls(self):
        self.directory.replace(self.rows, 10)
        status = deepcopy(self.rows[0])
        status['mod_revision'] = 11
        status['value']['state'] = 'draining'
        self.directory.apply([{'kv': status}], 11)
        self.assertIsNone(self.resolve())
        self.directory.apply([{'type': 'DELETE', 'kv': {
            'key': status['key'], 'mod_revision': 12}}], 12)
        self.assertIsNone(self.resolve())


if __name__ == '__main__':
    unittest.main()
