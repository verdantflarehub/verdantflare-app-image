import json
import os
import unittest
from unittest.mock import patch

import httpx
from src.registry import EtcdRegistry


class RegistryAuthenticationTest(unittest.IsolatedAsyncioTestCase):
    def make(self, handler, endpoints="http://registry.test:2379", authenticated=True):
        values = {"ETCD_USERNAME": "writer-test" if authenticated else "",
                  "ETCD_PASSWORD": "synthetic-password" if authenticated else ""}
        with patch.dict(os.environ, values):
            registry = EtcdRegistry("test", [], etcd_endpoints=endpoints)
        registry._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        self.addAsyncCleanup(registry.stop)
        return registry

    async def test_legacy_registration_without_credentials(self):
        paths = []
        def handler(request):
            paths.append(request.url.path)
            self.assertNotIn("authorization", request.headers)
            return httpx.Response(200, json={"ID": "17", "TTL": "300"})
        registry = self.make(handler, authenticated=False)
        self.assertTrue(await registry.register())
        self.assertEqual(paths, ["/v3/lease/grant", "/v3/kv/put"])

    async def test_authenticate_once_and_keep_secrets_out_of_metadata(self):
        paths = []
        def handler(request):
            paths.append(request.url.path)
            if request.url.path == "/v3/auth/authenticate":
                self.assertEqual(json.loads(request.content), {"name": "writer-test", "password": "synthetic-password"})
                return httpx.Response(200, json={"token": "synthetic-token"})
            self.assertEqual(request.headers["authorization"], "synthetic-token")
            return httpx.Response(200, json={"ID": "17", "TTL": "300"})
        registry = self.make(handler)
        self.assertTrue(await registry.register())
        self.assertEqual(paths.count("/v3/auth/authenticate"), 1)
        metadata = json.dumps(registry.build_metadata())
        for secret in ["writer-test", "synthetic-password", "synthetic-token"]:
            self.assertNotIn(secret, metadata)

    async def test_expired_token_reauthenticates_once(self):
        auth_count = 0
        sent = []
        def handler(request):
            nonlocal auth_count
            if request.url.path == "/v3/auth/authenticate":
                auth_count += 1
                return httpx.Response(200, json={"token": "token-" + str(auth_count)})
            sent.append(request.headers.get("authorization"))
            if sent[-1] == "token-1":
                return httpx.Response(401, json={"code": 16})
            return httpx.Response(200, json={"ID": "17"})
        registry = self.make(handler)
        self.assertEqual(await registry._request("/v3/lease/grant", {"TTL": 300}), {"ID": "17"})
        self.assertEqual(sent, ["token-1", "token-2"])
        self.assertEqual(auth_count, 2)

    async def test_repeated_unauthenticated_response_is_bounded(self):
        calls = []
        def handler(request):
            calls.append(request.url.path)
            if request.url.path == "/v3/auth/authenticate":
                return httpx.Response(200, json={"token": "token"})
            return httpx.Response(200, json={"code": 16, "error": "expired"})
        registry = self.make(handler)
        self.assertIsNone(await registry._request("/v3/kv/put", {}))
        self.assertEqual(len(calls), 4)
        self.assertEqual(registry._auth_tokens, {})

    async def test_permission_denial_never_retries_anonymously(self):
        calls = []
        def handler(request):
            calls.append(request.url.path)
            if request.url.path == "/v3/auth/authenticate":
                return httpx.Response(200, json={"token": "token"})
            self.assertEqual(request.headers.get("authorization"), "token")
            return httpx.Response(403, json={"code": 7, "error": "synthetic-password"})
        registry = self.make(handler)
        with self.assertLogs("etcd_registry") as logs:
            self.assertIsNone(await registry._request("/v3/kv/put", {}))
        self.assertEqual(len(calls), 2)
        self.assertNotIn("synthetic-password", " ".join(logs.output))

    async def test_failed_login_never_sends_registry_operation(self):
        calls = []
        def handler(request):
            calls.append(request.url.path)
            return httpx.Response(401, json={"error": "invalid"})
        registry = self.make(handler)
        self.assertIsNone(await registry._request("/v3/kv/put", {}))
        self.assertEqual(calls, ["/v3/auth/authenticate"])

    async def test_redirect_does_not_forward_credentials(self):
        hosts = []
        def handler(request):
            hosts.append(request.url.host)
            return httpx.Response(307, headers={"location": "https://untrusted.invalid/collect"})
        registry = self.make(handler)
        self.assertIsNone(await registry._request("/v3/kv/put", {}))
        self.assertEqual(hosts, ["registry.test"])

    async def test_malformed_auth_reply_does_not_log_body(self):
        registry = self.make(lambda request: httpx.Response(200, content=b"synthetic-password not JSON"))
        with self.assertLogs("etcd_registry") as logs:
            self.assertIsNone(await registry._request("/v3/kv/put", {}))
        self.assertNotIn("synthetic-password", " ".join(logs.output))

    async def test_endpoint_tokens_are_not_reused_on_failover(self):
        operations = []
        def handler(request):
            host = request.url.host
            if request.url.path == "/v3/auth/authenticate":
                self.assertNotIn("authorization", request.headers)
                return httpx.Response(200, json={"token": "token-" + host})
            operations.append(host)
            self.assertEqual(request.headers["authorization"], "token-" + host)
            if host == "first.test":
                return httpx.Response(503)
            return httpx.Response(200, json={"header": {}})
        registry = self.make(handler, "http://first.test:2379,http://second.test:2379")
        self.assertEqual(await registry._request("/v3/kv/put", {}), {"header": {}})
        self.assertEqual(operations, ["first.test", "second.test"])
        await registry.stop()
        self.assertEqual(registry._auth_tokens, {})

    def test_partial_credentials_rejected(self):
        for user, password in [("writer", ""), ("", "password")]:
            with patch.dict(os.environ, {"ETCD_USERNAME": user, "ETCD_PASSWORD": password}):
                with self.assertRaisesRegex(ValueError, "incomplete_etcd_credentials"):
                    EtcdRegistry("test", [])

    def test_endpoint_must_not_embed_identity_or_extra_path(self):
        for endpoint in ["http://user:password@host:2379", "http://host:2379/extra", "http://host:2379?q=secret", "file:///etc/config"]:
            with patch.dict(os.environ, {"ETCD_USERNAME": "", "ETCD_PASSWORD": ""}):
                with self.assertRaises(ValueError):
                    EtcdRegistry("test", [], etcd_endpoints=endpoint)

    async def test_rejected_registration_is_retried_after_permission_recovery(self):
        grants = []
        puts = []
        revoked = []
        def handler(request):
            data = json.loads(request.content)
            if request.url.path.endswith("/grant"):
                lease = str(len(grants) + 1)
                grants.append(lease)
                return httpx.Response(200, json={"ID": lease})
            if request.url.path.endswith("/put"):
                puts.append(data["lease"])
                if len(puts) == 1:
                    return httpx.Response(403, json={"code": 7})
                return httpx.Response(200, json={"header": {}})
            if request.url.path.endswith("/revoke"):
                revoked.append(data["ID"])
                return httpx.Response(200, json={"header": {}})
            raise AssertionError("must republish rather than renew unpublished lease")
        registry = self.make(handler, authenticated=False)
        self.assertFalse(await registry.register())
        self.assertIsNone(registry._lease_id)
        self.assertEqual(revoked, ["1"])
        self.assertTrue(await registry.keepalive())
        self.assertEqual(puts, ["1", "2"])

    async def test_missing_ttl_or_wrong_lease_cannot_report_online(self):
        for result in [{"ID": "1"}, {"ID": "1", "TTL": "0"}, {"ID": "wrong", "TTL": "300"}]:
            grants = []
            def handler(request):
                if request.url.path.endswith("/grant"):
                    lease = str(len(grants) + 1)
                    grants.append(lease)
                    return httpx.Response(200, json={"ID": lease})
                if request.url.path.endswith("/keepalive"):
                    return httpx.Response(200, json={"result": result})
                return httpx.Response(200, json={"header": {}})
            registry = self.make(handler, authenticated=False)
            self.assertTrue(await registry.register())
            self.assertTrue(await registry.keepalive())
            self.assertEqual(grants, ["1", "2"])
            self.assertEqual(registry._lease_id, "2")
