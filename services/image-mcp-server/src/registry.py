from __future__ import annotations

import asyncio
import base64
from datetime import datetime, timezone
import json
import logging
import os
from typing import Any
from urllib.parse import urlsplit

import httpx

logger = logging.getLogger("etcd_registry")


class EtcdRegistry:
    """Dynamic service registration client for etcd v3 gRPC-Gateway HTTP API.

    Adheres to Studio MCP Gateway v1 contract:
    - 5-minute (300s) TTL lease.
    - 1-minute (60s) keepalive heartbeat.
    - Immediate graceful lease revocation on shutdown.
    - Non-blocking graceful degradation if etcd is temporarily unreachable.
    """

    def __init__(
        self,
        domain: str,
        tools: list[dict[str, Any]],
        endpoint: str | None = None,
        health_endpoint: str | None = None,
        version: str | None = None,
        etcd_endpoints: str | None = None,
        lease_ttl: int = 300,
        heartbeat_interval: int = 60,
    ) -> None:
        self._username = os.environ.get("ETCD_USERNAME", "")
        self._password = os.environ.get("ETCD_PASSWORD", "")
        if bool(self._username) != bool(self._password):
            raise ValueError("incomplete_etcd_credentials")
        self._auth_tokens: dict[str, str] = {}
        self._auth_lock = asyncio.Lock()
        self.domain = domain
        self.tools = tools
        self.lease_ttl = lease_ttl
        self.heartbeat_interval = heartbeat_interval

        raw_etcd = etcd_endpoints or os.environ.get(
            "ETCD_ENDPOINTS", "http://etcd.verdantflare-station.svc.cluster.local:2379"
        )
        self.endpoints = [ep.strip().rstrip("/") for ep in raw_etcd.split(",") if ep.strip()]
        if not self.endpoints:
            self.endpoints = ["http://127.0.0.1:2379"]

        for address in self.endpoints:
            parsed = urlsplit(address)
            if (parsed.scheme not in {"http", "https"} or not parsed.hostname
                    or parsed.username is not None or parsed.password is not None
                    or parsed.path not in {"", "/"} or parsed.query or parsed.fragment
                    or any(c.isspace() for c in address)):
                raise ValueError("invalid_etcd_endpoint")

        self.service_endpoint = endpoint or os.environ.get(
            "MCP_SERVICE_ENDPOINT",
            f"http://{domain}-mcp-server.verdantflare-{domain}.svc.cluster.local:8000/mcp",
        )
        self.health_endpoint = health_endpoint or os.environ.get(
            "MCP_HEALTH_ENDPOINT",
            f"http://{domain}-mcp-server.verdantflare-{domain}.svc.cluster.local:8000/health",
        )
        self.version = version or os.environ.get("SERVICE_VERSION", "v1.0.0")

        self._lease_id: str | None = None
        self._client: httpx.AsyncClient | None = None
        self._heartbeat_task: asyncio.Task[None] | None = None
        self._running = False

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=5.0, trust_env=False, follow_redirects=False)
        return self._client

    def build_metadata(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "endpoint": self.service_endpoint,
            "health_endpoint": self.health_endpoint,
            "version": self.version,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "tools": self.tools,
        }

    async def _authenticate(self, client: httpx.AsyncClient, endpoint: str) -> bool:
        response = await client.post(endpoint + "/v3/auth/authenticate",
                                     json={"name": self._username, "password": self._password},
                                     follow_redirects=False)
        if response.status_code != 200:
            return False
        data = response.json()
        token = data.get("token") if isinstance(data, dict) else None
        if not isinstance(token, str) or not token or len(token) > 16384 or any(c.isspace() for c in token):
            return False
        self._auth_tokens[endpoint] = token
        return True

    async def _request(self, path: str, json_body: dict[str, Any]) -> dict[str, Any] | None:
        # Serialize refresh and requests so concurrent keepalive/shutdown cannot
        # replace a newer token with a rejected one. Never retry anonymously.
        async with self._auth_lock:
            client = self._get_client()
            for endpoint in self.endpoints:
                try:
                    for attempt in range(2):
                        if self._username and endpoint not in self._auth_tokens:
                            if not await self._authenticate(client, endpoint):
                                break
                        token = self._auth_tokens.get(endpoint)
                        headers = {"Authorization": token} if token else {}
                        response = await client.post(endpoint + path, json=json_body,
                                                     headers=headers, follow_redirects=False)
                        try:
                            data = response.json()
                        except ValueError:
                            data = None
                        unauthenticated = response.status_code == 401 or (
                            isinstance(data, dict) and data.get("code") in (16, "16"))
                        if self._username and unauthenticated:
                            self._auth_tokens.pop(endpoint, None)
                            if attempt == 0:
                                continue
                        if response.status_code == 200 and isinstance(data, dict) and "error" not in data and "code" not in data:
                            return data
                        logger.warning("[EtcdRegistry] Registry request rejected (HTTP %s)", response.status_code)
                        break
                except (httpx.HTTPError, ValueError, TypeError):
                    # Upstream responses and exception text may contain credentials.
                    logger.warning("[EtcdRegistry] Registry transport unavailable")
            return None

    async def register(self) -> bool:
        """Grants a lease and registers service metadata under /verdantflare/mcp/services/{domain}."""
        grant_resp = await self._request("/v3/lease/grant", {"TTL": self.lease_ttl})
        if not grant_resp or "ID" not in grant_resp:
            logger.warning("[EtcdRegistry] Failed to obtain lease from etcd")
            return False

        self._lease_id = str(grant_resp["ID"])

        key = f"/verdantflare/mcp/services/{self.domain}"
        value = json.dumps(self.build_metadata(), ensure_ascii=False)

        key_b64 = base64.b64encode(key.encode("utf-8")).decode("ascii")
        value_b64 = base64.b64encode(value.encode("utf-8")).decode("ascii")

        put_resp = await self._request(
            "/v3/kv/put",
            {"key": key_b64, "value": value_b64, "lease": self._lease_id},
        )
        if put_resp is not None:
            logger.info(
                "[EtcdRegistry] Successfully registered '%s' (lease=%s) -> %s",
                self.domain,
                self._lease_id,
                self.service_endpoint,
            )
            return True
        await self.deregister()
        logger.warning("[EtcdRegistry] Failed to put service key into etcd")
        return False

    async def keepalive(self) -> bool:
        if not self._lease_id:
            return await self.register()

        resp = await self._request("/v3/lease/keepalive", {"ID": self._lease_id})
        result = resp.get("result") if isinstance(resp, dict) else None
        if isinstance(result, dict) and str(result.get("ID")) == self._lease_id:
            try:
                if int(result.get("TTL", 0)) > 0:
                    return True
            except (TypeError, ValueError):
                pass

        logger.warning("[EtcdRegistry] Lease keepalive failed, attempting re-registration")
        return await self.register()

    async def deregister(self) -> None:
        """Revokes the lease, immediately removing the key from etcd."""
        if not self._lease_id:
            return
        lease_to_revoke = self._lease_id
        self._lease_id = None
        try:
            await self._request("/v3/lease/revoke", {"ID": lease_to_revoke})
            logger.info("[EtcdRegistry] Revoked lease %s for domain '%s'", lease_to_revoke, self.domain)
        except Exception as err:
            logger.warning("[EtcdRegistry] Error revoking lease %s: %s", lease_to_revoke, err)

    async def _heartbeat_loop(self) -> None:
        while self._running:
            try:
                await asyncio.sleep(self.heartbeat_interval)
                if not self._running:
                    break
                await self.keepalive()
            except asyncio.CancelledError:
                break
            except Exception as err:
                logger.warning("[EtcdRegistry] Error in heartbeat loop: %s", err)

    async def start(self) -> None:
        """Starts dynamic registration and the background heartbeat task."""
        self._running = True
        try:
            registered = await self.register()
            if not registered:
                logger.warning(
                    "[EtcdRegistry] Initial registration unsuccessful, will retry in heartbeat loop"
                )
        except Exception as err:
            logger.warning("[EtcdRegistry] Could not connect to etcd (%s); continuing in standalone mode", err)

        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

    async def stop(self) -> None:
        """Stops background heartbeat, revokes lease, and closes client."""
        self._running = False
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
            self._heartbeat_task = None

        await self.deregister()
        self._auth_tokens.clear()

        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
