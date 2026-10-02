from __future__ import annotations

import asyncio
import base64
from datetime import datetime, timezone
import json
import logging
import os
from typing import Any

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
            self._client = httpx.AsyncClient(timeout=5.0)
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

    async def _request(self, path: str, json_body: dict[str, Any]) -> dict[str, Any] | None:
        client = self._get_client()
        for base_url in self.endpoints:
            url = f"{base_url}{path}"
            try:
                resp = await client.post(url, json=json_body)
                if resp.status_code == 200:
                    return resp.json()
                logger.warning(
                    "[EtcdRegistry] HTTP %s from %s: %s", resp.status_code, url, resp.text
                )
            except Exception as err:
                logger.debug("[EtcdRegistry] Failed to call %s: %s", url, err)
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
        logger.warning("[EtcdRegistry] Failed to put service key into etcd")
        return False

    async def keepalive(self) -> bool:
        if not self._lease_id:
            return await self.register()

        resp = await self._request("/v3/lease/keepalive", {"ID": self._lease_id})
        if resp and "result" in resp:
            return True

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

        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
