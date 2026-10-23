"""Shared GPU task lease and on-demand scale for image-moss-api.

Follows the 5090 GPU lease protocol (`gpu-task-moss-image`):
- Write this role's Lease first, then read conflicting peers (`music3`).
- A live conflicting peer (`gpu-task-music3`) means release and retry.
- `moss-image` does not conflict with `h3` or `indextts2`.
- Scales `Deployment/image-moss-api` from 0 -> 1 when held and back to 0 after
  the 10-minute (600s) idle window.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
import random
import threading
import time
from typing import Callable

import httpx

LEASES = {
    "music3": "gpu-task-music3",
    "h3": "gpu-task-h3",
    "indextts2": "gpu-task-indextts2",
    "moss-image": "gpu-task-moss-image",
}
CONFLICTS = {
    "music3": ("h3", "indextts2", "moss-image"),
    "h3": ("music3",),
    "indextts2": ("music3",),
    "moss-image": ("music3",),
}


class LeaseBusy(RuntimeError):
    def __init__(self, holder: str | None = None) -> None:
        self.holder = holder or ""
        super().__init__("waiting_gpu_lease")


def format_time(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000000Z")


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def replicas_from_scale(payload: dict) -> int:
    spec = payload.get("spec") or {}
    if "replicas" not in spec:
        return 0
    return int(spec["replicas"])


def held_by_other(lease: dict | None, holder: str, now: datetime) -> str | None:
    spec = (lease or {}).get("spec") or {}
    identity = spec.get("holderIdentity") or ""
    if not identity or identity == holder:
        return None
    renew = spec.get("renewTime")
    duration = int(spec.get("leaseDurationSeconds") or 60)
    if not renew:
        return identity
    if parse_time(renew) + timedelta(seconds=duration) > now:
        return identity
    return None


class MemoryApi:
    """In-memory Lease and scale API for unit tests."""

    def __init__(self) -> None:
        self.leases: dict[str, dict] = {}
        self.scales: dict[str, int] = {}

    def get_lease(self, name: str) -> dict | None:
        item = self.leases.get(name)
        return (
            None
            if item is None
            else {
                "metadata": dict(item["metadata"]),
                "spec": dict(item["spec"]),
            }
        )

    def replace_lease(self, name: str, spec: dict, resource_version: str | None) -> int:
        current = self.leases.get(name)
        current_version = None if current is None else current["metadata"].get("resourceVersion")
        if current is not None and current_version != resource_version:
            return 409
        version = str(int(current_version or "0") + 1)
        self.leases[name] = {
            "metadata": {"name": name, "resourceVersion": version},
            "spec": dict(spec),
        }
        return 200

    def get_scale(self, name: str) -> int:
        return self.scales.get(name, 0)

    def patch_scale(self, name: str, replicas: int) -> None:
        self.scales[name] = replicas


class K8sApi:
    def __init__(
        self,
        namespace: str,
        host: str = "https://kubernetes.default.svc",
        token_path: str = "/var/run/secrets/kubernetes.io/serviceaccount/token",
        ca_path: str = "/var/run/secrets/kubernetes.io/serviceaccount/ca.crt",
    ) -> None:
        self.namespace = namespace
        self.host = host.rstrip("/")
        self.token_path = token_path
        self.ca_path = ca_path

    def _request(self, method: str, path: str, **kwargs: object) -> httpx.Response:
        token = Path(self.token_path).read_text(encoding="utf-8").strip()
        headers = {"Authorization": f"Bearer {token}"}
        extra = kwargs.pop("headers", None)
        if isinstance(extra, dict):
            headers.update(extra)
        with httpx.Client(verify=self.ca_path, timeout=10, trust_env=False) as client:
            return client.request(method, self.host + path, headers=headers, **kwargs)

    def get_lease(self, name: str) -> dict | None:
        path = f"/apis/coordination.k8s.io/v1/namespaces/{self.namespace}/leases/{name}"
        response = self._request("GET", path)
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response.json()

    def replace_lease(self, name: str, spec: dict, resource_version: str | None) -> int:
        path = f"/apis/coordination.k8s.io/v1/namespaces/{self.namespace}/leases/{name}"
        body = {
            "apiVersion": "coordination.k8s.io/v1",
            "kind": "Lease",
            "metadata": {"name": name, "namespace": self.namespace},
            "spec": spec,
        }
        if resource_version:
            body["metadata"]["resourceVersion"] = resource_version
            response = self._request("PUT", path, json=body)
        else:
            response = self._request(
                "POST",
                f"/apis/coordination.k8s.io/v1/namespaces/{self.namespace}/leases",
                json=body,
            )
            if response.status_code == 409:
                current = self.get_lease(name) or {}
                body["metadata"]["resourceVersion"] = current.get("metadata", {}).get("resourceVersion")
                response = self._request("PUT", path, json=body)
        return response.status_code

    def get_scale(self, name: str) -> int:
        path = f"/apis/apps/v1/namespaces/{self.namespace}/deployments/{name}/scale"
        response = self._request("GET", path)
        response.raise_for_status()
        return replicas_from_scale(response.json())

    def patch_scale(self, name: str, replicas: int) -> None:
        path = f"/apis/apps/v1/namespaces/{self.namespace}/deployments/{name}/scale"
        response = self._request(
            "PATCH",
            path,
            json={"spec": {"replicas": replicas}},
            headers={"Content-Type": "application/merge-patch+json"},
        )
        response.raise_for_status()


class GpuCoordinator:
    def __init__(
        self,
        api: object | None,
        role: str = "moss-image",
        holder: str = "moss-image/local",
        deployment: str | None = None,
        enabled: bool = False,
        idle_seconds: int = 600,
        queue_seconds: int = 1200,
        ready_seconds: int = 300,
        lease_seconds: int = 90,
        now=None,
        sleep=None,
        monotonic=None,
    ) -> None:
        self.api = api
        self.role = role
        self.holder = holder
        self.deployment = deployment or None
        self.enabled = enabled
        self.idle_seconds = idle_seconds
        self.queue_seconds = queue_seconds
        self.ready_seconds = ready_seconds
        self.lease_seconds = lease_seconds
        self._now = now or (lambda: datetime.now(timezone.utc))
        self._sleep = sleep or time.sleep
        self._monotonic = monotonic or time.monotonic
        self.in_flight = 0
        self.idle_at: float | None = None
        self._lock = threading.Lock()

    @property
    def lease_name(self) -> str:
        return LEASES[self.role]

    @classmethod
    def from_environment(cls) -> "GpuCoordinator":
        role = os.environ.get("MOSS_GPU_TASK_ROLE", os.environ.get("GPU_TASK_ROLE", "")).strip()
        if role not in CONFLICTS:
            return cls(api=None, role="moss-image", holder="", enabled=False)
        namespace = os.environ.get("MOSS_GPU_TASK_NAMESPACE", os.environ.get("GPU_TASK_NAMESPACE", "")).strip()
        namespace_file = Path("/var/run/secrets/kubernetes.io/serviceaccount/namespace")
        if not namespace and namespace_file.is_file():
            namespace = namespace_file.read_text(encoding="utf-8").strip()
        namespace = namespace or "verdantflare-app"
        holder = (
            os.environ.get("MOSS_GPU_TASK_HOLDER", "").strip()
            or f"{role}/{os.environ.get('HOSTNAME', 'local')}"
        )
        deployment = os.environ.get("MOSS_GPU_TASK_DEPLOYMENT", "image-moss-api").strip() or None
        return cls(
            api=K8sApi(namespace),
            role=role,
            holder=holder,
            deployment=deployment,
            enabled=True,
            idle_seconds=int(os.environ.get("MOSS_GPU_IDLE_SECONDS", "600")),
            queue_seconds=int(os.environ.get("MOSS_GPU_QUEUE_SECONDS", "1200")),
            ready_seconds=int(os.environ.get("MOSS_GPU_READY_SECONDS", "300")),
            lease_seconds=int(os.environ.get("MOSS_GPU_LEASE_SECONDS", "90")),
        )

    def peer_holder(self) -> str | None:
        if not self.enabled:
            return None
        now = self._now()
        for peer in CONFLICTS[self.role]:
            holder = held_by_other(self.api.get_lease(LEASES[peer]), self.holder, now)
            if holder:
                return holder
        return None

    def replicas(self) -> int | None:
        if not self.enabled or not self.deployment:
            return None
        try:
            return int(self.api.get_scale(self.deployment))
        except Exception:
            return None

    def scale_to(self, replicas: int) -> None:
        if not self.enabled or not self.deployment:
            return
        if self.api.get_scale(self.deployment) != replicas:
            self.api.patch_scale(self.deployment, replicas)

    def _spec(self, holder: str, acquire_time: str | None = None) -> dict:
        now = format_time(self._now())
        return {
            "holderIdentity": holder,
            "leaseDurationSeconds": self.lease_seconds,
            "acquireTime": acquire_time or now,
            "renewTime": now,
        }

    def _write(self, holder: str) -> bool:
        current = self.api.get_lease(self.lease_name)
        version = None if current is None else current.get("metadata", {}).get("resourceVersion")
        acquire_time = None
        if current and (current.get("spec") or {}).get("holderIdentity") == self.holder:
            acquire_time = (current.get("spec") or {}).get("acquireTime")
        return self.api.replace_lease(self.lease_name, self._spec(holder, acquire_time), version) in {200, 201}

    def try_acquire(self) -> bool:
        if not self.enabled:
            return True
        with self._lock:
            now = self._now()
            current = self.api.get_lease(self.lease_name)
            if held_by_other(current, self.holder, now):
                return False
            if not self._write(self.holder):
                return False
            if self.peer_holder():
                self._write("")
                return False
            return True

    def acquire_for_queue(self) -> None:
        if not self.enabled:
            return
        deadline = self._monotonic() + self.queue_seconds
        delay = 0.2
        while True:
            if self.try_acquire():
                with self._lock:
                    self.in_flight += 1
                    self.idle_at = None
                return
            if self._monotonic() >= deadline:
                raise LeaseBusy(self.peer_holder())
            delay = min(5.0, delay * 2)
            self._sleep(delay * (0.5 + random.random()))

    def ensure_worker_ready(self, ready_check: Callable[[], bool]) -> None:
        if not self.enabled:
            return
        self.scale_to(1)
        deadline = self._monotonic() + self.ready_seconds
        while self._monotonic() < deadline:
            if ready_check():
                return
            self.renew()
            self._sleep(1.0)
        raise TimeoutError(f"Worker {self.deployment} did not become ready within {self.ready_seconds}s")

    def renew(self) -> None:
        if not self.enabled:
            return
        current = self.api.get_lease(self.lease_name)
        identity = ((current or {}).get("spec") or {}).get("holderIdentity")
        if identity not in {None, "", self.holder}:
            return
        self._write(self.holder)

    def release(self) -> None:
        if not self.enabled:
            return
        with self._lock:
            current = self.api.get_lease(self.lease_name)
            identity = ((current or {}).get("spec") or {}).get("holderIdentity")
            if identity in {None, "", self.holder}:
                self._write("")
            if self.in_flight:
                self.in_flight -= 1
            if self.in_flight == 0:
                self.idle_at = self._monotonic()

    def maybe_scale_down(self) -> None:
        if not self.enabled or not self.deployment or self.in_flight or self.idle_at is None:
            return
        if self._monotonic() - self.idle_at < self.idle_seconds:
            return
        current = self.api.get_lease(self.lease_name)
        if ((current or {}).get("spec") or {}).get("holderIdentity") == self.holder:
            return
        if self.replicas() not in {None, 0}:
            self.scale_to(0)

    def hold(self) -> "_Hold":
        return _Hold(self)


class _Hold:
    def __init__(self, coord: GpuCoordinator) -> None:
        self.coord = coord
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "_Hold":
        self.coord.acquire_for_queue()
        if self.coord.enabled:
            self._thread = threading.Thread(target=self._renew, daemon=True)
            self._thread.start()
        return self

    def _renew(self) -> None:
        interval = max(5, self.coord.lease_seconds // 3)
        while not self._stop.wait(interval):
            try:
                self.coord.renew()
            except Exception:
                pass

    def __exit__(self, exc_type, exc, tb) -> bool:
        self._stop.set()
        self.coord.release()
        return False
