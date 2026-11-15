"""Shared GPU task lease and on-demand scale for image-moss.

Follows the 5090 GPU lease protocol (`gpu-task-moss-image`):
- Write this role's Lease first, then read conflicting peers (`music3`).
- A live conflicting peer (`gpu-task-music3`) means release and retry.
- `moss-image` does not conflict with `h3` or `indextts2`.
- Scales `Deployment/image-moss` from 0 -> 1 when held and back to 0 after
  the 10-minute (600s) idle window.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
import logging
import os
from pathlib import Path
import random
import threading
import time
from typing import Any, Callable
import uuid

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
        deployment = (
            os.environ.get("MOSS_GPU_TASK_DEPLOYMENT")
            or os.environ.get("MOSS_GPU_DEPLOYMENT_NAME")
            or "image-moss"
        ).strip() or None
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
        return self.replicas_for(self.deployment)

    def replicas_for(self, deployment: str | None) -> int | None:
        if not self.enabled or not deployment:
            return None
        try:
            return int(self.api.get_scale(deployment))
        except Exception:
            return None

    def scale_to(self, replicas: int) -> None:
        self.scale_deployment_to(self.deployment, replicas)

    def scale_deployment_to(self, deployment: str | None, replicas: int) -> None:
        if not self.enabled or not deployment:
            return
        if self.api.get_scale(deployment) != replicas:
            self.api.patch_scale(deployment, replicas)

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
        return _Hold(self, blocking=True)

    def try_hold(self) -> "_Hold":
        return _Hold(self, blocking=False)


class _Hold:
    def __init__(self, coord: GpuCoordinator, *, blocking: bool = True) -> None:
        self.coord = coord
        self.blocking = blocking
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "_Hold":
        if self.blocking:
            self.coord.acquire_for_queue()
        elif self.coord.enabled:
            if not self.coord.try_acquire():
                raise LeaseBusy(self.coord.peer_holder())
            with self.coord._lock:
                self.coord.in_flight += 1
                self.coord.idle_at = None
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


class SchedulerError(RuntimeError):
    def __init__(self, code: str, message: str = "", status_code: int = 500) -> None:
        self.code = code
        self.message = message
        self.status_code = status_code
        super().__init__(f"{code}: {message}" if message else code)


SCHEDULER_CONSUMER_SPECS: dict[str, dict[str, Any]] = {
    "moss-image": {
        "card_count": 1,
        "peak_budget_mib": 18432,
        "startup_free_min_mib": 19456,
        "priority_class": "exclusive_batch",
        "ttl_seconds": 240,
        "default_deployment": "image-moss",
    },
    "face-fusion": {
        "card_count": 1,
        "peak_budget_mib": 6144,
        "startup_free_min_mib": 6656,
        "priority_class": "shared_slice",
        "ttl_seconds": 180,
        "default_deployment": "image-face-fusion",
    },
}


@dataclass
class SchedulerLeaseGrant:
    lease_id: str
    fencing_token: int
    attempt_id: str
    stage_id: str
    consumer_kind: str
    holder_app_id: str
    parent_task_id: str = ""
    version: int = 1
    gpu_uuid: str = ""
    assigned_gpu_uuids: tuple[str, ...] = ()
    pod_uid: str = ""
    container_id: str = ""
    boot_id: str = ""
    expires_at: str = ""

    @property
    def is_target_bound(self) -> bool:
        return bool(self.pod_uid and self.container_id and self.boot_id)

    def headers(self) -> dict[str, str]:
        gpu_val = ",".join(self.assigned_gpu_uuids) if self.assigned_gpu_uuids else self.gpu_uuid
        hdrs = {
            "X-VF-GPU-Lease-ID": self.lease_id,
            "X-VF-GPU-Fencing-Token": str(self.fencing_token),
            "X-VF-GPU-Attempt-ID": self.attempt_id,
            "X-VF-GPU-Stage-ID": self.stage_id,
            "X-VF-GPU-Consumer-Kind": self.consumer_kind,
            "X-VF-GPU-Holder-App-ID": self.holder_app_id,
            # Keep short-form aliases for backwards compatibility
            "X-VF-Fencing-Token": str(self.fencing_token),
            "X-VF-Attempt-ID": self.attempt_id,
            "X-VF-Stage-ID": self.stage_id,
            "X-VF-Consumer-Kind": self.consumer_kind,
            "X-VF-Holder-App-ID": self.holder_app_id,
        }
        if self.parent_task_id:
            hdrs["X-VF-GPU-Parent-Task-ID"] = self.parent_task_id
        if self.pod_uid:
            hdrs["X-VF-GPU-Pod-UID"] = self.pod_uid
        if self.container_id:
            hdrs["X-VF-GPU-Container-ID"] = self.container_id
        if self.boot_id:
            hdrs["X-VF-GPU-Boot-ID"] = self.boot_id
        if gpu_val:
            hdrs["X-VF-GPU-UUID"] = gpu_val
        if self.expires_at:
            hdrs["X-VF-GPU-Expires-At"] = self.expires_at
        return hdrs


def _fsync_directory(dir_path: Path) -> None:
    if os.name == "nt":
        return
    dfd = os.open(str(dir_path), os.O_RDONLY)
    try:
        os.fsync(dfd)
    finally:
        os.close(dfd)


class SchedulerLeaseClient:
    """Station Runtime :5056 GPU Scheduler lease client for image-mcp."""

    def __init__(
        self,
        base_url: str = "",
        token: str = "",
        enabled: bool | None = None,
        holder_app_id: str = "image-mcp",
        queue_seconds: float = 1200.0,
        poll_interval_seconds: float = 2.0,
        heartbeat_interval_seconds: float = 15.0,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] | None = None,
        monotonic: Callable[[], float] | None = None,
        pending_releases_path: str | Path | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.enabled = bool(self.base_url) if enabled is None else bool(enabled)
        self.holder_app_id = holder_app_id
        self.queue_seconds = queue_seconds
        self.poll_interval_seconds = poll_interval_seconds
        self.heartbeat_interval_seconds = heartbeat_interval_seconds
        self._client = client
        self._sleep = sleep or time.sleep
        self._monotonic = monotonic or time.monotonic
        self._pending_lock = threading.RLock()
        raw_journal = (
            str(pending_releases_path).strip()
            if pending_releases_path is not None
            else os.environ.get("STATION_GPU_PENDING_RELEASES_PATH", "").strip()
        )
        self._pending_releases_path: Path | None = Path(raw_journal) if raw_journal else None
        self._pending_releases: dict[str, dict[str, Any]] = {}
        self._load_pending_releases_locked()

    @classmethod
    def from_environment(cls, *, pending_releases_path: str | Path | None = None) -> "SchedulerLeaseClient":
        url = os.environ.get("STATION_GPU_SCHEDULER_URL", "").strip()
        tok = (
            os.environ.get("STATION_GPU_SCHEDULER_TOKEN", "").strip()
            or os.environ.get("IMAGE_GPU_SCHEDULER_TOKEN", "").strip()
        )
        tok_file = os.environ.get("STATION_GPU_SCHEDULER_TOKEN_FILE", "").strip()
        if not tok and tok_file and Path(tok_file).is_file():
            tok = Path(tok_file).read_text(encoding="utf-8").strip()
        queue_sec = float(os.environ.get("MOSS_GPU_QUEUE_SECONDS", "1200"))
        holder_app_id = os.environ.get("STATION_GPU_HOLDER_APP_ID", "image-mcp").strip() or "image-mcp"
        env_journal = os.environ.get("STATION_GPU_PENDING_RELEASES_PATH", "").strip()
        return cls(
            base_url=url,
            token=tok,
            enabled=bool(url),
            holder_app_id=holder_app_id,
            queue_seconds=queue_sec,
            pending_releases_path=env_journal or pending_releases_path,
        )

    def configure_journal(self, path: str | Path | None) -> None:
        if path is None:
            return
        with self._pending_lock:
            if self._pending_releases_path is not None:
                return
            self._pending_releases_path = Path(path)
            existing_mem = dict(self._pending_releases)
            self._load_pending_releases_locked()
            if existing_mem:
                self._pending_releases.update(existing_mem)
                self._persist_pending_releases_locked()

    def _load_pending_releases_locked(self) -> None:
        if self._pending_releases_path is None or not os.path.lexists(self._pending_releases_path):
            return
        if os.path.islink(self._pending_releases_path) or not self._pending_releases_path.is_file():
            raise SchedulerError("JOURNAL_CORRUPTED", "pending releases journal is not a regular file", 503)
        try:
            payload = json.loads(self._pending_releases_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise SchedulerError("JOURNAL_CORRUPTED", f"failed reading pending releases journal: {exc}", 503) from exc
        if not isinstance(payload, dict):
            raise SchedulerError("JOURNAL_CORRUPTED", "pending releases journal root must be an object", 503)
        loaded: dict[str, dict[str, Any]] = {}
        for k, v in payload.items():
            if not isinstance(k, str) or not k.strip() or not isinstance(v, dict):
                raise SchedulerError("JOURNAL_CORRUPTED", "invalid pending release journal entry", 503)
            loaded[k.strip()] = dict(v)
        self._pending_releases = loaded

    def _persist_pending_releases_locked(self) -> None:
        if self._pending_releases_path is None:
            return
        self._pending_releases_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self._pending_releases_path.with_name(
            f"{self._pending_releases_path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
        )
        data = json.dumps(self._pending_releases, separators=(",", ":"), sort_keys=True).encode("utf-8")
        try:
            fd = os.open(str(tmp_path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            try:
                view = memoryview(data)
                while view:
                    written = os.write(fd, view)
                    if written <= 0:
                        raise OSError("short_write_zero_bytes")
                    view = view[written:]
                os.fsync(fd)
            finally:
                os.close(fd)
            os.replace(str(tmp_path), str(self._pending_releases_path))
            _fsync_directory(self._pending_releases_path.parent)
        except OSError as exc:
            try:
                if os.path.lexists(tmp_path):
                    os.unlink(tmp_path)
            except OSError:
                pass
            raise SchedulerError("JOURNAL_PERSIST_FAILED", str(exc), 503) from exc

    @staticmethod
    def _is_confirmed_release_receipt(lease_id: str, payload: Any) -> bool:
        if not isinstance(payload, dict):
            return False
        if str(payload.get("lease_id") or "").strip() != lease_id:
            return False
        return str(payload.get("state") or "").strip().lower() in ("released", "expired")

    def _auth_headers(self) -> dict[str, str]:
        if not self.token:
            return {}
        return {"Authorization": f"Bearer {self.token}"}

    def _request_raw(self, method: str, path: str, json_body: dict[str, Any] | None = None) -> httpx.Response:
        url = f"{self.base_url}{path}"
        headers = self._auth_headers()
        if self._client is not None:
            return self._client.request(method, url, json=json_body, headers=headers, timeout=15.0)
        with httpx.Client(timeout=15.0, trust_env=False) as client:
            return client.request(method, url, json=json_body, headers=headers)

    @staticmethod
    def _parse_error(resp: httpx.Response) -> SchedulerError:
        code = f"HTTP_{resp.status_code}"
        msg = resp.text
        try:
            data = resp.json() if resp.content else {}
        except ValueError:
            data = {}
        if isinstance(data, dict):
            top_code = data.get("error_code")
            top_msg = data.get("message")
            nested = data.get("error")
            if isinstance(top_code, str) and top_code:
                code = top_code
                if isinstance(top_msg, str) and top_msg:
                    msg = top_msg
            elif isinstance(nested, dict):
                if isinstance(nested.get("code"), str) and nested["code"]:
                    code = nested["code"]
                if isinstance(nested.get("message"), str) and nested["message"]:
                    msg = nested["message"]
        return SchedulerError(code, msg, status_code=resp.status_code)

    def get_pending_release(self, lease_id: str) -> dict[str, Any] | None:
        with self._pending_lock:
            entry = self._pending_releases.get(lease_id)
            return dict(entry) if isinstance(entry, dict) else None

    def flush_pending_releases(self, *, include_quarantined: bool = False) -> int:
        if not self.enabled:
            return 0
        with self._pending_lock:
            if not self._pending_releases:
                return 0
            snapshot = [
                (lid, dict(b))
                for lid, b in self._pending_releases.items()
                if include_quarantined or not bool(b.get("_quarantined_400"))
            ]
        if not snapshot:
            return 0
        flushed = 0
        for lease_id, body in snapshot:
            retry_body = {
                k: v
                for k, v in body.items()
                if k != "expected_version" and not str(k).startswith("_")
            }
            try:
                resp = self._request_raw("POST", f"/internal/v1/gpu-scheduler/leases/{lease_id}/release", retry_body)
                if resp.status_code == 200:
                    payload = resp.json() if resp.content else {}
                    if self._is_confirmed_release_receipt(lease_id, payload):
                        with self._pending_lock:
                            self._pending_releases.pop(lease_id, None)
                            self._persist_pending_releases_locked()
                        flushed += 1
                elif resp.status_code == 400:
                    err = self._parse_error(resp)
                    logging.error(
                        "image_gpu_release_quarantined_400 lease_id=%s code=%s message=%s",
                        lease_id,
                        err.code,
                        err.message,
                    )
                    with self._pending_lock:
                        cur = self._pending_releases.get(lease_id)
                        if isinstance(cur, dict):
                            upd = dict(cur)
                            upd["_quarantined_400"] = True
                            upd["_last_error"] = f"{err.code}:{err.message}"
                            self._pending_releases[lease_id] = upd
                            self._persist_pending_releases_locked()
            except Exception:
                continue
        return flushed

    @staticmethod
    def _extract_gpu_uuids(data: dict[str, Any]) -> tuple[str, ...]:
        raw_list = data.get("gpu_uuids") or data.get("assigned_gpu_uuids") or ()
        if isinstance(raw_list, (list, tuple)):
            items = tuple(str(u) for u in raw_list if u)
            if items:
                return items
        single = str(data.get("gpu_uuid") or "").strip()
        return (single,) if single else ()

    def acquire_for_queue(
        self,
        *,
        consumer_kind: str,
        task_id: str,
        stage_id: str = "execute",
        attempt_id: str | None = None,
        k8s_slots_required: int = 0,
        pod_uid: str = "",
        container_id: str = "",
        boot_id: str = "",
        visible_gpu_uuids: list[str] | tuple[str, ...] | None = None,
    ) -> SchedulerLeaseGrant | None:
        if not self.enabled:
            return None
        self.flush_pending_releases()
        spec = SCHEDULER_CONSUMER_SPECS[consumer_kind]
        att = attempt_id or f"att-{uuid.uuid4().hex[:12]}"
        idem = f"{self.holder_app_id}:{consumer_kind}:{task_id}:{att}:{stage_id}:acquire"
        body: dict[str, Any] = {
            "idempotency_key": idem,
            "holder_app_id": self.holder_app_id,
            "consumer_kind": consumer_kind,
            "task_id": task_id,
            "attempt_id": att,
            "stage_id": stage_id,
            "pod_uid": pod_uid,
            "container_id": container_id,
            "boot_id": boot_id,
            "card_count": int(spec["card_count"]),
            "k8s_slots_required": int(k8s_slots_required),
            "peak_budget_mib": int(spec["peak_budget_mib"]),
            "startup_free_min_mib": int(spec["startup_free_min_mib"]),
            "priority_class": str(spec["priority_class"]),
            "ttl_seconds": int(spec["ttl_seconds"]),
        }
        if visible_gpu_uuids:
            body["candidate_gpu_uuids"] = [str(u) for u in visible_gpu_uuids if str(u)]
        deadline = self._monotonic() + self.queue_seconds
        resp = self._request_raw("POST", "/internal/v1/gpu-scheduler/leases/acquire", body)
        if resp.status_code not in (200, 201, 202):
            raise LeaseBusy(f"scheduler_acquire_http_{resp.status_code}")
        data = resp.json()
        lease_id = str(data.get("lease_id") or "")
        while str(data.get("state") or "") in ("queued", "draining"):
            if self._monotonic() >= deadline:
                try:
                    self.release_raw(
                        lease_id,
                        int(data.get("fencing_token") or 0),
                        worker_settled=False,
                        reclaim_mode="abort_unstarted",
                    )
                except Exception:
                    pass
                raise LeaseBusy(str(data.get("blocking_reason") or data.get("queue_reason") or "waiting_gpu_lease"))
            self._sleep(self.poll_interval_seconds)
            poll = self._request_raw("GET", f"/internal/v1/gpu-scheduler/leases/{lease_id}")
            if poll.status_code != 200:
                raise LeaseBusy(f"scheduler_poll_http_{poll.status_code}")
            data = poll.json()

        if str(data.get("state") or "") not in ("leased", "running", "granted"):
            raise LeaseBusy(str(data.get("state") or "lease_not_granted"))

        assigned = self._extract_gpu_uuids(data)
        default_exp = format_time(datetime.now(timezone.utc) + timedelta(seconds=int(spec["ttl_seconds"])))
        return SchedulerLeaseGrant(
            lease_id=lease_id,
            fencing_token=int(data.get("fencing_token") or 1),
            attempt_id=str(data.get("attempt_id") or att),
            stage_id=str(data.get("stage_id") or stage_id),
            consumer_kind=consumer_kind,
            holder_app_id=self.holder_app_id,
            parent_task_id=task_id,
            version=int(data.get("version") or 1),
            gpu_uuid=assigned[0] if assigned else "",
            assigned_gpu_uuids=assigned,
            pod_uid=str(data.get("pod_uid") or pod_uid or ""),
            container_id=str(data.get("container_id") or container_id or ""),
            boot_id=str(data.get("boot_id") or boot_id or ""),
            expires_at=str(data.get("expires_at") or default_exp),
        )

    def bind_target(
        self,
        grant: SchedulerLeaseGrant,
        *,
        runtime_identity: dict[str, Any],
        ttl_seconds: int = 180,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        pod_uid = str(runtime_identity.get("pod_uid") or "").strip()
        container_id = str(runtime_identity.get("container_id") or "").strip()
        boot_id = str(runtime_identity.get("boot_id") or "").strip()
        if not pod_uid or not container_id or not boot_id:
            raise SchedulerError("INVALID_TARGET_IDENTITY", "bind_target requires pod_uid, container_id, and boot_id", 400)
        visible = runtime_identity.get("visible_gpu_uuids") or ()
        actual_gpus = [str(u) for u in visible if str(u)] if isinstance(visible, (list, tuple)) else []
        body: dict[str, Any] = {
            "idempotency_key": idempotency_key or f"bind:{grant.lease_id}:{grant.fencing_token}:{pod_uid}",
            "fencing_token": int(grant.fencing_token),
            "expected_version": int(grant.version),
            "ttl_seconds": int(ttl_seconds),
            "pod_uid": pod_uid,
            "container_id": container_id,
            "boot_id": boot_id,
        }
        if actual_gpus:
            body["actual_gpu_uuids"] = actual_gpus
        resp = self._request_raw("POST", f"/internal/v1/gpu-scheduler/leases/{grant.lease_id}/bind-target", body)
        if resp.status_code != 200:
            raise self._parse_error(resp)
        payload = resp.json() if resp.content else {}
        if isinstance(payload.get("version"), int):
            grant.version = int(payload["version"])
        if isinstance(payload.get("fencing_token"), int) and payload["fencing_token"] > 0:
            grant.fencing_token = int(payload["fencing_token"])
        assigned = self._extract_gpu_uuids(payload)
        if assigned:
            grant.assigned_gpu_uuids = assigned
            grant.gpu_uuid = assigned[0]
        elif actual_gpus:
            grant.assigned_gpu_uuids = tuple(actual_gpus)
            grant.gpu_uuid = actual_gpus[0]
        grant.pod_uid = str(payload.get("pod_uid") or pod_uid)
        grant.container_id = str(payload.get("container_id") or container_id)
        grant.boot_id = str(payload.get("boot_id") or boot_id)
        if payload.get("expires_at"):
            grant.expires_at = str(payload["expires_at"])
        return payload

    def heartbeat(
        self,
        grant: SchedulerLeaseGrant,
        *,
        ttl_seconds: int = 180,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "idempotency_key": f"hb:{grant.lease_id}:{uuid.uuid4().hex[:10]}",
            "fencing_token": grant.fencing_token,
            "expected_version": grant.version,
            "ttl_seconds": ttl_seconds,
        }
        resp = self._request_raw("POST", f"/internal/v1/gpu-scheduler/leases/{grant.lease_id}/heartbeat", body)
        if resp.status_code != 200:
            raise self._parse_error(resp)
        payload = resp.json() if resp.content else {}
        if isinstance(payload.get("version"), int):
            grant.version = int(payload["version"])
        if payload.get("expires_at"):
            grant.expires_at = str(payload["expires_at"])
        return payload

    def release_raw(
        self,
        lease_id: str,
        fencing_token: int,
        *,
        worker_settled: bool,
        runtime_identity: dict[str, Any] | None = None,
        reclaim_mode: str = "",
        reclaim_proof: dict[str, Any] | None = None,
        expected_version: int = 0,
    ) -> dict[str, Any]:
        ident = runtime_identity or {}
        body: dict[str, Any] = {
            "idempotency_key": f"rel:{lease_id}:{fencing_token}:{reclaim_mode or 'plain'}",
            "fencing_token": int(fencing_token),
            "worker_settled": bool(worker_settled),
        }
        if expected_version > 0:
            body["expected_version"] = int(expected_version)
        if ident.get("pod_uid"):
            body["pod_uid"] = str(ident["pod_uid"])
        if ident.get("container_id"):
            body["container_id"] = str(ident["container_id"])
        if ident.get("boot_id"):
            body["boot_id"] = str(ident["boot_id"])
        if ident.get("supervisor_pid"):
            body["supervisor_pid"] = int(ident["supervisor_pid"])
        if ident.get("supervisor_starttime_ticks"):
            body["supervisor_starttime_ticks"] = int(ident["supervisor_starttime_ticks"])
        if reclaim_mode:
            body["reclaim_mode"] = reclaim_mode
        if reclaim_proof is not None:
            body["reclaim_proof"] = reclaim_proof
        retry_body = {k: v for k, v in body.items() if k != "expected_version"}
        with self._pending_lock:
            self._pending_releases[lease_id] = retry_body
            self._persist_pending_releases_locked()
        try:
            resp = self._request_raw("POST", f"/internal/v1/gpu-scheduler/leases/{lease_id}/release", body)
        except Exception as exc:
            raise SchedulerError("RELEASE_TRANSPORT_FAILED", str(exc), 503) from exc
        if resp.status_code != 200:
            err = self._parse_error(resp)
            if resp.status_code == 400:
                logging.error(
                    "image_gpu_release_quarantined_400 lease_id=%s code=%s message=%s",
                    lease_id,
                    err.code,
                    err.message,
                )
                with self._pending_lock:
                    cur = self._pending_releases.get(lease_id)
                    if isinstance(cur, dict):
                        upd = dict(cur)
                        upd["_quarantined_400"] = True
                        upd["_last_error"] = f"{err.code}:{err.message}"
                        self._pending_releases[lease_id] = upd
                        self._persist_pending_releases_locked()
            raise err
        try:
            payload = resp.json() if resp.content else {}
        except ValueError:
            payload = {}
        if not self._is_confirmed_release_receipt(lease_id, payload):
            state = payload.get("state") if isinstance(payload, dict) else None
            raise SchedulerError("RELEASE_NOT_TERMINAL", f"Unconfirmed release receipt for {lease_id}: state={state!r}", 502)
        with self._pending_lock:
            self._pending_releases.pop(lease_id, None)
            self._persist_pending_releases_locked()
        return payload

    def release(
        self,
        grant: SchedulerLeaseGrant | None,
        *,
        worker_settled: bool,
        runtime_identity: dict[str, Any] | None = None,
        reclaim_mode: str = "",
        reclaim_proof: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if grant is None or not self.enabled:
            return {"skipped": True}
        return self.release_raw(
            grant.lease_id,
            grant.fencing_token,
            worker_settled=worker_settled,
            runtime_identity=runtime_identity,
            reclaim_mode=reclaim_mode,
            reclaim_proof=reclaim_proof,
            expected_version=grant.version,
        )
