"""6-tuple GPU lease fencing, runtime identity, fail-closed durable token sealing, and S2->S3 VRAM eviction guard for image-moss."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hmac
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import threading
import time
from typing import Any, Callable, Mapping
import uuid


_CONTAINER_ID_RE = re.compile(r"([0-9a-f]{64})")
_GPU_UUID_RE = re.compile(r"^GPU-[0-9a-fA-F-]{36}$")


def normalize_container_id(raw: str) -> str:
    val = (raw or "").strip()
    for prefix in ("containerd://", "docker://", "cri-o://"):
        if val.startswith(prefix):
            val = val[len(prefix) :].strip()
    return val


def is_unknown_identity_value(raw: str) -> bool:
    val = (raw or "").strip().lower()
    return (not val) or val.startswith("unknown") or val.startswith("local-")


def parse_expires_epoch(raw: str) -> float | None:
    val = (raw or "").strip()
    if not val:
        return None
    try:
        return float(val)
    except ValueError:
        pass
    try:
        return datetime.fromisoformat(val.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _fsync_directory(dir_path: Path) -> None:
    if os.name == "nt":
        return
    dfd = os.open(str(dir_path), os.O_RDONLY)
    try:
        os.fsync(dfd)
    finally:
        os.close(dfd)


@dataclass(frozen=True)
class RuntimeIdentity:
    pod_uid: str
    container_id: str
    boot_id: str
    supervisor_pid: int
    supervisor_starttime_ticks: int
    visible_gpu_uuids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "pod_uid": self.pod_uid,
            "container_id": self.container_id,
            "boot_id": self.boot_id,
            "supervisor_pid": self.supervisor_pid,
            "supervisor_starttime_ticks": self.supervisor_starttime_ticks,
            "visible_gpu_uuids": list(self.visible_gpu_uuids),
        }


@dataclass(frozen=True)
class LeaseGrant:
    lease_id: str
    fencing_token: int
    attempt_id: str
    stage_id: str
    consumer_kind: str
    holder_app_id: str
    pod_uid: str = ""
    container_id: str = ""
    boot_id: str = ""
    gpu_uuids: tuple[str, ...] = ()
    expires_epoch: float | None = None
    exec_id: str = ""


class LeaseFenceError(Exception):
    def __init__(self, status_code: int, error_code: str, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error_code = error_code
        self.message = message


class FencePersistenceError(LeaseFenceError):
    def __init__(self, message: str) -> None:
        super().__init__(503, "FENCE_STATE_CORRUPTED", message)


def _read_first_line(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip().splitlines()[0].strip()
    except Exception:
        return ""


def _detect_container_id() -> str:
    env_cid = os.environ.get("CONTAINER_ID", "").strip()
    if env_cid:
        return env_cid
    try:
        for line in Path("/proc/self/cgroup").read_text(encoding="utf-8", errors="replace").splitlines():
            m = _CONTAINER_ID_RE.search(line)
            if m:
                return f"containerd://{m.group(1)}"
    except Exception:
        pass
    return f"local-{os.getpid()}"


def _detect_starttime_ticks(pid: int) -> int:
    env_ticks = (
        os.environ.get("SUPERVISOR_STARTTIME_TICKS", "").strip()
        or os.environ.get("VF_SUPERVISOR_STARTTIME_TICKS", "").strip()
    )
    if env_ticks.isdigit():
        return int(env_ticks)
    try:
        stat_raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8", errors="replace")
        rparen = stat_raw.rfind(")")
        if rparen != -1:
            fields = stat_raw[rparen + 2 :].split()
            if len(fields) >= 20:
                return int(fields[19])
    except Exception:
        pass
    return 1


def _detect_visible_gpu_uuids() -> tuple[str, ...]:
    override = os.environ.get("VF_VISIBLE_GPU_UUIDS", "").strip()
    if override:
        return tuple(item.strip() for item in override.split(",") if item.strip())
    env_vis = os.environ.get("NVIDIA_VISIBLE_DEVICES", "").strip()
    if env_vis and env_vis not in ("all", "void", "none"):
        concrete = [u.strip() for u in env_vis.split(",") if _GPU_UUID_RE.match(u.strip())]
        if concrete:
            return tuple(concrete)
    try:
        values = [
            entry.split(b"=", 1)[1].decode("ascii")
            for entry in Path("/proc/self/environ").read_bytes().split(b"\0")
            if entry.startswith(b"NVIDIA_VISIBLE_DEVICES=")
        ]
        for v in values:
            parts = [u.strip() for u in v.split(",") if _GPU_UUID_RE.match(u.strip())]
            if parts:
                return tuple(parts)
    except Exception:
        pass
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
            text=True,
            timeout=5,
            stderr=subprocess.DEVNULL,
        ).strip().splitlines()
        parsed = [line.strip() for line in out if _GPU_UUID_RE.match(line.strip())]
        if parsed:
            return tuple(parsed)
    except Exception:
        pass
    return ()


def detect_runtime_identity() -> RuntimeIdentity:
    pid = os.getpid()
    pod_uid = (
        os.environ.get("POD_UID", "").strip()
        or _read_first_line(Path("/etc/podinfo/uid"))
        or f"local-pod-{pid}"
    )
    boot_id = (
        os.environ.get("NODE_BOOT_ID", "").strip()
        or _read_first_line(Path("/proc/sys/kernel/random/boot_id"))
        or "local-boot"
    )
    return RuntimeIdentity(
        pod_uid=pod_uid,
        container_id=_detect_container_id(),
        boot_id=boot_id,
        supervisor_pid=pid,
        supervisor_starttime_ticks=_detect_starttime_ticks(pid),
        visible_gpu_uuids=_detect_visible_gpu_uuids(),
    )


def default_guard_state_path(service_slug: str = "moss-image") -> Path:
    env_raw = os.environ.get("VF_GPU_GUARD_STATE_PATH", "").strip()
    if env_raw:
        return Path(env_raw)
    return Path(tempfile.gettempdir()) / f"vf-gpu-guard-{service_slug}.json"


class LeaseFenceGuard:
    """Validates 6-tuple GPU grant headers, enforces identity/expiry, and fail-closed durably seals tokens."""

    def __init__(
        self,
        expected_consumer_kind: str = "moss-image",
        policy: str | None = None,
        identity: RuntimeIdentity | None = None,
        state_path: str | Path | None = None,
        *,
        write_fn: Callable[[int, Any], int] | None = None,
        fsync_fn: Callable[[int], None] | None = None,
        replace_fn: Callable[[str, str], None] | None = None,
        fsync_dir_fn: Callable[[Path], None] | None = None,
    ) -> None:
        self.expected_consumer_kind = expected_consumer_kind
        self._policy_override = policy
        self.identity = identity or detect_runtime_identity()
        raw_state = (
            str(state_path).strip()
            if state_path is not None
            else os.environ.get("VF_GPU_GUARD_STATE_PATH", "").strip()
        )
        self._state_path = Path(raw_state) if raw_state else None
        self._write_fn = write_fn or os.write
        self._fsync_fn = fsync_fn or os.fsync
        self._replace_fn = replace_fn or os.replace
        self._fsync_dir_fn = fsync_dir_fn or _fsync_directory
        self._lock = threading.RLock()
        self._corrupted = False
        self._corrupt_reason = ""
        self._revoked_leases: set[str] = set()
        self._consumed_tokens: set[int] = set()
        self._max_fencing_token: int = 0
        self._active_lease_fencing: dict[str, int] = {}
        self._inflight_tasks: dict[str, dict[str, Any]] = {}
        self._active_executions: set[str] = set()
        self._load_state_locked()

    def _ensure_healthy_locked(self) -> None:
        if self._corrupted:
            raise FencePersistenceError(self._corrupt_reason or "fence_state_corrupted")

    def _load_state_locked(self) -> None:
        if self._state_path is None or not os.path.lexists(self._state_path):
            return
        try:
            if os.path.islink(self._state_path) or not self._state_path.is_file():
                raise ValueError("state_path_not_regular_file")
            payload = json.loads(self._state_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("state_payload_not_dict")
            raw_max = payload.get("max_fencing_token", 0)
            if isinstance(raw_max, bool) or not isinstance(raw_max, int) or raw_max < 0:
                raise ValueError("invalid_max_fencing_token")
            raw_revoked = payload.get("revoked_leases", [])
            if not isinstance(raw_revoked, list):
                raise ValueError("invalid_revoked_leases")
            revoked: set[str] = set()
            for item in raw_revoked:
                if not isinstance(item, str) or not item.strip():
                    raise ValueError("invalid_revoked_lease_entry")
                revoked.add(item.strip())
            raw_consumed = payload.get("consumed_tokens", [])
            if not isinstance(raw_consumed, list):
                raise ValueError("invalid_consumed_tokens")
            consumed: set[int] = set()
            for item in raw_consumed:
                if isinstance(item, bool) or not isinstance(item, int) or item <= 0:
                    raise ValueError("invalid_consumed_token_entry")
                consumed.add(item)
            raw_inflight = payload.get("inflight_tasks", {})
            if not isinstance(raw_inflight, dict):
                raise ValueError("invalid_inflight_tasks")
            cur_pid = int(self.identity.supervisor_pid)
            cur_ticks = int(self.identity.supervisor_starttime_ticks)
            inflight: dict[str, dict[str, Any]] = {}
            needs_repersist = False
            for exec_id, meta in raw_inflight.items():
                if not isinstance(exec_id, str) or not isinstance(meta, dict):
                    raise ValueError("invalid_inflight_entry")
                owner_pid = int(meta.get("owner_pid") or 0)
                owner_ticks = int(meta.get("owner_ticks") or 0)
                if owner_pid == cur_pid and owner_ticks == cur_ticks:
                    inflight[exec_id] = dict(meta)
                else:
                    lid = str(meta.get("lease_id") or "").strip()
                    tok = int(meta.get("fencing_token") or 0)
                    if lid:
                        revoked.add(lid)
                    if tok > 0:
                        consumed.add(tok)
                        if tok > raw_max:
                            raw_max = tok
                    needs_repersist = True
            self._max_fencing_token = raw_max
            self._revoked_leases = revoked
            self._consumed_tokens = consumed
            self._inflight_tasks = inflight
            if needs_repersist:
                self._commit_state_locked(
                    max_fencing_token=raw_max,
                    revoked_leases=revoked,
                    consumed_tokens=consumed,
                    inflight_tasks=inflight,
                )
        except Exception as exc:
            self._corrupted = True
            self._corrupt_reason = f"fence_state_corrupted:{type(exc).__name__}:{exc}"

    def _commit_state_locked(
        self,
        *,
        max_fencing_token: int,
        revoked_leases: set[str],
        consumed_tokens: set[int],
        inflight_tasks: dict[str, dict[str, Any]],
    ) -> None:
        self._ensure_healthy_locked()
        if self._state_path is not None:
            tmp_path = self._state_path.with_name(
                f"{self._state_path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
            )
            try:
                self._state_path.parent.mkdir(parents=True, exist_ok=True)
                data = json.dumps(
                    {
                        "max_fencing_token": max_fencing_token,
                        "revoked_leases": sorted(revoked_leases),
                        "consumed_tokens": sorted(consumed_tokens),
                        "inflight_tasks": inflight_tasks,
                    },
                    separators=(",", ":"),
                ).encode("utf-8")
                fd = os.open(str(tmp_path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
                try:
                    view = memoryview(data)
                    while view:
                        written = self._write_fn(fd, view)
                        if written <= 0:
                            raise OSError("short_write_zero_bytes")
                        view = view[written:]
                    self._fsync_fn(fd)
                finally:
                    os.close(fd)
                self._replace_fn(str(tmp_path), str(self._state_path))
                self._fsync_dir_fn(self._state_path.parent)
            except Exception as exc:
                try:
                    if os.path.lexists(tmp_path):
                        os.unlink(tmp_path)
                except OSError:
                    pass
                self._corrupted = True
                self._corrupt_reason = f"fence_state_persist_failed:{type(exc).__name__}:{exc}"
                raise FencePersistenceError(self._corrupt_reason) from exc

        self._max_fencing_token = max_fencing_token
        self._revoked_leases = revoked_leases
        self._consumed_tokens = consumed_tokens
        self._inflight_tasks = inflight_tasks

    @property
    def policy(self) -> str:
        raw = (
            self._policy_override
            if self._policy_override is not None
            else os.environ.get("IMAGE_GPU_GRANT_POLICY", "permissive")
        )
        val = raw.strip().lower()
        return "enforcing" if val in ("enforcing", "enforce", "require") else "permissive"

    def verify_runtime_token(self, authorization: str | None) -> None:
        expected = (
            os.environ.get("IMAGE_RUNTIME_TOKEN", "").strip()
            or os.environ.get("STATION_GPU_SCHEDULER_TOKEN", "").strip()
        )
        if not expected:
            raise LeaseFenceError(401, "RUNTIME_TOKEN_UNCONFIGURED", "Runtime bearer token is not configured")
        if not authorization or not authorization.lower().startswith("bearer "):
            raise LeaseFenceError(401, "UNAUTHORIZED", "Missing Bearer token")
        provided = authorization[7:].strip()
        if not hmac.compare_digest(provided, expected):
            raise LeaseFenceError(403, "FORBIDDEN", "Invalid runtime Bearer token")

    def _visible_gpus(self) -> tuple[str, ...]:
        if self.identity.visible_gpu_uuids:
            return self.identity.visible_gpu_uuids
        return _detect_visible_gpu_uuids()

    def begin_unheadered_execution(self) -> str:
        exec_id = f"unheadered-{uuid.uuid4().hex}"
        with self._lock:
            self._ensure_healthy_locked()
            self._active_executions.add(exec_id)
        return exec_id

    def end_unheadered_execution(self, exec_id: str | None) -> None:
        if not exec_id:
            return
        with self._lock:
            self._active_executions.discard(exec_id)

    def validate_request(self, headers: Mapping[str, str]) -> LeaseGrant | None:
        with self._lock:
            self._ensure_healthy_locked()
        lower = {k.lower(): v.strip() for k, v in headers.items() if isinstance(v, str)}
        lease_id = lower.get("x-vf-gpu-lease-id", "")
        token_raw = lower.get("x-vf-gpu-fencing-token", "") or lower.get("x-vf-fencing-token", "")
        attempt_id = lower.get("x-vf-gpu-attempt-id", "") or lower.get("x-vf-attempt-id", "")
        stage_id = lower.get("x-vf-gpu-stage-id", "") or lower.get("x-vf-stage-id", "")
        consumer_kind = lower.get("x-vf-gpu-consumer-kind", "") or lower.get("x-vf-consumer-kind", "")
        holder_app_id = lower.get("x-vf-gpu-holder-app-id", "") or lower.get("x-vf-holder-app-id", "")
        pod_uid = lower.get("x-vf-gpu-pod-uid", "")
        container_id = lower.get("x-vf-gpu-container-id", "")
        boot_id = lower.get("x-vf-gpu-boot-id", "")
        gpu_uuid_raw = lower.get("x-vf-gpu-uuid", "")
        expires_raw = lower.get("x-vf-gpu-expires-at", "")

        enforcing = self.policy == "enforcing"
        if enforcing and self._state_path is None:
            raise FencePersistenceError("durable state_path is required in enforcing mode")

        present_any = any(
            bool(x)
            for x in (
                lease_id,
                token_raw,
                attempt_id,
                stage_id,
                consumer_kind,
                holder_app_id,
                pod_uid,
                container_id,
                boot_id,
                gpu_uuid_raw,
                expires_raw,
            )
        )
        if not present_any:
            if enforcing:
                raise LeaseFenceError(
                    403,
                    "GPU_GRANT_REQUIRED",
                    f"6-tuple X-VF-GPU-* grant headers required for {self.expected_consumer_kind}",
                )
            return None

        if consumer_kind and consumer_kind != self.expected_consumer_kind:
            raise LeaseFenceError(
                403,
                "CONSUMER_KIND_MISMATCH",
                f"Expected X-VF-GPU-Consumer-Kind={self.expected_consumer_kind!r}, got {consumer_kind!r}",
            )

        core_present = all(bool(x) for x in (lease_id, token_raw, attempt_id, stage_id, consumer_kind, holder_app_id))
        if not core_present:
            raise LeaseFenceError(403, "GPU_GRANT_INCOMPLETE", "Incomplete 6-tuple X-VF-GPU-* grant headers")

        visible_gpus = self._visible_gpus()
        if enforcing:
            if (
                is_unknown_identity_value(self.identity.pod_uid)
                or is_unknown_identity_value(self.identity.container_id)
                or is_unknown_identity_value(self.identity.boot_id)
                or not visible_gpus
            ):
                raise LeaseFenceError(
                    503,
                    "GPU_IDENTITY_UNAVAILABLE",
                    "Receiver runtime identity or visible GPU UUID is unavailable in enforcing mode",
                )
            if not all(bool(x) for x in (pod_uid, container_id, boot_id, gpu_uuid_raw, expires_raw)):
                raise LeaseFenceError(
                    403,
                    "GPU_GRANT_INCOMPLETE",
                    "Missing target pod_uid, container_id, boot_id, gpu_uuid, or expires_at in enforcing mode",
                )

        try:
            fencing_token = int(token_raw)
        except ValueError as exc:
            raise LeaseFenceError(403, "STALE_FENCING_TOKEN", "X-VF-GPU-Fencing-Token must be a positive integer") from exc
        if fencing_token <= 0:
            raise LeaseFenceError(403, "STALE_FENCING_TOKEN", "X-VF-GPU-Fencing-Token must be > 0")

        if pod_uid and pod_uid != self.identity.pod_uid:
            raise LeaseFenceError(403, "POD_UID_MISMATCH", f"Pod UID mismatch: expected {self.identity.pod_uid!r}")
        if container_id and normalize_container_id(container_id) != normalize_container_id(self.identity.container_id):
            raise LeaseFenceError(403, "CONTAINER_ID_MISMATCH", "Container ID mismatch")
        if boot_id and boot_id != self.identity.boot_id:
            raise LeaseFenceError(403, "BOOT_ID_MISMATCH", "Node boot_id mismatch")

        granted_gpus = tuple(u.strip() for u in gpu_uuid_raw.split(",") if u.strip())
        if granted_gpus and visible_gpus:
            for g in granted_gpus:
                if g not in visible_gpus:
                    raise LeaseFenceError(403, "GPU_UUID_MISMATCH", f"Granted GPU {g!r} not in visible GPUs")

        expires_epoch = parse_expires_epoch(expires_raw)
        if expires_raw and expires_epoch is None:
            raise LeaseFenceError(403, "GPU_GRANT_INCOMPLETE", "Malformed X-VF-GPU-Expires-At timestamp")
        if expires_epoch is not None and expires_epoch <= time.time():
            raise LeaseFenceError(409, "LEASE_EXPIRED", "GPU lease grant has expired")

        exec_id = f"grant-{uuid.uuid4().hex}"
        with self._lock:
            self._ensure_healthy_locked()
            if lease_id in self._revoked_leases:
                raise LeaseFenceError(409, "LEASE_REVOKED", f"GPU lease {lease_id} has been revoked or sealed")
            if fencing_token in self._consumed_tokens or fencing_token <= self._max_fencing_token:
                raise LeaseFenceError(
                    409,
                    "STALE_FENCING_TOKEN",
                    f"Stale or already consumed fencing token {fencing_token} (max={self._max_fencing_token})",
                )
            next_inflight = dict(self._inflight_tasks)
            next_inflight[exec_id] = {
                "lease_id": lease_id,
                "fencing_token": fencing_token,
                "owner_pid": int(self.identity.supervisor_pid),
                "owner_ticks": int(self.identity.supervisor_starttime_ticks),
            }
            self._commit_state_locked(
                max_fencing_token=fencing_token,
                revoked_leases=set(self._revoked_leases),
                consumed_tokens=set(self._consumed_tokens),
                inflight_tasks=next_inflight,
            )
            self._active_lease_fencing[lease_id] = fencing_token
            self._active_executions.add(exec_id)

        return LeaseGrant(
            lease_id=lease_id,
            fencing_token=fencing_token,
            attempt_id=attempt_id,
            stage_id=stage_id,
            consumer_kind=consumer_kind,
            holder_app_id=holder_app_id,
            pod_uid=pod_uid or self.identity.pod_uid,
            container_id=container_id or self.identity.container_id,
            boot_id=boot_id or self.identity.boot_id,
            gpu_uuids=granted_gpus,
            expires_epoch=expires_epoch,
            exec_id=exec_id,
        )

    def revoke_lease(self, lease_id: str, fencing_token: int = 0) -> dict[str, Any]:
        clean_id = (lease_id or "").strip()
        if not clean_id:
            raise LeaseFenceError(400, "INVALID_ARGUMENT", "lease_id is required")
        with self._lock:
            self._ensure_healthy_locked()
            next_revoked = set(self._revoked_leases)
            next_revoked.add(clean_id)
            next_consumed = set(self._consumed_tokens)
            next_max = self._max_fencing_token
            prev_tok = self._active_lease_fencing.get(clean_id, 0)
            for tok in (prev_tok, fencing_token):
                if tok > 0:
                    next_consumed.add(tok)
                    if tok > next_max:
                        next_max = tok
            self._commit_state_locked(
                max_fencing_token=next_max,
                revoked_leases=next_revoked,
                consumed_tokens=next_consumed,
                inflight_tasks=dict(self._inflight_tasks),
            )
            self._active_lease_fencing.pop(clean_id, None)
            consumed = self._max_fencing_token
        return {
            "revoked": True,
            "revoked_lease_id": clean_id,
            "consumed_fencing_token": consumed,
            "proof_source": "receiver_guard",
            "token_sealed": True,
            "sealed_pod_uid": self.identity.pod_uid,
            "sealed_container_id": self.identity.container_id,
            "sealed_boot_id": self.identity.boot_id,
            "supervisor_pid": self.identity.supervisor_pid,
            "supervisor_starttime_ticks": self.identity.supervisor_starttime_ticks,
        }

    def finish_failed_request(self, grant: LeaseGrant | None) -> None:
        if grant is None:
            return
        with self._lock:
            self._active_executions.discard(grant.exec_id)
            if self._corrupted:
                return
            next_revoked = set(self._revoked_leases)
            next_consumed = set(self._consumed_tokens)
            next_max = self._max_fencing_token
            next_inflight = dict(self._inflight_tasks)
            next_inflight.pop(grant.exec_id, None)
            if grant.lease_id:
                next_revoked.add(grant.lease_id)
            if grant.fencing_token > 0:
                next_consumed.add(grant.fencing_token)
                if grant.fencing_token > next_max:
                    next_max = grant.fencing_token
            self._commit_state_locked(
                max_fencing_token=next_max,
                revoked_leases=next_revoked,
                consumed_tokens=next_consumed,
                inflight_tasks=next_inflight,
            )
            self._active_lease_fencing.pop(grant.lease_id, None)

    def reclaim_and_seal(
        self,
        *,
        lease_id: str = "",
        fencing_token: int = 0,
        active_tasks: int = 0,
    ) -> dict[str, Any]:
        clean_lease = (lease_id or "").strip()
        tok = int(fencing_token or 0)
        with self._lock:
            self._ensure_healthy_locked()
            next_revoked = set(self._revoked_leases)
            next_consumed = set(self._consumed_tokens)
            next_max = self._max_fencing_token
            need_commit = False
            if clean_lease and clean_lease not in next_revoked:
                next_revoked.add(clean_lease)
                need_commit = True
            prev_tok = self._active_lease_fencing.get(clean_lease, 0) if clean_lease else 0
            for candidate in (prev_tok, tok):
                if candidate > 0 and candidate not in next_consumed:
                    next_consumed.add(candidate)
                    need_commit = True
                if candidate > next_max:
                    next_max = candidate
                    need_commit = True
            if need_commit:
                self._commit_state_locked(
                    max_fencing_token=next_max,
                    revoked_leases=next_revoked,
                    consumed_tokens=next_consumed,
                    inflight_tasks=dict(self._inflight_tasks),
                )
            if clean_lease:
                self._active_lease_fencing.pop(clean_lease, None)
            effective_active = max(
                int(active_tasks),
                len(self._active_executions),
                len(self._inflight_tasks),
            )
            consumed = tok if tok > 0 else self._max_fencing_token
            is_settled = effective_active == 0
            return {
                "reclaimed": is_settled,
                "active_tasks": int(effective_active),
                "proof_source": "receiver_guard",
                "token_sealed": True,
                "supervisor_pid": self.identity.supervisor_pid,
                "supervisor_starttime_ticks": self.identity.supervisor_starttime_ticks,
                "sealed_pod_uid": self.identity.pod_uid,
                "sealed_container_id": self.identity.container_id,
                "sealed_boot_id": self.identity.boot_id,
                "revoked_lease_id": clean_lease,
                "consumed_fencing_token": consumed,
            }

    def seal_and_build_reclaim_proof(
        self,
        grant: LeaseGrant | None,
        *,
        lease_id: str = "",
        fencing_token: int = 0,
        active_tasks: int,
        reclaimed: bool,
    ) -> dict[str, Any]:
        resolved_lease = (grant.lease_id if grant else lease_id).strip()
        resolved_token = int(grant.fencing_token if grant else fencing_token)
        with self._lock:
            self._ensure_healthy_locked()
            if grant is not None and grant.exec_id:
                self._active_executions.discard(grant.exec_id)
            next_revoked = set(self._revoked_leases)
            next_consumed = set(self._consumed_tokens)
            next_max = self._max_fencing_token
            next_inflight = dict(self._inflight_tasks)
            if grant is not None and grant.exec_id:
                next_inflight.pop(grant.exec_id, None)
            if resolved_lease:
                next_revoked.add(resolved_lease)
            if resolved_token > 0:
                next_consumed.add(resolved_token)
                if resolved_token > next_max:
                    next_max = resolved_token
            self._commit_state_locked(
                max_fencing_token=next_max,
                revoked_leases=next_revoked,
                consumed_tokens=next_consumed,
                inflight_tasks=next_inflight,
            )
            if resolved_lease:
                self._active_lease_fencing.pop(resolved_lease, None)
            effective_active = max(
                int(active_tasks),
                len(self._active_executions),
                len(self._inflight_tasks),
            )
            consumed = resolved_token if resolved_token > 0 else self._max_fencing_token
        is_settled = bool(reclaimed and effective_active == 0)
        return {
            "reclaimed": is_settled,
            "active_tasks": int(effective_active),
            "proof_source": "receiver_guard",
            "token_sealed": True,
            "supervisor_pid": self.identity.supervisor_pid,
            "supervisor_starttime_ticks": self.identity.supervisor_starttime_ticks,
            "sealed_pod_uid": self.identity.pod_uid,
            "sealed_container_id": self.identity.container_id,
            "sealed_boot_id": self.identity.boot_id,
            "revoked_lease_id": resolved_lease,
            "consumed_fencing_token": consumed,
        }

    def build_reclaim_proof(
        self,
        grant: LeaseGrant | None,
        *,
        active_tasks: int,
        reclaimed: bool,
    ) -> dict[str, Any]:
        return self.seal_and_build_reclaim_proof(
            grant,
            active_tasks=active_tasks,
            reclaimed=reclaimed,
        )
