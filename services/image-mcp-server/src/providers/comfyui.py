from __future__ import annotations

import hashlib
import json
import math
import os
import re
import secrets
import ssl
import time
import urllib.parse
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import httpx

from ..artifacts import ArtifactError, ArtifactNotFound, ArtifactStore
from ..channel_authority import CALLER

_UUID7_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)
_UUID4_OR_7_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[47][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)
_COMFYUI_MODEL_ID_RE = re.compile(
    r"^comfyui:([a-z][a-z0-9-]{2,62}[a-z0-9])@([0-9]+\.[0-9]+\.[0-9]+)$"
)
_SAFE_JS_INT_MAX = 9007199254740991
_MAX_OUTPUT_IMAGE_BYTES = 64 * 1024 * 1024


class ComfyUIProviderError(RuntimeError):
    def __init__(self, code: str, *, uncertain: bool = False) -> None:
        super().__init__(code)
        self.code = code
        self.uncertain = uncertain


def new_uuid7() -> str:
    ts_ms = time.time_ns() // 1_000_000
    value = (ts_ms << 80) | (7 << 76)
    value |= (secrets.randbits(12) << 64) | (2 << 62) | secrets.randbits(62)
    return str(uuid.UUID(int=value))


def deterministic_station_user_uuid7(organization_id: str) -> str:
    digest = hashlib.sha256(f"station-image-mcp:{organization_id}".encode("utf-8")).digest()
    raw = int.from_bytes(digest[:16], "big")
    raw &= ~((0xF << 76) | (0x3 << 62))
    raw |= (7 << 76) | (2 << 62)
    return str(uuid.UUID(int=raw))


def is_uuid7(value: Any) -> bool:
    return isinstance(value, str) and bool(_UUID7_RE.fullmatch(value.strip()))


def is_uuid4_or_7(value: Any) -> bool:
    return isinstance(value, str) and bool(_UUID4_OR_7_RE.fullmatch(value.strip()))


def parse_comfyui_model_id(model_id: Any) -> tuple[str, str]:
    if not isinstance(model_id, str):
        raise ValueError("invalid_comfyui_model_id")
    matched = _COMFYUI_MODEL_ID_RE.fullmatch(model_id.strip())
    if not matched:
        raise ValueError("invalid_comfyui_model_id")
    return matched.group(1), matched.group(2)


def _normalize_origin_entry(raw: str) -> str:
    parsed = urllib.parse.urlsplit(raw.strip())
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError("comfyui_origin_invalid")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("comfyui_origin_invalid")
    if parsed.path not in ("", "/"):
        raise ValueError("comfyui_origin_invalid")
    host = parsed.hostname.lower()
    if ":" in host and not host.startswith("["):
        netloc = f"[{host}]"
    else:
        netloc = host
    if parsed.port is not None:
        netloc = f"{netloc}:{parsed.port}"
    return f"{parsed.scheme}://{netloc}"


def validate_internal_origin(
    raw_origin: str,
    *,
    allowed_origins_raw: str = "",
    allow_loopback: bool = False,
) -> str:
    if not raw_origin or not raw_origin.strip():
        raise ValueError("comfyui_not_configured")
    normalized = _normalize_origin_entry(raw_origin)
    parsed = urllib.parse.urlsplit(normalized)
    host = (parsed.hostname or "").lower()

    explicit_allowlist: set[str] = set()
    for item in allowed_origins_raw.split(","):
        if item.strip():
            explicit_allowlist.add(_normalize_origin_entry(item))

    if normalized in explicit_allowlist:
        return normalized
    if host.endswith(".svc.cluster.local") or host.endswith(".svc"):
        return normalized
    if allow_loopback and host in {"127.0.0.1", "localhost", "::1"}:
        return normalized
    raise ValueError("comfyui_origin_not_allowed")


@dataclass(frozen=True)
class _Principal:
    organization_id: str
    user_id: str
    caller_verified: bool = False


@dataclass(frozen=True)
class ComfyUICancelResult:
    confirmed_terminal: bool
    terminal_status: str | None = None
    remote_state: str | None = None
    reason: str = ""

    def __bool__(self) -> bool:
        return self.confirmed_terminal


def _call_authorize_fn(
    authorize_fn: Callable[..., dict[str, str]],
    project_id: str,
    authority: Any = None,
    *,
    require_write: bool = True,
) -> dict[str, str]:
    try:
        return authorize_fn(project_id, authority, require_write=require_write)
    except TypeError:
        return authorize_fn(project_id, authority)


class ComfyUIWorkflowProvider:
    """Dynamic local channel adapter projecting published ComfyUI image workflow APIs."""

    def __init__(
        self,
        *,
        origin: str | None = None,
        token: str | None = None,
        artifact_service_url: str | None = None,
        artifact_service_token: str | None = None,
        allowed_origins: str | None = None,
        allow_loopback: bool | None = None,
        poll_interval_seconds: float | None = None,
        timeout_seconds: float | None = None,
        http_client: httpx.Client | None = None,
    ) -> None:
        self._explicit_origin = origin
        self._explicit_token = token
        self._explicit_artifact_url = artifact_service_url
        self._explicit_artifact_token = artifact_service_token
        self._explicit_allowed_origins = allowed_origins
        self._explicit_allow_loopback = allow_loopback
        self._poll_interval = (
            poll_interval_seconds
            if poll_interval_seconds is not None
            else max(0.05, float(os.environ.get("COMFYUI_POLL_INTERVAL_SECONDS", "1.0")))
        )
        self._timeout_seconds = (
            timeout_seconds
            if timeout_seconds is not None
            else max(5.0, float(os.environ.get("COMFYUI_INVOKE_TIMEOUT_SECONDS", "300.0")))
        )
        self._client = http_client or self._create_default_http_client()

    @staticmethod
    def _create_default_http_client() -> httpx.Client:
        ssl_ctx = ssl.create_default_context()
        ca_file = (
            os.environ.get("IMAGE_ARTIFACT_CA_FILE", "").strip()
            or os.environ.get("IMAGE_PROJECT_CA_FILE", "").strip()
        )
        if ca_file:
            ssl_ctx.load_verify_locations(cafile=ca_file)
        ca_pem = (
            os.environ.get("IMAGE_ARTIFACT_CA_PEM", "").strip()
            or os.environ.get("IMAGE_PROJECT_CA_PEM", "").strip()
        )
        if ca_pem:
            ssl_ctx.load_verify_locations(cadata=ca_pem)
        return httpx.Client(verify=ssl_ctx, trust_env=False, follow_redirects=False, timeout=30.0)

    def _allow_loopback(self) -> bool:
        if self._explicit_allow_loopback is not None:
            return self._explicit_allow_loopback
        return os.environ.get("COMFYUI_ALLOW_LOOPBACK_ORIGIN", "").strip().lower() in {"1", "true", "yes"}

    def _allowed_origins_raw(self) -> str:
        if self._explicit_allowed_origins is not None:
            return self._explicit_allowed_origins
        return os.environ.get("COMFYUI_ALLOWED_ORIGINS", "")

    def _resolved_origin_and_token(self) -> tuple[str, str]:
        raw_origin = (
            self._explicit_origin
            if self._explicit_origin is not None
            else os.environ.get("COMFYUI_INTERNAL_ORIGIN", "")
        )
        raw_token = (
            self._explicit_token
            if self._explicit_token is not None
            else os.environ.get("COMFYUI_INTERNAL_TOKEN", "")
        )
        origin = validate_internal_origin(
            raw_origin,
            allowed_origins_raw=self._allowed_origins_raw(),
            allow_loopback=self._allow_loopback(),
        )
        token = raw_token.strip()
        if not token:
            raise ValueError("comfyui_not_configured")
        return origin, token

    def _resolved_artifact_service(self) -> tuple[str, str]:
        raw_url = (
            self._explicit_artifact_url
            if self._explicit_artifact_url is not None
            else (
                os.environ.get("IMAGE_ARTIFACT_SERVICE_URL", "").strip()
                or os.environ.get("ARTIFACT_SERVICE_URL", "").strip()
            )
        )
        raw_token = (
            self._explicit_artifact_token
            if self._explicit_artifact_token is not None
            else (
                os.environ.get("IMAGE_ARTIFACT_SERVICE_TOKEN", "").strip()
                or os.environ.get("ARTIFACT_SERVICE_TOKEN", "").strip()
            )
        )
        if not raw_url or not raw_token:
            raise ComfyUIProviderError("comfyui_artifact_bridge_not_configured", uncertain=False)
        try:
            origin = validate_internal_origin(
                raw_url,
                allowed_origins_raw=self._allowed_origins_raw(),
                allow_loopback=self._allow_loopback(),
            )
        except ValueError as exc:
            raise ComfyUIProviderError(str(exc), uncertain=False) from None
        return origin, raw_token

    def is_configured(self) -> bool:
        try:
            self._resolved_origin_and_token()
            return True
        except ValueError:
            return False

    @staticmethod
    def resolve_discovery_organization_id(explicit_organization_id: str | None = None) -> str | None:
        if explicit_organization_id is not None:
            cand = explicit_organization_id.strip()
            return cand if is_uuid7(cand) else None
        caller = CALLER.get()
        if isinstance(caller, dict):
            raw_org = str(caller.get("organization_id") or "").strip()
            raw_user = str(caller.get("user_id") or "").strip()
            if raw_org or raw_user:
                return raw_org if is_uuid7(raw_org) else None
        env_org = (
            os.environ.get("COMFYUI_ORGANIZATION_ID", "").strip()
            or os.environ.get("STATION_ORGANIZATION_ID", "").strip()
        )
        return env_org if is_uuid7(env_org) else None

    def _resolve_execution_principal(
        self,
        project_id: str,
        authorize_fn: Callable[..., dict[str, str]] | None,
        *,
        require_write: bool = True,
    ) -> _Principal:
        caller = CALLER.get()
        caller_org = str(caller.get("organization_id") or "").strip() if isinstance(caller, dict) else ""
        caller_user = str(caller.get("user_id") or "").strip() if isinstance(caller, dict) else ""
        has_caller_identity = bool(caller_org or caller_user)
        project_auth_url = os.environ.get("IMAGE_PROJECT_SERVICE_URL", "").strip()
        if has_caller_identity or project_auth_url:
            if has_caller_identity and (not is_uuid7(caller_org) or not is_uuid7(caller_user)):
                raise ValueError("verified_studio_identity_required")
            if authorize_fn is not None and project_auth_url:
                verified = _call_authorize_fn(authorize_fn, project_id, None, require_write=require_write)
                org_id = str(verified.get("organization_id") or "").strip()
                user_id = str(verified.get("user_id") or "").strip()
            elif authorize_fn is not None and not project_auth_url and has_caller_identity:
                try:
                    verified = _call_authorize_fn(authorize_fn, project_id, None, require_write=require_write)
                    org_id = str(verified.get("organization_id") or "").strip()
                    user_id = str(verified.get("user_id") or "").strip()
                except ValueError as exc:
                    if str(exc) != "project_authorization_not_configured":
                        raise
                    org_id = caller_org
                    user_id = caller_user
            else:
                raise ValueError("project_authorization_not_configured")
            if not is_uuid7(org_id) or not is_uuid7(user_id):
                raise ValueError("verified_studio_identity_required")
            if caller_org and org_id != caller_org:
                raise ValueError("project_access_denied")
            return _Principal(organization_id=org_id, user_id=user_id, caller_verified=True)

        org_id = self.resolve_discovery_organization_id()
        if not org_id:
            raise ValueError("comfyui_organization_required")
        env_user = os.environ.get("COMFYUI_SYSTEM_USER_ID", "").strip()
        user_id = env_user if is_uuid7(env_user) else deterministic_station_user_uuid7(org_id)
        return _Principal(organization_id=org_id, user_id=user_id, caller_verified=False)

    def authorize_task_caller(
        self,
        *,
        project_id: str,
        params: dict[str, Any],
        authorize_fn: Callable[..., dict[str, str]] | None,
        require_write: bool,
    ) -> dict[str, str]:
        auth_info = params.get("comfyui_authority") if isinstance(params, dict) else None
        task_org_id = str((auth_info or {}).get("organization_id") or "").strip()
        caller_verified_at_create = bool((auth_info or {}).get("caller_verified"))
        caller = CALLER.get()
        has_caller_identity = isinstance(caller, dict) and bool(
            str(caller.get("organization_id") or "").strip()
            or str(caller.get("user_id") or "").strip()
        )
        if caller_verified_at_create and not has_caller_identity:
            raise ValueError("verified_studio_identity_required")
        principal = self._resolve_execution_principal(
            project_id,
            authorize_fn,
            require_write=require_write,
        )
        if is_uuid7(task_org_id) and principal.organization_id != task_org_id:
            raise ValueError("project_access_denied")
        return {
            "organization_id": principal.organization_id,
            "user_id": principal.user_id,
        }

    def discover_models(self, organization_id: str | None = None) -> list[dict[str, Any]]:
        """Return active org-scoped ComfyUI workflows published to mount_channel='image'."""
        try:
            origin, token = self._resolved_origin_and_token()
        except ValueError:
            return []
        org_id = self.resolve_discovery_organization_id(organization_id)
        if not org_id:
            return []
        try:
            resp = self._client.post(
                f"{origin}/internal/v1/comfyui/published-channels",
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
                json={
                    "organization_id": org_id,
                    "mount_channel": "image",
                },
                timeout=10.0,
            )
            if resp.status_code != 200:
                return []
            payload = resp.json()
        except Exception:
            return []

        raw_items = payload.get("items") if isinstance(payload, dict) else None
        if not isinstance(raw_items, list):
            return []
        valid_items: list[dict[str, Any]] = []
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            slug = str(item.get("endpoint_slug") or "").strip()
            ver = str(item.get("version") or "").strip()
            model_id = str(item.get("catalog_model_id") or "").strip()
            spec = item.get("spec")
            if (
                item.get("engine") != "comfyui"
                or item.get("mount_channel") != "image"
                or item.get("output_modality") != "image"
                or item.get("organization_id") != org_id
                or model_id != f"comfyui:{slug}@{ver}"
                or not isinstance(spec, dict)
                or not isinstance(spec.get("exposed_inputs"), list)
            ):
                continue
            try:
                parse_comfyui_model_id(model_id)
            except ValueError:
                continue
            valid_items.append(item)
        return valid_items

    @staticmethod
    def _slot_type(slot: dict[str, Any]) -> str:
        raw = str(slot.get("param_type") or slot.get("value_type") or "").strip()
        return {
            "int": "integer",
            "float": "number",
            "bool": "boolean",
        }.get(raw, raw)

    @staticmethod
    def _build_catalog_entry(item: dict[str, Any]) -> dict[str, Any]:
        spec = item.get("spec") or {}
        exposed = spec.get("exposed_inputs") or []
        img_slots = sum(
            1
            for s in exposed
            if isinstance(s, dict)
            and ComfyUIWorkflowProvider._slot_type(s) == "artifact_image"
        )
        mask_slots = sum(
            1
            for s in exposed
            if isinstance(s, dict)
            and ComfyUIWorkflowProvider._slot_type(s) == "artifact_mask"
        )
        ops = ["generate"]
        if img_slots > 0:
            ops.append("edit")
        if img_slots > 0 and mask_slots > 0:
            ops.append("inpaint")

        if img_slots == 0 and mask_slots == 0:
            input_summary = "纯文本生成 (T2I) · 无参考图槽位"
        elif mask_slots > 0:
            input_summary = f"最多 {img_slots}张参考图 · {mask_slots}张遮罩"
        else:
            input_summary = f"最多 {img_slots}张参考底图 (I2I)"

        slug = item["endpoint_slug"]
        ver = item["version"]
        vram_mib = int(item.get("vram_budget_mib") or 0)
        repro = str(item.get("reproducibility_level") or "strict")
        exposed_summary: list[dict[str, Any]] = []
        for s in exposed:
            if not isinstance(s, dict) or not s.get("name"):
                continue
            st = ComfyUIWorkflowProvider._slot_type(s)
            entry_s: dict[str, Any] = {
                "name": s.get("name"),
                "param_type": st,
                "value_type": st,
                "required": bool(s.get("required")),
            }
            if "default_value" in s:
                entry_s["default"] = s["default_value"]
            elif "default" in s:
                entry_s["default"] = s["default"]
            exposed_summary.append(entry_s)

        return {
            "model_id": item["catalog_model_id"],
            "display_name": str(item.get("display_name") or slug),
            "official_name": f"{slug}@{ver}",
            "channel": "local",
            "channel_name": "本地",
            "source_badge": "本地",
            "mark": "LOC",
            "biz": "本地工作流 API",
            "pricing_line": "本地算力 · 共享 GPU",
            "release_meta": f"v{ver} · {repro} · 显存预算 {vram_mib} MiB",
            "input_summary": input_summary,
            "desc": f"ComfyUI 已发布工作流 API（{slug}@{ver}），来源：本地。",
            "operations": ops,
            "endpoint_slug": slug,
            "version": ver,
            "published_api_id": item.get("published_api_id"),
            "vram_budget_mib": vram_mib,
            "reproducibility_level": repro,
            "seed_policy": item.get("seed_policy") or "caller_or_random",
            "exposed_inputs": exposed_summary,
        }

    def capability(self, organization_id: str | None = None) -> dict[str, Any]:
        configured = self.is_configured()
        discovered = self.discover_models(organization_id=organization_id) if configured else []
        catalog = [self._build_catalog_entry(item) for item in discovered]
        supported = [entry["model_id"] for entry in catalog]
        return {
            "engine": "comfyui",
            "channel": "local",
            "channel_name": "本地",
            "source_badge": "本地",
            "model": supported[0] if supported else "",
            "enabled": configured and len(supported) > 0,
            "configured": configured,
            "supported_models": supported,
            "model_catalog": catalog,
        }

    @staticmethod
    def _coerce_and_validate_slot_value(
        slot: dict[str, Any],
        raw_val: Any,
        *,
        project_id: str,
        artifacts: ArtifactStore,
    ) -> Any:
        vtype = ComfyUIWorkflowProvider._slot_type(slot)
        min_bound = slot.get("minimum") if slot.get("minimum") is not None else slot.get("min")
        max_bound = slot.get("maximum") if slot.get("maximum") is not None else slot.get("max")
        is_seed_slot = vtype == "seed" or (
            vtype == "integer" and slot.get("input_key") == "seed"
        )
        if vtype in ("integer", "seed"):
            if isinstance(raw_val, bool) or not isinstance(raw_val, int):
                raise ValueError("invalid_workflow_parameter_type")
            lower = 0 if is_seed_slot else -_SAFE_JS_INT_MAX
            if raw_val < lower or raw_val > _SAFE_JS_INT_MAX:
                raise ValueError("workflow_parameter_out_of_range")
            if min_bound is not None and raw_val < min_bound:
                raise ValueError("workflow_parameter_out_of_range")
            if max_bound is not None and raw_val > max_bound:
                raise ValueError("workflow_parameter_out_of_range")
            return raw_val

        if vtype == "number":
            if isinstance(raw_val, bool) or not isinstance(raw_val, (int, float)):
                raise ValueError("invalid_workflow_parameter_type")
            fval = float(raw_val)
            if not math.isfinite(fval):
                raise ValueError("invalid_workflow_parameter_type")
            if min_bound is not None and fval < float(min_bound):
                raise ValueError("workflow_parameter_out_of_range")
            if max_bound is not None and fval > float(max_bound):
                raise ValueError("workflow_parameter_out_of_range")
            return fval

        if vtype == "boolean":
            if not isinstance(raw_val, bool):
                raise ValueError("invalid_workflow_parameter_type")
            return raw_val

        if vtype == "string":
            if not isinstance(raw_val, str):
                raise ValueError("invalid_workflow_parameter_type")
            min_len = slot.get("min_length")
            max_len = int(slot.get("max_length") if slot.get("max_length") is not None else 4000)
            if min_len is not None and len(raw_val) < int(min_len):
                raise ValueError("workflow_parameter_too_short")
            if len(raw_val) > max_len:
                raise ValueError("workflow_parameter_too_long")
            return raw_val

        if vtype == "enum":
            if not isinstance(raw_val, str):
                raise ValueError("invalid_workflow_parameter_type")
            allowed = slot.get("enum_values") or []
            if raw_val not in allowed:
                raise ValueError("workflow_parameter_enum_mismatch")
            return raw_val

        if vtype in ("artifact_image", "artifact_mask"):
            if not isinstance(raw_val, str) or not raw_val.strip():
                raise ValueError("invalid_workflow_artifact_id")
            aid = raw_val.strip()
            if is_uuid4_or_7(aid):
                return aid
            try:
                rec = artifacts.get(aid, project_id)
            except (ArtifactNotFound, ArtifactError) as exc:
                raise ValueError("artifact_not_found") from exc
            if rec.project_id != project_id:
                raise ValueError("artifact_not_found")
            return aid

        raise ValueError("unsupported_workflow_slot_type")

    def normalize(
        self,
        params: dict[str, Any],
        project_id: str,
        artifacts: ArtifactStore,
        authorize_fn: Callable[[str, Any], dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        if not is_uuid7(project_id):
            raise ValueError("invalid_comfyui_project_id")
        if not self.is_configured():
            raise ValueError("comfyui_not_configured")
        if params.get("channel_account_id") is not None or params.get("budget_limit") is not None:
            raise ValueError("channel_parameters_not_supported")

        model_id = str(params.get("model") or "").strip()
        slug, version = parse_comfyui_model_id(model_id)
        principal = self._resolve_execution_principal(project_id, authorize_fn)

        discovered = self.discover_models(organization_id=principal.organization_id)
        matched = next(
            (
                item
                for item in discovered
                if item.get("catalog_model_id") == model_id
                and item.get("endpoint_slug") == slug
                and item.get("version") == version
            ),
            None,
        )
        if matched is None:
            raise ValueError("comfyui_workflow_unavailable")

        spec = matched["spec"]
        exposed_list = [s for s in (spec.get("exposed_inputs") or []) if isinstance(s, dict) and s.get("name")]
        exposed_by_name = {str(s["name"]): s for s in exposed_list}

        raw_wf_params = params.get("workflow_params")
        if raw_wf_params is not None and not isinstance(raw_wf_params, dict):
            raise ValueError("invalid_workflow_params")
        mapped: dict[str, Any] = {}
        if isinstance(raw_wf_params, dict):
            for k, v in raw_wf_params.items():
                if k not in exposed_by_name:
                    raise ValueError("unsupported_workflow_parameter")
                if v is not None:
                    mapped[k] = v

        def _fill_first(candidates: tuple[str, ...], value: Any, *, types: set[str] | None = None) -> None:
            if value is None:
                return
            for name in candidates:
                slot = exposed_by_name.get(name)
                if slot is None or name in mapped:
                    continue
                if types is not None and self._slot_type(slot) not in types:
                    continue
                mapped[name] = value
                return

        prompt_val = params.get("prompt")
        if isinstance(prompt_val, str) and prompt_val != "":
            _fill_first(("prompt", "positive_prompt", "text"), prompt_val, types={"string"})
        neg_val = params.get("negative_prompt")
        if isinstance(neg_val, str) and neg_val != "":
            _fill_first(("negative_prompt",), neg_val, types={"string"})

        if params.get("seed") is not None:
            seed_slots = tuple(
                [
                    s["name"]
                    for s in exposed_list
                    if self._slot_type(s) == "seed" or s.get("input_key") == "seed"
                ]
                + ["seed", "noise_seed"]
            )
            _fill_first(seed_slots, params["seed"], types={"seed", "integer"})
        if params.get("steps") is not None:
            _fill_first(("steps",), params["steps"], types={"integer"})
        cfg_val = params.get("cfg_scale") if params.get("cfg_scale") is not None else params.get("guidance")
        if cfg_val is not None:
            _fill_first(("cfg", "cfg_scale", "guidance"), cfg_val, types={"number", "integer"})
        if params.get("sampler") is not None:
            _fill_first(("sampler_name", "sampler"), params["sampler"], types={"enum", "string"})
        if params.get("scheduler") is not None:
            _fill_first(("scheduler",), params["scheduler"], types={"enum", "string"})
        if params.get("denoise") is not None:
            _fill_first(("denoise",), params["denoise"], types={"number"})
        if params.get("width") is not None:
            _fill_first(("width",), params["width"], types={"integer"})
        if params.get("height") is not None:
            _fill_first(("height",), params["height"], types={"integer"})

        raw_img_refs = [
            params.get("source_artifact_id"),
            params.get("face_reference_artifact_id"),
            params.get("source_identity_artifact_id"),
            *(params.get("reference_artifact_ids") or []),
            *(params.get("reference_images") or []),
        ]
        img_refs = list(
            dict.fromkeys(
                str(r).strip() for r in raw_img_refs if isinstance(r, str) and r.strip()
            )
        )
        free_img_slots = [
            s["name"]
            for s in exposed_list
            if self._slot_type(s) == "artifact_image" and s["name"] not in mapped
        ]
        if img_refs:
            if len(img_refs) > len(free_img_slots):
                raise ValueError("unsupported_reference_image")
            for slot_name, ref_id in zip(free_img_slots, img_refs):
                mapped[slot_name] = ref_id

        mask_ref = str(params.get("mask_artifact_id") or params.get("mask") or "").strip()
        free_mask_slots = [
            s["name"]
            for s in exposed_list
            if self._slot_type(s) == "artifact_mask" and s["name"] not in mapped
        ]
        if mask_ref:
            if not free_mask_slots:
                raise ValueError("unsupported_mask_parameter")
            mapped[free_mask_slots[0]] = mask_ref

        validated_inputs: dict[str, Any] = {}
        media_slot_types: dict[str, str] = {}
        for slot in exposed_list:
            sname = str(slot["name"])
            stype = self._slot_type(slot)
            has_default = "default_value" in slot or "default" in slot
            if sname in mapped:
                validated_inputs[sname] = self._coerce_and_validate_slot_value(
                    slot,
                    mapped[sname],
                    project_id=project_id,
                    artifacts=artifacts,
                )
                if stype in ("artifact_image", "artifact_mask"):
                    media_slot_types[sname] = stype
            elif slot.get("required") and not has_default:
                raise ValueError(f"missing_required_workflow_input:{sname}")

        out = dict(params)
        out["model"] = model_id
        out["channel"] = "local"
        out["source_badge"] = "本地"
        out["comfyui_authority"] = {
            "organization_id": principal.organization_id,
            "user_id": principal.user_id,
            "caller_verified": principal.caller_verified,
        }
        out["comfyui_workflow"] = {
            "published_api_id": matched.get("published_api_id"),
            "endpoint_slug": slug,
            "version": version,
            "display_name": matched.get("display_name") or slug,
            "resolved_params": validated_inputs,
            "media_slot_types": media_slot_types,
        }
        return out

    def _call_headless_tool(
        self,
        principal: _Principal,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        try:
            origin, token = self._resolved_origin_and_token()
        except ValueError as exc:
            raise ComfyUIProviderError(str(exc), uncertain=False) from None

        headers = {
            "Authorization": f"Bearer {token}",
            "X-Organization-Id": principal.organization_id,
            "X-User-Id": principal.user_id,
            "X-Request-Id": new_uuid7(),
            "Content-Type": "application/json",
        }
        try:
            resp = self._client.post(
                f"{origin}/internal/apps/comfyui-api/call",
                headers=headers,
                json={"name": tool_name, "arguments": arguments},
                timeout=max(30.0, self._timeout_seconds),
            )
        except httpx.HTTPError as exc:
            raise ComfyUIProviderError(
                "comfyui_transport_error",
                uncertain=(tool_name == "api.invoke"),
            ) from exc

        try:
            body = resp.json()
        except Exception:
            body = {}

        if resp.status_code != 200 or (isinstance(body, dict) and body.get("isError") is True):
            err_code = "comfyui_call_failed"
            if isinstance(body, dict):
                sc = body.get("structuredContent")
                if isinstance(sc, dict) and isinstance(sc.get("error"), dict):
                    err_code = f"comfyui_{str(sc['error'].get('code') or 'call_failed').lower()}"
                elif isinstance(body.get("code"), str) and body["code"].strip():
                    err_code = f"comfyui_{body['code'].strip().lower()}"
                elif isinstance(body.get("error"), str) and body["error"].strip():
                    err_code = f"comfyui_{body['error'].strip().lower()}"
            uncertain = resp.status_code >= 500 and tool_name == "api.invoke"
            raise ComfyUIProviderError(err_code, uncertain=uncertain)

        if isinstance(body, dict) and isinstance(body.get("structuredContent"), dict):
            return body["structuredContent"]
        if isinstance(body, dict):
            return body
        raise ComfyUIProviderError("comfyui_invalid_response", uncertain=False)

    def _bridge_local_input_to_studio(
        self,
        *,
        principal: _Principal,
        project_id: str,
        task_id: str,
        local_artifact_id: str,
        artifacts: ArtifactStore,
    ) -> str:
        if is_uuid4_or_7(local_artifact_id):
            return local_artifact_id
        rec = artifacts.get(local_artifact_id, project_id)
        raw = artifacts.content_path(rec).read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != rec.sha256:
            raise ComfyUIProviderError("input_artifact_integrity_failed", uncertain=False)

        base_url, svc_token = self._resolved_artifact_service()
        base_headers = {
            "Authorization": f"Bearer {svc_token}",
            "X-User-Id": principal.user_id,
            "X-Organization-Id": principal.organization_id,
        }
        try:
            prep_resp = self._client.post(
                f"{base_url}/v2/artifacts/uploads",
                headers={**base_headers, "X-Request-Id": new_uuid7(), "Content-Type": "application/json"},
                json={
                    "project_id": project_id,
                    "filename": rec.filename,
                    "mime_type": rec.media_type,
                    "size_bytes": len(raw),
                    "sha256": digest,
                    "source": "image_mcp_bridge",
                    "source_task_id": task_id,
                },
                timeout=30.0,
            )
            if prep_resp.status_code == 403:
                raise ComfyUIProviderError("comfyui_forbidden", uncertain=False)
            prep_resp.raise_for_status()
            prep = prep_resp.json()
            upload_id = str(prep["upload_id"])
            studio_artifact_id = str(prep["artifact_id"])

            put_resp = self._client.put(
                f"{base_url}/v2/artifacts/uploads/{upload_id}/content",
                headers={
                    **base_headers,
                    "X-Request-Id": new_uuid7(),
                    "Content-Type": "application/octet-stream",
                    "Content-Length": str(len(raw)),
                },
                content=raw,
                timeout=60.0,
            )
            put_resp.raise_for_status()

            commit_resp = self._client.post(
                f"{base_url}/v2/artifacts/uploads/{upload_id}/commit",
                headers={**base_headers, "X-Request-Id": new_uuid7(), "Content-Type": "application/json"},
                json={"sha256": digest, "size_bytes": len(raw)},
                timeout=30.0,
            )
            commit_resp.raise_for_status()
            committed = commit_resp.json() if commit_resp.content else {}
            if isinstance(committed, dict) and committed.get("artifact_id"):
                studio_artifact_id = str(committed["artifact_id"])
        except ComfyUIProviderError:
            raise
        except Exception as exc:
            raise ComfyUIProviderError("comfyui_input_artifact_upload_failed", uncertain=False) from exc

        if not is_uuid4_or_7(studio_artifact_id):
            raise ComfyUIProviderError("comfyui_invalid_studio_artifact_id", uncertain=False)
        return studio_artifact_id

    def _download_studio_output_bytes(
        self,
        *,
        principal: _Principal,
        project_id: str,
        output_meta: dict[str, Any],
    ) -> tuple[bytes, str, str]:
        studio_aid = str(output_meta.get("artifact_id") or "").strip()
        expected_sha = str(output_meta.get("sha256") or "").strip().lower()
        expected_size = output_meta.get("size_bytes")
        if not is_uuid4_or_7(studio_aid) or len(expected_sha) != 64:
            raise ComfyUIProviderError("comfyui_output_metadata_invalid", uncertain=False)

        base_url, svc_token = self._resolved_artifact_service()
        query = urllib.parse.urlencode({"project_id": project_id, "artifact_id": studio_aid})
        try:
            resp = self._client.get(
                f"{base_url}/v2/artifacts/{studio_aid}/content?{query}",
                headers={
                    "Authorization": f"Bearer {svc_token}",
                    "X-User-Id": principal.user_id,
                    "X-Organization-Id": principal.organization_id,
                    "X-Request-Id": new_uuid7(),
                },
                timeout=60.0,
            )
            if resp.status_code == 403:
                raise ComfyUIProviderError("comfyui_output_forbidden", uncertain=False)
            if resp.status_code == 404:
                raise ComfyUIProviderError("comfyui_output_not_found", uncertain=False)
            resp.raise_for_status()
            raw = resp.content
        except ComfyUIProviderError:
            raise
        except Exception as exc:
            raise ComfyUIProviderError("comfyui_output_download_unavailable", uncertain=True) from exc

        if len(raw) > _MAX_OUTPUT_IMAGE_BYTES:
            raise ComfyUIProviderError("comfyui_output_too_large", uncertain=False)
        if isinstance(expected_size, int) and len(raw) != expected_size:
            raise ComfyUIProviderError("comfyui_output_size_mismatch", uncertain=False)
        if hashlib.sha256(raw).hexdigest() != expected_sha:
            raise ComfyUIProviderError("comfyui_output_sha256_mismatch", uncertain=False)

        mime_type = str(output_meta.get("mime_type") or "image/png").split(";")[0].strip() or "image/png"
        ext = {"image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp"}.get(mime_type, ".png")
        filename = str(output_meta.get("filename") or f"comfyui{ext}")
        return raw, mime_type, filename

    def execute_or_recover(
        self,
        *,
        task_id: str,
        project_id: str,
        params: dict[str, Any],
        artifacts: ArtifactStore,
        journal_path: Path,
        on_phase: Callable[[str], None] | None = None,
    ) -> tuple[bytes, str, str, dict[str, Any]]:
        auth_info = params.get("comfyui_authority") or {}
        wf_info = params.get("comfyui_workflow") or {}
        org_id = str(auth_info.get("organization_id") or "").strip()
        user_id = str(auth_info.get("user_id") or "").strip()
        if not is_uuid7(org_id) or not is_uuid7(user_id):
            raise ComfyUIProviderError("comfyui_authority_missing", uncertain=False)
        principal = _Principal(organization_id=org_id, user_id=user_id)

        slug = str(wf_info.get("endpoint_slug") or "").strip()
        version = str(wf_info.get("version") or "").strip()
        if not slug or not version:
            slug, version = parse_comfyui_model_id(params.get("model"))

        journal: dict[str, Any] = {}
        if journal_path.exists():
            try:
                journal = json.loads(journal_path.read_text(encoding="utf-8"))
            except Exception:
                journal = {}

        remote_task = journal.get("last_remote_task") if isinstance(journal.get("last_remote_task"), dict) else None
        remote_task_id = str(journal.get("remote_task_id") or "").strip()

        if not remote_task_id:
            resolved_params = dict(wf_info.get("resolved_params") or {})
            media_slots = dict(wf_info.get("media_slot_types") or {})
            bridged_inputs = dict(journal.get("bridged_inputs") or {})
            for slot_name in media_slots:
                if slot_name not in resolved_params:
                    continue
                if slot_name not in bridged_inputs:
                    bridged_inputs[slot_name] = self._bridge_local_input_to_studio(
                        principal=principal,
                        project_id=project_id,
                        task_id=task_id,
                        local_artifact_id=str(resolved_params[slot_name]),
                        artifacts=artifacts,
                    )
                    journal["bridged_inputs"] = bridged_inputs
                    self._write_journal(journal_path, journal)
                resolved_params[slot_name] = bridged_inputs[slot_name]

            if on_phase is not None:
                on_phase("submitting")
            journal["submitted"] = True
            self._write_journal(journal_path, journal)

            idem_key = f"imgmcp-{task_id}"
            remote_task = self._call_headless_tool(
                principal,
                "api.invoke",
                {
                    "project_id": project_id,
                    "idempotency_key": idem_key,
                    "endpoint_slug": slug,
                    "version": version,
                    "params": resolved_params,
                },
            )
            remote_task_id = str(remote_task.get("task_id") or "").strip()
            if not is_uuid4_or_7(remote_task_id):
                raise ComfyUIProviderError("comfyui_invalid_task_id", uncertain=False)
            journal["remote_task_id"] = remote_task_id
            journal["last_remote_task"] = remote_task
            self._write_journal(journal_path, journal)

        in_flight_states = {
            "accepted",
            "validated",
            "queued",
            "dispatching",
            "running",
            "collecting",
            "reconciling",
        }
        deadline = time.monotonic() + self._timeout_seconds
        while True:
            if remote_task is None or remote_task.get("state") in in_flight_states:
                if remote_task is not None and remote_task.get("state") in {
                    "queued",
                    "dispatching",
                    "running",
                    "collecting",
                    "reconciling",
                }:
                    if on_phase is not None:
                        on_phase(
                            "collecting"
                            if remote_task.get("state") == "collecting"
                            else "generating"
                        )
                    if time.monotonic() >= deadline:
                        raise ComfyUIProviderError("comfyui_poll_timeout", uncertain=True)
                    time.sleep(self._poll_interval)
                remote_task = self._call_headless_tool(
                    principal,
                    "task.get",
                    {
                        "project_id": project_id,
                        "task_id": remote_task_id,
                    },
                )
                journal["last_remote_task"] = remote_task
                self._write_journal(journal_path, journal)
                if remote_task.get("state") in in_flight_states:
                    if time.monotonic() >= deadline:
                        raise ComfyUIProviderError("comfyui_poll_timeout", uncertain=True)
                    continue

            state = str(remote_task.get("state") or "")
            if state == "succeeded":
                break
            if state in {"failed", "cancelled"}:
                err_obj = remote_task.get("error")
                code = "comfyui_execution_cancelled" if state == "cancelled" else "comfyui_execution_failed"
                if isinstance(err_obj, dict) and err_obj.get("code"):
                    code = f"comfyui_{str(err_obj['code']).lower()}"
                elif isinstance(err_obj, str) and err_obj.strip():
                    code = f"comfyui_{err_obj.strip().lower()}"
                raise ComfyUIProviderError(code, uncertain=False)
            raise ComfyUIProviderError("comfyui_unexpected_task_state", uncertain=True)

        if on_phase is not None:
            on_phase("collecting")
        results = remote_task.get("result")
        if not isinstance(results, list) or not results or not isinstance(results[0], dict):
            raise ComfyUIProviderError("comfyui_empty_result", uncertain=False)
        primary_out = results[0]
        raw_bytes, mime_type, filename = self._download_studio_output_bytes(
            principal=principal,
            project_id=project_id,
            output_meta=primary_out,
        )
        metadata = {
            "engine": "comfyui",
            "channel": "local",
            "source_badge": "本地",
            "model": f"comfyui:{slug}@{version}",
            "endpoint_slug": slug,
            "version": version,
            "published_api_id": wf_info.get("published_api_id"),
            "comfyui_task_id": remote_task_id,
            "studio_artifact_id": primary_out.get("artifact_id"),
            "task_id": task_id,
        }
        return raw_bytes, mime_type, filename, metadata

    def cancel_remote_task(
        self,
        *,
        task_id: str,
        project_id: str,
        params: dict[str, Any],
        journal_path: Path,
    ) -> bool:
        if not journal_path.exists():
            return False
        try:
            journal = json.loads(journal_path.read_text(encoding="utf-8"))
        except Exception:
            return False
        remote_task_id = str(journal.get("remote_task_id") or "").strip()
        auth_info = params.get("comfyui_authority") or {}
        org_id = str(auth_info.get("organization_id") or "").strip()
        user_id = str(auth_info.get("user_id") or "").strip()
        if not is_uuid4_or_7(remote_task_id) or not is_uuid7(org_id) or not is_uuid7(user_id):
            return False
        principal = _Principal(organization_id=org_id, user_id=user_id)
        try:
            self._call_headless_tool(
                principal,
                "task.cancel",
                {
                    "project_id": project_id,
                    "idempotency_key": f"cancel-{task_id}",
                    "task_id": remote_task_id,
                },
            )
            return True
        except Exception:
            return False

    @staticmethod
    def _write_journal(path: Path, data: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)
