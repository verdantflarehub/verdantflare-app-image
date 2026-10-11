"""Grok JSON REST adapter. No automatic retry of a chargeable request."""
from __future__ import annotations

import base64
import hashlib
import io
import json
import os
from urllib.parse import urlsplit

import httpx
from PIL import Image, ImageOps

ASPECT_RATIOS = ["16:9", "9:16", "1:1", "4:3", "3:4"]
IMAGE_MODELS = {"grok-imagine-image", "grok-imagine-image-2.0", "grok-imagine-image-quality"}
ADAPTER_VERSION = "grok-image-v1"
MAX_IMAGE_BYTES = 32 * 1024 * 1024


class GrokError(ValueError):
    def __init__(self, code: str, *, uncertain: bool = False):
        super().__init__(code)
        self.code = code
        self.uncertain = uncertain


def image_info(data: bytes, *, source: bool = False) -> dict:
    if not data or len(data) > (10 * 1024 * 1024 if source else MAX_IMAGE_BYTES):
        raise GrokError("image_size_exceeded")
    try:
        with Image.open(io.BytesIO(data)) as im:
            if im.format not in {"PNG", "JPEG", "WEBP"} or getattr(im, "n_frames", 1) != 1:
                raise ValueError
            if im.width * im.height > 16 * 1024 * 1024:
                raise ValueError
            fmt = im.format
            im.verify()
        with Image.open(io.BytesIO(data)) as im:
            im.load()
            width, height = ImageOps.exif_transpose(im).size
        return {"width": width, "height": height, "media_type": Image.MIME[fmt],
                "extension": {"JPEG": "jpg", "PNG": "png", "WEBP": "webp"}[fmt]}
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        raise GrokError("image_integrity_failed") from exc


class GrokImageProvider:
    def __init__(self, client: httpx.Client | None = None):
        self.base = os.getenv("XAI_BASE_URL", "https://api.x.ai/v1").rstrip("/")
        self.key = os.getenv("XAI_API_KEY", "")
        self.model = os.getenv("XAI_IMAGE_MODEL", "grok-imagine-image-2.0")
        self.enabled = os.getenv("XAI_IMAGE_ENABLED", "false").lower() == "true"
        self.client = client or httpx.Client(follow_redirects=False, timeout=300)
        self.profile = hashlib.sha256((self.base + "\0" + self.key).encode()).hexdigest()

    def configured(self) -> bool:
        p = urlsplit(self.base)
        return bool(self.key and p.scheme == "https" and p.hostname and not
                    (p.username or p.password or p.query or p.fragment) and self.model in IMAGE_MODELS)

    def capability(self) -> dict:
        return {"engine": "grok", "provider": "xai", "model": self.model,
                "enabled": self.enabled, "configured": self.configured(),
                "readiness": "not_checked", "verification_status": "not_recorded",
                "adapter_version": ADAPTER_VERSION, "modes": ["generate", "edit"],
                "aspect_ratios": ASPECT_RATIOS, "resolutions": ["1k", "2k"],
                "max_reference_images": 1, "inpaint": False}

    def normalize(self, params: dict, project_id: str, artifacts) -> dict:
        p = dict(params)
        if not self.enabled or not self.configured():
            raise GrokError("provider_not_configured")
        if os.getenv("XAI_IMAGE_RESPONSE_MODE", "sync_base64") != "sync_base64":
            raise GrokError("unsupported_response_mode")
        p["model"] = p.get("model") or self.model
        if p["model"] != self.model:
            raise GrokError("model_route_mismatch")
        if not isinstance(p.get("prompt"), str) or not p["prompt"].strip():
            raise GrokError("invalid_prompt")
        if (p.get("quality") is not None or p.get("background") not in {None, "auto"}
                or p.get("instructions") or p.get("size") or p.get("mask_artifact_id")
                or p.get("reference_artifact_ids") or p.get("action") not in {"generate", "edit"}):
            raise GrokError("unsupported_parameter")
        p["resolution"] = p.get("resolution") or "2k"
        if p["resolution"] not in {"1k", "2k"}:
            raise GrokError("unsupported_resolution")
        if p["action"] == "generate":
            p["aspect_ratio"] = p.get("aspect_ratio") or "16:9"
            if p["aspect_ratio"] not in ASPECT_RATIOS:
                raise GrokError("unsupported_aspect_ratio")
        else:
            source = artifacts.get(p.get("source_artifact_id", ""), project_id)
            data = artifacts.content_path(source).read_bytes()
            if hashlib.sha256(data).hexdigest() != source.sha256:
                raise GrokError("source_integrity_failed")
            image_info(data, source=True)
            p["source_sha256"] = source.sha256
            p.pop("aspect_ratio", None)
        for name in ("quality", "background", "instructions", "size", "mask_artifact_id", "reference_artifact_ids"):
            p.pop(name, None)
        p["endpoint_profile_id"] = self.profile
        p["adapter_version"] = ADAPTER_VERSION
        return p

    def generate(self, params: dict, project_id: str, artifacts) -> tuple[bytes, dict]:
        if params.get("endpoint_profile_id") != self.profile:
            raise GrokError("provider_configuration_changed")
        payload = {k: params[k] for k in ("model", "prompt", "resolution")}
        payload.update(n=1, response_format="b64_json")
        endpoint = "/images/generations"
        if params["action"] == "edit":
            source = artifacts.get(params["source_artifact_id"], project_id)
            data = artifacts.content_path(source).read_bytes()
            if hashlib.sha256(data).hexdigest() != params["source_sha256"]:
                raise GrokError("source_integrity_failed")
            info = image_info(data, source=True)
            payload["image"] = {"url": "data:" + info["media_type"] + ";base64," + base64.b64encode(data).decode()}
            endpoint = "/images/edits"
        else:
            payload["aspect_ratio"] = params["aspect_ratio"]
        try:
            with self.client.stream("POST", self.base + endpoint,
                                    headers={"Authorization": "Bearer " + self.key}, json=payload) as r:
                if r.status_code in {401, 403}:
                    raise GrokError("provider_auth_failed")
                if r.status_code in {400, 422}:
                    raise GrokError("provider_rejected")
                if not r.is_success:
                    raise GrokError(f"provider_http_{r.status_code}", uncertain=True)
                raw = bytearray()
                for chunk in r.iter_bytes():
                    raw.extend(chunk)
                    if len(raw) > MAX_IMAGE_BYTES * 4 // 3 + 1024 * 1024:
                        raise GrokError("provider_response_exceeded", uncertain=True)
            body = json.loads(raw)
            if not isinstance(body.get("data"), list) or len(body["data"]) != 1:
                raise ValueError
            data = base64.b64decode(body["data"][0]["b64_json"], validate=True)
        except httpx.HTTPError as exc:
            raise GrokError("submission_unknown", uncertain=True) from exc
        except (ValueError, KeyError, TypeError) as exc:
            if isinstance(exc, GrokError):
                raise
            raise GrokError("provider_protocol_error", uncertain=True) from exc
        info = image_info(data)
        info.update(engine="grok", model=params["model"], adapter_version=ADAPTER_VERSION,
                    requested_resolution=params["resolution"])
        return data, info
