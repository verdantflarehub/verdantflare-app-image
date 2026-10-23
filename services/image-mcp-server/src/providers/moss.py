"""MOSS-Image self-hosted provider adapter for image-mcp-server."""
from __future__ import annotations

import base64
import os
import re
from typing import Any

import httpx

DEFAULT_MOSS_MODEL = "moss-image-sft-int8-convrot"

FORBIDDEN_CONTROL_TOKENS: tuple[str, ...] = (
    "<|im_start|>",
    "<|im_end|>",
    "<think>",
    "</think>",
    "<|endoftext|>",
)

ALLOWED_SAMPLERS = {"euler", "er_sde"}
ALLOWED_SCHEDULERS = {"simple"}

ASPECT_SIZE_MAP_1K: dict[str, tuple[int, int]] = {
    "1:1": (1024, 1024),
    "16:9": (1344, 768),
    "9:16": (768, 1344),
    "4:3": (1152, 864),
    "3:4": (864, 1152),
}

ASPECT_SIZE_MAP_2K: dict[str, tuple[int, int]] = {
    "1:1": (1408, 1408),
    "16:9": (1792, 1024),
    "9:16": (1024, 1792),
    "4:3": (1600, 1200),
    "3:4": (1200, 1600),
}


class MossProviderError(RuntimeError):
    def __init__(self, error_code: str, message: str, status_code: int = 500) -> None:
        super().__init__(f"{error_code}: {message}")
        self.error_code = error_code
        self.message = message
        self.status_code = status_code


def _reject_control_tokens(text: str, field_name: str, error_code: str = "INVALID_ARGUMENT") -> None:
    for token in FORBIDDEN_CONTROL_TOKENS:
        if token in text:
            raise ValueError(f"{error_code}: {field_name} contains forbidden control token '{token}'")


def validate_moss_layout(raw_layout: Any, max_items: int = 32) -> dict[str, Any]:
    if not isinstance(raw_layout, dict):
        raise ValueError("INVALID_LAYOUT_JSON: layout must be an object")
    unknown_top = set(raw_layout.keys()) - {"meta", "items"}
    if unknown_top:
        raise ValueError(f"INVALID_LAYOUT_JSON: unsupported layout fields {sorted(unknown_top)}")
    meta = raw_layout.get("meta")
    if not isinstance(meta, str) or not meta.strip():
        raise ValueError("INVALID_LAYOUT_JSON: layout.meta must be a non-empty string")
    meta = meta.strip()
    if len(meta) > 2000:
        raise ValueError("INVALID_LAYOUT_JSON: layout.meta exceeds 2000 characters")
    _reject_control_tokens(meta, "layout.meta", "INVALID_LAYOUT_JSON")

    items = raw_layout.get("items")
    if not isinstance(items, list) or len(items) == 0:
        raise ValueError("INVALID_LAYOUT_JSON: layout.items must be a non-empty array")
    if len(items) > max_items:
        raise ValueError(f"INVALID_LAYOUT_JSON: layout.items count {len(items)} exceeds maximum {max_items}")

    seen_ids: set[int] = set()
    coord_mode: str | None = None
    cleaned_items: list[dict[str, Any]] = []
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"INVALID_LAYOUT_JSON: layout.items[{idx}] must be an object")
        unknown_keys = set(item.keys()) - {"id", "desc", "box_2d"}
        if unknown_keys:
            raise ValueError(f"INVALID_LAYOUT_JSON: layout.items[{idx}] has unknown fields {sorted(unknown_keys)}")
        item_id = item.get("id")
        if isinstance(item_id, bool) or not isinstance(item_id, int) or item_id < 1:
            raise ValueError(f"INVALID_LAYOUT_JSON: layout.items[{idx}].id must be a positive integer")
        if item_id in seen_ids:
            raise ValueError(f"INVALID_LAYOUT_JSON: duplicate layout item id {item_id}")
        seen_ids.add(item_id)

        desc = item.get("desc")
        if not isinstance(desc, str) or not desc.strip():
            raise ValueError(f"INVALID_LAYOUT_JSON: layout.items[{idx}].desc must be a non-empty string")
        desc = desc.strip()
        if len(desc) > 500:
            raise ValueError(f"INVALID_LAYOUT_JSON: layout.items[{idx}].desc exceeds 500 characters")
        _reject_control_tokens(desc, f"layout.items[{idx}].desc", "INVALID_LAYOUT_JSON")

        box = item.get("box_2d")
        if not isinstance(box, list) or len(box) != 4:
            raise ValueError(f"INVALID_LAYOUT_JSON: layout.items[{idx}].box_2d must have 4 numbers")
        for c in box:
            if isinstance(c, bool) or not isinstance(c, (int, float)):
                raise ValueError(f"INVALID_LAYOUT_JSON: layout.items[{idx}].box_2d elements must be numeric")
        x1, y1, x2, y2 = box
        if not (x1 < x2 and y1 < y2) or min(box) < 0:
            raise ValueError(f"INVALID_LAYOUT_JSON: layout.items[{idx}].box_2d must satisfy 0 <= x1 < x2 and 0 <= y1 < y2")

        has_float = any(isinstance(c, float) for c in box)
        max_val = max(box)
        if has_float or max_val <= 1.0:
            if max_val > 1.0:
                raise ValueError(f"INVALID_LAYOUT_JSON: layout.items[{idx}].box_2d float coords must be <= 1.0")
            item_mode = "normalized"
            clean_box = [round(float(c), 4) for c in box]
        else:
            if not all(isinstance(c, int) for c in box) or max_val > 1000:
                raise ValueError(f"INVALID_LAYOUT_JSON: layout.items[{idx}].box_2d int coords must be <= 1000")
            item_mode = "int1000"
            clean_box = [int(c) for c in box]

        if coord_mode is None:
            coord_mode = item_mode
        elif coord_mode != item_mode:
            raise ValueError("INVALID_LAYOUT_JSON: mixed normalized and integer box_2d coordinate systems")

        cleaned_items.append({"id": item_id, "desc": desc, "box_2d": clean_box})

    return {"meta": meta, "items": cleaned_items}


class MossProvider:
    model = DEFAULT_MOSS_MODEL

    def __init__(
        self,
        base_url: str | None = None,
        timeout: float | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        self.base_url = (base_url or os.environ.get("MOSS_IMAGE_API_URL", "http://image-moss-api:8000")).rstrip("/")
        self.timeout = timeout or float(os.environ.get("MOSS_IMAGE_TIMEOUT", "210"))
        self._client = client

    def capability(self) -> dict[str, Any]:
        return {
            "engine": "moss-image",
            "model": self.model,
            "provider": "self-hosted",
            "enabled": bool(self.base_url),
            "requires_channel_account": False,
            "budget_limit_required": False,
            "capabilities": ["text_to_image", "layout_to_image", "bilingual_typography"],
            "limits": {
                "min_dimension": 512,
                "max_dimension": 2048,
                "dimension_multiple": 16,
                "max_pixels": 2_097_152,
                "default_width": 1024,
                "default_height": 1024,
                "min_steps": 1,
                "max_steps": 50,
                "default_steps": 50,
                "default_cfg_scale": 3.0,
                "default_sampler": "euler",
                "default_scheduler": "simple",
                "default_shift": 5.0,
                "default_skip_single_layers": "22",
                "max_layout_items": 32,
            },
        }

    def normalize(self, params: dict[str, Any]) -> dict[str, Any]:
        out = dict(params)
        if out.get("action", "generate") != "generate":
            raise ValueError("capability_not_supported")
        if out.get("channel_account_id") is not None or out.get("budget_limit") is not None:
            raise ValueError("channel_parameters_not_supported")

        prompt = str(out.get("prompt") or "").strip()
        if not prompt:
            raise ValueError("INVALID_ARGUMENT: prompt is required")
        if len(prompt) > 4000:
            raise ValueError("INVALID_ARGUMENT: prompt exceeds 4000 characters")
        _reject_control_tokens(prompt, "prompt")
        out["prompt"] = prompt

        neg = str(out.get("negative_prompt") or "").strip()
        if len(neg) > 2000:
            raise ValueError("INVALID_ARGUMENT: negative_prompt exceeds 2000 characters")
        _reject_control_tokens(neg, "negative_prompt")
        out["negative_prompt"] = neg

        model = str(out.get("model") or "").strip()
        if model and model not in (DEFAULT_MOSS_MODEL, "moss-image", "moss-image-sft"):
            raise ValueError("capability_not_supported")
        out["model"] = DEFAULT_MOSS_MODEL

        width = out.get("width")
        height = out.get("height")
        if width is None and height is None:
            aspect = str(out.get("aspect_ratio") or "1:1").strip()
            res = str(out.get("resolution") or "1k").strip().lower()
            if res == "4k":
                raise ValueError("INVALID_DIMENSIONS: moss-image max_pixels is 2097152 (4k is not supported)")
            size_table = ASPECT_SIZE_MAP_2K if res == "2k" and aspect in ASPECT_SIZE_MAP_2K else ASPECT_SIZE_MAP_1K
            width, height = size_table.get(aspect, (1024, 1024))
        elif width is None or height is None:
            raise ValueError("INVALID_DIMENSIONS: width and height must both be provided")

        if isinstance(width, bool) or isinstance(height, bool) or not isinstance(width, int) or not isinstance(height, int):
            raise ValueError("INVALID_DIMENSIONS: width and height must be integers")
        if not (512 <= width <= 2048 and 512 <= height <= 2048):
            raise ValueError(f"INVALID_DIMENSIONS: width and height must be in [512, 2048], got {width}x{height}")
        if width % 16 != 0 or height % 16 != 0:
            raise ValueError(f"INVALID_DIMENSIONS: width and height must be multiples of 16, got {width}x{height}")
        if width * height > 2_097_152:
            raise ValueError(f"INVALID_DIMENSIONS: total pixels {width * height} exceeds 2097152")
        out["width"] = width
        out["height"] = height

        steps = out.get("steps")
        if steps is None:
            steps = 50
        if isinstance(steps, bool) or not isinstance(steps, int) or not (1 <= steps <= 50):
            raise ValueError("INVALID_SAMPLING_PARAMS: steps must be an integer between 1 and 50")
        out["steps"] = steps

        cfg_scale = out.get("cfg_scale")
        if cfg_scale is None:
            cfg_scale = 3.0
        if isinstance(cfg_scale, bool) or not isinstance(cfg_scale, (int, float)) or not (1.0 <= float(cfg_scale) <= 6.0):
            raise ValueError("INVALID_SAMPLING_PARAMS: cfg_scale must be between 1.0 and 6.0")
        out["cfg_scale"] = float(cfg_scale)

        sampler = str(out.get("sampler") or "euler").strip()
        if sampler not in ALLOWED_SAMPLERS:
            raise ValueError(f"INVALID_SAMPLING_PARAMS: sampler must be one of {sorted(ALLOWED_SAMPLERS)}")
        out["sampler"] = sampler

        scheduler = str(out.get("scheduler") or "simple").strip()
        if scheduler not in ALLOWED_SCHEDULERS:
            raise ValueError(f"INVALID_SAMPLING_PARAMS: scheduler must be one of {sorted(ALLOWED_SCHEDULERS)}")
        out["scheduler"] = scheduler

        shift = out.get("shift")
        if shift is None:
            shift = 5.0
        if isinstance(shift, bool) or not isinstance(shift, (int, float)) or not (1.0 <= float(shift) <= 10.0):
            raise ValueError("INVALID_SAMPLING_PARAMS: shift must be between 1.0 and 10.0")
        out["shift"] = float(shift)

        skip_layers = str(out.get("skip_single_layers") if out.get("skip_single_layers") is not None else "22").strip()
        if skip_layers:
            if not re.fullmatch(r"\d+(\s*,\s*\d+)*", skip_layers):
                raise ValueError("INVALID_SAMPLING_PARAMS: skip_single_layers must be comma-separated integers")
            nums = [int(x.strip()) for x in skip_layers.split(",")]
            if any(n < 0 or n > 63 for n in nums):
                raise ValueError("INVALID_SAMPLING_PARAMS: skip_single_layers indices must be in [0, 63]")
            skip_layers = ",".join(str(n) for n in nums)
        out["skip_single_layers"] = skip_layers

        seed = out.get("seed")
        if seed is not None:
            if isinstance(seed, bool) or not isinstance(seed, int) or not (0 <= seed <= 2_147_483_647):
                raise ValueError("INVALID_SAMPLING_PARAMS: seed must be in [0, 2147483647]")
        out["seed"] = seed

        layout = out.get("layout")
        if layout is not None:
            out["layout"] = validate_moss_layout(layout)
        else:
            out["layout"] = None

        return out

    def health(self) -> bool:
        try:
            if self._client is not None:
                resp = self._client.get(f"{self.base_url}/health", timeout=5.0)
            else:
                with httpx.Client(timeout=5.0, trust_env=False) as client:
                    resp = client.get(f"{self.base_url}/health")
            return resp.status_code == 200 and resp.json().get("status") == "healthy"
        except Exception:
            return False

    def generate(self, params: dict[str, Any]) -> tuple[bytes, dict[str, Any]]:
        payload = {
            "prompt": params["prompt"],
            "negative_prompt": params.get("negative_prompt", ""),
            "width": params["width"],
            "height": params["height"],
            "steps": params.get("steps", 50),
            "cfg_scale": params.get("cfg_scale", 3.0),
            "sampler": params.get("sampler", "euler"),
            "scheduler": params.get("scheduler", "simple"),
            "shift": params.get("shift", 5.0),
            "skip_single_layers": params.get("skip_single_layers", "22"),
            "seed": params.get("seed"),
            "layout": params.get("layout"),
        }
        try:
            if self._client is not None:
                resp = self._client.post(f"{self.base_url}/v1/generate", json=payload, timeout=self.timeout)
            else:
                with httpx.Client(timeout=self.timeout, trust_env=False) as client:
                    resp = client.post(f"{self.base_url}/v1/generate", json=payload)
        except httpx.TimeoutException as exc:
            raise MossProviderError("UPSTREAM_TIMEOUT", f"image-moss-api timed out: {exc}", 504) from exc
        except httpx.HTTPError as exc:
            raise MossProviderError("MODEL_NOT_READY", f"image-moss-api unreachable: {exc}", 503) from exc

        if resp.status_code != 200:
            try:
                err_body = resp.json()
                code = err_body.get("error_code", "INTERNAL_ERROR")
                msg = err_body.get("message", resp.text)
            except Exception:
                code = "INTERNAL_ERROR"
                msg = resp.text
            raise MossProviderError(code, msg, resp.status_code)

        body = resp.json()
        img_bytes = base64.b64decode(body["image_base64"])
        meta = {
            "engine": "moss-image",
            "model": DEFAULT_MOSS_MODEL,
            "prompt": params["prompt"],
            "negative_prompt": params.get("negative_prompt", ""),
            "width": body.get("width", params["width"]),
            "height": body.get("height", params["height"]),
            "resolved_seed": body.get("resolved_seed", params.get("seed")),
            "steps": body.get("steps", params.get("steps", 50)),
            "cfg_scale": body.get("cfg_scale", params.get("cfg_scale", 3.0)),
            "sampler": body.get("sampler", params.get("sampler", "euler")),
            "scheduler": body.get("scheduler", params.get("scheduler", "simple")),
            "shift": body.get("shift", params.get("shift", 5.0)),
            "skip_single_layers": params.get("skip_single_layers", "22"),
            "prompt_mode": body.get("prompt_mode", "layout" if params.get("layout") else "text"),
            "elapsed_ms": body.get("elapsed_ms", 0),
        }
        if params.get("layout") is not None:
            meta["layout"] = params["layout"]
        return img_bytes, meta
