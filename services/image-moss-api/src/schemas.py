"""Request/response schemas and domain validation for image-moss-api."""
from __future__ import annotations

import base64
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

FORBIDDEN_CONTROL_TOKENS: tuple[str, ...] = (
    "<|im_start|>",
    "<|im_end|>",
    "<think>",
    "</think>",
    "<|endoftext|>",
)

ALLOWED_SAMPLERS: frozenset[str] = frozenset({"euler", "er_sde"})
ALLOWED_SCHEDULERS: frozenset[str] = frozenset({"simple"})
MAX_REFERENCE_IMAGES = 3
MAX_EDIT_REFERENCE_IMAGES = 4


class MossApiError(Exception):
    """Structured domain error mapped to HTTP status and error_code."""

    def __init__(self, status_code: int, error_code: str, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error_code = error_code
        self.message = message


def reject_control_tokens(text: str, field_name: str, *, error_code: str = "INVALID_ARGUMENT") -> None:
    for token in FORBIDDEN_CONTROL_TOKENS:
        if token in text:
            raise MossApiError(
                400,
                error_code,
                f"{field_name} contains forbidden control token '{token}'",
            )


class LayoutItem(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=False)

    id: int
    desc: str
    box_2d: list[int | float]


class LayoutSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=False)

    meta: str
    items: list[LayoutItem]


class GenerateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=False)

    prompt: str = Field(default="")
    negative_prompt: str = Field(default="")
    width: int = Field(default=1024)
    height: int = Field(default=1024)
    steps: int = Field(default=50)
    cfg_scale: float = Field(default=3.0)
    sampler: str = Field(default="euler")
    scheduler: str = Field(default="simple")
    shift: float = Field(default=5.0)
    skip_single_layers: str = Field(default="22")
    seed: int | None = Field(default=None)
    layout: dict[str, Any] | None = Field(default=None)
    reference_images_base64: list[str] | None = Field(default=None)
    # Experimental Edit mode (Moss-Image-Edit LoRA + reference latents); references keep request order
    edit: bool = Field(default=False)
    edit_lora_strength: float = Field(default=1.0)
    reference_max_edge: int = Field(default=512)


class GenerateResponse(BaseModel):
    status: str = "ok"
    image_base64: str
    mime_type: str = "image/png"
    width: int
    height: int
    resolved_seed: int
    steps: int
    cfg_scale: float
    sampler: str
    scheduler: str
    shift: float
    prompt_mode: str
    elapsed_ms: int


class HealthResponse(BaseModel):
    status: str
    service: str = "image-moss-api"
    model: str = "moss-image-sft-int8-convrot"
    comfyui_version: str
    comfyui_commit: str
    models_verified: bool
    edit_available: bool = False
    comfyui_reachable: bool
    weights: dict[str, str]
    gpu: dict[str, Any]


def validate_layout(raw_layout: Any, *, max_items: int = 32) -> dict[str, Any]:
    if not isinstance(raw_layout, dict):
        raise MossApiError(400, "INVALID_LAYOUT_JSON", "layout must be a JSON object")
    unknown_top = set(raw_layout.keys()) - {"meta", "items"}
    if unknown_top:
        raise MossApiError(
            400,
            "INVALID_LAYOUT_JSON",
            f"layout contains unsupported top-level fields: {sorted(unknown_top)}",
        )
    meta = raw_layout.get("meta")
    if not isinstance(meta, str) or not meta.strip():
        raise MossApiError(400, "INVALID_LAYOUT_JSON", "layout.meta must be a non-empty string")
    meta = meta.strip()
    if len(meta) > 2000:
        raise MossApiError(400, "INVALID_LAYOUT_JSON", "layout.meta must not exceed 2000 characters")
    reject_control_tokens(meta, "layout.meta", error_code="INVALID_LAYOUT_JSON")

    items = raw_layout.get("items")
    if not isinstance(items, list) or len(items) == 0:
        raise MossApiError(400, "INVALID_LAYOUT_JSON", "layout.items must be a non-empty array")
    if len(items) > max_items:
        raise MossApiError(
            400,
            "INVALID_LAYOUT_JSON",
            f"layout.items count {len(items)} exceeds maximum {max_items}",
        )

    seen_ids: set[int] = set()
    coord_mode: str | None = None  # "normalized" or "int1000"
    validated_items: list[dict[str, Any]] = []

    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            raise MossApiError(400, "INVALID_LAYOUT_JSON", f"layout.items[{idx}] must be an object")
        unknown_item_keys = set(item.keys()) - {"id", "desc", "box_2d"}
        if unknown_item_keys:
            raise MossApiError(
                400,
                "INVALID_LAYOUT_JSON",
                f"layout.items[{idx}] contains unsupported fields: {sorted(unknown_item_keys)}",
            )
        item_id = item.get("id")
        if isinstance(item_id, bool) or not isinstance(item_id, int) or item_id < 1:
            raise MossApiError(
                400,
                "INVALID_LAYOUT_JSON",
                f"layout.items[{idx}].id must be a positive integer",
            )
        if item_id in seen_ids:
            raise MossApiError(
                400,
                "INVALID_LAYOUT_JSON",
                f"Duplicate layout.items id: {item_id}",
            )
        seen_ids.add(item_id)

        desc = item.get("desc")
        if not isinstance(desc, str) or not desc.strip():
            raise MossApiError(
                400,
                "INVALID_LAYOUT_JSON",
                f"layout.items[{idx}].desc must be a non-empty string",
            )
        desc = desc.strip()
        if len(desc) > 500:
            raise MossApiError(
                400,
                "INVALID_LAYOUT_JSON",
                f"layout.items[{idx}].desc must not exceed 500 characters",
            )
        reject_control_tokens(desc, f"layout.items[{idx}].desc", error_code="INVALID_LAYOUT_JSON")

        box = item.get("box_2d")
        if not isinstance(box, list) or len(box) != 4:
            raise MossApiError(
                400,
                "INVALID_LAYOUT_JSON",
                f"layout.items[{idx}].box_2d must be a 4-element array [x1, y1, x2, y2]",
            )
        for coord in box:
            if isinstance(coord, bool) or not isinstance(coord, (int, float)):
                raise MossApiError(
                    400,
                    "INVALID_LAYOUT_JSON",
                    f"layout.items[{idx}].box_2d elements must be numbers",
                )

        x1, y1, x2, y2 = box
        if not (x1 < x2 and y1 < y2):
            raise MossApiError(
                400,
                "INVALID_LAYOUT_JSON",
                f"layout.items[{idx}].box_2d must satisfy x1 < x2 and y1 < y2",
            )

        has_float = any(isinstance(c, float) for c in box)
        max_val = max(box)
        min_val = min(box)
        if min_val < 0:
            raise MossApiError(
                400,
                "INVALID_LAYOUT_JSON",
                f"layout.items[{idx}].box_2d coordinates must be non-negative",
            )

        if has_float or max_val <= 1.0:
            if max_val > 1.0:
                raise MossApiError(
                    400,
                    "INVALID_LAYOUT_JSON",
                    f"layout.items[{idx}].box_2d floating-point coordinates must be within [0.0, 1.0]",
                )
            item_mode = "normalized"
            clean_box = [round(float(c), 4) for c in box]
        else:
            if not all(isinstance(c, int) for c in box) or max_val > 1000:
                raise MossApiError(
                    400,
                    "INVALID_LAYOUT_JSON",
                    f"layout.items[{idx}].box_2d integer coordinates must be within [0, 1000]",
                )
            item_mode = "int1000"
            clean_box = [int(c) for c in box]

        if coord_mode is None:
            coord_mode = item_mode
        elif coord_mode != item_mode:
            raise MossApiError(
                400,
                "INVALID_LAYOUT_JSON",
                "layout.items must not mix normalized [0.0, 1.0] and integer [0, 1000] box_2d coordinates",
            )

        validated_items.append(
            {
                "id": item_id,
                "desc": desc,
                "box_2d": clean_box,
            }
        )

    return {"meta": meta, "items": validated_items}


def validate_generate_request(
    req: GenerateRequest,
    *,
    max_pixels: int = 2_097_152,
    max_steps: int = 50,
    max_layout_items: int = 32,
) -> dict[str, Any]:
    prompt = (req.prompt or "").strip()
    if not prompt:
        raise MossApiError(400, "INVALID_ARGUMENT", "prompt must be a non-empty string")
    if len(prompt) > 4000:
        raise MossApiError(400, "INVALID_ARGUMENT", "prompt must not exceed 4000 characters")
    reject_control_tokens(prompt, "prompt")

    negative_prompt = (req.negative_prompt or "").strip()
    if len(negative_prompt) > 2000:
        raise MossApiError(400, "INVALID_ARGUMENT", "negative_prompt must not exceed 2000 characters")
    reject_control_tokens(negative_prompt, "negative_prompt")

    if not (512 <= req.width <= 2048 and 512 <= req.height <= 2048):
        raise MossApiError(
            400,
            "INVALID_DIMENSIONS",
            f"width and height must be within [512, 2048], got {req.width}x{req.height}",
        )
    if req.width % 16 != 0 or req.height % 16 != 0:
        raise MossApiError(
            400,
            "INVALID_DIMENSIONS",
            f"width and height must be multiples of 16, got {req.width}x{req.height}",
        )
    if req.width * req.height > max_pixels:
        raise MossApiError(
            400,
            "INVALID_DIMENSIONS",
            f"Total pixels {req.width * req.height} exceeds maximum {max_pixels}",
        )

    if not (1 <= req.steps <= max_steps):
        raise MossApiError(
            400,
            "INVALID_SAMPLING_PARAMS",
            f"steps must be between 1 and {max_steps}, got {req.steps}",
        )
    if not (1.0 <= req.cfg_scale <= 6.0):
        raise MossApiError(
            400,
            "INVALID_SAMPLING_PARAMS",
            f"cfg_scale must be between 1.0 and 6.0, got {req.cfg_scale}",
        )
    if req.sampler not in ALLOWED_SAMPLERS:
        raise MossApiError(
            400,
            "INVALID_SAMPLING_PARAMS",
            f"sampler must be one of {sorted(ALLOWED_SAMPLERS)}, got '{req.sampler}'",
        )
    if req.scheduler not in ALLOWED_SCHEDULERS:
        raise MossApiError(
            400,
            "INVALID_SAMPLING_PARAMS",
            f"scheduler must be one of {sorted(ALLOWED_SCHEDULERS)}, got '{req.scheduler}'",
        )
    if not (1.0 <= req.shift <= 10.0):
        raise MossApiError(
            400,
            "INVALID_SAMPLING_PARAMS",
            f"shift must be between 1.0 and 10.0, got {req.shift}",
        )

    skip_layers = (req.skip_single_layers or "").strip()
    if skip_layers:
        if not re.fullmatch(r"\d+(\s*,\s*\d+)*", skip_layers):
            raise MossApiError(
                400,
                "INVALID_SAMPLING_PARAMS",
                "skip_single_layers must be comma-separated layer indices",
            )
        layer_nums = [int(x.strip()) for x in skip_layers.split(",")]
        if any(n < 0 or n > 63 for n in layer_nums):
            raise MossApiError(
                400,
                "INVALID_SAMPLING_PARAMS",
                "skip_single_layers indices must be within [0, 63]",
            )
        skip_layers = ",".join(str(n) for n in layer_nums)

    if req.seed is not None and not (0 <= req.seed <= 2_147_483_647):
        raise MossApiError(
            400,
            "INVALID_SAMPLING_PARAMS",
            "seed must be within [0, 2147483647]",
        )

    validated_layout = None
    if req.layout is not None:
        validated_layout = validate_layout(req.layout, max_items=max_layout_items)

    validated_refs: list[bytes] = []
    max_refs = MAX_EDIT_REFERENCE_IMAGES if req.edit else MAX_REFERENCE_IMAGES
    if req.edit:
        if not req.reference_images_base64:
            raise MossApiError(400, "INVALID_ARGUMENT", "edit requires at least 1 reference image")
        if validated_layout is not None:
            raise MossApiError(400, "INVALID_ARGUMENT", "edit does not support layout")
        if not (0.0 <= req.edit_lora_strength <= 2.0):
            raise MossApiError(
                400,
                "INVALID_SAMPLING_PARAMS",
                f"edit_lora_strength must be between 0.0 and 2.0, got {req.edit_lora_strength}",
            )
        if not (256 <= req.reference_max_edge <= 1024) or req.reference_max_edge % 16 != 0:
            raise MossApiError(
                400,
                "INVALID_ARGUMENT",
                f"reference_max_edge must be a multiple of 16 within [256, 1024], got {req.reference_max_edge}",
            )
    if req.reference_images_base64:
        if len(req.reference_images_base64) > max_refs:
            raise MossApiError(
                400,
                "INVALID_ARGUMENT",
                f"reference_images_base64 supports at most {max_refs} images, got {len(req.reference_images_base64)}",
            )
        for idx, raw_b64 in enumerate(req.reference_images_base64):
            if not isinstance(raw_b64, str) or not raw_b64.strip():
                raise MossApiError(
                    400,
                    "INVALID_ARGUMENT",
                    f"reference_images_base64[{idx}] must be a non-empty base64 string",
                )
            clean_b64 = raw_b64.strip()
            if "," in clean_b64 and clean_b64.startswith("data:"):
                clean_b64 = clean_b64.split(",", 1)[1]
            try:
                decoded = base64.b64decode(clean_b64, validate=True)
            except Exception as exc:
                raise MossApiError(
                    400,
                    "INVALID_ARGUMENT",
                    f"reference_images_base64[{idx}] is not valid base64: {exc}",
                ) from exc
            if len(decoded) < 16 or len(decoded) > 20 * 1024 * 1024:
                raise MossApiError(
                    400,
                    "INVALID_ARGUMENT",
                    f"reference_images_base64[{idx}] decoded size must be between 16B and 20MB",
                )
            validated_refs.append(decoded)

    return {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "width": req.width,
        "height": req.height,
        "steps": req.steps,
        "cfg_scale": float(req.cfg_scale),
        "sampler": req.sampler,
        "scheduler": req.scheduler,
        "shift": float(req.shift),
        "skip_single_layers": skip_layers,
        "seed": req.seed,
        "layout": validated_layout,
        "reference_images": validated_refs,
        "edit": bool(req.edit),
        "edit_lora_strength": float(req.edit_lora_strength),
        "reference_max_edge": int(req.reference_max_edge),
    }
