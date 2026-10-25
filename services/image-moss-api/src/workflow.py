"""Fixed ComfyUI API workflow builder and MOSS layout JSON serializer."""
from __future__ import annotations

import json
import re
from typing import Any

DIFFUSION_MODEL_FILE = "moss_image_sft_int8_convrot.safetensors"
TEXT_ENCODER_FILE = "qwen3.5_9b_int8_convrot.safetensors"
VAE_FILE = "flux2-vae_small.safetensors"
EDIT_LORA_FILE = "moss_image_edit.safetensors"
# TextEncodeMOSSImageEdit layout for chinoll/Moss-Image-Edit: references go to the DiT as latents only
EDIT_FORMAT = "Moss-Image-Edit"

VISION_TOKEN_SPAN = "<|vision_start|><|image_pad|><|vision_end|>"
REFERENCE_CONDITIONING_INSTRUCTION = (
    "保持参考图主体面部身份与核心特征一致，严格按以下描述生成：\n"
)
REFERENCE_CONDITIONING_PREFIX = f"{VISION_TOKEN_SPAN}{REFERENCE_CONDITIONING_INSTRUCTION}"
DEFAULT_REFERENCE_RESOLUTION = 384

_STRAY_VISION_TOKEN_RE = re.compile(
    r"(?:Picture\s+\d+\s*:\s*)?<\|vision_start\|>\s*(?:<\|image_pad\|>\s*)*<\|vision_end\|>\s*"
    r"|<\|(?:vision_start|image_pad|vision_end)\|>"
)


def _strip_stray_vision_tokens(text: str) -> str:
    return _STRAY_VISION_TOKEN_RE.sub("", text).strip()


def _build_reference_prefix(reference_image_count: int) -> str:
    return f"{VISION_TOKEN_SPAN * reference_image_count}{REFERENCE_CONDITIONING_INSTRUCTION}"


def serialize_moss_prompt(
    prompt: str,
    layout: dict[str, Any] | None,
    *,
    has_reference_image: bool = False,
    reference_image_count: int = 0,
) -> tuple[str, str]:
    """Serialize prompt or layout JSON into the exact string expected by CLIPTextEncode / TextEncodeQwenImage21.

    Matches upstream ComfyUI `MOSSImageLayoutPrompt` (`comfy_extras/nodes_moss_image.py`),
    which formats `{"meta": ..., "items": ...}` via `json.dumps(..., ensure_ascii=False, separators=(",", ":"))`
    before `comfy/text_encoders/moss_image.py` wraps it into the Qwen3.5 user/think template.
    When reference images are present, prepends one `<|vision_start|><|image_pad|><|vision_end|>` span
    per reference image so `MOSSImageTokenizer` + `Qwen35VisionModel` replaces 100% of `<|image_pad|>` (248056)
    tokens and preserves the trailing `<|im_start|>assistant\n<think>\n` template.
    """
    effective_ref_count = (
        int(reference_image_count)
        if reference_image_count > 0
        else (1 if has_reference_image else 0)
    )
    clean_prompt = _strip_stray_vision_tokens(prompt)
    if layout is None:
        if effective_ref_count > 0:
            return f"{_build_reference_prefix(effective_ref_count)}{clean_prompt}", "reference"
        return clean_prompt, "text"
    meta = _strip_stray_vision_tokens(str(layout["meta"]))
    if clean_prompt and clean_prompt != meta and clean_prompt not in meta:
        combined_meta = f"{meta} {clean_prompt}".strip()
    else:
        combined_meta = meta
    payload = {
        "meta": combined_meta,
        "items": [
            {
                "id": item["id"],
                "desc": _strip_stray_vision_tokens(str(item["desc"])),
                "box_2d": item["box_2d"],
            }
            for item in layout["items"]
        ],
    }
    serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    if effective_ref_count > 0:
        return f"{_build_reference_prefix(effective_ref_count)}{serialized}", "reference_layout"
    return serialized, "layout"


def build_comfy_workflow(
    *,
    positive_text: str,
    negative_text: str,
    width: int,
    height: int,
    steps: int,
    cfg_scale: float,
    sampler: str,
    scheduler: str,
    shift: float,
    skip_single_layers: str,
    seed: int,
    filename_prefix: str = "moss_image",
    reference_image_filename: str | None = None,
    reference_image_filenames: list[str] | None = None,
    reference_resolution: int = DEFAULT_REFERENCE_RESOLUTION,
) -> dict[str, Any]:
    """Build the fixed ComfyUI API workflow for MOSS-Image (with optional Qwen3.5-9B Vision reference input)."""
    if reference_image_filenames:
        ref_files = [str(f) for f in reference_image_filenames if f]
    elif reference_image_filename:
        ref_files = [str(reference_image_filename)]
    else:
        ref_files = []

    if ref_files:
        positive_inputs: dict[str, Any] = {
            "clip": ["2", 0],
            "prompt": positive_text,
            "negative_prompt": "",
            "resolution": int(reference_resolution),
        }
        for idx in range(1, len(ref_files) + 1):
            positive_inputs[f"images.image_{idx}"] = [str(11 + idx), 0]
        positive_node: dict[str, Any] = {
            "class_type": "TextEncodeQwenImage21",
            "inputs": positive_inputs,
        }
    else:
        positive_node = {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "clip": ["2", 0],
                "text": positive_text,
            },
        }

    wf: dict[str, Any] = {
        "1": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": DIFFUSION_MODEL_FILE,
                "weight_dtype": "default",
            },
        },
        "2": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": TEXT_ENCODER_FILE,
                "type": "moss_image",
                "device": "default",
            },
        },
        "3": {
            "class_type": "VAELoader",
            "inputs": {
                "vae_name": VAE_FILE,
            },
        },
        "4": {
            "class_type": "ModelSamplingAuraFlow",
            "inputs": {
                "model": ["1", 0],
                "shift": shift,
            },
        },
        "5": {
            "class_type": "SkipLayerGuidanceDiTSimple",
            "inputs": {
                "model": ["4", 0],
                "double_layers": "",
                "single_layers": skip_single_layers,
                "start_percent": 0.0,
                "end_percent": 0.5,
            },
        },
        "6": positive_node,
        "7": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "clip": ["2", 0],
                "text": negative_text,
            },
        },
        "8": {
            "class_type": "EmptyFlux2LatentImage",
            "inputs": {
                "width": width,
                "height": height,
                "batch_size": 1,
            },
        },
        "9": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["5", 0],
                "positive": ["6", 0],
                "negative": ["7", 0],
                "latent_image": ["8", 0],
                "seed": seed,
                "steps": steps,
                "cfg": cfg_scale,
                "sampler_name": sampler,
                "scheduler": scheduler,
                "denoise": 1.0,
            },
        },
        "10": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["9", 0],
                "vae": ["3", 0],
            },
        },
        "11": {
            "class_type": "SaveImage",
            "inputs": {
                "images": ["10", 0],
                "filename_prefix": filename_prefix,
            },
        },
    }
    for idx, ref_file in enumerate(ref_files, start=1):
        wf[str(11 + idx)] = {
            "class_type": "LoadImage",
            "inputs": {
                "image": ref_file,
            },
        }
    return wf


def build_edit_workflow(
    *,
    positive_text: str,
    negative_text: str,
    width: int,
    height: int,
    steps: int,
    cfg_scale: float,
    sampler: str,
    scheduler: str,
    shift: float,
    skip_single_layers: str,
    seed: int,
    reference_image_filenames: list[str],
    lora_strength: float = 1.0,
    filename_prefix: str = "moss_image_edit",
) -> dict[str, Any]:
    """Build the fixed Edit workflow: Moss-Image-Edit LoRA plus core TextEncodeMOSSImageEdit reference latents.

    References must already be scaled by the caller; the node encodes them at the size they come in.
    The same encode node is used for the negative prompt, as upstream requires.
    """
    if not reference_image_filenames:
        raise ValueError("edit workflow requires at least one reference image")
    wf = build_comfy_workflow(
        positive_text=positive_text,
        negative_text=negative_text,
        width=width,
        height=height,
        steps=steps,
        cfg_scale=cfg_scale,
        sampler=sampler,
        scheduler=scheduler,
        shift=shift,
        skip_single_layers=skip_single_layers,
        seed=seed,
        filename_prefix=filename_prefix,
    )
    wf["20"] = {
        "class_type": "LoraLoaderModelOnly",
        "inputs": {
            "model": ["1", 0],
            "lora_name": EDIT_LORA_FILE,
            "strength_model": float(lora_strength),
        },
    }
    wf["4"]["inputs"]["model"] = ["20", 0]
    images = {f"images.image_{idx}": [str(11 + idx), 0] for idx in range(1, len(reference_image_filenames) + 1)}
    for node_id, text in (("6", positive_text), ("7", negative_text)):
        wf[node_id] = {
            "class_type": "TextEncodeMOSSImageEdit",
            "inputs": {
                "clip": ["2", 0],
                "prompt": text,
                "vae": ["3", 0],
                "edit_format": EDIT_FORMAT,
                **images,
            },
        }
    for idx, ref_file in enumerate(reference_image_filenames, start=1):
        wf[str(11 + idx)] = {
            "class_type": "LoadImage",
            "inputs": {
                "image": str(ref_file),
            },
        }
    return wf
