"""Fixed ComfyUI API workflow builder and MOSS layout JSON serializer."""
from __future__ import annotations

import json
from typing import Any

DIFFUSION_MODEL_FILE = "moss_image_sft_int8_convrot.safetensors"
TEXT_ENCODER_FILE = "qwen3.5_9b_int8_convrot.safetensors"
VAE_FILE = "flux2-vae_small.safetensors"

REFERENCE_CONDITIONING_PREFIX = (
    "<|vision_start|><|image_pad|><|vision_end|>"
    "严格保持参考图中人物的面部骨相、精致鹅蛋脸型、中分微卷深棕长发配额前两三缕极细碎发（露出饱满额头，无厚齐刘海）、清澈杏仁眼及微笑时左脸颊（画面右侧）明显的深圆酒窝"
    "（仅参考人物面部五官与发型神态，完全忽略参考图中的针织毛衣与背景，严格按以下描述的动作姿态与服装场景生成）：\n"
)


def serialize_moss_prompt(
    prompt: str,
    layout: dict[str, Any] | None,
    *,
    has_reference_image: bool = False,
) -> tuple[str, str]:
    """Serialize prompt or layout JSON into the exact string expected by CLIPTextEncode / TextEncodeQwenImageEdit.

    Matches upstream ComfyUI `MOSSImageLayoutPrompt` (`comfy_extras/nodes_moss_image.py`),
    which formats `{"meta": ..., "items": ...}` via `json.dumps(..., ensure_ascii=False, separators=(",", ":"))`
    before `comfy/text_encoders/moss_image.py` wraps it into the Qwen3.5 user/think template.
    When `has_reference_image=True`, prepends `<|vision_start|><|image_pad|><|vision_end|>` so
    `Qwen35ImageTokenizer` + `Qwen35VisionModel` encodes the reference image into the DiT conditioning sequence.
    """
    if layout is None:
        if has_reference_image:
            return f"{REFERENCE_CONDITIONING_PREFIX}{prompt}", "reference"
        return prompt, "text"
    meta = layout["meta"]
    if prompt and prompt != meta and prompt not in meta:
        combined_meta = f"{meta} {prompt}".strip()
    else:
        combined_meta = meta
    payload = {
        "meta": combined_meta,
        "items": [
            {
                "id": item["id"],
                "desc": item["desc"],
                "box_2d": item["box_2d"],
            }
            for item in layout["items"]
        ],
    }
    serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    if has_reference_image:
        return f"{REFERENCE_CONDITIONING_PREFIX}{serialized}", "reference_layout"
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
) -> dict[str, Any]:
    """Build the fixed ComfyUI API workflow for MOSS-Image (with optional Qwen3.5-9B Vision reference input)."""
    if reference_image_filename:
        positive_node: dict[str, Any] = {
            "class_type": "TextEncodeQwenImageEdit",
            "inputs": {
                "clip": ["2", 0],
                "prompt": positive_text,
                "image": ["12", 0],
            },
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
    if reference_image_filename:
        wf["12"] = {
            "class_type": "LoadImage",
            "inputs": {
                "image": reference_image_filename,
            },
        }
    return wf
