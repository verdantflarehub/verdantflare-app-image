"""Service configuration for image-moss-api."""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path

_UPSTREAM = json.loads((Path(__file__).resolve().parents[1] / "upstream.lock.json").read_text(encoding="utf-8"))
COMFYUI_VERSION: str = _UPSTREAM["version"]
COMFYUI_COMMIT: str = _UPSTREAM["commit"]


@dataclass(frozen=True)
class Settings:
    comfyui_root: str = "/opt/comfyui"
    model_root: str = "/models/moss-image"
    comfyui_host: str = "127.0.0.1"
    comfyui_port: int = 8188
    startup_timeout: float = 180.0
    inference_timeout: float = 180.0
    max_pixels: int = 2_097_152
    max_steps: int = 50
    max_layout_items: int = 32
    max_queue: int = 2
    verify_sha256_on_boot: bool = False
    auto_start_comfyui: bool = True
    enable_edit: bool = True

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            comfyui_root=os.getenv("MOSS_COMFYUI_ROOT", cls.comfyui_root),
            model_root=os.getenv("MOSS_IMAGE_MODEL_ROOT", cls.model_root),
            comfyui_host=os.getenv("MOSS_COMFYUI_HOST", cls.comfyui_host),
            comfyui_port=int(os.getenv("MOSS_COMFYUI_PORT", str(cls.comfyui_port))),
            startup_timeout=float(os.getenv("MOSS_STARTUP_TIMEOUT", str(cls.startup_timeout))),
            inference_timeout=float(os.getenv("MOSS_INFERENCE_TIMEOUT", str(cls.inference_timeout))),
            max_pixels=int(os.getenv("MOSS_MAX_PIXELS", str(cls.max_pixels))),
            max_steps=int(os.getenv("MOSS_MAX_STEPS", str(cls.max_steps))),
            max_layout_items=int(os.getenv("MOSS_MAX_LAYOUT_ITEMS", str(cls.max_layout_items))),
            max_queue=int(os.getenv("MOSS_MAX_QUEUE", str(cls.max_queue))),
            verify_sha256_on_boot=os.getenv("MOSS_VERIFY_SHA256_ON_BOOT", "false").lower() in ("1", "true", "yes"),
            auto_start_comfyui=os.getenv("MOSS_AUTO_START_COMFYUI", "true").lower() in ("1", "true", "yes"),
            enable_edit=os.getenv("MOSS_ENABLE_EDIT", "true").lower() in ("1", "true", "yes"),
        )

    def __post_init__(self) -> None:
        if self.comfyui_host != "127.0.0.1":
            raise ValueError("MOSS_COMFYUI_HOST must be 127.0.0.1")
        if not (1024 <= self.comfyui_port <= 65535):
            raise ValueError("Invalid MOSS_COMFYUI_PORT")
        if not (0 < self.startup_timeout <= 3600 and 0 < self.inference_timeout <= 3600):
            raise ValueError("Invalid timeout settings")
        if self.max_pixels <= 0 or self.max_steps <= 0 or self.max_layout_items <= 0 or self.max_queue < 0:
            raise ValueError("Invalid limit settings")

    @property
    def comfyui_base_url(self) -> str:
        return f"http://{self.comfyui_host}:{self.comfyui_port}"
