"""Service configuration; upstream code is loaded only in the isolated worker."""
import os
import json
from pathlib import Path
from dataclasses import dataclass

_UPSTREAM = json.loads((Path(__file__).resolve().parents[1] / "upstream.lock.json").read_text())
FACEFUSION_VERSION = _UPSTREAM["version"]
FACEFUSION_COMMIT = _UPSTREAM["commit"]


@dataclass(frozen=True)
class Settings:
    source_root: str = "/opt/facefusion"
    model_root: str = "/models/facefusion"
    provider: str = "cuda"
    swapper: str = "hyperswap_1a_256"
    pixel_boost: str = "256x256"
    mask_types: tuple = ("box", "occlusion", "region")
    enhancer: str = "codeformer"
    enhancer_blend: int = 80
    timeout: float = 120.0
    startup_timeout: float = 180.0
    max_pixels: int = 40_000_000

    @classmethod
    def from_env(cls):
        return cls(
            source_root=os.getenv("FACEFUSION_SOURCE_ROOT", cls.source_root),
            model_root=os.getenv("FACEFUSION_MODEL_ROOT", cls.model_root),
            provider=os.getenv("FACEFUSION_PROVIDER", cls.provider),
            swapper=os.getenv("FACEFUSION_SWAPPER", cls.swapper),
            pixel_boost=os.getenv("FACEFUSION_PIXEL_BOOST", cls.pixel_boost),
            mask_types=tuple(os.getenv("FACEFUSION_MASK_TYPES", "box,occlusion,region").split(",")),
            enhancer=os.getenv("FACEFUSION_ENHANCER", cls.enhancer),
            enhancer_blend=int(os.getenv("FACEFUSION_ENHANCER_BLEND", "80")),
            timeout=float(os.getenv("FACEFUSION_TIMEOUT", "120")),
            startup_timeout=float(os.getenv("FACEFUSION_STARTUP_TIMEOUT", "180")),
            max_pixels=int(os.getenv("FACEFUSION_MAX_PIXELS", "40000000")),
        )

    def __post_init__(self):
        if self.provider not in ("cuda", "cpu"):
            raise ValueError("FACEFUSION_PROVIDER must be cuda or cpu")
        if not 0 <= self.enhancer_blend <= 100:
            raise ValueError("Invalid enhancer blend")
        if not self.mask_types or not set(self.mask_types) <= {"box", "occlusion", "region", "area"}:
            raise ValueError("Invalid face mask types")
        if not (0 < self.timeout <= 3600 and 0 < self.startup_timeout <= 3600 and self.max_pixels > 0):
            raise ValueError("Invalid worker limits")
