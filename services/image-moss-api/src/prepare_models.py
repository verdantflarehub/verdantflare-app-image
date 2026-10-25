"""Explicit model preparation for MOSS-Image INT8-ConvRot weights.

Downloads pinned revisions from ModelScope Comfy-Org/MOSS-Image (or verifies
existing files without overwriting) and validates byte length and SHA-256.
The serving worker remains offline and never downloads weights at runtime.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen

MODELSCOPE_REPO = "Comfy-Org/MOSS-Image"

MODEL_MANIFEST: tuple[dict[str, object], ...] = (
    {
        "key": "diffusion_model",
        "path": "diffusion_models/moss_image_sft_int8_convrot.safetensors",
        "revision": "05cf96d24fa7b4755289b923ac5729b16cc0d9ba",
        "size": 5413698958,
        "sha256": "b4d1475bc7eeffd7b809e82ea13a1fba1fb85c4e18365af3813722faa5c0fe4f",
    },
    {
        "key": "text_encoder",
        "path": "text_encoders/qwen3.5_9b_int8_convrot.safetensors",
        "revision": "05cf96d24fa7b4755289b923ac5729b16cc0d9ba",
        "size": 9748121532,
        "sha256": "32e3787472f708fa5f83624acbf5ef23f7fc89c439e05065ea57cfe5a97f025a",
    },
    {
        "key": "vae",
        "path": "vae/flux2-vae_small.safetensors",
        "revision": "3b6958fde4d8ecd98c25c2973ef13df54fb0c4de",
        "size": 249521340,
        "sha256": "d8d52ba036475f5fb07c8b435e176d3d97ebfa82f0d1a1c317f9cc1e25bd013b",
    },
)

# Optional Edit LoRA (moss-edit-lora format, read by ComfyUI core LoRA loader). Prepared and
# served only when MOSS_ENABLE_EDIT is on; T2I / LAYOUT / R2I never depend on it.
EDIT_LORA_MANIFEST: tuple[dict[str, object], ...] = (
    {
        "key": "edit_lora",
        "path": "loras/moss_image_edit.safetensors",
        "repo": "ChenkinLab/Moss-Image-Edit",
        "source_path": "adapter_model.safetensors",
        "revision": "0d42c24245ddf2d67d74285962c188ab3bee6ffb",
        "size": 297382216,
        "sha256": "17a48b8c605dcffdedb92a2b887d18cf316505be90802717669fb6556098dcdf",
    },
)


def edit_enabled() -> bool:
    return os.getenv("MOSS_ENABLE_EDIT", "true").lower() in ("1", "true", "yes")


def sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    total = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
            total += len(chunk)
    return digest.hexdigest(), total


def verify_model_file(
    path: Path,
    expected_sha256: str,
    expected_size: int,
    *,
    check_sha256: bool = True,
) -> dict[str, object]:
    if path.is_symlink():
        raise ValueError(f"Refusing symlink model path: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Missing model file: {path}")
    stat_size = path.stat().st_size
    if stat_size != expected_size:
        raise ValueError(
            f"Size mismatch for {path.name}: expected {expected_size}, got {stat_size}"
        )
    if not check_sha256:
        return {"path": path.name, "bytes": stat_size, "sha256": expected_sha256}
    actual_sha, actual_size = sha256_file(path)
    if actual_size != expected_size:
        raise ValueError(
            f"Size mismatch for {path.name}: expected {expected_size}, got {actual_size}"
        )
    if actual_sha != expected_sha256:
        raise ValueError(
            f"SHA-256 mismatch for {path.name}: expected {expected_sha256}, got {actual_sha}"
        )
    return {"path": path.name, "bytes": actual_size, "sha256": actual_sha}


def prepare_one(
    model_root: Path,
    item: dict[str, object],
    *,
    repo: str = MODELSCOPE_REPO,
) -> dict[str, object]:
    rel_path = str(item["path"])
    revision = str(item["revision"])
    expected_size = int(item["size"])
    expected_sha256 = str(item["sha256"])

    target = model_root / rel_path
    if target.is_symlink():
        raise ValueError(f"Refusing symlink model path: {target}")
    if target.exists():
        print(json.dumps({"event": "verifying_existing", "path": rel_path}), flush=True)
        verified = verify_model_file(target, expected_sha256, expected_size, check_sha256=True)
        target.chmod(0o644)
        return {
            "path": rel_path,
            "revision": revision,
            "bytes": verified["bytes"],
            "sha256": verified["sha256"],
        }

    target.parent.mkdir(parents=True, exist_ok=True)
    repo = str(item.get("repo", repo))
    encoded_path = quote(str(item.get("source_path", rel_path)), safe="")
    url = f"https://modelscope.cn/api/v1/models/{repo}/repo?Revision={revision}&FilePath={encoded_path}"
    print(
        json.dumps(
            {
                "event": "downloading",
                "path": rel_path,
                "revision": revision,
                "expected_bytes": expected_size,
            }
        ),
        flush=True,
    )
    fd, tmp_name = tempfile.mkstemp(prefix=".prepare-", dir=target.parent)
    tmp_path = Path(tmp_name)
    try:
        digest = hashlib.sha256()
        written = 0
        req = Request(url, headers={"User-Agent": "VerdantFlare-ModelPrepare/1.0"})
        with os.fdopen(fd, "wb") as out, urlopen(req, timeout=180) as resp:
            if urlparse(resp.geturl()).scheme != "https":
                raise ValueError("Insecure redirect during model download")
            while True:
                chunk = resp.read(4 * 1024 * 1024)
                if not chunk:
                    break
                written += len(chunk)
                if written > expected_size:
                    raise ValueError(
                        f"Download exceeded expected size {expected_size} for {rel_path}"
                    )
                digest.update(chunk)
                out.write(chunk)
        actual_sha = digest.hexdigest()
        if written != expected_size or actual_sha != expected_sha256:
            raise ValueError(
                f"Downloaded file verification failed for {rel_path}: "
                f"size={written} (expected {expected_size}), "
                f"sha256={actual_sha} (expected {expected_sha256})"
            )
        tmp_path.chmod(0o644)
        try:
            os.link(tmp_path, target)
        except FileExistsError:
            verified = verify_model_file(target, expected_sha256, expected_size, check_sha256=True)
            target.chmod(0o644)
            return {
                "path": rel_path,
                "revision": revision,
                "bytes": verified["bytes"],
                "sha256": verified["sha256"],
            }
        return {
            "path": rel_path,
            "revision": revision,
            "bytes": written,
            "sha256": actual_sha,
        }
    finally:
        tmp_path.unlink(missing_ok=True)


def prepare_all(model_root: Path | None = None) -> list[dict[str, object]]:
    root = model_root or Path(os.getenv("MOSS_IMAGE_MODEL_ROOT", "/models/moss-image"))
    repo = os.getenv("MOSS_IMAGE_MODELSCOPE_REPO", MODELSCOPE_REPO)
    specs = MODEL_MANIFEST + (EDIT_LORA_MANIFEST if edit_enabled() else ())
    results = [prepare_one(root, dict(spec), repo=repo) for spec in specs]
    print(
        json.dumps(
            {
                "event": "moss_image_models_prepared",
                "repository": repo,
                "models": results,
            }
        ),
        flush=True,
    )
    return results


if __name__ == "__main__":
    prepare_all()
