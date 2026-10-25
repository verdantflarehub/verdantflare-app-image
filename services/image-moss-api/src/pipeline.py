"""Inference pipeline managing headless ComfyUI on 127.0.0.1 and GPU execution."""
from __future__ import annotations

import asyncio
import base64
import io
import json
from pathlib import Path
import secrets
import subprocess
import tempfile
import time
from typing import Any, Callable
import urllib.error
import urllib.parse
import urllib.request
import uuid

from src.prepare_models import EDIT_LORA_MANIFEST, MODEL_MANIFEST, verify_model_file
from src.schemas import GenerateRequest, MossApiError, validate_generate_request
from src.settings import COMFYUI_COMMIT, COMFYUI_VERSION, Settings
from src.workflow import _strip_stray_vision_tokens, build_comfy_workflow, build_edit_workflow, serialize_moss_prompt


def _query_gpu_stats() -> dict[str, Any]:
    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=uuid,name,memory.total,memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ]
        line = subprocess.check_output(cmd, text=True, timeout=5).strip().splitlines()[0]
        parts = [p.strip() for p in line.split(",")]
        return {
            "available": True,
            "uuid": parts[0],
            "name": parts[1],
            "memory_total_mib": int(parts[2]),
            "memory_used_mib": int(parts[3]),
            "memory_free_mib": int(parts[4]),
        }
    except Exception as exc:
        return {"available": False, "detail": str(exc)}


def _scale_reference(im: Any, max_edge: int) -> Any:
    """Shrink (never enlarge) so the long edge fits max_edge, both sides multiples of 16, aspect kept."""
    from PIL import Image

    w, h = im.size
    scale = min(1.0, max_edge / max(w, h))
    nw, nh = max(16, int(w * scale) // 16 * 16), max(16, int(h * scale) // 16 * 16)
    if (nw, nh) == (w, h):
        return im
    return im.resize((nw, nh), Image.LANCZOS)


class MossImagePipeline:
    """Manages model verification, local ComfyUI process, and serialized inference."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        comfy_runner: Callable[[dict[str, Any], float], bytes] | None = None,
    ) -> None:
        self.settings = settings or Settings.from_env()
        self._comfy_runner = comfy_runner
        self._proc: subprocess.Popen[bytes] | None = None
        self._extra_paths_file: Path | None = None
        self._models_verified = False
        self._verified_weights: dict[str, str] = {}
        self._edit_available = False
        self._last_error: str | None = None
        self._lock = asyncio.Lock()
        self._active_and_queued = 0

    def verify_models(self) -> dict[str, str]:
        root = Path(self.settings.model_root)
        weights: dict[str, str] = {}
        for spec in MODEL_MANIFEST:
            rel_path = str(spec["path"])
            expected_sha = str(spec["sha256"])
            expected_size = int(spec["size"])
            info = verify_model_file(
                root / rel_path,
                expected_sha,
                expected_size,
                check_sha256=self.settings.verify_sha256_on_boot,
            )
            weights[str(spec["key"])] = str(info["path"])
        self._edit_available = False
        if self.settings.enable_edit:
            for spec in EDIT_LORA_MANIFEST:
                try:
                    info = verify_model_file(
                        root / str(spec["path"]),
                        str(spec["sha256"]),
                        int(spec["size"]),
                        check_sha256=self.settings.verify_sha256_on_boot,
                    )
                except (FileNotFoundError, ValueError) as exc:
                    self._last_error = f"Edit LoRA unavailable: {exc}"
                    break
                weights[str(spec["key"])] = str(info["path"])
            else:
                self._edit_available = True
        self._models_verified = True
        self._verified_weights = weights
        return weights

    def start(self) -> None:
        try:
            self.verify_models()
        except Exception as exc:
            self._models_verified = False
            self._last_error = f"Model verification failed: {exc}"
            if self.settings.auto_start_comfyui:
                raise

        if not self.settings.auto_start_comfyui or self._comfy_runner is not None:
            return

        if self._is_comfyui_reachable():
            return

        fd, tmp_yaml = tempfile.mkstemp(prefix="moss-extra-paths-", suffix=".yaml")
        with open(fd, "w", encoding="utf-8") as f:
            f.write(
                f"moss_image:\n"
                f"  base_path: {self.settings.model_root}\n"
                f"  diffusion_models: diffusion_models\n"
                f"  text_encoders: text_encoders\n"
                f"  vae: vae\n"
                f"  loras: loras\n"
            )
        self._extra_paths_file = Path(tmp_yaml)

        output_dir = Path(tempfile.gettempdir()) / "moss-comfy-output"
        temp_dir = Path(tempfile.gettempdir()) / "moss-comfy-temp"
        user_dir = Path(tempfile.gettempdir()) / "moss-comfy-user"
        output_dir.mkdir(parents=True, exist_ok=True)
        temp_dir.mkdir(parents=True, exist_ok=True)
        user_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            "python",
            "main.py",
            "--listen",
            "127.0.0.1",
            "--port",
            str(self.settings.comfyui_port),
            "--extra-model-paths-config",
            str(self._extra_paths_file),
            "--output-directory",
            str(output_dir),
            "--temp-directory",
            str(temp_dir),
            "--user-directory",
            str(user_dir),
            "--disable-auto-launch",
        ]
        self._proc = subprocess.Popen(
            cmd,
            cwd=self.settings.comfyui_root,
        )

        deadline = time.monotonic() + self.settings.startup_timeout
        while time.monotonic() < deadline:
            if self._proc.poll() is not None:
                raise RuntimeError(
                    f"Headless ComfyUI exited prematurely with code {self._proc.returncode}"
                )
            if self._is_comfyui_reachable():
                return
            time.sleep(0.5)
        raise TimeoutError(
            f"Headless ComfyUI did not become ready within {self.settings.startup_timeout}s"
        )

    def close(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=10)
            except Exception:
                self._proc.kill()
        self._proc = None
        if self._extra_paths_file is not None:
            self._extra_paths_file.unlink(missing_ok=True)
            self._extra_paths_file = None

    def _is_comfyui_reachable(self) -> bool:
        if self._comfy_runner is not None:
            return True
        try:
            with urllib.request.urlopen(
                f"{self.settings.comfyui_base_url}/system_stats", timeout=2.0
            ) as resp:
                return resp.status == 200
        except Exception:
            return False

    def health(self) -> dict[str, Any]:
        if not self._models_verified:
            try:
                self.verify_models()
            except Exception as exc:
                self._last_error = str(exc)
        reachable = self._is_comfyui_reachable()
        gpu_info = _query_gpu_stats() if self._comfy_runner is None else {"available": True, "mock": True}
        healthy = bool(self._models_verified and reachable and gpu_info.get("available", False))
        return {
            "status": "healthy" if healthy else "unhealthy",
            "service": "image-moss-api",
            "model": "moss-image-sft-int8-convrot",
            "comfyui_version": COMFYUI_VERSION,
            "comfyui_commit": COMFYUI_COMMIT,
            "models_verified": self._models_verified,
            "edit_available": self._edit_available,
            "comfyui_reachable": reachable,
            "weights": dict(self._verified_weights),
            "gpu": gpu_info,
        }

    async def generate(self, req: GenerateRequest) -> dict[str, Any]:
        params = validate_generate_request(
            req,
            max_pixels=self.settings.max_pixels,
            max_steps=self.settings.max_steps,
            max_layout_items=self.settings.max_layout_items,
        )
        if not self._models_verified:
            try:
                self.verify_models()
            except Exception as exc:
                raise MossApiError(503, "MODEL_NOT_READY", f"Model weights not ready: {exc}") from exc

        # 1 active + max_queue waiting
        max_in_flight = 1 + self.settings.max_queue
        if self._active_and_queued >= max_in_flight:
            raise MossApiError(
                429,
                "RATE_LIMITED",
                f"Inference queue is full (max_queue={self.settings.max_queue})",
            )

        resolved_seed = (
            int(params["seed"])
            if params["seed"] is not None
            else secrets.randbelow(2_147_483_647)
        )
        ref_images: list[bytes] = params.get("reference_images") or []
        ref_count = len(ref_images)
        edit = bool(params.get("edit"))
        if edit:
            if not self._edit_available:
                raise MossApiError(503, "MODEL_NOT_READY", "Edit LoRA is not available on this instance")
            positive_text, prompt_mode = _strip_stray_vision_tokens(params["prompt"]), "edit"
        else:
            positive_text, prompt_mode = serialize_moss_prompt(
                params["prompt"],
                params["layout"],
                has_reference_image=ref_count > 0,
                reference_image_count=ref_count,
            )

        ref_filenames: list[str] = []
        ref_paths: list[Path] = []
        if ref_count > 0:
            input_dir = Path(self.settings.comfyui_root) / "input"
            if not input_dir.parent.exists():
                input_dir = Path(tempfile.gettempdir()) / "moss-comfy-input"
            input_dir.mkdir(parents=True, exist_ok=True)
            for ref_bytes in ref_images:
                ref_filename = f"moss_ref_{uuid.uuid4().hex}.jpg"
                ref_path = input_dir / ref_filename
                normalized_bytes = ref_bytes
                try:
                    from PIL import Image, ImageOps

                    with Image.open(io.BytesIO(ref_bytes)) as im:
                        im_rgb = ImageOps.exif_transpose(im).convert("RGB")
                        if edit:
                            im_rgb = _scale_reference(im_rgb, params["reference_max_edge"])
                        out_buf = io.BytesIO()
                        im_rgb.save(out_buf, format="JPEG", quality=95)
                        normalized_bytes = out_buf.getvalue()
                except Exception:
                    pass
                ref_path.write_bytes(normalized_bytes)
                ref_filenames.append(ref_filename)
                ref_paths.append(ref_path)

        sampling = dict(
            positive_text=positive_text,
            negative_text=params["negative_prompt"],
            width=params["width"],
            height=params["height"],
            steps=params["steps"],
            cfg_scale=params["cfg_scale"],
            sampler=params["sampler"],
            scheduler=params["scheduler"],
            shift=params["shift"],
            skip_single_layers=params["skip_single_layers"],
            seed=resolved_seed,
            reference_image_filenames=ref_filenames,
        )
        if edit:
            workflow = build_edit_workflow(**sampling, lora_strength=params["edit_lora_strength"])
        else:
            workflow = build_comfy_workflow(**sampling)

        self._active_and_queued += 1
        try:
            async with self._lock:
                t0 = time.perf_counter()
                try:
                    img_bytes = await asyncio.wait_for(
                        asyncio.to_thread(
                            self._execute_workflow,
                            workflow,
                            self.settings.inference_timeout,
                        ),
                        timeout=self.settings.inference_timeout + 5.0,
                    )
                except (TimeoutError, asyncio.TimeoutError) as exc:
                    self._interrupt_comfyui()
                    raise MossApiError(
                        504,
                        "UPSTREAM_TIMEOUT",
                        f"Inference exceeded timeout of {self.settings.inference_timeout}s",
                    ) from exc
                elapsed_ms = int(round((time.perf_counter() - t0) * 1000))
        finally:
            self._active_and_queued -= 1
            for ref_path in ref_paths:
                ref_path.unlink(missing_ok=True)

        return {
            "status": "ok",
            "image_base64": base64.b64encode(img_bytes).decode("ascii"),
            "mime_type": "image/png",
            "width": params["width"],
            "height": params["height"],
            "resolved_seed": resolved_seed,
            "steps": params["steps"],
            "cfg_scale": params["cfg_scale"],
            "sampler": params["sampler"],
            "scheduler": params["scheduler"],
            "shift": params["shift"],
            "prompt_mode": prompt_mode,
            "elapsed_ms": elapsed_ms,
        }

    def _interrupt_comfyui(self) -> None:
        if self._comfy_runner is not None:
            return
        try:
            req = urllib.request.Request(
                f"{self.settings.comfyui_base_url}/interrupt",
                data=b"{}",
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            urllib.request.urlopen(req, timeout=3.0).read()
        except Exception:
            pass

    def _execute_workflow(self, workflow: dict[str, Any], timeout_s: float) -> bytes:
        if self._comfy_runner is not None:
            return self._comfy_runner(workflow, timeout_s)

        payload = json.dumps({"prompt": workflow, "client_id": str(uuid.uuid4())}).encode("utf-8")
        req = urllib.request.Request(
            f"{self.settings.comfyui_base_url}/prompt",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=15.0) as resp:
                submit_data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise MossApiError(503, "MODEL_NOT_READY", f"Local ComfyUI unreachable: {exc}") from exc

        prompt_id = submit_data.get("prompt_id")
        if not prompt_id:
            raise MossApiError(500, "INTERNAL_ERROR", "ComfyUI did not return prompt_id")

        deadline = time.monotonic() + timeout_s
        history_entry: dict[str, Any] | None = None
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(
                    f"{self.settings.comfyui_base_url}/history/{prompt_id}", timeout=10.0
                ) as hresp:
                    hist = json.loads(hresp.read().decode("utf-8"))
            except Exception as exc:
                raise MossApiError(503, "MODEL_NOT_READY", f"Error polling ComfyUI history: {exc}") from exc

            if prompt_id in hist:
                history_entry = hist[prompt_id]
                status = history_entry.get("status", {})
                if status.get("status_str") == "error":
                    raw_msg = json.dumps(status, ensure_ascii=False)
                    if "out of memory" in raw_msg.lower() or "oom" in raw_msg.lower():
                        raise MossApiError(503, "GPU_OUT_OF_MEMORY", "CUDA out of memory during inference")
                    raise MossApiError(500, "INTERNAL_ERROR", f"ComfyUI execution failed: {raw_msg}")
                if status.get("completed") or history_entry.get("outputs"):
                    break
            time.sleep(0.3)
        else:
            self._interrupt_comfyui()
            raise TimeoutError(f"Inference timed out after {timeout_s}s")

        outputs = (history_entry or {}).get("outputs", {})
        node_out = outputs.get("11", {})
        images = node_out.get("images", [])
        if not images:
            raise MossApiError(500, "INTERNAL_ERROR", "ComfyUI workflow completed without output image")
        img_info = images[0]
        query = urllib.parse.urlencode(
            {
                "filename": img_info["filename"],
                "subfolder": img_info.get("subfolder", ""),
                "type": img_info.get("type", "output"),
            }
        )
        with urllib.request.urlopen(
            f"{self.settings.comfyui_base_url}/view?{query}", timeout=15.0
        ) as vresp:
            return vresp.read()
