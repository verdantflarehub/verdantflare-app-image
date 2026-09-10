from __future__ import annotations

import base64
import json
import mimetypes
import os
import urllib.error
import urllib.request
import uuid
from typing import Any

# 官方规范：Codex 生图模型主力为 gpt-image-2.5-sunburst (画质基准)，亦支持 gpt-image-2.5-flare (极速响应) 与 gpt-image-2 (向下兼容)
DEFAULT_IMAGE_MODEL = "gpt-image-2.5-sunburst"
DEFAULT_SESSION_MODEL = "gpt-6-astra"
DEFAULT_SIZE = "2048x1152"
DEFAULT_4K_SIZE = "3840x2160"
DEFAULT_QUALITY = "high"
DEFAULT_BACKGROUND = "auto"

# 官方规范方案 A：注入逐字直传系统指令，彻底防止中介模型 (如 gpt-6-astra) 二次改写、概括或稀释专业提示词
DEFAULT_VERBATIM_INSTRUCTIONS = (
    "When invoking the image_generation tool, use the user's image prompt verbatim. "
    "Do not rewrite, expand, summarize, embellish, translate, normalize punctuation, "
    "or add or remove visual details or constraints. Preserve the original language, "
    "wording, capitalization, quotes, and punctuation exactly."
)

# 官方支持质量档位：auto, low, medium, high, xhigh, max
VALID_QUALITIES = {"auto", "low", "medium", "high", "xhigh", "max"}
# 官方支持背景模式：auto, opaque, transparent
VALID_BACKGROUNDS = {"auto", "opaque", "transparent"}


def normalize_quality(quality: str | None) -> str:
    if not quality:
        return "auto"
    q = quality.lower().strip()
    if q in VALID_QUALITIES:
        return q
    # 历史兼容映射
    if q == "hd":
        return "high"
    if q == "standard":
        return "medium"
    return "auto"


def normalize_background(background: str | None) -> str:
    if not background:
        return "auto"
    bg = background.lower().strip()
    if bg in VALID_BACKGROUNDS:
        return bg
    return "auto"


class CodexProviderError(Exception):
    pass


class CodexProvider:
    def __init__(self, base_url: str | None = None, api_key: str | None = None) -> None:
        raw_url = (
            base_url
            or os.environ.get("OPENAI_BASE_URL")
            or os.environ.get("CODEX_RELAY_BASE_URL")
            or "https://api.openai.com/v1"
        )
        norm_url = raw_url.rstrip("/")
        if not norm_url.endswith("/v1"):
            norm_url = f"{norm_url}/v1"
        self.base_url = norm_url
        self.api_key = (
            api_key
            or os.environ.get("OPENAI_API_KEY")
            or os.environ.get("CODEX_RELAY_API_KEY")
            or ""
        )

    def _resolve_target_size(self, size: str, prefer_4k: bool) -> str:
        if not prefer_4k:
            return size
        # 4K 分辨率动态映射，严格遵循 GPT Image 2.5 官方单边上限 <= 3840 像素且为 16 的整数倍
        mapping_4k = {
            "2048x1152": "3840x2160",  # 16:9 4K (8,294,400 像素，官方允许上限)
            "1152x2048": "2160x3840",  # 9:16 4K
            "1024x1024": "2048x2048",  # 1:1 2K 正方形
            "1792x1344": "2880x2160",  # 4:3 4K
            "1344x1792": "2160x2880",  # 3:4 4K
            "1536x1024": "3072x2048",  # 3:2 4K
            "1024x1536": "2048x3072",  # 2:3 4K
        }
        if size in mapping_4k:
            return mapping_4k[size]
        if "x" in size:
            try:
                w, h = map(int, size.split("x", 1))
                if w < h:
                    return "2160x3840"
                if w == h:
                    return "2048x2048"
                return "3840x2160"
            except ValueError:
                pass
        return DEFAULT_4K_SIZE

    def _ensure_auth(self) -> None:
        if not self.api_key:
            raise CodexProviderError("缺少 API 凭据，请配置 OPENAI_API_KEY (或 OPENAI_BASE_URL)")

    def _parse_sse(self, response: urllib.response.addinfourl) -> list[str]:
        data_lines: list[str] = []
        final_results: list[str] = []
        partial_results: dict[int, str] = {}

        def flush_item() -> None:
            nonlocal data_lines
            if not data_lines:
                return
            raw = "\n".join(data_lines).strip()
            data_lines = []
            if not raw or raw == "[DONE]":
                return
            try:
                item = json.loads(raw)
            except json.JSONDecodeError:
                return

            event_type = item.get("type", "")
            if event_type == "response.image_generation_call.partial_image":
                b64 = item.get("partial_image_b64")
                idx = item.get("partial_image_index", len(partial_results))
                if isinstance(b64, str) and b64:
                    try:
                        partial_results[int(idx)] = b64
                    except (TypeError, ValueError):
                        pass
            elif event_type in {"response.completed", "response.output_item.done", "image_generation_call"}:
                self._collect_b64(item, final_results)

        for raw_line in response:
            line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
            if not line:
                flush_item()
                continue
            if line.startswith(":"):
                continue
            if line.startswith("data:"):
                data_lines.append(line[len("data:") :].strip())

        flush_item()
        if final_results:
            return final_results
        if partial_results:
            return [partial_results[idx] for idx in sorted(partial_results)]
        raise CodexProviderError("Responses SSE 流中未收到有效图片 Base64 数据")

    def _collect_b64(self, val: Any, res: list[str]) -> None:
        if isinstance(val, dict):
            for k, v in val.items():
                if k in {"result", "b64_json", "partial_image_b64"} and isinstance(v, str) and v:
                    res.append(v)
                else:
                    self._collect_b64(v, res)
        elif isinstance(val, list):
            for item in val:
                self._collect_b64(item, res)

    def generate(
        self,
        prompt: str,
        size: str = DEFAULT_SIZE,
        prefer_4k: bool = False,
        quality: str = DEFAULT_QUALITY,
        background: str = DEFAULT_BACKGROUND,
        model: str = DEFAULT_IMAGE_MODEL,
        instructions: str | None = None,
    ) -> bytes:
        self._ensure_auth()
        target_size = self._resolve_target_size(size, prefer_4k)
        target_quality = normalize_quality(quality)
        target_bg = normalize_background(background)
        target_instructions = (
            instructions
            or os.environ.get("CODEX_VERBATIM_INSTRUCTIONS")
            or DEFAULT_VERBATIM_INSTRUCTIONS
        )

        tool: dict[str, Any] = {
            "type": "image_generation",
            "model": model,  # gpt-image-2.5-sunburst 或 gpt-image-2.5-flare
            "size": target_size,
            "quality": target_quality,
            "output_format": "png",
            "partial_images": 3,
        }
        if target_bg != "auto":
            tool["background"] = target_bg

        payload = {
            "model": DEFAULT_SESSION_MODEL,
            "instructions": target_instructions,
            "input": prompt,
            "tools": [tool],
            "tool_choice": {"type": "image_generation"},
            "stream": True,
        }

        endpoint = f"{self.base_url}/responses"
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            endpoint,
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                images = self._parse_sse(resp)
                return base64.b64decode(images[-1])
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            # 4K 回退支持
            if prefer_4k and target_size != size:
                return self.generate(
                    prompt,
                    size=size,
                    prefer_4k=False,
                    quality=quality,
                    background=background,
                    model=model,
                )
            raise CodexProviderError(f"Codex 生图请求失败 HTTP {exc.code}: {detail}") from exc
        except Exception as exc:
            raise CodexProviderError(f"Codex 生成执行异常: {exc}") from exc

    def edit(
        self,
        prompt: str,
        source_bytes: bytes | list[bytes],
        mask_bytes: bytes | None = None,
        size: str = DEFAULT_SIZE,
        quality: str = DEFAULT_QUALITY,
        background: str = DEFAULT_BACKGROUND,
        model: str = DEFAULT_IMAGE_MODEL,
    ) -> bytes:
        self._ensure_auth()
        endpoint = f"{self.base_url}/images/edits"
        boundary = f"----CodexImageBoundary{uuid.uuid4().hex}"
        body = bytearray()

        target_quality = normalize_quality(quality)
        target_bg = normalize_background(background)

        fields: dict[str, Any] = {
            "model": model,  # gpt-image-2.5-sunburst 或 gpt-image-2.5-flare
            "prompt": prompt,
            "size": size,
            "output_format": "png",
            "n": 1,
        }
        if target_quality != "auto":
            fields["quality"] = target_quality
        if target_bg != "auto":
            fields["background"] = target_bg

        for k, v in fields.items():
            body.extend(f"--{boundary}\r\n".encode("utf-8"))
            body.extend(f'Content-Disposition: form-data; name="{k}"\r\n\r\n'.encode("utf-8"))
            body.extend(str(v).encode("utf-8"))
            body.extend(b"\r\n")

        # 支持多参考图（单个 bytes 或 list[bytes]）
        raw_list = source_bytes if isinstance(source_bytes, list) else [source_bytes]
        files = []
        for idx, b_data in enumerate(raw_list):
            field_name = "image" if idx == 0 else f"image_{idx}"
            filename = f"source_{idx}.png" if idx > 0 else "source.png"
            files.append((field_name, filename, b_data, "image/png"))

        if mask_bytes:
            files.append(("mask", "mask.png", mask_bytes, "image/png"))

        for name, filename, content, ctype in files:
            body.extend(f"--{boundary}\r\n".encode("utf-8"))
            body.extend(f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'.encode("utf-8"))
            body.extend(f"Content-Type: {ctype}\r\n\r\n".encode("utf-8"))
            body.extend(content)
            body.extend(b"\r\n")

        body.extend(f"--{boundary}--\r\n".encode("utf-8"))

        req = urllib.request.Request(
            endpoint,
            data=bytes(body),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "Accept": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                items = data.get("data", [])
                if not items or "b64_json" not in items[0]:
                    raise CodexProviderError("Edits 未返回图片数据")
                return base64.b64decode(items[0]["b64_json"])
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise CodexProviderError(f"Codex 图片编辑失败 HTTP {exc.code}: {detail}") from exc
