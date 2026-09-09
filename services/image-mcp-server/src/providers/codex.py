from __future__ import annotations

import base64
import json
import mimetypes
import os
import urllib.error
import urllib.request
import uuid
from typing import Any

# 用户明确规范：Codex 生图模型权威指定为 gpt-image-2
DEFAULT_IMAGE_MODEL = "gpt-image-2"
DEFAULT_SESSION_MODEL = "gpt-6-astra"
DEFAULT_SIZE = "2048x1152"
DEFAULT_4K_SIZE = "4096x2304"


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
        quality: str = "auto",
        model: str = DEFAULT_IMAGE_MODEL,
    ) -> bytes:
        self._ensure_auth()
        target_size = DEFAULT_4K_SIZE if prefer_4k else size

        tool = {
            "type": "image_generation",
            "model": model,  # 指定 gpt-image-2
            "size": target_size,
            "quality": quality,
            "output_format": "png",
            "partial_images": 3,
        }
        payload = {
            "model": DEFAULT_SESSION_MODEL,
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
            if prefer_4k and target_size == DEFAULT_4K_SIZE:
                return self.generate(prompt, size=size, prefer_4k=False, quality=quality, model=model)
            raise CodexProviderError(f"Codex 生图请求失败 HTTP {exc.code}: {detail}") from exc
        except Exception as exc:
            raise CodexProviderError(f"Codex 生成执行异常: {exc}") from exc

    def edit(
        self,
        prompt: str,
        source_bytes: bytes,
        mask_bytes: bytes | None = None,
        size: str = DEFAULT_SIZE,
        model: str = DEFAULT_IMAGE_MODEL,
    ) -> bytes:
        self._ensure_auth()
        endpoint = f"{self.base_url}/images/edits"
        boundary = f"----CodexImageBoundary{uuid.uuid4().hex}"
        body = bytearray()

        fields = {
            "model": model,  # gpt-image-2
            "prompt": prompt,
            "size": size,
            "n": 1,
        }
        for k, v in fields.items():
            body.extend(f"--{boundary}\r\n".encode("utf-8"))
            body.extend(f'Content-Disposition: form-data; name="{k}"\r\n\r\n'.encode("utf-8"))
            body.extend(str(v).encode("utf-8"))
            body.extend(b"\r\n")

        files = [("image", "source.png", source_bytes, "image/png")]
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
