from __future__ import annotations

import base64
import json
import os
import re
import time
import urllib.error
import urllib.request
from typing import Any

UNIVERSAL_CONSTRAINTS = (
    "No text, no letters, no words, no typography, no watermarks anywhere in the image. "
    "Pure illustrative and cinematic visual artwork."
)
FIX_TEXT = "The image is purely visual artwork with absolutely NO text, labels, words, or letters."
DEFAULT_GEMINI_MODEL = "gemini-3.1-flash-image"


class GeminiProviderError(Exception):
    pass


def _detect_mime_type(data: bytes) -> str:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith(b"RIFF") and b"WEBP" in data[:12]:
        return "image/webp"
    return "image/png"


class GeminiProvider:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        anthropic_base_url: str | None = None,
        anthropic_token: str | None = None,
    ) -> None:
        self.api_key = (
            api_key
            or os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
            or ""
        ).strip()
        self.base_url = (
            base_url
            or os.environ.get("GEMINI_BASE_URL")
            or "https://generativelanguage.googleapis.com"
        ).rstrip("/")

        self.anthropic_base_url = (
            anthropic_base_url or os.environ.get("ANTHROPIC_BASE_URL") or ""
        ).rstrip("/")
        self.anthropic_token = (
            anthropic_token or os.environ.get("ANTHROPIC_AUTH_TOKEN") or ""
        ).strip()

    def _ensure_auth(self) -> str:
        """返回鉴权通道: 优先使用 'google' 原生通道，回退到 'anthropic'"""
        if self.api_key:
            return "google"
        if self.anthropic_base_url and self.anthropic_token:
            return "anthropic"
        raise GeminiProviderError(
            "缺少有效的 Gemini API 凭据，请配置 GEMINI_API_KEY (+ GEMINI_BASE_URL) 或 ANTHROPIC_BASE_URL + ANTHROPIC_AUTH_TOKEN"
        )

    def _extract_image_bytes(self, data: dict[str, Any]) -> bytes:
        # 1. Anthropic 格式
        content = data.get("content", [])
        if isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type")
                if btype == "image":
                    src = block.get("source", {})
                    if src.get("type") == "base64" and src.get("data"):
                        return base64.b64decode(src["data"])
                elif btype == "text":
                    text = block.get("text", "")
                    m = re.search(r"data:image\/(?:png|jpeg|jpg|webp);base64,([A-Za-z0-9+/=\s]+)", text)
                    if m:
                        b64_clean = re.sub(r"\s+", "", m.group(1))
                        return base64.b64decode(b64_clean)

        # 2. Google REST 格式
        candidates = data.get("candidates", [])
        for cand in candidates:
            finish_reason = cand.get("finishReason")
            if finish_reason in ("IMAGE_SAFETY", "SAFETY"):
                raise GeminiProviderError(f"Gemini 安全审核拦截 (finishReason: {finish_reason})：内容触发了云端安全策略")
            if finish_reason == "IMAGE_OTHER":
                raise GeminiProviderError(f"Gemini 后置安全审查拦截 (finishReason: IMAGE_OTHER)：生成图像被云端安全策略阻断")

            parts = cand.get("content", {}).get("parts", [])
            for p in parts:
                inline = p.get("inlineData")
                if inline and inline.get("data"):
                    return base64.b64decode(inline["data"])

        raise GeminiProviderError("Gemini 响应未返回有效图像数据")

    def _generate_anthropic(
        self,
        prompt: str,
        model: str,
        source_images: list[bytes] | None = None,
        source_image_bytes: bytes | None = None,
        mask_bytes: bytes | None = None,
        timeout: int = 120,
    ) -> bytes:
        base = self.anthropic_base_url
        if base.endswith("/messages"):
            url = base
        elif base.endswith("/v1"):
            url = f"{base}/messages"
        else:
            url = f"{base}/v1/messages"

        content: list[dict[str, Any]] = []

        images: list[bytes] = []
        if source_images:
            images.extend(source_images)
        elif source_image_bytes:
            images.append(source_image_bytes)

        for img in images:
            b64 = base64.b64encode(img).decode("utf-8")
            mime = _detect_mime_type(img)
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": mime,
                    "data": b64,
                },
            })

        if mask_bytes:
            b64_mask = base64.b64encode(mask_bytes).decode("utf-8")
            mime_mask = _detect_mime_type(mask_bytes)
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": mime_mask,
                    "data": b64_mask,
                },
            })

        content.append({"type": "text", "text": prompt})

        payload = {
            "model": model,
            "max_tokens": 4096,
            "messages": [
                {
                    "role": "user",
                    "content": content if len(content) > 1 else prompt,
                }
            ],
        }

        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            headers={
                "x-api-key": self.anthropic_token,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return self._extract_image_bytes(data)

    def _generate_google(
        self,
        prompt: str,
        model: str,
        source_images: list[bytes] | None = None,
        source_image_bytes: bytes | None = None,
        mask_bytes: bytes | None = None,
        timeout: int = 120,
    ) -> bytes:
        base = self.base_url.rstrip("/")
        if base.endswith("/v1beta"):
            base = base[:-7]
        url = f"{base}/v1beta/models/{model}:generateContent?key={self.api_key}"

        parts: list[dict[str, Any]] = []

        images: list[bytes] = []
        if source_images:
            images.extend(source_images)
        elif source_image_bytes:
            images.append(source_image_bytes)

        for img in images:
            b64 = base64.b64encode(img).decode("utf-8")
            mime = _detect_mime_type(img)
            parts.append({"inlineData": {"mimeType": mime, "data": b64}})

        if mask_bytes:
            b64_mask = base64.b64encode(mask_bytes).decode("utf-8")
            mime_mask = _detect_mime_type(mask_bytes)
            parts.append({"inlineData": {"mimeType": mime_mask, "data": b64_mask}})

        parts.append({"text": prompt})
        payload = {
            "contents": [{"parts": parts}],
            "generationConfig": {
                "responseModalities": ["IMAGE", "TEXT"],
            },
            "safetySettings": [
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            ],
        }

        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key,
        }
        if self.api_key.startswith("sk-"):
            headers["Authorization"] = f"Bearer {self.api_key}"

        req = urllib.request.Request(
            url,
            data=body,
            headers=headers,
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return self._extract_image_bytes(data)

    def generate(
        self,
        prompt: str,
        model: str = DEFAULT_GEMINI_MODEL,
        source_images: list[bytes] | None = None,
        source_image_bytes: bytes | None = None,
        mask_bytes: bytes | None = None,
        retries: int = 2,
    ) -> bytes:
        channel = self._ensure_auth()

        clean_prompt = prompt.strip()
        if "no text" not in clean_prompt.lower() and "no watermarks" not in clean_prompt.lower():
            clean_prompt = f"{clean_prompt}\n\n{UNIVERSAL_CONSTRAINTS}"

        for attempt in range(1, retries + 1):
            curr_prompt = clean_prompt if attempt == 1 else f"{clean_prompt}\n\n{FIX_TEXT}"

            try:
                if channel == "google":
                    return self._generate_google(
                        prompt=curr_prompt,
                        model=model,
                        source_images=source_images,
                        source_image_bytes=source_image_bytes,
                        mask_bytes=mask_bytes,
                    )
                else:
                    return self._generate_anthropic(
                        prompt=curr_prompt,
                        model=model,
                        source_images=source_images,
                        source_image_bytes=source_image_bytes,
                        mask_bytes=mask_bytes,
                    )
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                if attempt == retries:
                    raise GeminiProviderError(f"Gemini API 失败 HTTP {exc.code}: {detail}") from exc
            except Exception as exc:
                if attempt == retries:
                    raise GeminiProviderError(f"Gemini 网络连接失败: {exc}") from exc

            time.sleep(2)

        raise GeminiProviderError("Gemini 响应未返回有效图像数据")

