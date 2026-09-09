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


class GeminiProvider:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        anthropic_base_url: str | None = None,
        anthropic_token: str | None = None,
    ) -> None:
        self.anthropic_base_url = (
            anthropic_base_url or os.environ.get("ANTHROPIC_BASE_URL") or ""
        ).rstrip("/")
        self.anthropic_token = (
            anthropic_token or os.environ.get("ANTHROPIC_AUTH_TOKEN") or ""
        ).strip()

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

    def _ensure_auth(self) -> str:
        """返回鉴权通道: 'anthropic' 或 'google'"""
        if self.anthropic_base_url and self.anthropic_token:
            return "anthropic"
        if self.api_key:
            return "google"
        raise GeminiProviderError(
            "缺少有效的 Gemini API 凭据，请配置 ANTHROPIC_BASE_URL + ANTHROPIC_AUTH_TOKEN 或 GEMINI_API_KEY"
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
        source_image_bytes: bytes | None,
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
        if source_image_bytes:
            b64 = base64.b64encode(source_image_bytes).decode("utf-8")
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": b64,
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
        source_image_bytes: bytes | None,
        timeout: int = 120,
    ) -> bytes:
        url = f"{self.base_url}/v1beta/models/{model}:generateContent?key={self.api_key}"
        parts: list[dict[str, Any]] = []
        if source_image_bytes:
            b64 = base64.b64encode(source_image_bytes).decode("utf-8")
            parts.append({"inlineData": {"mimeType": "image/png", "data": b64}})

        parts.append({"text": prompt})
        payload = {
            "contents": [{"parts": parts}],
            "generationConfig": {
                "responseModalities": ["IMAGE", "TEXT"],
            },
        }

        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return self._extract_image_bytes(data)

    def generate(
        self,
        prompt: str,
        model: str = DEFAULT_GEMINI_MODEL,
        source_image_bytes: bytes | None = None,
        retries: int = 2,
    ) -> bytes:
        channel = self._ensure_auth()

        clean_prompt = prompt.strip()
        if "no text" not in clean_prompt.lower() and "no watermarks" not in clean_prompt.lower():
            clean_prompt = f"{clean_prompt}\n\n{UNIVERSAL_CONSTRAINTS}"

        for attempt in range(1, retries + 1):
            curr_prompt = clean_prompt if attempt == 1 else f"{clean_prompt}\n\n{FIX_TEXT}"

            try:
                if channel == "anthropic":
                    return self._generate_anthropic(
                        prompt=curr_prompt,
                        model=model,
                        source_image_bytes=source_image_bytes,
                    )
                else:
                    return self._generate_google(
                        prompt=curr_prompt,
                        model=model,
                        source_image_bytes=source_image_bytes,
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

