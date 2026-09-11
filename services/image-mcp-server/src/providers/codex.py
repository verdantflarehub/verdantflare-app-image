from __future__ import annotations

import base64
import json
import mimetypes
import os
import re
import urllib.error
import urllib.request
import uuid
from typing import Any

# 官方规范：Codex 生图模型主力为 gpt-image-2.5-sunburst (画质基准)，亦支持 gpt-image-2.5-flare (极速响应) 与 gpt-image-2 (向下兼容)
DEFAULT_IMAGE_MODEL = "gpt-image-2.5-sunburst"
DEFAULT_SESSION_MODEL = "gpt-5.6-luna"
DEFAULT_SIZE = "2048x1152"
DEFAULT_4K_SIZE = "3840x2160"
DEFAULT_QUALITY = "high"
DEFAULT_BACKGROUND = "auto"

# 官方规范方案 A：注入逐字直传系统指令，彻底防止中介模型二次改写、概括或稀释专业提示词
DEFAULT_VERBATIM_INSTRUCTIONS = (
    "When invoking the image_generation tool, use the user's image prompt verbatim. "
    "Do not rewrite, expand, summarize, embellish, translate, normalize punctuation, "
    "or add or remove visual details or constraints. Preserve the original language, "
    "wording, capitalization, quotes, and punctuation exactly."
)

# 针对 OpenAI Responses / GPT Image 模型缺少负面提示词独立通道的防御性过滤器
# 彻底杜绝提示词末尾携带的"负面提示词/negative prompt"被底层扩散模型误作为正向关键词加权反噬
NEGATIVE_PROMPT_PATTERN = re.compile(
    r"(?:^|\r?\n)\s*(?:负面提示词|负向提示词|反向提示词|negative\s*prompts?)[\s:：].*?(?=(?:\r?\n\s*\r?\n)|\Z)",
    re.IGNORECASE | re.DOTALL,
)


def sanitize_prompt(prompt: str) -> str:
    """过滤提示词中由于历史习惯引入的负面提示词/反向词区块。

    说明：
    GPT Image (DALL-E / Responses API) 等官方模型不存在独立负面词通道。
    在方案 A 逐字透传指令生效后，若将负向词（如'塑料皮肤，过度磨皮，动漫脸'）直接送入底层模型，
    会被扩散注意力机制误解为目标特征正面加权，造成严重的画质涂抹与塑料感反噬。
    故此处对负向词进行自动化防御性剥离。
    """
    if not prompt:
        return ""
    cleaned = NEGATIVE_PROMPT_PATTERN.sub("\n", prompt)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


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


class CodexPolicyRefusalError(CodexProviderError):
    """上游安全审查或内容政策拦截拒止异常，包含模型的原始拒止说明与整改建议。"""
    pass


def extract_sse_model_texts_and_errors(events: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    """从 Responses SSE 事件流中提取中介模型的文本回复（如拒止理由、建议）以及工具执行错误。"""
    texts: list[str] = []
    errors: list[str] = []
    for ev in events:
        ev_type = str(ev.get("type", ""))
        if ev_type in ("response.failed", "error"):
            errors.append(str(ev.get("error") or ev))

        # 兼容包含 item 的各类事件（如 response.output_item.done, output_item.done 等）
        item = ev.get("item")
        if isinstance(item, dict):
            item_type = item.get("type")
            if item_type == "message":
                for c in item.get("content", []):
                    if isinstance(c, dict) and c.get("type") == "output_text":
                        t = c.get("text", "").strip()
                        if t and t not in texts:
                            texts.append(t)
            elif item_type == "image_generation_call":
                status = item.get("status")
                if status and status != "completed":
                    errors.append(f"image_generation_call status={status}")
    return texts, errors


def format_http_error_detail(status_code: int, raw_body: str) -> str:
    """格式化 HTTP 错误响应体为结构化可读文本，提取上游真实错误信息。"""
    try:
        data = json.loads(raw_body)
        err = data.get("error", {})
        if isinstance(err, dict):
            msg = err.get("message") or raw_body
            code = err.get("code")
            err_type = err.get("type")
            prefix = f"HTTP {status_code}"
            if code:
                prefix += f" [{code}]"
            elif err_type:
                prefix += f" [{err_type}]"
            return f"{prefix}: {msg}"
    except Exception:
        pass
    return f"HTTP {status_code}: {raw_body}"


def parse_sse_events(response) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    event_type = ""
    data_lines: list[str] = []

    def flush() -> None:
        nonlocal event_type, data_lines
        if not data_lines:
            event_type = ""
            return
        current_event_type = event_type
        raw = "\n".join(data_lines).strip()
        event_type = ""
        data_lines = []
        if not raw or raw == "[DONE]":
            return
        try:
            item = json.loads(raw)
        except json.JSONDecodeError:
            return
        if current_event_type and "type" not in item:
            item["type"] = current_event_type
        events.append(item)

    for raw_line in response:
        line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
        if not line:
            flush()
            continue
        if line.startswith(":"):
            continue
        if line.startswith("event:"):
            event_type = line[len("event:") :].strip()
            continue
        if line.startswith("data:"):
            data_lines.append(line[len("data:") :].strip())
            continue

    flush()
    return events


def collect_b64(val: Any) -> list[str]:
    res = []
    if isinstance(val, dict):
        for k, v in val.items():
            if k in {"result", "b64_json", "partial_image_b64"} and isinstance(v, str) and v:
                res.append(v)
            else:
                res.extend(collect_b64(v))
    elif isinstance(val, list):
        for item in val:
            res.extend(collect_b64(item))
    return res


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
        self.session_model = (
            os.environ.get("CODEX_SESSION_MODEL")
            or os.environ.get("SUB2API_IMAGES_MAIN_MODEL")
            or DEFAULT_SESSION_MODEL
        )
        self.auto_sanitize_negative_prompts = (
            os.environ.get("CODEX_AUTO_SANITIZE_NEGATIVE_PROMPTS", "true").lower()
            in {"true", "1", "yes"}
        )
        # 默认关闭 prefer_responses，全面落地方案 E：直接请求 sub2api 标准 /images/generations 与 /images/edits
        self.prefer_responses = (
            os.environ.get("CODEX_PREFER_RESPONSES", "false").lower()
            in {"true", "1", "yes"}
        )
        self.actor_auth = (
            os.environ.get("CODEX_ACTOR_AUTH")
            or os.environ.get("OPENAI_ACTOR_AUTH")
            or "local-image-extension"
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

    def _generate_via_responses(
        self,
        prompt: str,
        size: str,
        quality: str,
        background: str,
        model: str,
        source_bytes: list[bytes] | None = None,
        instructions: str | None = None,
    ) -> bytes:
        endpoint = f"{self.base_url}/responses"
        tool: dict[str, Any] = {
            "type": "image_generation",
            "model": model,
            "size": size,
            "quality": quality if quality != "auto" else "high",
            "background": background,
            "output_format": "png",
            "partial_images": 3,
        }

        user_content: list[dict[str, Any]] = []
        if source_bytes:
            for b_data in source_bytes:
                b64_img = base64.b64encode(b_data).decode("utf-8")
                user_content.append({"type": "input_image", "image_url": f"data:image/png;base64,{b64_img}"})

        user_content.append({"type": "input_text", "text": prompt})

        payload = {
            "model": self.session_model,
            "instructions": instructions or DEFAULT_VERBATIM_INSTRUCTIONS,
            "input": [{"type": "message", "role": "user", "content": user_content}],
            "tools": [tool],
            "tool_choice": {"type": "image_generation"},
            "stream": True,
        }

        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream, application/json",
        }
        if self.actor_auth:
            headers["x-openai-actor-authorization"] = self.actor_auth

        req = urllib.request.Request(
            endpoint,
            data=body,
            headers=headers,
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=300) as resp:
            events = parse_sse_events(resp)

        b64_list = collect_b64(events)
        if not b64_list:
            model_texts, sse_errors = extract_sse_model_texts_and_errors(events)
            if model_texts:
                refusal_msg = " ".join(model_texts)
                raise CodexPolicyRefusalError(f"上游安全审查拦截拒止: {refusal_msg}")
            if sse_errors:
                raise CodexProviderError(f"Responses 工具调用失败: {'; '.join(sse_errors)}")
            for ev in events:
                if ev.get("type") in ("response.failed", "error"):
                    raise CodexProviderError(f"Responses 失败: {ev}")
            raise CodexProviderError("Responses SSE 流未返回有效图像数据")
        return base64.b64decode(b64_list[-1])

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
        sanitized_prompt = (
            sanitize_prompt(prompt) if self.auto_sanitize_negative_prompts else prompt
        )

        # 1. 若显式配置 CODEX_PREFER_RESPONSES=true，优先使用 OpenAI Responses API 流式生成
        if self.prefer_responses:
            try:
                return self._generate_via_responses(
                    prompt=sanitized_prompt,
                    size=target_size,
                    quality=target_quality,
                    background=target_bg,
                    model=model,
                    instructions=instructions,
                )
            except CodexPolicyRefusalError:
                # 明确的安全审查拒止，严禁静默吞掉并回退（回退只会撞 502 并掩盖真相）
                raise
            except Exception:
                # 端点不可用或 mock 不支持时回退至 /images/generations
                pass

        # 2. 方案 E 正式链路：调用标准 /images/generations (由 sub2api 网关内部托管 gpt-5.6-luna + 逐字直传系统指令)
        payload: dict[str, Any] = {
            "model": model,  # gpt-image-2.5-sunburst 或 gpt-image-2.5-flare
            "prompt": sanitized_prompt,
            "size": target_size,
            "quality": target_quality,
            "response_format": "b64_json",
        }
        if target_bg != "auto":
            payload["background"] = target_bg

        endpoint = f"{self.base_url}/images/generations"
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
                resp_data = json.loads(resp.read().decode("utf-8"))
                items = resp_data.get("data", [])
                if not items or "b64_json" not in items[0]:
                    raise CodexProviderError("Codex /images/generations 未返回有效图片 Base64 数据")
                return base64.b64decode(items[0]["b64_json"])
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            formatted = format_http_error_detail(exc.code, detail)
            # 4K 回退支持
            if prefer_4k and target_size != size:
                return self.generate(
                    prompt,
                    size=size,
                    prefer_4k=False,
                    quality=quality,
                    background=background,
                    model=model,
                    instructions=instructions,
                )
            raise CodexProviderError(f"Codex 生图请求失败: {formatted}") from exc
        except CodexProviderError:
            raise
        except Exception as exc:
            raise CodexProviderError(f"Codex 生成执行异常: {exc}") from exc

    def edit(
        self,
        prompt: str,
        source_bytes: bytes | list[bytes],
        mask_bytes: bytes | None = None,
        size: str = DEFAULT_SIZE,
        prefer_4k: bool = False,
        quality: str = DEFAULT_QUALITY,
        background: str = DEFAULT_BACKGROUND,
        model: str = DEFAULT_IMAGE_MODEL,
        instructions: str | None = None,
    ) -> bytes:
        self._ensure_auth()
        raw_list = source_bytes if isinstance(source_bytes, list) else [source_bytes]
        target_size = self._resolve_target_size(size, prefer_4k)
        target_quality = normalize_quality(quality)
        target_bg = normalize_background(background)
        sanitized_prompt = (
            sanitize_prompt(prompt) if self.auto_sanitize_negative_prompts else prompt
        )

        # 1. 若显式开启 CODEX_PREFER_RESPONSES=true，无 mask 的参考图引导生成优先走 Responses API
        if self.prefer_responses and not mask_bytes:
            try:
                return self._generate_via_responses(
                    prompt=sanitized_prompt,
                    size=target_size,
                    quality=target_quality,
                    background=target_bg,
                    model=model,
                    source_bytes=raw_list,
                    instructions=instructions,
                )
            except CodexPolicyRefusalError:
                # 明确的安全审查拒止，严禁静默吞掉并回退
                raise
            except Exception:
                pass

        # 2. 方案 E 正式链路：调用标准 /images/edits (由 sub2api 网关内部托管 gpt-5.6-luna + 逐字直传系统指令)
        endpoint = f"{self.base_url}/images/edits"
        boundary = f"----CodexImageBoundary{uuid.uuid4().hex}"
        body = bytearray()

        fields: dict[str, Any] = {
            "model": model,  # gpt-image-2.5-sunburst 或 gpt-image-2.5-flare
            "prompt": sanitized_prompt,
            "size": target_size,
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
            formatted = format_http_error_detail(exc.code, detail)
            # 4K 回退支持
            if prefer_4k and target_size != size:
                return self.edit(
                    prompt=prompt,
                    source_bytes=source_bytes,
                    mask_bytes=mask_bytes,
                    size=size,
                    prefer_4k=False,
                    quality=quality,
                    background=background,
                    model=model,
                    instructions=instructions,
                )
            raise CodexProviderError(f"Codex 图片编辑失败: {formatted}") from exc
        except CodexProviderError:
            raise
        except Exception as exc:
            raise CodexProviderError(f"Codex 编辑执行异常: {exc}") from exc
