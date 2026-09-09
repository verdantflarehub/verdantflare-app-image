#!/usr/bin/env python3
"""
本地独立调试脚本：通过 Google Gemini API 或 Antigravity / Anthropic Messages 协议测试 Gemini 图像生成。

支持双协议驱动：
  1. Anthropic Messages 协议 (优先检测 ANTHROPIC_BASE_URL + ANTHROPIC_AUTH_TOKEN)
     默认模型：gemini-3.1-flash-image
  2. Google Official REST API (检测 GEMINI_API_KEY + 可选 GEMINI_BASE_URL)
     默认模型：gemini-3.1-flash-image

功能覆盖：
  - 文生图 (Text-to-Image)
  - 以图生图 / 参考图控制 (Image-to-Image via --image)
  - 局部重绘 / 局部编辑 (Inpainting via --image + prompt)
  - 4K / 高清出图提示词增强
  - 纯净图约束 (No-text/No-watermark)
  - 零重型 SDK 依赖，纯 Python 标准库 + Pillow
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import mimetypes
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any
from PIL import Image

DEFAULT_MODEL = "gemini-3.1-flash-image"

UNIVERSAL_CONSTRAINTS = (
    "No text, no letters, no words, no typography, no watermarks anywhere in the image. "
    "Pure illustrative and cinematic visual artwork."
)
FIX_TEXT = "The image is purely visual artwork with absolutely NO text, labels, words, or letters."


def load_dotenv() -> None:
    search_dirs = [Path.cwd(), Path.cwd().parent, Path(__file__).resolve().parents[2]]
    for d in search_dirs:
        env_file = d / ".env"
        if env_file.is_file():
            for line in env_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip("'\"")
                    if k and k not in os.environ:
                        os.environ[k] = v
            break


def extract_image_bytes(data: dict[str, Any]) -> bytes:
    """从 Anthropic Messages 或 Google Gemini REST API 响应中提取图像字节。"""
    # 1. Anthropic 格式 (content blocks)
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

    # 2. Google REST 格式 (candidates -> parts -> inlineData)
    candidates = data.get("candidates", [])
    for cand in candidates:
        parts = cand.get("content", {}).get("parts", [])
        for p in parts:
            inline = p.get("inlineData")
            if inline and inline.get("data"):
                return base64.b64decode(inline["data"])

    raise RuntimeError("响应数据中未包含有效图像数据 (未找到 Base64 图像或 inlineData)")


def generate_with_anthropic_api(
    prompt: str,
    base_url: str,
    auth_token: str,
    model: str = DEFAULT_MODEL,
    image_path: Path | None = None,
    timeout: int = 120,
) -> bytes:
    """使用 Anthropic Messages 协议调用 Gemini 图像模型。"""
    base = base_url.rstrip("/")
    if base.endswith("/messages"):
        url = base
    elif base.endswith("/v1"):
        url = f"{base}/messages"
    else:
        url = f"{base}/v1/messages"

    content: list[dict[str, Any]] = []
    if image_path and image_path.is_file():
        mime_type = mimetypes.guess_type(str(image_path))[0] or "image/png"
        b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": mime_type,
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
            "x-api-key": auth_token,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Anthropic API 请求失败 HTTP {exc.code}: {detail}") from exc
    except Exception as exc:
        raise RuntimeError(f"Anthropic API 网络异常: {exc}") from exc

    return extract_image_bytes(data)


def generate_with_google_api(
    prompt: str,
    api_key: str,
    model: str = DEFAULT_MODEL,
    base_url: str | None = None,
    image_path: Path | None = None,
    timeout: int = 120,
) -> bytes:
    """使用纯标准库调用 Google Gemini 原生 REST API 出图。"""
    endpoint = (base_url or "https://generativelanguage.googleapis.com").rstrip("/")
    url = f"{endpoint}/v1beta/models/{model}:generateContent?key={api_key}"

    parts: list[dict[str, Any]] = []
    if image_path and image_path.is_file():
        mime_type = mimetypes.guess_type(str(image_path))[0] or "image/png"
        b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
        parts.append({"inlineData": {"mimeType": mime_type, "data": b64}})

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

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Gemini REST API 请求失败 HTTP {exc.code}: {detail}") from exc
    except Exception as exc:
        raise RuntimeError(f"Gemini 网络连接异常: {exc}") from exc

    return extract_image_bytes(data)


def main() -> int:
    parser = argparse.ArgumentParser(description="本地独立调试脚本：测试 Gemini 图像生成 (支持 Anthropic 中继与官方 REST)")
    parser.add_argument("prompt", help="提示词文本")
    parser.add_argument("--image", help="输入参考图路径 (以图生图/局部重绘)")
    parser.add_argument("--output", "-o", default="test-gemini-output.png", help="输出图片路径")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"模型名称 (默认: {DEFAULT_MODEL})")
    parser.add_argument("--protocol", choices=["auto", "anthropic", "google"], default="auto", help="调用协议模式 (默认: auto 自动优选)")
    parser.add_argument("--retry", type=int, default=2, help="最大重试次数")
    parser.add_argument("--dry-run", action="store_true", help="仅校验配置，不发起实际网络请求")

    args = parser.parse_args()
    load_dotenv()

    anthropic_base_url = os.environ.get("ANTHROPIC_BASE_URL", "").strip()
    anthropic_token = os.environ.get("ANTHROPIC_AUTH_TOKEN", "").strip()
    gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or ""
    gemini_base_url = os.environ.get("GEMINI_BASE_URL", "").strip()

    # 确定调用通道
    channel = "unknown"
    if args.protocol == "anthropic" or (args.protocol == "auto" and anthropic_base_url and anthropic_token):
        channel = "anthropic"
    elif args.protocol == "google" or (args.protocol == "auto" and gemini_key):
        channel = "google"

    print("⚙️ Gemini 图像生成运行配置:")
    print(f"  - 调度模式: {channel.upper()} (protocol: {args.protocol})")
    print(f"  - 模型: {args.model}")
    print(f"  - 参考图: {args.image or '无 (纯文本生成)'}")
    print(f"  - 输出路径: {args.output}")
    if channel == "anthropic":
        print(f"  - Base URL: {anthropic_base_url}")
        print(f"  - Auth Token: {'已就绪 (ANTHROPIC_AUTH_TOKEN)' if anthropic_token else '缺失'}")
    else:
        print(f"  - Base URL: {gemini_base_url or '默认 (Google Official)'}")
        print(f"  - API Key: {'已就绪 (GEMINI_API_KEY)' if gemini_key else '缺失'}")

    if channel == "unknown":
        print("\n❌ 错误: 未检测到有效配置。请在 .env 中配置 ANTHROPIC_BASE_URL + ANTHROPIC_AUTH_TOKEN 或 GEMINI_API_KEY", file=sys.stderr)
        return 1

    if args.dry_run:
        print("✅ Dry-run 校验通过。")
        return 0

    prompt = args.prompt.strip()
    if "no text" not in prompt.lower() and "no watermarks" not in prompt.lower():
        prompt = f"{prompt}\n\n{UNIVERSAL_CONSTRAINTS}"

    start_time = time.time()
    img_ref = Path(args.image) if args.image else None
    if img_ref and not img_ref.is_file():
        print(f"\n❌ 错误: 参考图不存在: {img_ref}", file=sys.stderr)
        return 1

    for attempt in range(1, args.retry + 1):
        curr_prompt = prompt if attempt == 1 else f"{prompt}\n\n{FIX_TEXT}"
        try:
            print(f"🚀 发起 Gemini 生成请求 (Attempt {attempt}/{args.retry} via {channel})...")
            if channel == "anthropic":
                img_bytes = generate_with_anthropic_api(
                    prompt=curr_prompt,
                    base_url=anthropic_base_url,
                    auth_token=anthropic_token,
                    model=args.model,
                    image_path=img_ref,
                )
            else:
                img_bytes = generate_with_google_api(
                    prompt=curr_prompt,
                    api_key=gemini_key,
                    model=args.model,
                    base_url=gemini_base_url or None,
                    image_path=img_ref,
                )

            img = Image.open(io.BytesIO(img_bytes))
            out_p = Path(args.output)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            img.save(str(out_p))

            elapsed = time.time() - start_time
            print("✅ Gemini 生成成功！")
            print(f"  - 耗时: {elapsed:.2f}s")
            print(f"  - 尺寸: {img.size[0]}x{img.size[1]}")
            print(f"  - 格式: {img.format or 'PNG'}")
            print(f"  - 文件大小: {len(img_bytes)/1024:.1f} KB")
            print(f"  - 保存位置: {out_p.resolve()}")
            return 0
        except Exception as exc:
            print(f"❌ [Attempt {attempt}] 失败: {exc}", file=sys.stderr)
            if attempt < args.retry:
                time.sleep(2)

    print(f"❌ 所有 {args.retry} 次尝试均失败", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
