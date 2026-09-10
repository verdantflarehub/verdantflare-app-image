#!/usr/bin/env python3
"""
本地独立调试脚本：通过 OpenAI Responses / Images 协议或 Codex Relay 测试图像生成与编辑。

优先读取当前目录或父目录的 .env 文件；若未配置则回退读取 ~/.codex/ 目录。
支持模式：
  1. 文生图 (默认 responses 模式，带 image_generation 工具桥)
  2. 图生图 / 图像编辑 (--image 进入 edits 模式)
  3. 遮罩局部重绘 (--image + --mask)
  4. 4K 自动回退 (--prefer-4k)
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ModuleNotFoundError:
        tomllib = None  # type: ignore[assignment]


DEFAULT_MODEL = "gpt-image-2.5-sunburst"
DEFAULT_RESPONSES_FALLBACK_MODEL = "gpt-5.5"
DEFAULT_SIZE = "2048x1152"
DEFAULT_4K_SIZE = "4096x2304"
DEFAULT_QUALITY = "hd"
DEFAULT_BACKGROUND = "auto"
DEFAULT_FORMAT = "png"


def load_dotenv() -> None:
    """搜索并加载 .env 文件，但不覆盖已有的真实环境变量。"""
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


def resolve_auth_and_base_url() -> tuple[str, str]:
    """解析 base_url 和 api_key。

    优先级：
    1. 环境变量 OPENAI_BASE_URL + OPENAI_API_KEY
    2. 环境变量 CODEX_RELAY_BASE_URL + CODEX_RELAY_API_KEY
    3. ~/.codex/config.toml + auth.json
    """
    load_dotenv()

    base_url = os.environ.get("OPENAI_BASE_URL") or os.environ.get("CODEX_RELAY_BASE_URL")
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("CODEX_RELAY_API_KEY")

    if base_url and api_key:
        norm_url = base_url.rstrip("/")
        if not norm_url.endswith("/v1"):
            norm_url = f"{norm_url}/v1"
        return norm_url, api_key

    # 回退到 ~/.codex
    codex_home = Path(os.environ.get("CODEX_HOME", "~/.codex")).expanduser()
    config_path = codex_home / "config.toml"
    auth_path = codex_home / "auth.json"

    if config_path.is_file() and auth_path.is_file():
        if tomllib is None:
            raise RuntimeError("Python 3.11+ is required for tomllib fallback.")
        with config_path.open("rb") as fh:
            cfg = tomllib.load(fh)
        provider = cfg.get("model_provider", "beagle")
        providers = cfg.get("model_providers", {})
        prov_cfg = providers.get(provider, {})
        base_url = prov_cfg.get("base_url", "https://api.openai.com/v1")

        auth_data = json.loads(auth_path.read_text(encoding="utf-8"))
        api_key = auth_data.get("OPENAI_API_KEY")

        if base_url and api_key:
            norm_url = base_url.rstrip("/")
            if not norm_url.endswith("/v1"):
                norm_url = f"{norm_url}/v1"
            return norm_url, api_key

    raise RuntimeError("无法找到 API Key 和 Base URL。请在 .env 或环境变量中设置 OPENAI_BASE_URL 和 OPENAI_API_KEY。")


def build_responses_payload(args: argparse.Namespace, model: str) -> dict[str, Any]:
    tool: dict[str, Any] = {
        "type": "image_generation",
        "size": args.size,
        "quality": args.quality,
        "background": args.background,
        "output_format": args.format,
        "partial_images": 3,
    }
    if args.image:
        image_path = Path(args.image)
        if not image_path.is_file():
            raise RuntimeError(f"输入底图不存在: {image_path}")
        mime_type = mimetypes.guess_type(str(image_path))[0] or "image/png"
        image_b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
        data_url = f"data:{mime_type};base64,{image_b64}"
        user_content = [
            {"type": "input_image", "image_url": data_url},
            {"type": "input_text", "text": args.prompt},
        ]
        user_input: Any = [{"type": "message", "role": "user", "content": user_content}]
    else:
        user_input = args.prompt

    return {
        "model": model,
        "input": user_input,
        "tools": [tool],
        "tool_choice": {"type": "image_generation"},
        "stream": True,
    }


def parse_sse_data_events(response: urllib.response.addinfourl) -> list[dict[str, Any]]:
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


def collect_b64_values(value: Any) -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            if key in {"result", "b64_json", "partial_image_b64"} and isinstance(child, str) and child:
                found.append(child)
            else:
                found.extend(collect_b64_values(child))
    elif isinstance(value, list):
        for child in value:
            found.extend(collect_b64_values(child))
    return found


def request_responses_images(endpoint: str, api_key: str, payload: dict[str, Any], timeout: int = 300) -> list[str]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            events = parse_sse_data_events(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"API 请求失败 HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"网络连接失败: {exc.reason}") from exc

    final_results: list[str] = []
    partial_results: dict[int, str] = {}

    for event in events:
        event_type = event.get("type")
        if event_type == "response.image_generation_call.partial_image":
            b64 = event.get("partial_image_b64")
            index = event.get("partial_image_index", len(partial_results))
            if isinstance(b64, str) and b64:
                try:
                    partial_results[int(index)] = b64
                except (TypeError, ValueError):
                    pass
            continue

        if event_type in {"response.completed", "response.output_item.done", "image_generation_call"}:
            final_results.extend(collect_b64_values(event))

    if final_results:
        return final_results
    if partial_results:
        return [partial_results[index] for index in sorted(partial_results)]
    raise RuntimeError("响应流中未找到有效图片数据。")


def request_images_edits(endpoint: str, api_key: str, args: argparse.Namespace, model: str) -> bytes:
    if not args.image:
        raise RuntimeError("图片编辑模式必须指定 --image")
    image_path = Path(args.image)
    if not image_path.is_file():
        raise RuntimeError(f"底图文件未找到: {image_path}")

    boundary = f"----ImagegenBoundary{uuid.uuid4().hex}"
    body = bytearray()

    fields = {
        "model": model,
        "prompt": args.prompt,
        "n": 1,
        "size": args.size,
    }
    for name, value in fields.items():
        if value is not None:
            body.extend(f"--{boundary}\r\n".encode("utf-8"))
            body.extend(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8"))
            body.extend(str(value).encode("utf-8"))
            body.extend(b"\r\n")

    files = [("image", image_path.name, image_path.read_bytes(), "image/png")]
    if args.mask:
        mask_path = Path(args.mask)
        if not mask_path.is_file():
            raise RuntimeError(f"遮罩文件未找到: {mask_path}")
        files.append(("mask", mask_path.name, mask_path.read_bytes(), "image/png"))

    for name, filename, content, content_type in files:
        body.extend(f"--{boundary}\r\n".encode("utf-8"))
        body.extend(f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'.encode("utf-8"))
        body.extend(f"Content-Type: {content_type}\r\n\r\n".encode("utf-8"))
        body.extend(content)
        body.extend(b"\r\n")

    body.extend(f"--{boundary}--\r\n".encode("utf-8"))

    request = urllib.request.Request(
        endpoint,
        data=bytes(body),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Accept": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            raw = response.read().decode("utf-8")
            data = json.loads(raw)
            items = data.get("data", [])
            if not items:
                raise RuntimeError("Edits API 未返回任何图片")
            b64_json = items[0].get("b64_json")
            if not b64_json:
                raise RuntimeError("Edits API 未返回 b64_json")
            return base64.b64decode(b64_json)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Edits 请求失败 HTTP {exc.code}: {detail}") from exc


def main() -> int:
    parser = argparse.ArgumentParser(description="本地独立调试脚本：测试 Codex/OpenAI 图像生成")
    parser.add_argument("prompt", help="提示词文本")
    parser.add_argument("--image", help="底图路径 (传入则自动进入 edits 模式)")
    parser.add_argument("--mask", help="遮罩图路径 (仅用于 edits 模式)")
    parser.add_argument("--mode", choices=("responses", "edits"), help="强制指定模式")
    parser.add_argument("--output", "-o", default="test-codex-output.png", help="输出图片文件路径")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"模型名 (默认: {DEFAULT_MODEL})")
    parser.add_argument("--size", default=DEFAULT_SIZE, help=f"尺寸 (默认: {DEFAULT_SIZE})")
    parser.add_argument("--prefer-4k", action="store_true", help="尝试 4K (4096x2304)，失败则回退")
    parser.add_argument("--quality", default=DEFAULT_QUALITY, help="质量")
    parser.add_argument("--background", default=DEFAULT_BACKGROUND, help="背景偏好")
    parser.add_argument("--format", default=DEFAULT_FORMAT, help="格式")
    parser.add_argument("--dry-run", action="store_true", help="仅打印解析配置，不发送实际网络请求")

    args = parser.parse_args()

    mode = args.mode or ("edits" if args.image else "responses")

    try:
        base_url, api_key = resolve_auth_and_base_url()
    except Exception as exc:
        print(f"❌ 配置加载错误: {exc}", file=sys.stderr)
        return 1

    requested_size = args.size
    if args.prefer_4k and args.size == DEFAULT_SIZE:
        args.size = DEFAULT_4K_SIZE

    endpoint = f"{base_url}/images/edits" if mode == "edits" else f"{base_url}/responses"

    print(f"⚙️ 运行配置:")
    print(f"  - 模式: {mode}")
    print(f"  - Endpoint: {endpoint}")
    print(f"  - 模型: {args.model}")
    print(f"  - 尺寸: {args.size}")
    print(f"  - 输出路径: {args.output}")
    print(f"  - API 凭据: 已就绪 (长度: {len(api_key)} 字符)")

    if args.dry_run:
        print("✅ Dry-run 完成，配置有效。")
        return 0

    print("🚀 开始请求图像生成...")
    start_time = time.time()

    try:
        if mode == "edits":
            image_bytes = request_images_edits(endpoint, api_key, args, args.model)
        else:
            payload = build_responses_payload(args, args.model)
            try:
                images_b64 = request_responses_images(endpoint, api_key, payload)
            except RuntimeError as exc:
                if args.prefer_4k and args.size == DEFAULT_4K_SIZE:
                    print(f"⚠️ 4K 请求失败，正在自动回退至 {requested_size}: {exc}")
                    args.size = requested_size
                    payload = build_responses_payload(args, args.model)
                    images_b64 = request_responses_images(endpoint, api_key, payload)
                else:
                    raise
            image_bytes = base64.b64decode(images_b64[-1])

        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(image_bytes)

        elapsed = time.time() - start_time
        file_size_kb = len(image_bytes) / 1024
        print(f"✅ 生成成功！")
        print(f"  - 耗时: {elapsed:.2f}s")
        print(f"  - 文件大小: {file_size_kb:.1f} KB")
        print(f"  - 保存位置: {out_path.resolve()}")
        return 0

    except Exception as exc:
        print(f"❌ 生成失败: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
