import json
import unittest
from unittest.mock import MagicMock, patch
from src.providers.codex import (
    CodexProvider,
    CodexProviderError,
    CodexPolicyRefusalError,
    format_http_error_detail,
    DEFAULT_VERBATIM_INSTRUCTIONS,
    DEFAULT_SESSION_MODEL,
    DEFAULT_IMAGE_MODEL,
    sanitize_prompt,
)


class TestCodexProvider(unittest.TestCase):
    def test_ensure_auth_missing(self):
        with patch.dict("os.environ", {}, clear=True):
            provider = CodexProvider(base_url="https://api.openai.com", api_key="")
            with self.assertRaises(CodexProviderError):
                provider._ensure_auth()

    def test_default_verbatim_instructions_constant(self):
        self.assertIn("use the user's image prompt verbatim", DEFAULT_VERBATIM_INSTRUCTIONS)
        self.assertIn("Do not rewrite, expand, summarize", DEFAULT_VERBATIM_INSTRUCTIONS)

    @patch("urllib.request.urlopen")
    def test_generate_payload_format(self, mock_urlopen):
        captured_payload = {}
        captured_url = ""

        def fake_urlopen(req, timeout=300):
            nonlocal captured_payload, captured_url
            captured_url = req.full_url
            captured_payload = json.loads(req.data.decode("utf-8"))
            mock_resp = MagicMock()
            import base64
            fake_b64 = base64.b64encode(b"fake-codex-image").decode("utf-8")
            resp_body = json.dumps({"data": [{"b64_json": fake_b64}]}).encode("utf-8")
            mock_resp.read.return_value = resp_body
            mock_resp.__enter__.return_value = mock_resp
            return mock_resp

        mock_urlopen.side_effect = fake_urlopen

        provider = CodexProvider(base_url="https://api.openai.com/v1", api_key="sk-test-12345")
        result = provider.generate(
            prompt="A photorealistic dancer in high resolution",
            size="1152x2048",
            quality="high",
        )

        self.assertEqual(result, b"fake-codex-image")
        self.assertEqual(captured_url, "https://api.openai.com/v1/images/generations")
        self.assertEqual(captured_payload.get("model"), DEFAULT_IMAGE_MODEL)
        self.assertEqual(captured_payload.get("prompt"), "A photorealistic dancer in high resolution")
        self.assertEqual(captured_payload.get("size"), "1152x2048")
        self.assertEqual(captured_payload.get("quality"), "high")
        self.assertEqual(captured_payload.get("response_format"), "b64_json")

    def test_sanitize_prompt_removes_chinese_negative_block(self):
        raw = "9:16 真实写真，舞蹈室自拍。\n\n负面提示词： 未成年外观，儿童化面容，塑料皮肤，过度磨皮，动漫脸，CG脸。"
        cleaned = sanitize_prompt(raw)
        self.assertEqual(cleaned, "9:16 真实写真，舞蹈室自拍。")

    def test_sanitize_prompt_removes_english_negative_block(self):
        raw = "A realistic studio portrait.\n\nNegative prompt: blurry, bad anatomy, cartoon, watermark"
        cleaned = sanitize_prompt(raw)
        self.assertEqual(cleaned, "A realistic studio portrait.")

    @patch("urllib.request.urlopen")
    def test_generate_auto_sanitizes_negative_prompt_token_pollution(self, mock_urlopen):
        captured_payload = {}

        def fake_urlopen(req, timeout=300):
            nonlocal captured_payload
            captured_payload = json.loads(req.data.decode("utf-8"))
            mock_resp = MagicMock()
            import base64
            fake_b64 = base64.b64encode(b"fake-sanitized-image").decode("utf-8")
            resp_body = json.dumps({"data": [{"b64_json": fake_b64}]}).encode("utf-8")
            mock_resp.read.return_value = resp_body
            mock_resp.__enter__.return_value = mock_resp
            return mock_resp

        mock_urlopen.side_effect = fake_urlopen
        provider = CodexProvider(base_url="https://api.openai.com/v1", api_key="sk-test-12345")
        dirty_prompt = "真实写真，高清摄影。\n\n负面提示词： 塑料皮肤，过度磨皮，动漫脸，CG脸。"
        provider.generate(prompt=dirty_prompt)

        self.assertEqual(captured_payload.get("prompt"), "真实写真，高清摄影。")

    @patch("urllib.request.urlopen")
    def test_generate_via_responses_payload_format(self, mock_urlopen):
        captured_payload = {}
        captured_url = ""

        def fake_urlopen(req, timeout=300):
            nonlocal captured_payload, captured_url
            captured_url = req.full_url
            captured_payload = json.loads(req.data.decode("utf-8"))
            mock_resp = MagicMock()
            import base64
            fake_b64 = base64.b64encode(b"fake-sse-image").decode("utf-8")
            sse_body = [
                b"event: response.output_item.added\r\n",
                f'data: {{"type": "image_generation", "b64_json": "{fake_b64}"}}\r\n'.encode("utf-8"),
                b"\r\n",
                b"event: response.completed\r\n",
                b"data: [DONE]\r\n",
                b"\r\n",
            ]
            mock_resp.__iter__.return_value = iter(sse_body)
            mock_resp.__enter__.return_value = mock_resp
            return mock_resp

        mock_urlopen.side_effect = fake_urlopen
        provider = CodexProvider(base_url="https://api.openai.com/v1", api_key="sk-test-12345")
        result = provider._generate_via_responses(
            prompt="A portrait in high quality",
            size="1152x2048",
            quality="high",
            background="auto",
            model="gpt-image-2.5-sunburst",
        )

        self.assertEqual(result, b"fake-sse-image")
        self.assertEqual(captured_url, "https://api.openai.com/v1/responses")
        self.assertEqual(captured_payload.get("model"), provider.session_model)
        self.assertEqual(captured_payload.get("instructions"), DEFAULT_VERBATIM_INSTRUCTIONS)
        self.assertEqual(captured_payload.get("tools")[0].get("model"), "gpt-image-2.5-sunburst")
        self.assertEqual(captured_payload.get("tools")[0].get("action"), "generate")
        self.assertTrue(captured_payload.get("stream"))

    @patch("urllib.request.urlopen")
    def test_edit_via_responses_payload_format(self, mock_urlopen):
        captured_payload = {}
        captured_url = ""

        def fake_urlopen(req, timeout=300):
            nonlocal captured_payload, captured_url
            captured_url = req.full_url
            captured_payload = json.loads(req.data.decode("utf-8"))
            mock_resp = MagicMock()
            import base64
            fake_b64 = base64.b64encode(b"fake-sse-edit-image").decode("utf-8")
            sse_body = [
                b"event: response.output_item.added\r\n",
                f'data: {{"type": "image_generation", "b64_json": "{fake_b64}"}}\r\n'.encode("utf-8"),
                b"\r\n",
                b"event: response.completed\r\n",
                b"data: [DONE]\r\n",
                b"\r\n",
            ]
            mock_resp.__iter__.return_value = iter(sse_body)
            mock_resp.__enter__.return_value = mock_resp
            return mock_resp

        mock_urlopen.side_effect = fake_urlopen
        provider = CodexProvider(base_url="https://api.openai.com/v1", api_key="sk-test-12345")
        provider.prefer_responses = True
        result = provider.edit(
            prompt="Change background to sunset",
            source_bytes=b"fake-source-bytes",
            size="1152x2048",
            quality="high",
            model="gpt-image-2.5-sunburst",
        )

        self.assertEqual(result, b"fake-sse-edit-image")
        self.assertEqual(captured_url, "https://api.openai.com/v1/responses")
        self.assertEqual(captured_payload.get("model"), provider.session_model)
        self.assertEqual(captured_payload.get("instructions"), DEFAULT_VERBATIM_INSTRUCTIONS)
        tool = captured_payload.get("tools")[0]
        self.assertEqual(tool.get("type"), "image_generation")
        self.assertEqual(tool.get("action"), "edit")
        self.assertEqual(tool.get("model"), "gpt-image-2.5-sunburst")
        self.assertEqual(tool.get("size"), "1152x2048")

        user_content = captured_payload.get("input")[0].get("content")
        self.assertEqual(len(user_content), 2)
        self.assertEqual(user_content[0].get("type"), "input_image")
        self.assertTrue(user_content[0].get("image_url", "").startswith("data:image/png;base64,"))
        self.assertEqual(user_content[1].get("type"), "input_text")
        self.assertEqual(user_content[1].get("text"), "Change background to sunset")

    @patch("urllib.request.urlopen")
    def test_edit_payload_format_and_4k(self, mock_urlopen):
        captured_data = b""
        captured_url = ""

        def fake_urlopen(req, timeout=300):
            nonlocal captured_data, captured_url
            captured_url = req.full_url
            captured_data = req.data
            mock_resp = MagicMock()
            import base64
            fake_b64 = base64.b64encode(b"fake-edit-image").decode("utf-8")
            resp_body = json.dumps({"data": [{"b64_json": fake_b64}]}).encode("utf-8")
            mock_resp.read.return_value = resp_body
            mock_resp.__enter__.return_value = mock_resp
            return mock_resp

        mock_urlopen.side_effect = fake_urlopen
        provider = CodexProvider(base_url="https://api.openai.com/v1", api_key="sk-test-12345")
        result = provider.edit(
            prompt="Change clothes to white summer dress",
            source_bytes=b"fake-source-png",
            size="1152x2048",
            prefer_4k=True,
            quality="high",
            model="gpt-image-2.5-sunburst",
        )

        self.assertEqual(result, b"fake-edit-image")
        self.assertEqual(captured_url, "https://api.openai.com/v1/images/edits")
        # 验证 4K 映射生效：9:16 从 1152x2048 映射为 2160x3840
        self.assertIn(b'name="size"\r\n\r\n2160x3840\r\n', captured_data)
        self.assertIn(b'name="model"\r\n\r\ngpt-image-2.5-sunburst\r\n', captured_data)

    @patch("urllib.request.urlopen")
    def test_generate_via_responses_extracts_chinese_refusal_text(self, mock_urlopen):
        mock_resp = MagicMock()
        sse_body = [
            b"event: response.output_item.done\r\n",
            b'data: {"type": "output_item.done", "item": {"type": "message", "content": [{"type": "output_text", "text": "\xe6\x8a\xb1\xe6\xad\x89\xef\xbc\x8c\xe8\xbf\x99\xe4\xb8\xaa\xe8\xaf\xb7\xe6\xb1\x82\xe5\x8c\x85\xe5\x90\xab\xe5\xb8\xa6\xe6\x9c\x89\xe6\x80\xa7\xe5\x8c\x96\xe6\x84\x8f\xe5\x91\xb3\xe7\x9a\x84\xe8\xa7\x86\xe8\xa7\x89\xef\xbc\x8c\xe6\x88\x91\xe6\x97\xa0\xe6\xb3\x95\xe7\x94\x9f\xe6\x88\x90\xe3\x80\x82"}]}}\r\n',
            b"\r\n",
            b"event: response.completed\r\n",
            b"data: [DONE]\r\n",
            b"\r\n",
        ]
        mock_resp.__iter__.return_value = iter(sse_body)
        mock_resp.__enter__.return_value = mock_resp
        mock_urlopen.return_value = mock_resp

        provider = CodexProvider(base_url="https://api.openai.com/v1", api_key="sk-test-12345")
        with self.assertRaises(CodexPolicyRefusalError) as ctx:
            provider._generate_via_responses(
                prompt="测试敏感提示词",
                size="1152x2048",
                quality="high",
                background="auto",
                model="gpt-image-2.5-sunburst",
            )
        self.assertIn("上游安全审查拦截拒止", str(ctx.exception))
        self.assertIn("带有性化意味", str(ctx.exception))

    @patch("urllib.request.urlopen")
    def test_generate_via_responses_extracts_english_sexualized_refusal(self, mock_urlopen):
        mock_resp = MagicMock()
        sse_body = [
            b"event: response.output_item.done\r\n",
            b'data: {"type": "output_item.done", "item": {"type": "message", "content": [{"type": "output_text", "text": "I can\xe2\x80\x99t generate that image because the no-pants styling was flagged as sexualized."}]}}\r\n',
            b"\r\n",
            b"event: response.completed\r\n",
            b"data: [DONE]\r\n",
            b"\r\n",
        ]
        mock_resp.__iter__.return_value = iter(sse_body)
        mock_resp.__enter__.return_value = mock_resp
        mock_urlopen.return_value = mock_resp

        provider = CodexProvider(base_url="https://api.openai.com/v1", api_key="sk-test-12345")
        with self.assertRaises(CodexPolicyRefusalError) as ctx:
            provider._generate_via_responses(
                prompt="no-pants styling",
                size="1152x2048",
                quality="high",
                background="auto",
                model="gpt-image-2.5-sunburst",
            )
        self.assertIn("flagged as sexualized", str(ctx.exception))

    @patch("urllib.request.urlopen")
    def test_generate_re_raises_policy_refusal_without_fallback(self, mock_urlopen):
        mock_resp = MagicMock()
        sse_body = [
            b"event: response.output_item.done\r\n",
            b'data: {"type": "output_item.done", "item": {"type": "message", "content": [{"type": "output_text", "text": "Flagged as sexualized."}]}}\r\n',
            b"\r\n",
            b"event: response.completed\r\n",
            b"data: [DONE]\r\n",
            b"\r\n",
        ]
        mock_resp.__iter__.return_value = iter(sse_body)
        mock_resp.__enter__.return_value = mock_resp
        mock_urlopen.return_value = mock_resp

        provider = CodexProvider(base_url="https://api.openai.com/v1", api_key="sk-test-12345")
        provider.prefer_responses = True
        with self.assertRaises(CodexPolicyRefusalError) as ctx:
            provider.generate(prompt="test prompt")
        self.assertIn("Flagged as sexualized", str(ctx.exception))

    def test_format_http_error_detail(self):
        body = json.dumps({"error": {"message": "Upstream refused", "code": "content_policy_violation"}})
        formatted = format_http_error_detail(400, body)
        self.assertEqual(formatted, "HTTP 400 [content_policy_violation]: Upstream refused")


if __name__ == "__main__":
    unittest.main()
