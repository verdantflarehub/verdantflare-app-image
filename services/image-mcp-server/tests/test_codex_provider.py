import json
import unittest
from unittest.mock import MagicMock, patch
from src.providers.codex import (
    CodexProvider,
    CodexProviderError,
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
        self.assertEqual(captured_payload.get("model"), "gpt-image-2.5-sunburst")
        self.assertTrue(captured_payload.get("stream"))


if __name__ == "__main__":
    unittest.main()
