import json
import unittest
from unittest.mock import MagicMock, patch
from src.providers.codex import (
    CodexProvider,
    CodexProviderError,
    DEFAULT_VERBATIM_INSTRUCTIONS,
    DEFAULT_SESSION_MODEL,
    DEFAULT_IMAGE_MODEL,
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
    def test_generate_payload_includes_verbatim_instructions(self, mock_urlopen):
        captured_payload = {}

        def fake_urlopen(req, timeout=300):
            nonlocal captured_payload
            captured_payload = json.loads(req.data.decode("utf-8"))
            mock_resp = MagicMock()
            import base64
            fake_b64 = base64.b64encode(b"fake-codex-image").decode("utf-8")
            data_json = json.dumps({"type": "response.completed", "result": fake_b64})
            lines = [
                f"data: {data_json}\r\n".encode("utf-8"),
                b"\r\n",
                b"data: [DONE]\r\n",
                b"\r\n",
            ]
            mock_resp.__enter__.return_value = lines
            return mock_resp

        mock_urlopen.side_effect = fake_urlopen

        provider = CodexProvider(base_url="https://api.openai.com/v1", api_key="sk-test-12345")
        result = provider.generate(
            prompt="A photorealistic dancer in high resolution",
            size="1152x2048",
            quality="high",
        )

        self.assertEqual(result, b"fake-codex-image")
        self.assertEqual(captured_payload.get("model"), DEFAULT_SESSION_MODEL)
        self.assertEqual(captured_payload.get("instructions"), DEFAULT_VERBATIM_INSTRUCTIONS)
        self.assertEqual(captured_payload.get("input"), "A photorealistic dancer in high resolution")

        tools = captured_payload.get("tools", [])
        self.assertEqual(len(tools), 1)
        self.assertEqual(tools[0].get("type"), "image_generation")
        self.assertEqual(tools[0].get("model"), DEFAULT_IMAGE_MODEL)
        self.assertEqual(tools[0].get("size"), "1152x2048")
        self.assertEqual(tools[0].get("quality"), "high")

    @patch("urllib.request.urlopen")
    def test_generate_custom_instructions_override(self, mock_urlopen):
        captured_payload = {}

        def fake_urlopen(req, timeout=300):
            nonlocal captured_payload
            captured_payload = json.loads(req.data.decode("utf-8"))
            mock_resp = MagicMock()
            import base64
            fake_b64 = base64.b64encode(b"fake-codex-image-override").decode("utf-8")
            data_json = json.dumps({"type": "response.completed", "result": fake_b64})
            lines = [
                f"data: {data_json}\r\n".encode("utf-8"),
                b"\r\n",
                b"data: [DONE]\r\n",
                b"\r\n",
            ]
            mock_resp.__enter__.return_value = lines
            return mock_resp

        mock_urlopen.side_effect = fake_urlopen

        provider = CodexProvider(base_url="https://api.openai.com/v1", api_key="sk-test-12345")
        custom_instructions = "Custom prompt instruction override"
        provider.generate(
            prompt="Test prompt",
            instructions=custom_instructions,
        )
        self.assertEqual(captured_payload.get("instructions"), custom_instructions)


if __name__ == "__main__":
    unittest.main()
