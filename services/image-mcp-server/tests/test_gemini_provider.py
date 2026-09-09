import base64
import unittest
from unittest.mock import patch
from src.providers.gemini import GeminiProvider, GeminiProviderError


class TestGeminiProvider(unittest.TestCase):
    def test_ensure_auth_anthropic(self):
        provider = GeminiProvider(
            anthropic_base_url="https://sub2api.example.com",
            anthropic_token="secret-token",
        )
        self.assertEqual(provider._ensure_auth(), "anthropic")

    def test_ensure_auth_google(self):
        provider = GeminiProvider(api_key="ai-za-google-key")
        self.assertEqual(provider._ensure_auth(), "google")

    def test_ensure_auth_missing(self):
        with patch.dict("os.environ", {}, clear=True):
            provider = GeminiProvider(api_key="", anthropic_base_url="", anthropic_token="")
            with self.assertRaises(GeminiProviderError):
                provider._ensure_auth()

    def test_extract_anthropic_text_data_uri(self):
        raw_b64 = base64.b64encode(b"fake-jpeg-binary").decode("utf-8")
        resp_data = {
            "content": [
                {
                    "type": "text",
                    "text": f"Here is the generated image:\n![image](data:image/jpeg;base64,{raw_b64})\nDone.",
                }
            ]
        }
        provider = GeminiProvider(anthropic_base_url="https://fake", anthropic_token="token")
        extracted = provider._extract_image_bytes(resp_data)
        self.assertEqual(extracted, b"fake-jpeg-binary")

    def test_extract_anthropic_image_block(self):
        raw_b64 = base64.b64encode(b"fake-png-binary").decode("utf-8")
        resp_data = {
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": raw_b64,
                    },
                }
            ]
        }
        provider = GeminiProvider(anthropic_base_url="https://fake", anthropic_token="token")
        extracted = provider._extract_image_bytes(resp_data)
        self.assertEqual(extracted, b"fake-png-binary")

    def test_extract_google_inline_data(self):
        raw_b64 = base64.b64encode(b"google-img-binary").decode("utf-8")
        resp_data = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"inlineData": {"mimeType": "image/png", "data": raw_b64}}
                        ]
                    }
                }
            ]
        }
        provider = GeminiProvider(api_key="fake-key")
        extracted = provider._extract_image_bytes(resp_data)
        self.assertEqual(extracted, b"google-img-binary")

    def test_extract_missing_image_raises(self):
        resp_data = {"content": [{"type": "text", "text": "Sorry, I could not generate."}]}
        provider = GeminiProvider(anthropic_base_url="https://fake", anthropic_token="token")
        with self.assertRaises(GeminiProviderError):
            provider._extract_image_bytes(resp_data)


if __name__ == "__main__":
    unittest.main()

