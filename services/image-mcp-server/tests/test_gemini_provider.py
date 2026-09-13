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


    def test_ensure_auth_prefers_google(self):
        provider = GeminiProvider(
            api_key="ai-za-google-key",
            anthropic_base_url="https://sub2api.example.com",
            anthropic_token="secret-token",
        )
        self.assertEqual(provider._ensure_auth(), "google")

    def test_detect_mime_type(self):
        from src.providers.gemini import _detect_mime_type

        png_header = b"\x89PNG\r\n\x1a\n\x00\x00"
        jpeg_header = b"\xff\xd8\xff\xe0\x00\x10"
        webp_header = b"RIFF\x00\x00\x00\x00WEBPVP8"

        self.assertEqual(_detect_mime_type(png_header), "image/png")
        self.assertEqual(_detect_mime_type(jpeg_header), "image/jpeg")
        self.assertEqual(_detect_mime_type(webp_header), "image/webp")

    def test_generate_google_multi_image_payload(self):
        import io
        import json
        from unittest.mock import MagicMock

        provider = GeminiProvider(api_key="sk-sub2api-key", base_url="https://sub2api.example.com/v1beta")
        raw_b64 = base64.b64encode(b"result-img").decode("utf-8")
        mock_response = io.BytesIO(
            json.dumps({
                "candidates": [
                    {"content": {"parts": [{"inlineData": {"mimeType": "image/png", "data": raw_b64}}]}}
                ]
            }).encode("utf-8")
        )

        sent_requests = []

        def fake_urlopen(req, timeout=120):
            sent_requests.append(req)
            return mock_response

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            img_bytes = provider.generate(
                prompt="Replace clothing",
                source_images=[b"\x89PNG\r\n\x1a\n-person", b"\x89PNG\r\n\x1a\n-clothing"],
                mask_bytes=b"\x89PNG\r\n\x1a\n-mask",
            )

        self.assertEqual(img_bytes, b"result-img")
        self.assertEqual(len(sent_requests), 1)
        req = sent_requests[0]

        # Verify URL stripped duplicate v1beta
        self.assertEqual(req.full_url, "https://sub2api.example.com/v1beta/models/gemini-3.1-flash-image:generateContent?key=sk-sub2api-key")
        self.assertEqual(req.headers.get("X-goog-api-key"), "sk-sub2api-key")
        self.assertEqual(req.headers.get("Authorization"), "Bearer sk-sub2api-key")

        payload = json.loads(req.data.decode("utf-8"))
        parts = payload["contents"][0]["parts"]
        # 2 source images + 1 mask + 1 text prompt = 4 parts
        self.assertEqual(len(parts), 4)
        self.assertIn("inlineData", parts[0])
        self.assertIn("inlineData", parts[1])
        self.assertIn("inlineData", parts[2])
        self.assertIn("text", parts[3])
        self.assertEqual(payload["generationConfig"]["responseModalities"], ["IMAGE", "TEXT"])


if __name__ == "__main__":
    unittest.main()

