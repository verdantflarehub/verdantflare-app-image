import os

os.environ.setdefault("IMAGE_ARTIFACT_ROOT", "/tmp/image-test-artifacts")
os.environ.setdefault("IMAGE_MCP_BEARER_TOKEN", "test-secret-token")

import unittest

try:
    from starlette.testclient import TestClient
    from src.server import app
    HAS_STARLETTE = True
except ImportError:
    HAS_STARLETTE = False


@unittest.skipUnless(HAS_STARLETTE, "需要安装 starlette 与 mcp 库才能运行集成测试")
class TestServer(unittest.TestCase):
    def setUp(self):
        os.environ["IMAGE_ARTIFACT_ROOT"] = "/tmp/image-test-artifacts"
        os.environ["IMAGE_MCP_BEARER_TOKEN"] = "test-secret-token"

    def test_health_endpoint(self):
        with TestClient(app) as client:
            resp = client.get("/health")
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertEqual(data["status"], "ok")
            self.assertEqual(data["service"], "image-mcp-server")

    def test_bearer_auth(self):
        with TestClient(app) as client:
            resp = client.post("/image", json={})
            self.assertEqual(resp.status_code, 401)
            self.assertEqual(resp.json(), {"error": "unauthorized"})

            resp = client.post("/image", json={}, headers={"Authorization": "Bearer wrong-token"})
            self.assertEqual(resp.status_code, 401)

            resp_health = client.get("/health")
            self.assertEqual(resp_health.status_code, 200)

            # 测试携带正确 Bearer Token 的 MCP 初始化握手
            init_payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "test-agent", "version": "1.0"},
                },
            }
            resp_mcp = client.post(
                "/image",
                json=init_payload,
                headers={
                    "Authorization": "Bearer test-secret-token",
                    "Content-Type": "application/json",
                    "Accept": "application/json, text/event-stream",
                },
            )
            self.assertEqual(resp_mcp.status_code, 200)
            mcp_data = resp_mcp.json()
            self.assertEqual(mcp_data["jsonrpc"], "2.0")
            self.assertEqual(mcp_data["id"], 1)
            self.assertIn("capabilities", mcp_data["result"])

    def test_artifact_import_tool(self):
        from unittest.mock import patch, MagicMock
        from src.server import artifact_import

        with patch("src.server.artifacts.import_from_url") as mock_import:
            from src.artifacts import ArtifactRecord
            mock_rec = ArtifactRecord(
                artifact_id="art-test-123",
                project_id="test-proj",
                filename="test.png",
                media_type="image/png",
                size_bytes=100,
                sha256="abc123sha",
                created_at="2026-09-11T00:00:00Z",
            )
            mock_import.return_value = mock_rec

            res = artifact_import(
                project_id="test-proj",
                source_url="https://example.com/test.png",
                filename="test.png",
                expected_sha256="abc123sha",
            )
            self.assertIsNotNone(res)
            self.assertEqual(len(res.content), 1)
            import json
            data = json.loads(res.content[0].text)
            self.assertEqual(data["status"], "completed")
            self.assertEqual(data["artifact"]["artifact_id"], "art-test-123")

    def test_api_upload_artifact(self):
        with TestClient(app) as client:
            upload_files = {"file": ("test.png", b"fake-png-content", "image/png")}
            data = {"project_id": "upload-test-project"}
            headers = {"Authorization": "Bearer test-secret-token"}
            resp = client.post("/api/artifacts/upload", files=upload_files, data=data, headers=headers)
            self.assertEqual(resp.status_code, 201)
            resp_data = resp.json()
            self.assertEqual(resp_data["status"], "completed")
            self.assertEqual(resp_data["project_id"], "upload-test-project")
            self.assertTrue(resp_data["artifact"]["artifact_id"].startswith("art-"))
            self.assertEqual(resp_data["artifact"]["filename"], "test.png")


if __name__ == "__main__":
    unittest.main()

