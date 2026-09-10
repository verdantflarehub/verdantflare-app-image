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


if __name__ == "__main__":
    unittest.main()

