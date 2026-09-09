import os

os.environ.setdefault("IMAGE_ARTIFACT_ROOT", "/tmp/image-dashboard-test")
os.environ.setdefault("IMAGE_MCP_BEARER_TOKEN", "dash-secret-token")

import unittest

try:
    from starlette.testclient import TestClient
    from src.server import app, tasks
    HAS_STARLETTE = True
except ImportError:
    HAS_STARLETTE = False


@unittest.skipUnless(HAS_STARLETTE, "需要安装 starlette 才能运行 Dashboard 测试")
class TestDashboard(unittest.TestCase):
    def setUp(self):
        os.environ["IMAGE_ARTIFACT_ROOT"] = "/tmp/image-dashboard-test"
        os.environ["IMAGE_MCP_BEARER_TOKEN"] = "dash-secret-token"

    def test_dashboard_html_page(self):
        with TestClient(app) as client:
            resp = client.get("/dashboard")
            self.assertEqual(resp.status_code, 200)
            self.assertIn("VerdantFlare", resp.text)
            self.assertIn("IMAGE STATION", resp.text.upper())

    def test_api_tasks_auth(self):
        with TestClient(app) as client:
            # 未提供 token -> 401
            resp = client.get("/api/tasks")
            self.assertEqual(resp.status_code, 401)

            # Query param 提供 token -> 200
            resp_query = client.get("/api/tasks?token=dash-secret-token")
            self.assertEqual(resp_query.status_code, 200)
            data = resp_query.json()
            self.assertIn("tasks", data)
            self.assertIn("total", data)

            # Header 提供 token -> 200
            resp_header = client.get(
                "/api/tasks",
                headers={"Authorization": "Bearer dash-secret-token"},
            )
            self.assertEqual(resp_header.status_code, 200)

    def test_api_tasks_stats(self):
        with TestClient(app) as client:
            resp = client.get("/api/tasks/stats?token=dash-secret-token")
            self.assertEqual(resp.status_code, 200)
            stats = resp.json()
            self.assertIn("queued", stats)
            self.assertIn("running", stats)
            self.assertIn("completed", stats)
            self.assertIn("failed", stats)
            self.assertIn("avg_duration", stats)


if __name__ == "__main__":
    unittest.main()

