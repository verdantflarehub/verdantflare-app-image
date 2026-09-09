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

            # 测试网关前缀 /image/dashboard 也正常返回 HTML，无需鉴权拦截
            resp_image = client.get("/image/dashboard")
            self.assertEqual(resp_image.status_code, 200)
            self.assertIn("VerdantFlare", resp_image.text)
            self.assertIn("tokenModal", resp_image.text)

    def test_api_tasks_auth(self):
        with TestClient(app) as client:
            # 未提供 token -> 401
            resp = client.get("/api/tasks")
            self.assertEqual(resp.status_code, 401)

            resp_image_unauth = client.get("/image/api/tasks")
            self.assertEqual(resp_image_unauth.status_code, 401)

            # Query param 提供 token -> 200
            resp_query = client.get("/api/tasks?token=dash-secret-token")
            self.assertEqual(resp_query.status_code, 200)
            data = resp_query.json()
            self.assertIn("tasks", data)
            self.assertIn("total", data)

            # Header 提供 token -> 200
            resp_header = client.get(
                "/image/api/tasks",
                headers={"Authorization": "Bearer dash-secret-token"},
            )
            self.assertEqual(resp_header.status_code, 200)

            # X-MCP-Token Header 提供 token -> 200
            resp_x_header = client.get(
                "/image/api/tasks",
                headers={"X-MCP-Token": "dash-secret-token"},
            )
            self.assertEqual(resp_x_header.status_code, 200)

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

            # 测试 /image/api/tasks/stats
            resp_image = client.get("/image/api/tasks/stats", headers={"Authorization": "Bearer dash-secret-token"})
            self.assertEqual(resp_image.status_code, 200)
            stats_image = resp_image.json()
            self.assertEqual(stats_image["queued"], stats["queued"])

    def test_api_create_task(self):
        with TestClient(app) as client:
            # 未提供 token -> 401
            resp = client.post("/image/api/tasks", json={"prompt": "test prompt"})
            self.assertEqual(resp.status_code, 401)

            # 缺少 prompt -> 400
            resp_no_prompt = client.post(
                "/image/api/tasks",
                json={"prompt": ""},
                headers={"Authorization": "Bearer dash-secret-token"},
            )
            self.assertEqual(resp_no_prompt.status_code, 400)

            # 正常创建 -> 201
            resp_ok = client.post(
                "/image/api/tasks",
                json={
                    "project_id": "test-proj",
                    "prompt": "a test prompt",
                    "engine": "gemini",
                },
                headers={"Authorization": "Bearer dash-secret-token"},
            )
            self.assertEqual(resp_ok.status_code, 201)
            created = resp_ok.json()
            self.assertIn("task_id", created)
            self.assertEqual(created["project_id"], "test-proj")
            self.assertEqual(created["status"], "queued")


if __name__ == "__main__":
    unittest.main()

