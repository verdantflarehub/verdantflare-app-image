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


class TestDashboardStaticHtml(unittest.TestCase):
    def test_static_index_html_v12_contracts(self):
        from pathlib import Path
        html_path = Path(__file__).resolve().parent.parent / "src" / "static" / "index.html"
        html = html_path.read_text(encoding="utf-8")
        self.assertIn("VerdantFlare", html)
        self.assertIn("IMAGE STATION", html.upper())
        self.assertIn("tokenModal", html)
        self.assertIn("--vf-hairline-outer", html)
        self.assertIn("--vf-hairline-inner", html)
        self.assertIn("--vf-shadow-card", html)
        self.assertIn("--vf-shadow-card-hover", html)
        self.assertIn("--vf-ease-spring", html)
        self.assertIn("--vf-shimmer", html)
        self.assertIn("tabular-nums", html)
        self.assertNotIn("btnThemeToggle", html)
        self.assertIn("channel-group", html)
        self.assertIn("model-cap-row", html)
        self.assertIn("model-quota-strip", html)
        self.assertIn("detail-shell", html)
        self.assertIn("inspect-head", html)
        self.assertIn("inspect-layout", html)
        self.assertIn("output-pane", html)
        self.assertIn("input-pane", html)
        self.assertIn("result-stage", html)
        self.assertIn("inspect-tech", html)
        self.assertIn("zoomDialog", html)
        self.assertIn("prompt-ref-chip", html)
        self.assertIn("ref-image-grid", html)
        self.assertIn("object-fit: contain", html)
        self.assertIn("ref-empty-box", html)
        self.assertIn("openZoomModal", html)
        self.assertNotIn("compare-container", html)
        self.assertNotIn("btnScrollPrompt", html)
        self.assertNotIn("creationConsole", html)
        self.assertNotIn("令牌直通", html)
        self.assertNotIn("inspTriSteps", html)
        self.assertIn('html[data-embedded="true"] #imageBreadcrumb', html)
        self.assertIn("M22.282 9.821", html)
        self.assertRegex(
            html,
            r'(?s)<div class="output-pane">.*?id="inspOutputMeta".*?id="inspPromptBox".*?</div>\s*<div class="input-pane">\s*<!-- 2\. 参考图',
        )


if __name__ == "__main__":
    unittest.main()


