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

    def test_api_tasks_direct_measured_specs(self):
        import io
        from PIL import Image
        from src.server import artifacts

        buf = io.BytesIO()
        Image.new("RGB", (1280, 720), color=(100, 150, 200)).save(buf, format="PNG")
        png_bytes = buf.getvalue()

        with TestClient(app) as client:
            art = artifacts.create_from_bytes(
                project_id="spec-proj",
                filename="out.png",
                data=png_bytes,
                media_type="image/png",
            )
            task = tasks.create(
                project_id="spec-proj",
                idempotency_key=f"spec-{art.artifact_id}",
                engine="gemini",
                model="gemini-nano-banana-2.1",
                request_params={"prompt": "spec test", "resolution": "2k", "aspect_ratio": "1:1"},
            )
            tasks.update_status(task.task_id, "completed", artifact_id=art.artifact_id)

            resp = client.get(
                "/api/tasks?project_id=spec-proj",
                headers={"Authorization": "Bearer dash-secret-token"},
            )
            self.assertEqual(resp.status_code, 200)
            items = resp.json()["tasks"]
            matched = next(t for t in items if t["task_id"] == task.task_id)
            self.assertEqual(matched["outputWidth"], 1280)
            self.assertEqual(matched["outputHeight"], 720)
            self.assertEqual(matched["outputFormat"], "PNG")
            self.assertEqual(matched["outputSizeBytes"], len(png_bytes))
            self.assertEqual(matched["outputSha256"], art.sha256)
            self.assertEqual(matched["media"]["width"], 1280)
            self.assertEqual(matched["media"]["height"], 720)
            self.assertEqual(matched["media"]["format"], "PNG")
            self.assertEqual(matched["media"]["size_bytes"], len(png_bytes))

            detail_resp = client.get(
                f"/api/tasks/{task.task_id}",
                headers={"Authorization": "Bearer dash-secret-token"},
            )
            self.assertEqual(detail_resp.status_code, 200)
            detail = detail_resp.json()
            self.assertEqual(detail["outputWidth"], 1280)
            self.assertEqual(detail["outputHeight"], 720)
            self.assertEqual(detail["outputFormat"], "PNG")
            self.assertEqual(detail["outputSizeBytes"], len(png_bytes))


class TestDashboardStaticHtml(unittest.TestCase):
    def test_enrich_task_dict_without_starlette(self):
        import io
        import tempfile
        from pathlib import Path
        from PIL import Image
        from src.artifacts import ArtifactStore
        from src.tasks import TaskStore
        from src.dashboard import _enrich_task_dict

        buf = io.BytesIO()
        Image.new("RGB", (1920, 1080), color=(30, 60, 90)).save(buf, format="PNG")
        png_bytes = buf.getvalue()

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            art_store = ArtifactStore(root_dir=root / "projects")
            task_store = TaskStore(db_path=root / "tasks.db")
            task_store.ensure_ready()
            art = art_store.create_from_bytes(
                project_id="unit-proj",
                filename="test.png",
                data=png_bytes,
                media_type="image/png",
            )
            task = task_store.create(
                project_id="unit-proj",
                idempotency_key="idem-unit-1",
                engine="gemini",
                model="gemini-nano-banana-2.1",
                request_params={"prompt": "test", "resolution": "2k", "aspect_ratio": "1:1"},
            )
            task_store.update_status(task.task_id, "completed", artifact_id=art.artifact_id)
            updated_task = task_store.get(task.task_id)
            enriched = _enrich_task_dict(updated_task, art_store, None)
            self.assertEqual(enriched["outputWidth"], 1920)
            self.assertEqual(enriched["outputHeight"], 1080)
            self.assertEqual(enriched["outputFormat"], "PNG")
            self.assertEqual(enriched["outputSizeBytes"], len(png_bytes))
            self.assertEqual(enriched["outputMediaType"], "image/png")
            self.assertEqual(enriched["outputSha256"], art.sha256)
            self.assertEqual(enriched["media"]["width"], 1920)
            self.assertEqual(enriched["media"]["height"], 1080)
            self.assertEqual(enriched["media"]["format"], "PNG")
            self.assertEqual(enriched["media"]["size_bytes"], len(png_bytes))

            from starlette.applications import Starlette
            from starlette.routing import Route
            from starlette.testclient import TestClient as DirectTestClient
            from src.dashboard import (
                api_artifact_content,
                api_artifact_thumbnail,
                api_get_task,
                api_list_tasks,
            )

            class _DummyQueue:
                def capabilities(self):
                    return {}

            mini_app = Starlette(
                routes=[
                    Route("/api/tasks", api_list_tasks, methods=["GET"]),
                    Route("/api/tasks/{task_id}", api_get_task, methods=["GET"]),
                    Route("/artifacts/{artifact_id}/content", api_artifact_content, methods=["GET", "HEAD"]),
                    Route("/artifacts/{artifact_id}/thumbnail", api_artifact_thumbnail, methods=["GET", "HEAD"]),
                ]
            )
            mini_app.state.tasks = task_store
            mini_app.state.artifacts = art_store
            mini_app.state.ratings = None
            mini_app.state.queue_manager = _DummyQueue()

            with DirectTestClient(mini_app) as client:
                list_resp = client.get(
                    "/api/tasks?project_id=unit-proj",
                    headers={"Authorization": "Bearer dash-secret-token"},
                )
                self.assertEqual(list_resp.status_code, 200)
                t_item = list_resp.json()["tasks"][0]
                self.assertEqual(t_item["outputWidth"], 1920)
                self.assertEqual(t_item["outputHeight"], 1080)
                self.assertEqual(t_item["outputFormat"], "PNG")
                self.assertEqual(t_item["outputSizeBytes"], len(png_bytes))
                self.assertEqual(t_item["outputSha256"], art.sha256)

                get_resp = client.get(
                    f"/api/tasks/{task.task_id}",
                    headers={"Authorization": "Bearer dash-secret-token"},
                )
                self.assertEqual(get_resp.status_code, 200)
                self.assertEqual(get_resp.json()["outputWidth"], 1920)
                self.assertEqual(get_resp.json()["outputSizeBytes"], len(png_bytes))

                # Thumbnail 320 + immutable cache headers + ETag + 304
                thumb_resp = client.get(f"/artifacts/{art.artifact_id}/thumbnail?edge=320")
                self.assertEqual(thumb_resp.status_code, 200)
                self.assertEqual(thumb_resp.headers.get("content-type"), "image/webp")
                self.assertEqual(
                    thumb_resp.headers.get("cache-control"),
                    "private, max-age=2592000, immutable",
                )
                etag = thumb_resp.headers.get("etag")
                self.assertEqual(etag, f'"{art.sha256}-320-v1"')
                with Image.open(io.BytesIO(thumb_resp.content)) as t_img:
                    self.assertEqual(t_img.size, (320, 180))

                # Conditional GET with If-None-Match -> 304
                not_mod = client.get(
                    f"/artifacts/{art.artifact_id}/thumbnail?edge=320",
                    headers={"If-None-Match": etag},
                )
                self.assertEqual(not_mod.status_code, 304)
                self.assertEqual(not_mod.content, b"")

                # Invalid edge -> 400 with no-store
                bad_edge = client.get(f"/artifacts/{art.artifact_id}/thumbnail?edge=999")
                self.assertEqual(bad_edge.status_code, 400)
                self.assertEqual(bad_edge.headers.get("cache-control"), "no-store")

                # Content GET + ETag + 304
                content_resp = client.get(f"/artifacts/{art.artifact_id}/content")
                self.assertEqual(content_resp.status_code, 200)
                self.assertEqual(
                    content_resp.headers.get("cache-control"),
                    "private, max-age=2592000, immutable",
                )
                c_etag = content_resp.headers.get("etag")
                self.assertEqual(c_etag, f'"{art.sha256}"')
                content_304 = client.get(
                    f"/artifacts/{art.artifact_id}/content",
                    headers={"If-None-Match": c_etag},
                )
                self.assertEqual(content_304.status_code, 304)

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
        self.assertIn("s.includes('moss')) return '本地'", html)
        self.assertNotIn("return String(ch);", html)
        self.assertIn("model-display-name", html)
        self.assertIn("model-id-sub", html)
        self.assertIn("Nano Banana 2.1", html)
        self.assertIn("gemini-nano-banana-2.1", html)
        self.assertIn("Nano Banana Pro", html)
        self.assertIn("gemini-3-pro-image", html)
        self.assertIn("Nano Banana 2", html)
        self.assertIn("gemini-3.1-flash-image", html)
        self.assertIn("Nano Banana 2 Lite", html)
        self.assertIn("gemini-3.1-flash-lite-image", html)
        self.assertIn("Nano Banana", html)
        self.assertIn("gemini-2.5-flash-image", html)
        self.assertIn("modelDisplayName", html)
        self.assertIn("normalizeModelCode", html)
        self.assertIn("最多 6张参考图 · 0张蒙版", html)
        self.assertIn("MOSS Image", html)
        self.assertIn("moss-image-sft-int8-convrot", html)
        self.assertIn("grok-imagine-image-2.0", html)
        self.assertIn("gpt-image-2.5-sunburst", html)
        self.assertIn("gpt-image-2.5-flare", html)
        for fake_entry in (
            "Imagen 4 Ultra",
            "fal.ai",
            "Higgsfield",
            "FLUX.1 Kontext",
            "FLUX.1 Fill",
            "FLUX1.1 [pro] Ultra",
            "AuraSR v2 Upscaler",
            "Soul 2.0",
            "Seedream 5.0",
            "Qwen-Image-Edit-2511",
            "Z-Image-Turbo",
            "InstantID + InsightFace",
            "SDXL ControlNet Union",
        ):
            self.assertNotIn(fake_entry, html)
        self.assertRegex(
            html,
            r'(?s)<div class="output-pane">.*?id="inspOutputMeta".*?id="inspPromptBox".*?</div>\s*<div class="input-pane">\s*<!-- 2\. 参考图',
        )


if __name__ == "__main__":
    unittest.main()


