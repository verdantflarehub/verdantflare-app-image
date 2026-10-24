from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from pathlib import Path

from PIL import Image
from starlette.testclient import TestClient

import src.server as server_mod
from src.artifacts import ArtifactStore
from src.content_rating import ContentRatingError, ContentRatingStore
from src.tasks import TaskNotFound, TaskStore


def _png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), color=(120, 30, 60)).save(buf, format="PNG")
    return buf.getvalue()


class TestContentRatingStore(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self._tmp.name) / "tasks.db"
        self.store = ContentRatingStore(self.db_path, domain="image")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_add_only_and_human_override_lifecycle(self) -> None:
        # 1. Rate general
        r1 = self.store.rate_content(
            version_id="art-1",
            project_id="proj-1",
            rating="general",
            reason="clean landscape poster",
            rubric_version="rubric v1.0",
        )
        self.assertEqual(r1["rating"], "general")
        self.assertFalse(r1["ignored"])

        # 2. Upgrade general -> nsfw
        r2 = self.store.rate_content(
            version_id="art-1",
            project_id="proj-1",
            rating="nsfw",
            reason="explicit_nudity",
            rubric_version="rubric v1.0",
        )
        self.assertEqual(r2["rating"], "nsfw")
        self.assertFalse(r2["ignored"])
        self.assertTrue(self.store.is_nsfw("art-1"))

        # 3. Attempt to downgrade nsfw -> general via MCP (must be ignored)
        r3 = self.store.rate_content(
            version_id="art-1",
            project_id="proj-1",
            rating="general",
            reason="attempt to clear via mcp",
            rubric_version="rubric v1.0",
        )
        self.assertEqual(r3["rating"], "nsfw")
        self.assertTrue(r3["ignored"])
        self.assertTrue(self.store.is_nsfw("art-1"))

        # 4. Clear by human
        r4 = self.store.clear_rating(
            version_id="art-1",
            project_id="proj-1",
            actor_user_id="user-admin",
            note="false positive mannequin",
        )
        self.assertEqual(r4["rating"], "general")
        self.assertEqual(r4["source"], "human")
        self.assertTrue(r4["human_cleared"])
        self.assertFalse(self.store.is_nsfw("art-1"))

        # 4b. AI attempts to re-rate nsfw after human clear -> ignored!
        r4b = self.store.rate_content(
            version_id="art-1",
            project_id="proj-1",
            rating="nsfw",
            reason="ai re-rate attempt",
            rubric_version="rubric v1.0",
        )
        self.assertEqual(r4b["rating"], "general")
        self.assertTrue(r4b["ignored"])
        self.assertFalse(self.store.is_nsfw("art-1"))

        # 5. Mark by human
        r5 = self.store.mark_rating(
            version_id="art-1",
            project_id="proj-1",
            actor_user_id="user-admin",
            reason="manual 18+ mark",
        )
        self.assertEqual(r5["rating"], "nsfw")
        self.assertEqual(r5["source"], "human")
        self.assertTrue(self.store.is_nsfw("art-1"))

        events = self.store.list_events("art-1")
        self.assertEqual(
            [e["event_type"] for e in events],
            ["rate", "rate", "rate", "clear", "ignored_after_clear", "rate"],
        )

    def test_validation_rejects_invalid_rating_or_reason(self) -> None:
        with self.assertRaises(ContentRatingError):
            self.store.rate_content(
                version_id="art-1",
                project_id="proj-1",
                rating="r18",
                reason="bad enum",
            )
        with self.assertRaises(ContentRatingError):
            self.store.rate_content(
                version_id="art-1",
                project_id="proj-1",
                rating="nsfw",
                reason="   ",
            )


class TestImageMcpAndHttpContentRating(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        os.environ["IMAGE_MCP_BEARER_TOKEN"] = "test-secret-token"
        self.orig_tasks = server_mod.tasks
        self.orig_artifacts = server_mod.artifacts
        self.orig_ratings = server_mod.ratings

        self.tasks = TaskStore(root / "tasks.db")
        self.artifacts = ArtifactStore(root / "artifacts")
        self.ratings = ContentRatingStore(self.tasks.db_path, domain="image")
        self.tasks.ensure_ready()
        self.artifacts.ensure_ready()
        self.ratings.ensure_ready()

        server_mod.tasks = self.tasks
        server_mod.artifacts = self.artifacts
        server_mod.ratings = self.ratings
        server_mod.app.state.tasks = self.tasks
        server_mod.app.state.artifacts = self.artifacts
        server_mod.app.state.ratings = self.ratings

        # Seed 1 general task (codex) and 1 moss-image task
        art_gen = self.artifacts.create_from_bytes(
            project_id="proj-a",
            data=_png_bytes(),
            filename="general.png",
            media_type="image/png",
            metadata={"engine": "codex", "width": 32, "height": 32},
        )
        t_gen = self.tasks.create(
            project_id="proj-a",
            idempotency_key="idem-gen",
            engine="codex",
            model="gpt-image-2.5-sunburst",
            request_params={"prompt": "sunny mountain"},
        )
        self.task_gen = self.tasks.update_status(t_gen.task_id, "completed", artifact_id=art_gen.artifact_id, duration_seconds=5.0)
        self.art_gen = art_gen

        art_moss = self.artifacts.create_from_bytes(
            project_id="proj-a",
            data=_png_bytes(),
            filename="moss.png",
            media_type="image/png",
            metadata={"engine": "moss-image", "width": 32, "height": 32},
        )
        t_moss = self.tasks.create(
            project_id="proj-a",
            idempotency_key="idem-moss",
            engine="moss-image",
            model="moss-image-sft-int8-convrot",
            request_params={"prompt": "成人向海报测试"},
        )
        self.task_moss = self.tasks.update_status(t_moss.task_id, "completed", artifact_id=art_moss.artifact_id, duration_seconds=46.0)
        self.art_moss = art_moss

    def tearDown(self) -> None:
        server_mod.tasks = self.orig_tasks
        server_mod.artifacts = self.orig_artifacts
        server_mod.ratings = self.orig_ratings
        server_mod.app.state.tasks = self.orig_tasks
        server_mod.app.state.artifacts = self.orig_artifacts
        server_mod.app.state.ratings = self.orig_ratings
        self._tmp.cleanup()

    def test_mcp_rate_content_hides_from_mcp_and_http_supports_local_switch_and_clear(self) -> None:
        # Initially both tasks are visible in MCP list
        initial_list = json.loads(server_mod.image_list(project_id="proj-a").content[0].text)
        self.assertEqual(initial_list["total"], 2)

        # Rate moss-image artifact as nsfw via MCP tool
        rate_res = json.loads(
            server_mod.image_rate_content(
                version_id=self.art_moss.artifact_id,
                rating="nsfw",
                reason="explicit_nudity",
                rubric_version="rubric v1.0",
            ).content[0].text
        )
        self.assertEqual(rate_res["rating"], "nsfw")
        self.assertFalse(rate_res["ignored"])

        # Attempting to downgrade via MCP general must not clear nsfw
        downgrade_res = json.loads(
            server_mod.image_rate_content(
                version_id=self.art_moss.artifact_id,
                rating="general",
                reason="try downgrade",
                rubric_version="rubric v1.0",
            ).content[0].text
        )
        self.assertEqual(downgrade_res["rating"], "nsfw")
        self.assertTrue(downgrade_res["ignored"])

        # MCP list now hides the nsfw task and decrements total (no placeholder)
        after_list = json.loads(server_mod.image_list(project_id="proj-a").content[0].text)
        self.assertEqual(after_list["total"], 1)
        self.assertEqual([t["task_id"] for t in after_list["tasks"]], [self.task_gen.task_id])

        # MCP status and result return NOT_FOUND for the nsfw task
        with self.assertRaises(TaskNotFound):
            server_mod.image_status(self.task_moss.task_id)
        with self.assertRaises(TaskNotFound):
            server_mod.image_result(self.task_moss.task_id)

        # Unknown or cross-project version_id raises TaskNotFound
        with self.assertRaises(TaskNotFound):
            server_mod.image_rate_content(
                version_id="art-nonexistent",
                rating="general",
                reason="missing",
            )
        with self.assertRaises(TaskNotFound):
            server_mod.image_rate_content(
                version_id=self.art_moss.artifact_id,
                rating="general",
                reason="wrong project",
                project_id="proj-other",
            )

        headers = {"Authorization": "Bearer test-secret-token"}
        client = TestClient(server_mod.app)

        # HTTP /api/tasks without X-VF-Include-NSFW hides the nsfw task
        resp_default = client.get("/api/tasks?project_id=proj-a", headers=headers)
        self.assertEqual(resp_default.status_code, 200)
        self.assertEqual(resp_default.json()["total"], 1)

        # HTTP /artifacts/{id}/content without X-VF-Include-NSFW returns 404
        resp_art_hidden = client.get(f"/artifacts/{self.art_moss.artifact_id}/content")
        self.assertEqual(resp_art_hidden.status_code, 404)

        # HTTP /api/tasks with X-VF-Include-NSFW: 1 includes the nsfw task with UI fields
        resp_inc = client.get(
            "/api/tasks?project_id=proj-a",
            headers={**headers, "X-VF-Include-NSFW": "1"},
        )
        self.assertEqual(resp_inc.status_code, 200)
        inc_data = resp_inc.json()
        self.assertEqual(inc_data["total"], 2)
        moss_item = next(t for t in inc_data["tasks"] if t["task_id"] == self.task_moss.task_id)
        self.assertEqual(moss_item["content_rating"]["rating"], "nsfw")
        self.assertTrue(moss_item["can_clear_rating"])
        self.assertFalse(moss_item["can_mark_rating"])

        # HTTP /artifacts/{id}/content with X-VF-Include-NSFW: 1 returns 200
        resp_art_shown = client.get(
            f"/artifacts/{self.art_moss.artifact_id}/content",
            headers={"X-VF-Include-NSFW": "1"},
        )
        self.assertEqual(resp_art_shown.status_code, 200)

        # Internal human clear endpoint restores general visibility
        resp_clear = client.post(
            f"/internal/content-ratings/{self.art_moss.artifact_id}/clear",
            json={"note": "误标撤销"},
            headers={**headers, "X-User-Id": "user-mengsk"},
        )
        self.assertEqual(resp_clear.status_code, 200)
        self.assertEqual(resp_clear.json()["rating"], "general")
        self.assertEqual(resp_clear.json()["source"], "human")
        self.assertTrue(resp_clear.json()["human_cleared"])

        # Now MCP list and status see both tasks again
        restored_list = json.loads(server_mod.image_list(project_id="proj-a").content[0].text)
        self.assertEqual(restored_list["total"], 2)
        status_res = json.loads(server_mod.image_status(self.task_moss.task_id).content[0].text)
        self.assertEqual(status_res["status"], "completed")

    def test_task_card_three_dots_actions_title_rating_and_delete(self) -> None:
        headers = {"Authorization": "Bearer test-secret-token", "X-User-Id": "user-mengsk"}
        client = TestClient(server_mod.app)

        # Dashboard HTML contains three-dots menu and no static card-action-text 查看
        dash_resp = client.get("/dashboard")
        self.assertEqual(dash_resp.status_code, 200)
        self.assertIn("data-menu-toggle", dash_resp.text)
        self.assertIn('data-task-action="edit"', dash_resp.text)
        self.assertIn('data-task-action="toggle-nsfw"', dash_resp.text)
        self.assertIn('data-task-action="delete"', dash_resp.text)
        self.assertNotIn('<span class="card-action-text">查看</span>', dash_resp.text)

        # 1. Edit task title
        title_resp = client.post(
            f"/api/tasks/{self.task_gen.task_id}/title",
            json={"title": "重命名后的山景定妆图"},
            headers=headers,
        )
        self.assertEqual(title_resp.status_code, 200)
        self.assertEqual(title_resp.json()["title"], "重命名后的山景定妆图")

        list_resp = client.get("/api/tasks?project_id=proj-a", headers=headers)
        gen_item = next(t for t in list_resp.json()["tasks"] if t["task_id"] == self.task_gen.task_id)
        self.assertEqual(gen_item["title"], "重命名后的山景定妆图")
        self.assertEqual(gen_item["request_params"]["prompt"], "sunny mountain")

        # 2. Mark task as NSFW via card action endpoint
        mark_resp = client.post(
            f"/api/tasks/{self.task_moss.task_id}/content-rating",
            json={"action": "mark", "reason": "manual_mark_nsfw"},
            headers=headers,
        )
        self.assertEqual(mark_resp.status_code, 200)
        self.assertEqual(mark_resp.json()["content_rating"]["rating"], "nsfw")

        # Hidden when X-VF-Include-NSFW is not set
        hidden_resp = client.get("/api/tasks?project_id=proj-a", headers=headers)
        self.assertEqual(hidden_resp.json()["total"], 1)

        # Visible when X-VF-Include-NSFW: 1
        shown_resp = client.get(
            "/api/tasks?project_id=proj-a",
            headers={**headers, "X-VF-Include-NSFW": "1"},
        )
        self.assertEqual(shown_resp.json()["total"], 2)

        # 3. Clear NSFW rating via card action endpoint
        clear_resp = client.post(
            f"/api/tasks/{self.task_moss.task_id}/content-rating",
            json={"action": "clear", "reason": "manual_clear_nsfw"},
            headers=headers,
        )
        self.assertEqual(clear_resp.status_code, 200)
        self.assertEqual(clear_resp.json()["content_rating"]["rating"], "general")

        after_clear = client.get("/api/tasks?project_id=proj-a", headers=headers)
        self.assertEqual(after_clear.json()["total"], 2)

        # 4. Delete task via POST /api/tasks/{id}/delete
        del_resp = client.post(
            f"/api/tasks/{self.task_moss.task_id}/delete",
            json={},
            headers=headers,
        )
        self.assertEqual(del_resp.status_code, 200)
        self.assertTrue(del_resp.json()["deleted"])

        after_del = client.get("/api/tasks?project_id=proj-a", headers=headers)
        self.assertEqual(after_del.json()["total"], 1)
        self.assertEqual([t["task_id"] for t in after_del.json()["tasks"]], [self.task_gen.task_id])


if __name__ == "__main__":
    unittest.main()
