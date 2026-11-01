import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from starlette.testclient import TestClient

from src import server
from src.artifacts import ArtifactStore
from src.channel_authority import CALLER
from src.content_rating import ContentRatingStore


PROJECT = "00000000-0000-7000-8000-000000000001"
OTHER = "00000000-0000-7000-8000-000000000002"


class ResultReadTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.artifacts = ArtifactStore(Path(self.tmp.name) / "artifacts")
        self.addCleanup(self.artifacts._thumb_executor.shutdown, wait=True)
        self.ratings = ContentRatingStore(Path(self.tmp.name) / "ratings.db")
        self.original = self.artifacts.create_from_bytes(PROJECT, "original.png", b"original")
        self.final = self.artifacts.create_from_bytes(PROJECT, "final.png", b"final")
        self.headers = {"Authorization": "Bearer result-read-test", "X-User-Id": "test-user",
                        "X-Organization-Id": "test-org"}
        for p in (patch.dict(os.environ, {"IMAGE_MCP_BEARER_TOKEN": "result-read-test"}),
                  patch.object(server, "artifacts", self.artifacts),
                  patch.object(server, "_active_ratings", return_value=self.ratings)):
            p.start()
            self.addCleanup(p.stop)
        self.auth = patch("src.result_read.authorize").start()
        self.addCleanup(patch.stopall)
        self.client = TestClient(server.app)
        self.addCleanup(self.client.close)

    def read(self, **kwargs):
        body = kwargs.pop("json", {"project_id": PROJECT, "artifact_id": self.final.artifact_id})
        response = self.client.post("/internal/results/read", json=body,
                                    headers=kwargs.pop("headers", self.headers), **kwargs)
        self.assertEqual(response.headers.get("cache-control"), "no-store")
        return response

    def mark(self, artifact):
        self.ratings.mark_rating(version_id=artifact.artifact_id, project_id=PROJECT,
                                 actor_user_id="test-user")

    def test_final_without_task_ignores_original_rating_and_passes_actor(self):
        self.mark(self.original)
        callers = []
        self.auth.side_effect = lambda *args, **kwargs: callers.append(CALLER.get())
        response = self.read()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {
            "artifact_id": self.final.artifact_id, "project_id": PROJECT,
            "sha256": self.final.sha256,
            "content_rating": {"rating": "unrated", "source": None, "human_cleared": False},
        })
        self.assertEqual(callers, [{"user_id": "test-user", "organization_id": "test-org"}])
        self.assertFalse(self.auth.call_args.kwargs["require_write"])

    def test_mark_include_and_clear_are_current_not_cached(self):
        self.assertEqual(self.read().status_code, 200)
        self.mark(self.final)
        hidden = self.read()
        missing = self.read(json={"project_id": PROJECT, "artifact_id": "art-0000000000000000"})
        self.assertEqual((hidden.status_code, hidden.json()), (404, missing.json()))
        included = self.read(headers={**self.headers, "X-VF-Include-NSFW": "1"})
        self.assertEqual(included.json()["content_rating"]["rating"], "nsfw")
        self.ratings.clear_rating(version_id=self.final.artifact_id, project_id=PROJECT,
                                  actor_user_id="test-user")
        self.assertEqual(self.read().json()["content_rating"],
                         {"rating": "general", "source": "human", "human_cleared": True})

    def test_other_project_cannot_resolve_artifact(self):
        self.assertEqual(self.read(json={"project_id": OTHER,
                                        "artifact_id": self.final.artifact_id}).status_code, 404)

    def test_closed_input_and_path_components(self):
        for body in ([], None, {}, {"project_id": "../escape", "artifact_id": self.final.artifact_id},
                     {"project_id": PROJECT, "artifact_id": "../escape"},
                     {"project_id": PROJECT, "artifact_id": self.final.artifact_id, "include_nsfw": True}):
            with self.subTest(body=body):
                self.assertEqual(self.read(json=body).status_code, 400)
        self.auth.assert_not_called()

    def test_query_token_and_identity_headers_are_not_service_auth(self):
        self.assertEqual(self.read(headers={"X-User-Id": "test-user"},
                                   params={"token": "result-read-test"}).status_code, 401)
        self.auth.assert_not_called()

    def test_authority_and_dependency_errors_are_not_hidden_or_visible(self):
        for error, status in (("verified_studio_identity_required", 401),
                              ("project_access_denied", 403),
                              ("project_authorization_unavailable", 503)):
            self.auth.side_effect = ValueError(error)
            self.assertEqual(self.read().status_code, status)

    def test_rating_store_failure_does_not_become_unrated(self):
        with patch.object(self.ratings, "get", side_effect=OSError("unavailable")):
            response = self.read()
        self.assertEqual(response.status_code, 503)
        self.assertNotIn("sha256", response.json())


if __name__ == "__main__":
    unittest.main()
