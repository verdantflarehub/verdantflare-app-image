import tempfile
import unittest
from pathlib import Path
from src.tasks import TaskStore
from src.content_rating import ContentRatingStore


class TaskSearchTests(unittest.TestCase):
    def test_search_before_pagination_and_literal_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = TaskStore(Path(tmp) / "tasks.db")
            ratings = ContentRatingStore(store.db_path, domain="image")
            target = store.create(project_id="special-project", idempotency_key="target", engine="gemini",
                                  request_params={"prompt": "x" * 400 + "尾部关键词", "title": "100%_TITLE"})
            for i in range(55):
                store.create(project_id="other", idempotency_key=str(i), engine="gemini", request_params={"prompt": "ordinary"})
            for query in [target.task_id.upper(), "special-project", "尾部关键词", "%_title"]:
                rows, total = store.list_tasks(q=query, limit=1)
                self.assertEqual(total, 1)
                self.assertEqual(rows[0].task_id, target.task_id)
                self.assertEqual(store.list_tasks(q=query, limit=1, offset=1), ([], 1))
            self.assertEqual(store.list_tasks(q="' OR 1=1 --"), ([], 0))
            self.assertEqual(store.list_tasks(q=target.task_id, project_id="other"), ([], 0))
            ratings.rate_content(version_id=target.task_id, task_id=target.task_id,
                                 project_id="special-project", rating="nsfw", reason="test")
            self.assertEqual(store.list_tasks(q=target.task_id), ([], 0))
            self.assertEqual(store.list_tasks(q=target.task_id, include_nsfw=True)[1], 1)

    def test_http_query_validation_and_forwarding(self):
        import asyncio
        import os
        from types import SimpleNamespace
        from unittest.mock import Mock, patch
        from starlette.requests import Request
        from src.dashboard import api_list_tasks
        store = Mock(db_path=Path("unused"))
        store.list_tasks.return_value = ([], 0)
        state = SimpleNamespace(tasks=store, ratings=SimpleNamespace(db_path=store.db_path),
                                artifacts=Mock(), queue_manager=Mock())
        state.queue_manager.capabilities.return_value = {}
        def call(query):
            req = Request({"type": "http", "method": "GET", "path": "/api/tasks", "query_string": query.encode(),
                           "headers": [], "app": SimpleNamespace(state=state)})
            return asyncio.run(api_list_tasks(req))
        with patch("src.dashboard.check_auth", return_value=True):
            for query in ["limit=10000", "limit=no", "offset=-1", "q=" + "a" * 501]:
                self.assertEqual(call(query).status_code, 400)
            store.list_tasks.assert_not_called()
            self.assertEqual(call("q=imgtask-needle&limit=50&offset=0").status_code, 200)
            self.assertEqual(store.list_tasks.call_args.kwargs["q"], "imgtask-needle")
