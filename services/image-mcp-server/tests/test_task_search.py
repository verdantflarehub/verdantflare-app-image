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
            for query in [
                "limit=10000",
                "limit=no",
                "offset=-1",
                "offset=9223372036854775808",
                "offset=" + ("9" * 40),
                "q=" + "a" * 501,
            ]:
                self.assertEqual(call(query).status_code, 400, query)
            store.list_tasks.assert_not_called()
            self.assertEqual(call("q=imgtask-needle&limit=50&offset=0").status_code, 200)
            self.assertEqual(store.list_tasks.call_args.kwargs["q"], "imgtask-needle")

            # SQLite 64-bit signed upper bound (2^63 - 1 = 9223372036854775807) is valid
            store.list_tasks.reset_mock()
            max_i64_query = "before_cursor=9223372036854775807%3A2026-11-15T00%3A00%3A00%2B00%3A00&limit=50&offset=0"
            self.assertEqual(call(max_i64_query).status_code, 200)
            self.assertEqual(
                store.list_tasks.call_args.kwargs["before_cursor"],
                ("2026-11-15T00:00:00+00:00", 9223372036854775807),
            )

            # Invalid before_cursor (including 2^63 overflow, long digits, leading zero, empty) or non-zero offset -> 400
            store.list_tasks.reset_mock()
            invalid_cursors = [
                "before_cursor=",
                "before_cursor=10%3A2026-11-15T00%3A00%3A00%2B00%3A00&offset=50",
                "before_cursor=0%3A2026-11-15T00%3A00%3A00%2B00%3A00",
                "before_cursor=010%3A2026-11-15T00%3A00%3A00%2B00%3A00",
                "before_cursor=-3%3A2026-11-15T00%3A00%3A00%2B00%3A00",
                "before_cursor=9223372036854775808%3A2026-11-15T00%3A00%3A00%2B00%3A00",
                "before_cursor=" + ("9" * 30) + "%3A2026-11-15T00%3A00%3A00%2B00%3A00",
                "before_cursor=abc%3A2026-11-15T00%3A00%3A00%2B00%3A00",
                "before_cursor=10%3Anot-an-iso-date",
                "before_cursor=10",
                "before_cursor=%20%20",
                "before_cursor=10%3A2026-11-15T00%3A00%3A00%00%2B00%3A00",
                "before_cursor=" + ("1" * 250) + "%3A2026-11-15T00%3A00%3A00%2B00%3A00",
            ]
            for query in invalid_cursors:
                self.assertEqual(call(query).status_code, 400, query)
            store.list_tasks.assert_not_called()

    def test_keyset_cursor_total_order_and_concurrent_mutations(self):
        import asyncio
        import json
        from types import SimpleNamespace
        from unittest.mock import Mock, patch
        from urllib.parse import quote
        from starlette.requests import Request
        from src.dashboard import api_get_task, api_list_tasks

        with tempfile.TemporaryDirectory() as tmp:
            store = TaskStore(Path(tmp) / "tasks.db")
            store.ensure_ready()
            fixed_ts = "2026-11-15T04:00:00+00:00"
            older_ts = "2026-11-15T03:00:00+00:00"
            created_ids: list[str] = []
            for i in range(6):
                rec = store.create(
                    project_id="cursor-proj",
                    idempotency_key=f"same-ts-{i}",
                    engine="gemini",
                    request_params={"prompt": f"item-{i}"},
                )
                created_ids.append(rec.task_id)
            with store._get_conn() as conn:
                for task_id in created_ids[:4]:
                    conn.execute("UPDATE tasks SET created_at = ? WHERE task_id = ?", (fixed_ts, task_id))
                for task_id in created_ids[4:]:
                    conn.execute("UPDATE tasks SET created_at = ? WHERE task_id = ?", (older_ts, task_id))
                conn.commit()

            # Same-second rows must sort strictly by rowid DESC (created_ids[3], [2], [1], [0], then [5], [4])
            expected_order = [
                created_ids[3],
                created_ids[2],
                created_ids[1],
                created_ids[0],
                created_ids[5],
                created_ids[4],
            ]
            all_rows, total = store.list_tasks(project_id="cursor-proj", limit=10)
            self.assertEqual(total, 6)
            self.assertEqual([r.task_id for r in all_rows], expected_order)
            for r in all_rows:
                self.assertNotIn("sort_rowid", r.to_dict())
                self.assertTrue(r.sort_cursor and ":" in r.sort_cursor)
                single = store.get(r.task_id)
                self.assertEqual(single.sort_cursor, r.sort_cursor)

            state = SimpleNamespace(
                tasks=store,
                ratings=None,
                artifacts=Mock(),
                queue_manager=Mock(),
            )
            state.queue_manager.capabilities.return_value = {}

            def call_list(query: str) -> dict:
                req = Request({
                    "type": "http",
                    "method": "GET",
                    "path": "/api/tasks",
                    "query_string": query.encode(),
                    "headers": [],
                    "app": SimpleNamespace(state=state),
                })
                resp = asyncio.run(api_list_tasks(req))
                self.assertEqual(resp.status_code, 200)
                return json.loads(resp.body.decode("utf-8"))

            def call_get(task_id: str) -> dict:
                req = Request({
                    "type": "http",
                    "method": "GET",
                    "path": f"/api/tasks/{task_id}",
                    "path_params": {"task_id": task_id},
                    "query_string": b"",
                    "headers": [],
                    "app": SimpleNamespace(state=state),
                })
                resp = asyncio.run(api_get_task(req))
                self.assertEqual(resp.status_code, 200)
                return json.loads(resp.body.decode("utf-8"))

            # Fetch page 1 (limit=2): gets [created_ids[3], created_ids[2]] (both at fixed_ts!)
            with patch("src.dashboard.check_auth", return_value=True):
                p1 = call_list("project_id=cursor-proj&limit=2&offset=0")
                self.assertTrue(p1["has_more"])
                self.assertEqual([t["task_id"] for t in p1["tasks"]], expected_order[:2])
                self.assertEqual(p1["next_cursor"], p1["tasks"][-1]["sort_cursor"])
                self.assertEqual(call_get(expected_order[0])["sort_cursor"], p1["tasks"][0]["sort_cursor"])

                # Mutate head concurrently BEFORE fetching page 2: insert 2 brand-new rows and delete expected_order[0]
                head_new_1 = store.create(
                    project_id="cursor-proj",
                    idempotency_key="head-new-1",
                    engine="gemini",
                    request_params={"prompt": "brand-new-1"},
                )
                head_new_2 = store.create(
                    project_id="cursor-proj",
                    idempotency_key="head-new-2",
                    engine="gemini",
                    request_params={"prompt": "brand-new-2"},
                )
                store.delete_task(expected_order[0])
                self.assertNotEqual(head_new_1.task_id, head_new_2.task_id)

                # Keyset pagination with before_cursor must still return exact remaining rows without duplicates or skips
                p2 = call_list(f"project_id=cursor-proj&limit=2&offset=0&before_cursor={quote(p1['next_cursor'], safe='')}")
                self.assertTrue(p2["has_more"])
                self.assertEqual([t["task_id"] for t in p2["tasks"]], expected_order[2:4])

                p3 = call_list(f"project_id=cursor-proj&limit=2&offset=0&before_cursor={quote(p2['next_cursor'], safe='')}")
                self.assertFalse(p3["has_more"])
                self.assertIsNone(p3["next_cursor"])
                self.assertEqual([t["task_id"] for t in p3["tasks"]], expected_order[4:6])

                # Real SQLite binding at MAX_SQLITE_INT (9223372036854775807) succeeds without OverflowError,
                # while MAX_SQLITE_INT + 1 (9223372036854775808) returns HTTP 400 cleanly.
                max_bound_cursor = quote(f"9223372036854775807:{fixed_ts}", safe="")
                p_max = call_list(f"project_id=cursor-proj&limit=2&offset=0&before_cursor={max_bound_cursor}")
                self.assertEqual(len(p_max["tasks"]), 2)
                overflow_req = Request({
                    "type": "http",
                    "method": "GET",
                    "path": "/api/tasks",
                    "query_string": f"project_id=cursor-proj&limit=2&offset=0&before_cursor={quote(f'9223372036854775808:{fixed_ts}', safe='')}".encode(),
                    "headers": [],
                    "app": SimpleNamespace(state=state),
                })
                self.assertEqual(asyncio.run(api_list_tasks(overflow_req)).status_code, 400)
