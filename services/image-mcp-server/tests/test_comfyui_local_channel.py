from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
from PIL import Image

from src.artifacts import ArtifactStore
from src.channel_authority import CALLER
from src.providers.codex import CodexProvider
from src.providers.comfyui import ComfyUIWorkflowProvider, validate_internal_origin
from src.providers.gemini import GEMINI_MODEL_CATALOG, GeminiProvider
from src.queue import TaskJob, TaskQueueManager
from src.tasks import TaskStore

ORG_A = "019a8700-0000-7000-8000-000000000001"
ORG_B = "019a8700-0000-7000-8000-000000000002"
USER_A = "019a8700-0000-7000-8000-000000000011"
PROJECT_A = "019a8700-0000-7000-8000-000000000101"
STUDIO_IN_AID = "019a8700-0000-7000-8000-000000000201"
STUDIO_OUT_AID = "019a8700-0000-7000-8000-000000000202"
REMOTE_TASK_ID = "019a8700-0000-7000-8000-000000000301"


def _make_png_bytes(color: tuple[int, int, int] = (32, 160, 120)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), color=color).save(buf, format="PNG")
    return buf.getvalue()


def _load_golden_spec() -> dict:
    repo_root = Path(__file__).resolve().parents[4]
    golden_path = repo_root / "docs" / "design" / "app" / "comfyui" / "fixtures" / "golden_sdxl_lora_spec.v1.json"
    raw = json.loads(golden_path.read_text(encoding="utf-8"))
    return raw["spec"] if isinstance(raw, dict) and isinstance(raw.get("spec"), dict) else raw


class ComfyUILocalChannelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.artifacts = ArtifactStore(self.root / "artifacts", prewarm_on_create=False)
        self.tasks = TaskStore(self.root / "tasks.db")
        self.golden_spec = _load_golden_spec()
        self.published_active = True
        self.output_png = _make_png_bytes((20, 180, 90))
        self.output_sha = hashlib.sha256(self.output_png).hexdigest()
        self.uploaded_inputs: dict[str, bytes] = {}
        self.headless_calls: list[dict] = []
        self.poll_count = 0

        def handler(request: httpx.Request) -> httpx.Response:
            host = request.url.host
            path = request.url.path
            auth = request.headers.get("Authorization", "")

            if host == "comfyui-manager.verdantflare-app.svc.cluster.local":
                if auth != "Bearer comfy-internal-secret":
                    return httpx.Response(401, json={"error": "UNAUTHORIZED"})
                if path == "/internal/v1/comfyui/published-channels":
                    body = json.loads(request.content.decode("utf-8"))
                    if set(body.keys()) != {"organization_id", "mount_channel"}:
                        return httpx.Response(400, json={"error": "INVALID_PARAMETERS"})
                    if body["organization_id"] != ORG_A or body["mount_channel"] != "image" or not self.published_active:
                        return httpx.Response(200, json={"items": []})
                    return httpx.Response(
                        200,
                        json={
                            "items": [
                                {
                                    "published_api_id": "019a8700-0000-7000-8000-000000000999",
                                    "catalog_model_id": "comfyui:sdxl-lora-img2img@1.0.0",
                                    "engine": "comfyui",
                                    "channel_id": "local",
                                    "organization_id": ORG_A,
                                    "project_id": PROJECT_A,
                                    "endpoint_slug": "sdxl-lora-img2img",
                                    "version": "1.0.0",
                                    "display_name": "SDXL Lightning LoRA 图生图",
                                    "output_modality": "image",
                                    "mount_channel": "image",
                                    "reproducibility_level": "strict",
                                    "seed_policy": "caller_or_random",
                                    "vram_budget_mib": 15629,
                                    "source_badge": "本地",
                                    "spec": self.golden_spec,
                                }
                            ]
                        },
                    )
                if path == "/internal/apps/comfyui-api/call":
                    for hdr in ("X-Organization-Id", "X-User-Id", "X-Request-Id"):
                        if not request.headers.get(hdr):
                            return httpx.Response(400, json={"error": f"MISSING_{hdr}"})
                    call_body = json.loads(request.content.decode("utf-8"))
                    self.headless_calls.append(call_body)
                    tname = call_body["name"]
                    targs = call_body["arguments"]
                    if tname == "api.invoke":
                        if not self.published_active:
                            return httpx.Response(
                                404,
                                json={
                                    "isError": True,
                                    "structuredContent": {"error": {"code": "PUBLISHED_API_NOT_FOUND"}},
                                },
                            )
                        return httpx.Response(
                            200,
                            json={
                                "task_id": REMOTE_TASK_ID,
                                "project_id": targs["project_id"],
                                "state": "running",
                                "execution_pool": "headless_api",
                                "result": None,
                            },
                        )
                    if tname == "task.get":
                        self.poll_count += 1
                        if self.poll_count < 2:
                            return httpx.Response(
                                200,
                                json={
                                    "task_id": REMOTE_TASK_ID,
                                    "project_id": targs["project_id"],
                                    "state": "running",
                                    "result": None,
                                },
                            )
                        return httpx.Response(
                            200,
                            json={
                                "task_id": REMOTE_TASK_ID,
                                "project_id": targs["project_id"],
                                "state": "succeeded",
                                "result": [
                                    {
                                        "artifact_id": STUDIO_OUT_AID,
                                        "version_id": "v1",
                                        "node_id": "9",
                                        "filename": "sdxl_out.png",
                                        "mime_type": "image/png",
                                        "size_bytes": len(self.output_png),
                                        "sha256": self.output_sha,
                                    }
                                ],
                            },
                        )
                    if tname == "task.cancel":
                        return httpx.Response(
                            200,
                            json={
                                "task_id": REMOTE_TASK_ID,
                                "project_id": targs["project_id"],
                                "state": "cancelled",
                                "cancel_requested": True,
                            },
                        )

            if host == "artifact.verdantflare-station.svc.cluster.local":
                if auth != "Bearer studio-artifact-secret":
                    return httpx.Response(401, json={"error": "UNAUTHORIZED"})
                if path == "/v2/artifacts/uploads" and request.method == "POST":
                    return httpx.Response(
                        200,
                        json={
                            "upload_id": "upl-1",
                            "artifact_id": STUDIO_IN_AID,
                            "version_id": "v1",
                        },
                    )
                if path == "/v2/artifacts/uploads/upl-1/content" and request.method == "PUT":
                    self.uploaded_inputs["upl-1"] = request.content
                    return httpx.Response(200, json={"ok": True})
                if path == "/v2/artifacts/uploads/upl-1/commit" and request.method == "POST":
                    meta = json.loads(request.content.decode("utf-8"))
                    raw = self.uploaded_inputs["upl-1"]
                    assert hashlib.sha256(raw).hexdigest() == meta["sha256"]
                    return httpx.Response(200, json={"artifact_id": STUDIO_IN_AID, "version_id": "v1"})
                if path == f"/v2/artifacts/{STUDIO_OUT_AID}/content" and request.method == "GET":
                    if request.url.params.get("project_id") != PROJECT_A:
                        return httpx.Response(403, json={"error": "FORBIDDEN"})
                    return httpx.Response(
                        200,
                        content=self.output_png,
                        headers={"Content-Type": "image/png"},
                    )

            return httpx.Response(404, json={"error": "NOT_FOUND"})

        self.http_client = httpx.Client(transport=httpx.MockTransport(handler))
        self.comfyui = ComfyUIWorkflowProvider(
            origin="http://comfyui-manager.verdantflare-app.svc.cluster.local:8091",
            token="comfy-internal-secret",
            artifact_service_url="https://artifact.verdantflare-station.svc.cluster.local:8094",
            artifact_service_token="studio-artifact-secret",
            poll_interval_seconds=0.01,
            timeout_seconds=10.0,
            http_client=self.http_client,
        )
        self.queue = TaskQueueManager(
            tasks=self.tasks,
            artifacts=self.artifacts,
            codex_provider=CodexProvider(),
            gemini_provider=GeminiProvider(),
            google_authorize=lambda pid, _auth: {"organization_id": ORG_A, "user_id": USER_A},
            comfyui_provider=self.comfyui,
        )

    def tearDown(self) -> None:
        self.http_client.close()
        self.tmp.cleanup()

    def test_origin_allowlist_blocks_external_and_allows_cluster_dns(self) -> None:
        self.assertEqual(
            validate_internal_origin("http://comfyui-manager.verdantflare-app.svc.cluster.local:8091"),
            "http://comfyui-manager.verdantflare-app.svc.cluster.local:8091",
        )
        with self.assertRaisesRegex(ValueError, "comfyui_origin_not_allowed"):
            validate_internal_origin("https://api.example.com")
        with self.assertRaisesRegex(ValueError, "comfyui_origin_not_allowed"):
            validate_internal_origin("http://127.0.0.1:8091", allow_loopback=False)
        with self.assertRaisesRegex(ValueError, "comfyui_origin_invalid"):
            validate_internal_origin("http://user:pass@comfyui-manager.svc.cluster.local:8091")
        with self.assertRaisesRegex(ValueError, "comfyui_origin_invalid"):
            validate_internal_origin("http://comfyui-manager.svc.cluster.local:8091/extra/path")

    def test_dynamic_catalog_org_isolation_and_chinese_local_badge(self) -> None:
        gemini_snapshot_before = tuple(dict(x) for x in GEMINI_MODEL_CATALOG)

        tok = CALLER.set({"organization_id": ORG_A, "user_id": USER_A})
        try:
            caps = self.queue.capabilities()
        finally:
            CALLER.reset(tok)

        comfy_cap = next(e for e in caps["engines"] if e["engine"] == "comfyui")
        self.assertEqual(comfy_cap["channel"], "local")
        self.assertEqual(comfy_cap["channel_name"], "本地")
        self.assertEqual(comfy_cap["source_badge"], "本地")
        self.assertEqual(comfy_cap["supported_models"], ["comfyui:sdxl-lora-img2img@1.0.0"])
        self.assertEqual(comfy_cap["model_catalog"][0]["source_badge"], "本地")
        self.assertEqual(GEMINI_MODEL_CATALOG, gemini_snapshot_before)

        # Cross-org caller sees zero ORG_A workflows
        tok_b = CALLER.set({"organization_id": ORG_B, "user_id": USER_A})
        try:
            caps_b = self.queue.capabilities()
        finally:
            CALLER.reset(tok_b)
        comfy_b = next(e for e in caps_b["engines"] if e["engine"] == "comfyui")
        self.assertEqual(comfy_b["supported_models"], [])

        # Malformed caller organization_id fails closed even if STATION_ORGANIZATION_ID is set
        with patch.dict(os.environ, {"STATION_ORGANIZATION_ID": ORG_A}, clear=False):
            tok_bad = CALLER.set({"organization_id": "not-a-uuid7", "user_id": USER_A})
            try:
                self.assertEqual(self.comfyui.discover_models(), [])
            finally:
                CALLER.reset(tok_bad)

    def test_end_to_end_execution_artifact_bridge_and_unpublish_drain(self) -> None:
        ref_bytes = _make_png_bytes((120, 40, 200))
        ref_art = self.artifacts.create_from_bytes(
            project_id=PROJECT_A,
            filename="ref.png",
            data=ref_bytes,
            media_type="image/png",
        )

        tok = CALLER.set({"organization_id": ORG_A, "user_id": USER_A})
        try:
            prepared = self.queue.prepare(
                "comfyui",
                PROJECT_A,
                {
                    "action": "edit",
                    "model": "comfyui:sdxl-lora-img2img@1.0.0",
                    "prompt": "cinematic neon alley portrait",
                    "negative_prompt": "blurry, watermark",
                    "source_artifact_id": ref_art.artifact_id,
                    "denoise": 0.55,
                    "steps": 4,
                    "seed": 42,
                    "sampler": "euler",
                },
            )
        finally:
            CALLER.reset(tok)

        rec = self.tasks.create(
            project_id=PROJECT_A,
            idempotency_key="idem-comfy-e2e-1",
            engine="comfyui",
            model=prepared["model"],
            request_params=prepared,
        )
        self.assertTrue(self.tasks.claim_task(rec.task_id))

        # Simulate initial submit + unpublish BEFORE task finishes polling!
        # Override poll behavior so unpublish happens while remote task is already running.
        original_poll = self.comfyui._call_headless_tool

        def intercept_call(principal, tool_name, arguments):
            res = original_poll(principal, tool_name, arguments)
            if tool_name == "api.invoke":
                # Workflow is immediately unpublished right after submission!
                self.published_active = False
            return res

        with patch.object(self.comfyui, "_call_headless_tool", side_effect=intercept_call):
            self.queue._execute_comfyui(
                TaskJob(
                    task_id=rec.task_id,
                    project_id=PROJECT_A,
                    engine="comfyui",
                    action="edit",
                    params=prepared,
                )
            )
            self.tasks.update_status(rec.task_id, "completed")

        # 1. In-flight task drained to completion and verified Studio artifact SHA-256
        done = self.tasks.get(rec.task_id)
        self.assertEqual(done.status, "completed")
        self.assertIsNotNone(done.artifact_id)
        out_art = self.artifacts.get(done.artifact_id, PROJECT_A)
        self.assertEqual(out_art.sha256, self.output_sha)
        self.assertEqual(out_art.metadata["source_badge"], "本地")
        self.assertEqual(out_art.metadata["comfyui_task_id"], REMOTE_TASK_ID)
        self.assertEqual(self.uploaded_inputs["upl-1"], ref_bytes)

        invoke_call = next(c for c in self.headless_calls if c["name"] == "api.invoke")
        self.assertEqual(invoke_call["arguments"]["params"]["ref_image"], STUDIO_IN_AID)
        self.assertEqual(invoke_call["arguments"]["params"]["sampler_name"], "euler")
        self.assertEqual(invoke_call["arguments"]["params"]["seed"], 42)

        # 2. After unpublish, catalog removes workflow & new prepare() calls are immediately rejected
        tok2 = CALLER.set({"organization_id": ORG_A, "user_id": USER_A})
        try:
            self.assertEqual(self.comfyui.capability()["supported_models"], [])
            with self.assertRaisesRegex(ValueError, "comfyui_workflow_unavailable"):
                self.queue.prepare(
                    "comfyui",
                    PROJECT_A,
                    {
                        "action": "generate",
                        "model": "comfyui:sdxl-lora-img2img@1.0.0",
                        "prompt": "rejected after unpublish",
                    },
                )
        finally:
            CALLER.reset(tok2)

    def test_exposed_inputs_rejects_unexposed_params_and_forwards_cancel(self) -> None:
        ref_art = self.artifacts.create_from_bytes(
            project_id=PROJECT_A,
            filename="ref2.png",
            data=_make_png_bytes((60, 80, 100)),
            media_type="image/png",
        )
        tok = CALLER.set({"organization_id": ORG_A, "user_id": USER_A})
        try:
            with self.assertRaisesRegex(ValueError, "unsupported_workflow_parameter"):
                self.queue.prepare(
                    "comfyui",
                    PROJECT_A,
                    {
                        "action": "generate",
                        "model": "comfyui:sdxl-lora-img2img@1.0.0",
                        "prompt": "valid prompt",
                        "source_artifact_id": ref_art.artifact_id,
                        "workflow_params": {"injected_node_override": "forbidden"},
                    },
                )
            with self.assertRaisesRegex(ValueError, "missing_required_workflow_input:ref_image"):
                self.queue.prepare(
                    "comfyui",
                    PROJECT_A,
                    {
                        "action": "generate",
                        "model": "comfyui:sdxl-lora-img2img@1.0.0",
                        "prompt": "missing ref image",
                    },
                )
            prepared = self.queue.prepare(
                "comfyui",
                PROJECT_A,
                {
                    "action": "edit",
                    "model": "comfyui:sdxl-lora-img2img@1.0.0",
                    "prompt": "cancel me",
                    "source_artifact_id": ref_art.artifact_id,
                },
            )
        finally:
            CALLER.reset(tok)

        rec = self.tasks.create(
            project_id=PROJECT_A,
            idempotency_key="idem-comfy-cancel-1",
            engine="comfyui",
            model=prepared["model"],
            request_params=prepared,
        )
        journal = self.queue._comfyui_journal(rec.task_id)
        journal.write_text(json.dumps({"remote_task_id": REMOTE_TASK_ID}), encoding="utf-8")
        self.assertTrue(self.queue.cancel_comfyui_task(rec))
        cancel_call = next(c for c in self.headless_calls if c["name"] == "task.cancel")
        self.assertEqual(cancel_call["arguments"]["task_id"], REMOTE_TASK_ID)


if __name__ == "__main__":
    unittest.main()
