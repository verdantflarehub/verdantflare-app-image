import hashlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from uuid import UUID, uuid4

import httpx
from PIL import Image

from src.artifacts import ArtifactStore
from src.execution_proof import proof
from src.providers.google_flow import FlowChannel, image_info, normalize
from src.queue import TaskJob, TaskQueueManager
from src.tasks import TaskStore


class FlowTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        output = io.BytesIO()
        Image.new("RGB", (8, 8)).save(output, format="PNG")
        self.raw = output.getvalue()
        self.digest = hashlib.sha256(self.raw).hexdigest()
        self.actor = dict(user_id=str(uuid4()), organization_id=str(uuid4()))
        self.project = str(uuid4())
        self.call = uuid4().hex
        self.token = uuid4().hex
        self.client = httpx.Client(transport=httpx.MockTransport(self.handle))
        self.addCleanup(self.client.close)
        self.url = "https://flow.invalid"
        self.channel = FlowChannel(self.url, self.token, self.client)
        self.uploads = {}
        self.versions = {}
        self.key = ""
        self.submissions = 0
        self.lost = None
        self.enterContext(patch.dict(os.environ, IMAGE_ARTIFACT_SERVICE_URL="https://artifact.invalid",
                                     IMAGE_ARTIFACT_SERVICE_TOKEN=uuid4().hex))

    def manager(self):
        return TaskQueueManager(TaskStore(self.root / "tasks.db"), ArtifactStore(self.root / "artifacts"),
                                Mock(), Mock(), flow_channel=self.channel, google_authorize=Mock(return_value=self.actor))

    def handle(self, req):
        path = req.url.path
        self.assertEqual(req.headers["x-organization-id"], self.actor["organization_id"])
        self.assertEqual(req.headers["x-user-id"], self.actor["user_id"])
        if path == "/v1/models":
            return httpx.Response(200, json={"models": {"image": ["nano2"]}})
        if path == "/v1/calls":
            body = json.loads(req.content)
            self.assertEqual(body["parameters"], {"prompt": "square", "aspect": "1:1"})
            self.assertEqual(body["capability"], "image")
            self.key = body["idempotency_key"]
            self.submissions += 1
            if self.lost == "submit":
                self.lost = None
                raise httpx.ReadError("response lost")
        if path in ("/v1/calls", "/v1/calls/by-idempotency/" + self.key):
            return httpx.Response(200, json=dict(call_id=self.call, request_id=self.key, accepted=True, state="succeeded"))
        if path.endswith("/result"):
            return httpx.Response(200, content=self.raw, headers={"content-type":"image/png","x-content-sha256":self.digest})
        if path == "/v2/artifacts/uploads":
            step = "prepare"
            body = json.loads(req.content)
            self.assertEqual(body["source"], dict(kind="task_output", service_id="image", project_id=self.project,
                                                   run_id=self.key, task_id=str(UUID(self.call))))
            upload = self.uploads.setdefault(body["write_id"], str(uuid4()))
            self.source = body["source"]
            value = dict(upload_id=upload)
        elif path.startswith("/v2/artifacts/uploads/"):
            upload = path.split("/")[4]
            self.assertIn(upload, self.uploads.values())
            if req.method == "GET":
                return httpx.Response(200, json={"state":"committed" if upload in self.versions else "prepared"})
            if req.method == "PUT":
                step, value = "put", {}
                self.assertEqual(req.content, self.raw)
                self.stored = req.content
            else:
                step = "commit"
                self.assertEqual(self.stored, self.raw)
                value = self.versions.setdefault(upload, dict(artifact_id=str(uuid4()),store_id=str(uuid4()),
                    version_id=str(uuid4()),source=self.source,sha256=self.digest,size=len(self.raw),mime="image/png",
                    created_by=self.actor["user_id"],organization_id=self.actor["organization_id"]))
        elif path.endswith("/content") and req.method == "GET":
            version, = self.versions.values()
            self.assertEqual(path.split("/")[3], version["version_id"])
            return httpx.Response(200, content=self.stored)
        else:
            self.fail(path)
        if self.lost == step:
            self.lost = None
            raise httpx.ReadError("response lost")
        return httpx.Response(200, json=value)

    def recover_after(self, step):
        manager = self.manager()
        params = manager.prepare("google-flow", self.project, dict(prompt="square",action="generate"))
        task = manager.tasks.create(self.project,"key","google-flow",params)
        manager.tasks.claim_task(task.task_id)
        job = TaskJob(task.task_id,self.project,"google-flow","generate",params)
        self.lost = step
        with self.assertRaises(httpx.ReadError):
            manager._execute_channel(job)
        manager = self.manager()
        manager.tasks.recover_hanging_tasks()
        self.assertFalse(manager.tasks.get_queued_tasks())
        manager._recover_channel_unknown(manager.tasks.get(task.task_id))
        result = proof(manager.tasks.get(task.task_id),manager.artifacts)
        self.assertEqual(result["call_id"], self.call)
        self.assertEqual(result["outputs"][0]["sha256"], self.digest)
        self.assertEqual((self.submissions,len(self.uploads),len(self.versions)), (1,1,1))

    def test_discovery_uses_underscore_url(self):
        from src.channel_discovery_client import ChannelDiscoveryClient
        with patch.dict(os.environ, {"CHANNEL_DISCOVERY_SCOPE_ID": "test",
                                     "IMAGE_GOOGLE_FLOW_CHANNEL_URL": self.url}, clear=True):
            with patch.object(ChannelDiscoveryClient, "__init__", return_value=None) as constructor:
                ChannelDiscoveryClient.from_environment()
        self.assertEqual(constructor.call_args.args[-1]["google-flow"], self.url)

    def test_restart_after_lost_submission_recovers_original_call(self):
        self.recover_after("submit")

    def test_normal_mcp_entry_and_flow_capabilities(self):
        import asyncio
        from src import server
        manager = self.manager()
        with patch.object(server, "queue_manager", manager), patch.object(server, "tasks", manager.tasks):
            async def create():
                await server.image_create(engine="google-flow", project_id=self.project,
                                          idempotency_key="normal-entry", prompt="square", aspect_ratio="1:1")
                return await manager._queue.get()
            job = asyncio.run(create())
        self.assertEqual(job.params["model"], "nano2")
        self.assertNotIn("budget_limit", job.params)
        self.assertTrue(manager.tasks.claim_task(job.task_id))
        manager._execute_channel(job)
        self.assertEqual(manager.tasks.get(job.task_id).status, "completed")
        caps = next(e for e in manager.capabilities()["engines"] if e["engine"] == "google-flow")
        self.assertTrue(caps["enabled"])
        self.assertEqual(caps["budget_currency"], "credits")

    def test_discovery_checks_flow_instance_and_lease(self):
        from datetime import datetime, timezone
        from src.channel_discovery import ChannelDirectory
        now = datetime.now(timezone.utc)
        directory = ChannelDirectory("test", {"google-flow": self.url})
        base = directory.prefix + "google-flow/"
        rows = [dict(key=base + "status", lease=1, mod_revision=1,
                     value=dict(state="enabled", observed_at=now.isoformat())),
                dict(key=base + "instances/one", lease=1, mod_revision=1,
                     value=dict(schema_version=1, generation=1, service_identity="google-flow",
                                execution_url=self.url, capabilities=[dict(capability_id="image",
                                execution_state="adapter_connected_unverified")]))]
        directory.replace(rows, 1)
        self.assertEqual(directory.resolve("google-flow", "image", now), self.url)
        rows[1]["lease"] = 2
        directory.replace(rows, 2)
        self.assertIsNone(directory.resolve("google-flow", "image", now))

    def test_lost_prepare(self):
        self.recover_after("prepare")

    def test_lost_put(self):
        self.recover_after("put")

    def test_lost_commit(self):
        self.recover_after("commit")

    def test_reject_invalid_media_and_paths(self):
        self.assertEqual(image_info(self.raw, "image/png", self.digest), (8,8))
        for raw, mime, digest in ((self.raw, "image/jpeg", self.digest), (self.raw, "image/png", "0"*64),
                                  (b"broken", "image/png", hashlib.sha256(b"broken").hexdigest())):
            with self.subTest(mime=mime), self.assertRaises((OSError,ValueError)):
                image_info(raw, mime, digest)
        with self.assertRaisesRegex(ValueError, "invalid_reference"):
            normalize(dict(prompt="edit", action="edit", source_artifact_id="../record"))
