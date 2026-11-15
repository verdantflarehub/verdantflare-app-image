"""P1b unit tests for image-mcp-server :5056 GPU Scheduler integration (moss-image & face-fusion)."""
from __future__ import annotations

import asyncio
import base64
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock

import httpx
from PIL import Image

from src.artifacts import ArtifactStore
from src.faceswap import handle_faceswap
from src.gpu_lease import GpuCoordinator, MemoryApi, SchedulerError, SchedulerLeaseClient
from src.providers.moss import DEFAULT_MOSS_MODEL, MossProvider
from src.queue import TaskJob, TaskQueueManager
from src.tasks import TaskStore

ALLOWED_RECLAIM_MODES = {"", "warm_idle_settled", "epoch_unload", "pod_deleted", "abort_unstarted"}


def _png_bytes(w: int = 64, h: int = 64) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (100, 140, 180)).save(buf, format="PNG")
    return buf.getvalue()


class ImageSchedulerP1bTest(unittest.TestCase):
    def test_moss_cold_start_binds_target_reconciles_gpu_and_uses_warm_idle_settled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = ArtifactStore(root / "artifacts")
            tasks = TaskStore(root / "tasks.db")
            artifacts.ensure_ready()
            tasks.ensure_ready()

            png = _png_bytes(64, 64)
            b64_png = base64.b64encode(png).decode("ascii")
            events: list[str] = []
            sched_bodies: dict[str, dict] = {}

            mem_api = MemoryApi()
            mem_api.scales["image-moss"] = 0

            orig_patch_scale = mem_api.patch_scale
            orig_replace_lease = mem_api.replace_lease

            def tracked_patch_scale(name: str, replicas: int) -> None:
                events.append(f"scale:{name}:{replicas}")
                orig_patch_scale(name, replicas)

            def tracked_replace_lease(name: str, spec: dict, rv: str | None) -> int:
                holder = spec.get("holderIdentity") or ""
                events.append(f"k8s_lease:{'acquire' if holder else 'release'}")
                return orig_replace_lease(name, spec, rv)

            mem_api.patch_scale = tracked_patch_scale  # type: ignore[method-assign]
            mem_api.replace_lease = tracked_replace_lease  # type: ignore[method-assign]

            moss_gpu = GpuCoordinator(
                api=mem_api,
                role="moss-image",
                holder="moss-image/test",
                deployment="image-moss",
                enabled=True,
            )

            def sched_handler(request: httpx.Request) -> httpx.Response:
                body = json.loads(request.content.decode("utf-8")) if request.content else {}
                if request.url.path == "/internal/v1/gpu-scheduler/leases/acquire":
                    events.append("sched:acquire")
                    sched_bodies["acquire"] = body
                    return httpx.Response(
                        200,
                        json={
                            "lease_id": "lease-moss-99",
                            "state": "queued",
                            "blocking_reason": "vram_pressure",
                            "fencing_token": 0,
                        },
                    )
                if request.url.path == "/internal/v1/gpu-scheduler/leases/lease-moss-99":
                    events.append("sched:poll")
                    return httpx.Response(
                        200,
                        json={
                            "lease_id": "lease-moss-99",
                            "state": "leased",
                            "fencing_token": 77,
                            "version": 2,
                            "attempt_id": sched_bodies["acquire"]["attempt_id"],
                            "stage_id": "generate",
                            "gpu_uuids": ["GPU-00000000-0000-0000-0000-000000000000"],
                            "expires_at": "2027-01-01T00:00:00Z",
                        },
                    )
                if request.url.path == "/internal/v1/gpu-scheduler/leases/lease-moss-99/bind-target":
                    events.append("sched:bind_target")
                    sched_bodies["bind_target"] = body
                    return httpx.Response(
                        200,
                        json={
                            "lease_id": "lease-moss-99",
                            "state": "leased",
                            "fencing_token": 78,
                            "version": 3,
                            "pod_uid": body["pod_uid"],
                            "container_id": body["container_id"],
                            "boot_id": body["boot_id"],
                            "gpu_uuids": body.get("actual_gpu_uuids", []),
                            "expires_at": "2027-01-01T00:05:00Z",
                        },
                    )
                if request.url.path.endswith("/release"):
                    events.append("sched:release")
                    sched_bodies["release"] = body
                    mode = body.get("reclaim_mode", "")
                    if mode not in ALLOWED_RECLAIM_MODES:
                        return httpx.Response(
                            400,
                            json={"error_code": "INVALID_REQUEST", "message": f"unsupported reclaim_mode {mode}"},
                        )
                    lid = request.url.path.split("/")[-2]
                    return httpx.Response(200, json={"lease_id": lid, "state": "released"})
                raise AssertionError(f"unexpected scheduler path {request.url.path}")

            def moss_handler(request: httpx.Request) -> httpx.Response:
                if request.url.path == "/health":
                    return httpx.Response(
                        200,
                        json={
                            "status": "healthy",
                            "runtime_identity": {
                                "pod_uid": "pod-moss-live",
                                "container_id": "cid-moss-live",
                                "boot_id": "boot-5090",
                                "supervisor_pid": 901,
                                "supervisor_starttime_ticks": 12345,
                                "visible_gpu_uuids": ["GPU-11111111-1111-1111-1111-111111111111"],
                            },
                        },
                    )
                if request.url.path == "/v1/generate":
                    events.append("moss:generate")
                    # Reconciled GPU UUID & bumped fencing token from /bind-target MUST be sent
                    self.assertEqual(request.headers.get("x-vf-gpu-lease-id"), "lease-moss-99")
                    self.assertEqual(request.headers.get("x-vf-gpu-fencing-token"), "78")
                    self.assertEqual(request.headers.get("x-vf-gpu-pod-uid"), "pod-moss-live")
                    self.assertEqual(request.headers.get("x-vf-gpu-container-id"), "cid-moss-live")
                    self.assertEqual(request.headers.get("x-vf-gpu-boot-id"), "boot-5090")
                    self.assertEqual(
                        request.headers.get("x-vf-gpu-uuid"),
                        "GPU-11111111-1111-1111-1111-111111111111",
                    )
                    self.assertEqual(request.headers.get("x-vf-gpu-expires-at"), "2027-01-01T00:05:00Z")
                    proof = {
                        "reclaimed": True,
                        "active_tasks": 0,
                        "proof_source": "receiver_guard",
                        "token_sealed": True,
                        "supervisor_pid": 901,
                        "supervisor_starttime_ticks": 12345,
                        "sealed_pod_uid": "pod-moss-live",
                        "sealed_container_id": "cid-moss-live",
                        "sealed_boot_id": "boot-5090",
                        "revoked_lease_id": "lease-moss-99",
                        "consumed_fencing_token": 78,
                    }
                    return httpx.Response(
                        200,
                        json={
                            "status": "ok",
                            "image_base64": b64_png,
                            "mime_type": "image/png",
                            "width": 1024,
                            "height": 1024,
                            "resolved_seed": 7,
                            "steps": 20,
                            "cfg_scale": 3.0,
                            "sampler": "euler",
                            "scheduler": "simple",
                            "shift": 5.0,
                            "prompt_mode": "text",
                            "elapsed_ms": 1200,
                        },
                        headers={"X-VF-Reclaim-Proof": json.dumps(proof)},
                    )
                raise AssertionError(f"unexpected moss path {request.url.path}")

            sched = SchedulerLeaseClient(
                base_url="http://station-runtime:5056",
                token="sched-tok",
                enabled=True,
                poll_interval_seconds=0.0,
                client=httpx.Client(transport=httpx.MockTransport(sched_handler)),
                sleep=lambda _: None,
            )
            moss_provider = MossProvider(
                base_url="http://image-moss:8000",
                client=httpx.Client(transport=httpx.MockTransport(moss_handler)),
            )
            qm = TaskQueueManager(
                tasks=tasks,
                artifacts=artifacts,
                codex_provider=MagicMock(),
                gemini_provider=MagicMock(),
                moss_provider=moss_provider,
                moss_gpu=moss_gpu,
                scheduler=sched,
            )
            params = qm.prepare("moss-image", "proj-1", {"action": "generate", "prompt": "courtyard", "width": 1024, "height": 1024})
            rec = tasks.create(project_id="proj-1", idempotency_key="idem-1", engine="moss-image", model=DEFAULT_MOSS_MODEL, request_params=params)
            qm._execute_moss(TaskJob(rec.task_id, "proj-1", "moss-image", "generate", params))

            self.assertEqual(sched_bodies["acquire"]["peak_budget_mib"], 18432)
            self.assertEqual(sched_bodies["acquire"]["k8s_slots_required"], 1)
            self.assertEqual(sched_bodies["bind_target"]["pod_uid"], "pod-moss-live")
            self.assertEqual(
                sched_bodies["bind_target"]["actual_gpu_uuids"],
                ["GPU-11111111-1111-1111-1111-111111111111"],
            )
            self.assertEqual(sched_bodies["release"]["reclaim_mode"], "warm_idle_settled")
            self.assertTrue(sched_bodies["release"]["worker_settled"])
            self.assertLess(events.index("sched:acquire"), events.index("scale:image-moss:1"))
            self.assertLess(events.index("scale:image-moss:1"), events.index("sched:bind_target"))
            self.assertLess(events.index("sched:bind_target"), events.index("moss:generate"))
            self.assertLess(events.index("k8s_lease:release"), events.index("sched:release"))

    def test_release_protocol_error_handling_and_pending_retry_queue(self) -> None:
        attempts = {"release": 0}

        def sched_handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content.decode("utf-8")) if request.content else {}
            if request.url.path.endswith("/release"):
                attempts["release"] += 1
                if body.get("reclaim_mode") == "warm_settled":
                    return httpx.Response(400, json={"error_code": "INVALID_REQUEST", "message": "bad mode"})
                if attempts["release"] == 2:
                    return httpx.Response(503, json={"error_code": "UNAVAILABLE", "message": "temporary outage"})
                lid = request.url.path.split("/")[-2]
                return httpx.Response(200, json={"lease_id": lid, "state": "released"})
            raise AssertionError(request.url.path)

        sched = SchedulerLeaseClient(
            base_url="http://station-runtime:5056",
            token="sched-tok",
            enabled=True,
            client=httpx.Client(transport=httpx.MockTransport(sched_handler)),
        )
        # Invalid reclaim_mode raises SchedulerError with parsed top-level error_code AND quarantines (never deletes!)
        proof_x = {
            "reclaimed": True,
            "active_tasks": 0,
            "proof_source": "receiver_guard",
            "token_sealed": True,
            "consumed_fencing_token": 1,
        }
        with self.assertRaises(SchedulerError) as ctx:
            sched.release_raw("lease-x", 1, worker_settled=True, reclaim_mode="warm_settled", reclaim_proof=proof_x)
        self.assertEqual(ctx.exception.code, "INVALID_REQUEST")
        quarantined_x = sched.get_pending_release("lease-x")
        self.assertIsNotNone(quarantined_x)
        self.assertTrue(quarantined_x["_quarantined_400"])  # type: ignore[index]
        self.assertEqual(quarantined_x["reclaim_proof"], proof_x)  # type: ignore[index]

        # 503 enqueues into pending_releases; auto-flush skips quarantined lease-x and flushes lease-y!
        with self.assertRaises(SchedulerError):
            sched.release_raw("lease-y", 2, worker_settled=False, reclaim_mode="abort_unstarted")
        self.assertEqual(set(sched._pending_releases.keys()), {"lease-x", "lease-y"})
        flushed = sched.flush_pending_releases()
        self.assertEqual(flushed, 1)
        self.assertEqual(set(sched._pending_releases.keys()), {"lease-x"})

        # Corrected request on quarantined lease-x reuses preserved reclaim_proof and clears journal on 200 receipt!
        recovered_x = sched.release_raw(
            "lease-x",
            1,
            worker_settled=True,
            reclaim_mode="warm_idle_settled",
            reclaim_proof=quarantined_x["reclaim_proof"],  # type: ignore[index]
        )
        self.assertEqual(recovered_x["state"], "released")
        self.assertEqual(len(sched._pending_releases), 0)

    def test_faceswap_exempts_cpu_mode_and_acquires_6144mib_slice_in_cuda_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = ArtifactStore(root / "artifacts")
            artifacts.ensure_ready()
            png = _png_bytes(64, 64)
            target_rec = artifacts.create_from_bytes(
                project_id="proj-ff",
                filename="target.png",
                data=png,
                media_type="image/png",
            )
            source_rec = artifacts.create_from_bytes(
                project_id="proj-ff",
                filename="source.png",
                data=png,
                media_type="image/png",
            )

            sched_calls: list[str] = []
            release_bodies: list[dict] = []

            def sched_handler(request: httpx.Request) -> httpx.Response:
                body = json.loads(request.content.decode("utf-8")) if request.content else {}
                sched_calls.append(request.url.path)
                if request.url.path.endswith("/acquire"):
                    self.assertEqual(body["consumer_kind"], "face-fusion")
                    self.assertEqual(body["peak_budget_mib"], 6144)
                    return httpx.Response(
                        200,
                        json={
                            "lease_id": "lease-ff-7",
                            "state": "leased",
                            "fencing_token": 88,
                            "attempt_id": body["attempt_id"],
                            "stage_id": "fuse",
                            "pod_uid": body.get("pod_uid", ""),
                            "container_id": body.get("container_id", ""),
                            "boot_id": body.get("boot_id", ""),
                            "gpu_uuids": ["GPU-ffffffff-1111-2222-3333-444444444444"],
                            "expires_at": "2027-01-01T00:00:00Z",
                        },
                    )
                if request.url.path.endswith("/release"):
                    release_bodies.append(body)
                    self.assertIn(body.get("reclaim_mode", ""), ALLOWED_RECLAIM_MODES)
                    lid = request.url.path.split("/")[-2]
                    return httpx.Response(200, json={"lease_id": lid, "state": "released"})
                raise AssertionError(request.url.path)

            mode = {"cuda": False}

            def ff_handler(request: httpx.Request) -> httpx.Response:
                if request.url.path == "/health":
                    return httpx.Response(
                        200,
                        json={
                            "status": "healthy",
                            "cuda_available": mode["cuda"],
                            "execution_provider": "CUDAExecutionProvider" if mode["cuda"] else "CPUExecutionProvider",
                            "runtime_identity": {
                                "pod_uid": "pod-ff-1",
                                "container_id": "cid-ff-1",
                                "boot_id": "boot-1",
                                "supervisor_pid": 55,
                                "supervisor_starttime_ticks": 66,
                                "visible_gpu_uuids": ["GPU-ffffffff-1111-2222-3333-444444444444"],
                            },
                        },
                    )
                if request.url.path == "/v1/fuse":
                    body = json.loads(request.content.decode("utf-8"))
                    Path(body["output_path"]).write_bytes(png)
                    if mode["cuda"]:
                        self.assertEqual(request.headers.get("x-vf-gpu-lease-id"), "lease-ff-7")
                        self.assertEqual(request.headers.get("x-vf-gpu-fencing-token"), "88")
                        self.assertEqual(request.headers.get("x-vf-gpu-pod-uid"), "pod-ff-1")
                        self.assertEqual(
                            request.headers.get("x-vf-gpu-uuid"),
                            "GPU-ffffffff-1111-2222-3333-444444444444",
                        )
                        proof = {
                            "reclaimed": True,
                            "active_tasks": 0,
                            "proof_source": "receiver_guard",
                            "token_sealed": True,
                            "supervisor_pid": 55,
                            "supervisor_starttime_ticks": 66,
                            "sealed_pod_uid": "pod-ff-1",
                            "sealed_container_id": "cid-ff-1",
                            "sealed_boot_id": "boot-1",
                            "revoked_lease_id": "lease-ff-7",
                            "consumed_fencing_token": 88,
                        }
                        return httpx.Response(
                            200,
                            json={
                                "status": "success",
                                "output_path": body["output_path"],
                                "detected_faces": 1,
                                "arcface_similarity": 0.92,
                                "inference_time_ms": 25,
                                "pipeline": "facefusion",
                                "execution": {},
                            },
                            headers={"X-VF-Reclaim-Proof": json.dumps(proof)},
                        )
                    self.assertIsNone(request.headers.get("x-vf-gpu-lease-id"))
                    return httpx.Response(
                        200,
                        json={
                            "status": "success",
                            "output_path": body["output_path"],
                            "detected_faces": 1,
                            "arcface_similarity": 0.92,
                            "inference_time_ms": 25,
                            "pipeline": "facefusion",
                            "execution": {},
                        },
                    )
                raise AssertionError(f"unexpected ff path {request.url.path}")

            sched = SchedulerLeaseClient(
                base_url="http://station-runtime:5056",
                token="sched-tok",
                enabled=True,
                client=httpx.Client(transport=httpx.MockTransport(sched_handler)),
            )

            async def run_both() -> None:
                async with httpx.AsyncClient(transport=httpx.MockTransport(ff_handler)) as ff_client:
                    res_cpu = await handle_faceswap(
                        artifacts=artifacts,
                        target_artifact_id=target_rec.artifact_id,
                        source_identity_artifact_id=source_rec.artifact_id,
                        project_id="proj-ff",
                        scheduler=sched,
                        http_client=ff_client,
                    )
                    self.assertEqual(res_cpu["status"], "completed")
                    self.assertEqual(sched_calls, [])

                    mode["cuda"] = True
                    res_cuda = await handle_faceswap(
                        artifacts=artifacts,
                        target_artifact_id=target_rec.artifact_id,
                        source_identity_artifact_id=source_rec.artifact_id,
                        project_id="proj-ff",
                        scheduler=sched,
                        http_client=ff_client,
                    )
                    self.assertEqual(res_cuda["status"], "completed")
                    self.assertEqual(
                        sched_calls,
                        [
                            "/internal/v1/gpu-scheduler/leases/acquire",
                            "/internal/v1/gpu-scheduler/leases/lease-ff-7/release",
                        ],
                    )
                    self.assertEqual(release_bodies[0]["reclaim_mode"], "warm_idle_settled")
                    self.assertTrue(release_bodies[0]["worker_settled"])

            asyncio.run(run_both())

    def test_r2_m1_durable_pending_release_journal_strict_receipt_and_restart_recovery(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            journal_path = Path(tmp) / "gpu_pending_releases.json"
            mode = {"resp": "disconnect"}
            seen_bodies: list[dict] = []

            def handler(request: httpx.Request) -> httpx.Response:
                body = json.loads(request.content.decode("utf-8")) if request.content else {}
                seen_bodies.append(body)
                lid = request.url.path.split("/")[-2]
                m = mode["resp"]
                if m == "disconnect":
                    raise RuntimeError("socket_reset")
                if m == "404":
                    return httpx.Response(404, json={"error_code": "LEASE_NOT_FOUND"})
                if m == "wrong_lease":
                    return httpx.Response(200, json={"lease_id": "other-lease", "state": "released"})
                if m == "reclaiming":
                    return httpx.Response(200, json={"lease_id": lid, "state": "reclaiming"})
                if m == "partial" and lid == "lease-a":
                    return httpx.Response(503, json={"error_code": "UNAVAILABLE"})
                return httpx.Response(200, json={"lease_id": lid, "state": "released"})

            http_client = httpx.Client(transport=httpx.MockTransport(handler))
            c1 = SchedulerLeaseClient(
                base_url="http://station-runtime:5056",
                token="sched-tok",
                enabled=True,
                client=http_client,
                pending_releases_path=journal_path,
            )
            with self.assertRaises(SchedulerError):
                c1.release_raw("lease-a", 10, worker_settled=True, expected_version=4)
            with self.assertRaises(SchedulerError):
                c1.release_raw("lease-b", 11, worker_settled=True, expected_version=5)
            self.assertTrue(journal_path.is_file())
            self.assertNotIn("expected_version", c1._pending_releases["lease-a"])

            # Restart client from disk; 404, wrong lease_id, and reclaiming MUST NOT delete entries
            c2 = SchedulerLeaseClient(
                base_url="http://station-runtime:5056",
                token="sched-tok",
                enabled=True,
                client=http_client,
                pending_releases_path=journal_path,
            )
            self.assertEqual(set(c2._pending_releases.keys()), {"lease-a", "lease-b"})
            for bad in ("404", "wrong_lease", "reclaiming"):
                mode["resp"] = bad
                self.assertEqual(c2.flush_pending_releases(), 0)
                self.assertEqual(set(c2._pending_releases.keys()), {"lease-a", "lease-b"})

            # Partial failure on lease-a still flushes lease-b (no early break)
            mode["resp"] = "partial"
            self.assertEqual(c2.flush_pending_releases(), 1)
            self.assertEqual(set(c2._pending_releases.keys()), {"lease-a"})

            # Full recovery clears remaining entry on disk
            mode["resp"] = "ok"
            self.assertEqual(c2.flush_pending_releases(), 1)
            self.assertEqual(c2._pending_releases, {})
            c3 = SchedulerLeaseClient(
                base_url="http://station-runtime:5056",
                token="sched-tok",
                enabled=True,
                client=http_client,
                pending_releases_path=journal_path,
            )
            self.assertEqual(c3._pending_releases, {})

