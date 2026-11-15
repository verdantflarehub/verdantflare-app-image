"""Unit tests for image-face-fusion CUDA 6-tuple GPU fencing, durable token sealing, and CPU-mode exemption."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import tempfile
from pathlib import Path
from unittest import mock

from PIL import Image
import pytest
from starlette.testclient import TestClient

import src.main as ff_main
from src.gpu_guard import LeaseFenceGuard, RuntimeIdentity

GPU_0 = "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
CID_HEX = "b" * 64


def _future_iso(seconds: int = 180) -> str:
    return (datetime.now(timezone.utc) + timedelta(seconds=seconds)).strftime("%Y-%m-%dT%H:%M:%SZ")


def _grant_headers(
    lease_id: str = "lease-ff-1",
    token: int = 12,
    consumer_kind: str = "face-fusion",
    attempt_id: str = "att-ff-1",
) -> dict[str, str]:
    return {
        "X-VF-GPU-Lease-ID": lease_id,
        "X-VF-GPU-Fencing-Token": str(token),
        "X-VF-GPU-Attempt-ID": attempt_id,
        "X-VF-GPU-Stage-ID": "fuse",
        "X-VF-GPU-Consumer-Kind": consumer_kind,
        "X-VF-GPU-Holder-App-ID": "image-mcp",
        "X-VF-GPU-Pod-UID": "pod-ff-1",
        "X-VF-GPU-Container-ID": CID_HEX,
        "X-VF-GPU-Boot-ID": "boot-ff-1",
        "X-VF-GPU-UUID": GPU_0,
        "X-VF-GPU-Expires-At": _future_iso(),
    }


@pytest.fixture
def ff_client(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("IMAGE_RUNTIME_TOKEN", "ff-secret")
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp)
        state_file = p / "ff_guard_state.json"
        ident = RuntimeIdentity(
            pod_uid="pod-ff-1",
            container_id=f"containerd://{CID_HEX}",
            boot_id="boot-ff-1",
            supervisor_pid=6543,
            supervisor_starttime_ticks=12345,
            visible_gpu_uuids=(GPU_0,),
        )
        guard = LeaseFenceGuard(expected_consumer_kind="face-fusion", identity=ident, state_path=state_file)
        monkeypatch.setattr(ff_main, "gpu_guard", guard)
        monkeypatch.setattr(ff_main.pipeline, "runtime_identity", ident)
        target = p / "target.png"
        source = p / "source.png"
        out = p / "out.png"
        Image.new("RGB", (64, 64), (120, 120, 120)).save(target, "PNG")
        Image.new("RGB", (64, 64), (180, 180, 180)).save(source, "PNG")
        with TestClient(ff_main.app) as client:
            yield client, target, source, out, state_file, ident


def test_cpu_mode_is_strictly_exempt_even_when_enforcing(ff_client, monkeypatch: pytest.MonkeyPatch) -> None:
    client, target, source, out, _, _ = ff_client
    monkeypatch.setenv("IMAGE_GPU_GRANT_POLICY", "enforcing")

    cpu_health = {
        "status": "healthy",
        "cuda_available": False,
        "execution_provider": "CPUExecutionProvider",
        "models_loaded": {"swapper": True},
    }
    fake_fuse = {
        "status": "success",
        "output_path": str(out),
        "detected_faces": 1,
        "arcface_similarity": 0.91,
        "inference_time_ms": 40,
        "pipeline": "cpu-test",
        "execution": {},
    }
    with mock.patch("src.main.pipeline.health", return_value=cpu_health), mock.patch(
        "src.main.pipeline.fuse", return_value=fake_fuse
    ):
        resp = client.post(
            "/v1/fuse",
            json={
                "target_image_path": str(target),
                "source_face_path": str(source),
                "output_path": str(out),
            },
        )
        assert resp.status_code == 200
        assert "X-VF-Reclaim-Proof" not in resp.headers


def test_cuda_mode_enforces_6tuple_seals_token_and_persists_across_restart(
    ff_client, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, target, source, out, state_file, ident = ff_client
    monkeypatch.setenv("IMAGE_GPU_GRANT_POLICY", "enforcing")

    cuda_health = {
        "status": "healthy",
        "cuda_available": True,
        "execution_provider": "CUDAExecutionProvider",
        "models_loaded": {"swapper": True},
    }
    fake_fuse = {
        "status": "success",
        "output_path": str(out),
        "detected_faces": 1,
        "arcface_similarity": 0.93,
        "inference_time_ms": 18,
        "pipeline": "cuda-test",
        "execution": {},
    }
    req_body = {
        "target_image_path": str(target),
        "source_face_path": str(source),
        "output_path": str(out),
    }
    with mock.patch("src.main.pipeline.health", return_value=cuda_health), mock.patch(
        "src.main.pipeline.fuse", return_value=fake_fuse
    ):
        denied = client.post("/v1/fuse", json=req_body)
        assert denied.status_code == 403
        assert denied.json()["error_code"] == "GPU_GRANT_REQUIRED"

        ok = client.post("/v1/fuse", json=req_body, headers=_grant_headers("lease-cuda-1", 31))
        assert ok.status_code == 200
        proof = json.loads(ok.headers["X-VF-Reclaim-Proof"])
        assert proof["reclaimed"] is True
        assert proof["proof_source"] == "receiver_guard"
        assert proof["token_sealed"] is True
        assert proof["sealed_pod_uid"] == "pod-ff-1"
        assert proof["consumed_fencing_token"] == 31

        # Replay after completion with new attempt_id or restart is rejected
        restarted = LeaseFenceGuard(expected_consumer_kind="face-fusion", identity=ident, state_path=state_file)
        monkeypatch.setattr(ff_main, "gpu_guard", restarted)
        after_restart = client.post(
            "/v1/fuse",
            json=req_body,
            headers=_grant_headers("lease-cuda-2", 31, attempt_id="att-ff-new"),
        )
        assert after_restart.status_code == 409
        assert after_restart.json()["error_code"] == "STALE_FENCING_TOKEN"


def test_r2_h1_reclaim_vs_fuse_barrier_and_r2_h3_corrupted_state_fails_closed(
    ff_client, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os
    import threading
    from src.gpu_guard import FencePersistenceError

    client, target, source, out, state_file, ident = ff_client
    monkeypatch.setenv("IMAGE_GPU_GRANT_POLICY", "enforcing")
    cuda_health = {
        "status": "healthy",
        "cuda_available": True,
        "execution_provider": "CUDAExecutionProvider",
        "models_loaded": {"swapper": True},
    }
    entered = threading.Barrier(2)
    allow_finish = threading.Event()

    def slow_fuse(**kwargs):
        entered.wait(timeout=5.0)
        allow_finish.wait(timeout=5.0)
        return {
            "status": "success",
            "output_path": str(out),
            "detected_faces": 1,
            "arcface_similarity": 0.94,
            "inference_time_ms": 20,
            "pipeline": "cuda-test",
            "execution": {},
        }

    # Short write + Barrier(2) active execution blocks /reclaim with 409 while sealing token
    guard = LeaseFenceGuard(
        expected_consumer_kind="face-fusion",
        identity=ident,
        state_path=state_file,
        write_fn=lambda fd, buf: os.write(fd, buf[:1]),
    )
    monkeypatch.setattr(ff_main, "gpu_guard", guard)
    codes: list[int] = []
    with mock.patch("src.main.pipeline.health", return_value=cuda_health), mock.patch(
        "src.main.pipeline.fuse", side_effect=slow_fuse
    ):
        def worker():
            r = client.post(
                "/v1/fuse",
                json={"target_image_path": str(target), "source_face_path": str(source), "output_path": str(out)},
                headers=_grant_headers("lease-ff-race", 60),
            )
            codes.append(r.status_code)

        t = threading.Thread(target=worker)
        t.start()
        entered.wait(timeout=5.0)
        rec = client.post(
            "/internal/v1/gpu-guard/reclaim",
            json={"lease_id": "lease-ff-race", "fencing_token": 60},
            headers={"Authorization": "Bearer ff-secret"},
        )
        assert rec.status_code == 409
        assert rec.json()["reclaimed"] is False
        assert rec.json()["token_sealed"] is True
        allow_finish.set()
        t.join(timeout=5.0)
        assert codes == [200]

    # Corrupted state file fails closed with 503
    bad_file = state_file.parent / "bad_ff_state.json"
    bad_file.write_text("{corrupt", encoding="utf-8")
    bad_guard = LeaseFenceGuard(expected_consumer_kind="face-fusion", identity=ident, state_path=bad_file)
    with pytest.raises(FencePersistenceError) as exc_info:
        bad_guard.validate_request(_grant_headers("lease-ff-9", 99), cuda_active=True)
    assert exc_info.value.status_code == 503

