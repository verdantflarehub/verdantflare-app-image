"""Unit tests for image-moss 6-tuple GPU lease fencing, durable token sealing, and S2->S3 VRAM eviction."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

import src.main as moss_main
from src.gpu_guard import LeaseFenceGuard, RuntimeIdentity
from src.pipeline import MossImagePipeline
from src.settings import Settings


PNG_1X1 = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\xcf\xc0"
    b"\x00\x00\x03\x01\x01\x00\xc9\xfe\x92\xef\x00\x00\x00\x00IEND\xaeB`\x82"
)

GPU_0 = "GPU-11111111-2222-3333-4444-555555555555"
CID_HEX = "a" * 64


def _future_iso(seconds: int = 180) -> str:
    return (datetime.now(timezone.utc) + timedelta(seconds=seconds)).strftime("%Y-%m-%dT%H:%M:%SZ")


def _grant_headers(
    lease_id: str = "lease-moss-1",
    token: int = 10,
    consumer_kind: str = "moss-image",
    attempt_id: str = "att-1",
    stage_id: str = "generate",
    pod_uid: str = "pod-moss-1",
    container_id: str = CID_HEX,
    boot_id: str = "boot-moss-1",
    gpu_uuid: str = GPU_0,
    expires_at: str | None = None,
) -> dict[str, str]:
    return {
        "X-VF-GPU-Lease-ID": lease_id,
        "X-VF-GPU-Fencing-Token": str(token),
        "X-VF-GPU-Attempt-ID": attempt_id,
        "X-VF-GPU-Stage-ID": stage_id,
        "X-VF-GPU-Consumer-Kind": consumer_kind,
        "X-VF-GPU-Holder-App-ID": "image-mcp",
        "X-VF-GPU-Pod-UID": pod_uid,
        "X-VF-GPU-Container-ID": container_id,
        "X-VF-GPU-Boot-ID": boot_id,
        "X-VF-GPU-UUID": gpu_uuid,
        "X-VF-GPU-Expires-At": expires_at if expires_at is not None else _future_iso(),
    }


@pytest.fixture
def configured_client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("IMAGE_RUNTIME_TOKEN", "runtime-secret")
    state_file = tmp_path / "moss_guard_state.json"
    monkeypatch.setenv("VF_GPU_GUARD_STATE_PATH", str(state_file))
    settings = Settings(auto_start_comfyui=False, verify_sha256_on_boot=False)
    pipe = MossImagePipeline(settings=settings, comfy_runner=lambda wf, t: PNG_1X1)

    def _fake_verify():
        pipe._models_verified = True
        pipe._verified_weights = {"diffusion": "/models/moss.safetensors"}
        return pipe._verified_weights

    monkeypatch.setattr(pipe, "verify_models", _fake_verify)
    _fake_verify()
    ident = RuntimeIdentity(
        pod_uid="pod-moss-1",
        container_id=f"containerd://{CID_HEX}",
        boot_id="boot-moss-1",
        supervisor_pid=4321,
        supervisor_starttime_ticks=98765,
        visible_gpu_uuids=(GPU_0,),
    )
    pipe.runtime_identity = ident
    guard = LeaseFenceGuard(expected_consumer_kind="moss-image", identity=ident, state_path=state_file)
    monkeypatch.setattr(moss_main, "pipeline", pipe)
    monkeypatch.setattr(moss_main, "gpu_guard", guard)
    with TestClient(moss_main.app) as client:
        yield client, pipe, guard, state_file, ident


def test_enforcing_rejects_missing_incomplete_unknown_identity_and_prefix_cid(
    configured_client, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, _, _, state_file, _ = configured_client
    monkeypatch.setenv("IMAGE_GPU_GRANT_POLICY", "enforcing")
    req_body = {"prompt": "a neon lantern", "width": 512, "height": 512, "steps": 4}

    resp = client.post("/v1/generate", json=req_body)
    assert resp.status_code == 403
    assert resp.json()["error_code"] == "GPU_GRANT_REQUIRED"

    bad = _grant_headers(consumer_kind="face-fusion")
    resp2 = client.post("/v1/generate", json=req_body, headers=bad)
    assert resp2.status_code == 403
    assert resp2.json()["error_code"] == "CONSUMER_KIND_MISMATCH"

    # Missing pod_uid/gpu_uuid/expires_at rejected
    incomplete = dict(_grant_headers())
    incomplete.pop("X-VF-GPU-UUID")
    resp_inc = client.post("/v1/generate", json=req_body, headers=incomplete)
    assert resp_inc.status_code == 403
    assert resp_inc.json()["error_code"] == "GPU_GRANT_INCOMPLETE"

    # Prefix-only container_id rejected (no startswith acceptance)
    prefix_cid = _grant_headers(container_id=CID_HEX[:12])
    resp_cid = client.post("/v1/generate", json=req_body, headers=prefix_cid)
    assert resp_cid.status_code == 403
    assert resp_cid.json()["error_code"] == "CONTAINER_ID_MISMATCH"

    # Expired grant rejected
    expired = _grant_headers(expires_at="2020-01-01T00:00:00Z")
    resp_exp = client.post("/v1/generate", json=req_body, headers=expired)
    assert resp_exp.status_code == 409
    assert resp_exp.json()["error_code"] == "LEASE_EXPIRED"

    # Unknown local identity fails closed in enforcing mode
    unknown_guard = LeaseFenceGuard(
        expected_consumer_kind="moss-image",
        policy="enforcing",
        identity=RuntimeIdentity(
            pod_uid="local-pod-1",
            container_id="local-1",
            boot_id="local-boot",
            supervisor_pid=1,
            supervisor_starttime_ticks=1,
            visible_gpu_uuids=(),
        ),
        state_path=state_file,
    )
    monkeypatch.setattr(moss_main, "gpu_guard", unknown_guard)
    resp_unk = client.post("/v1/generate", json=req_body, headers=_grant_headers())
    assert resp_unk.status_code == 503
    assert resp_unk.json()["error_code"] == "GPU_IDENTITY_UNAVAILABLE"


def test_valid_grant_durably_seals_token_blocks_new_attempt_and_restart_replay(
    configured_client, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, _, _, state_file, ident = configured_client
    monkeypatch.setenv("IMAGE_GPU_GRANT_POLICY", "enforcing")
    req_body = {"prompt": "a neon lantern", "width": 512, "height": 512, "steps": 4}

    ok = client.post("/v1/generate", json=req_body, headers=_grant_headers("lease-1", 25))
    assert ok.status_code == 200
    proof = json.loads(ok.headers["X-VF-Reclaim-Proof"])
    assert proof["reclaimed"] is True
    assert proof["proof_source"] == "receiver_guard"
    assert proof["token_sealed"] is True
    assert proof["sealed_pod_uid"] == "pod-moss-1"
    assert proof["consumed_fencing_token"] == 25

    # Immediate replay with SAME fencing token or SAME lease_id (even with new attempt_id/stage_id) is blocked
    replay_new_attempt = client.post(
        "/v1/generate",
        json=req_body,
        headers=_grant_headers("lease-2", 25, attempt_id="att-bypass", stage_id="stage-bypass"),
    )
    assert replay_new_attempt.status_code == 409
    assert replay_new_attempt.json()["error_code"] == "STALE_FENCING_TOKEN"

    replay_same_lease = client.post(
        "/v1/generate",
        json=req_body,
        headers=_grant_headers("lease-1", 26, attempt_id="att-bypass-2"),
    )
    assert replay_same_lease.status_code == 409
    assert replay_same_lease.json()["error_code"] == "LEASE_REVOKED"

    # Simulate process restart by instantiating a fresh LeaseFenceGuard reading persisted state_file
    restarted_guard = LeaseFenceGuard(expected_consumer_kind="moss-image", identity=ident, state_path=state_file)
    monkeypatch.setattr(moss_main, "gpu_guard", restarted_guard)

    replay_after_restart = client.post(
        "/v1/generate",
        json=req_body,
        headers=_grant_headers("lease-3", 25, attempt_id="att-restart"),
    )
    assert replay_after_restart.status_code == 409
    assert replay_after_restart.json()["error_code"] == "STALE_FENCING_TOKEN"

    # Explicit /internal/v1/gpu-guard/reclaim endpoint seals token 30
    reclaim_resp = client.post(
        "/internal/v1/gpu-guard/reclaim",
        json={"lease_id": "lease-30", "fencing_token": 30},
        headers={"Authorization": "Bearer runtime-secret"},
    )
    assert reclaim_resp.status_code == 200
    assert reclaim_resp.json()["proof_source"] == "receiver_guard"
    assert reclaim_resp.json()["token_sealed"] is True
    assert reclaim_resp.json()["consumed_fencing_token"] == 30

    after_reclaim = client.post("/v1/generate", json=req_body, headers=_grant_headers("lease-31", 30))
    assert after_reclaim.status_code == 409


def test_health_exposes_runtime_identity_and_evict_vram_transitions_s2_to_s3(
    configured_client, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, pipe, _, _, _ = configured_client
    monkeypatch.setenv("IMAGE_GPU_GRANT_POLICY", "permissive")

    ok = client.post("/v1/generate", json={"prompt": "courtyard", "width": 512, "height": 512, "steps": 4})
    assert ok.status_code == 200
    h1 = client.get("/health")
    assert h1.status_code == 200
    assert h1.json()["residency_state"] == "S2"
    assert h1.json()["runtime_identity"]["pod_uid"] == "pod-moss-1"

    ev = client.post(
        "/internal/v1/gpu-residency/evict-vram",
        headers={"Authorization": "Bearer runtime-secret"},
    )
    assert ev.status_code == 200
    assert ev.json()["residency_state"] == "S3"
    assert pipe.residency_state == "S3"


def test_r2_h1_concurrent_reclaim_vs_generate_barrier_and_r2_h3_fail_closed_persistence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import threading
    from src.gpu_guard import FencePersistenceError

    monkeypatch.setenv("IMAGE_RUNTIME_TOKEN", "runtime-secret")
    monkeypatch.setenv("IMAGE_GPU_GRANT_POLICY", "enforcing")
    state_file = tmp_path / "moss_r2_state.json"
    ident = RuntimeIdentity(
        pod_uid="pod-moss-1",
        container_id=f"containerd://{CID_HEX}",
        boot_id="boot-moss-1",
        supervisor_pid=4321,
        supervisor_starttime_ticks=98765,
        visible_gpu_uuids=(GPU_0,),
    )

    # 1. R2-H3: Short writes (1 byte per os.write) succeed and persist cleanly
    import os
    short_guard = LeaseFenceGuard(
        expected_consumer_kind="moss-image",
        identity=ident,
        state_path=state_file,
        write_fn=lambda fd, buf: os.write(fd, buf[:1]),
    )
    g1 = short_guard.validate_request(_grant_headers("lease-short", 5))
    assert g1 is not None
    short_guard.seal_and_build_reclaim_proof(g1, active_tasks=0, reclaimed=True)
    saved = json.loads(state_file.read_text(encoding="utf-8"))
    assert saved["max_fencing_token"] == 5
    assert 5 in saved["consumed_tokens"]

    # 2. R2-H3: Commit failure does NOT mutate in-memory state and quarantines guard (503)
    def fail_replace(src: str, dst: str) -> None:
        raise OSError("disk_full")

    failing_guard = LeaseFenceGuard(
        expected_consumer_kind="moss-image",
        identity=ident,
        state_path=state_file,
        replace_fn=fail_replace,
    )
    with pytest.raises(FencePersistenceError) as exc_info:
        failing_guard.validate_request(_grant_headers("lease-fail", 6))
    assert exc_info.value.status_code == 503
    assert failing_guard._max_fencing_token == 5
    with pytest.raises(FencePersistenceError):
        failing_guard.reclaim_and_seal(lease_id="lease-fail", fencing_token=6, active_tasks=0)

    # 3. R2-H3: Corrupted existing state file fails closed with 503 FENCE_STATE_CORRUPTED
    corrupt_path = tmp_path / "corrupt.json"
    corrupt_path.write_text("{truncated-json", encoding="utf-8")
    corrupt_guard = LeaseFenceGuard(expected_consumer_kind="moss-image", identity=ident, state_path=corrupt_path)
    with pytest.raises(FencePersistenceError):
        corrupt_guard.validate_request(_grant_headers("lease-c", 99))

    # 4. R2-H1: Barrier(2) concurrency — if generate validated grant before /reclaim, /reclaim MUST return 409 (reclaimed=False)
    race_state = tmp_path / "race_state.json"
    entered = threading.Barrier(2)
    release_runner = threading.Event()
    settings = Settings(auto_start_comfyui=False, verify_sha256_on_boot=False)

    def blocking_runner(wf, t):
        entered.wait(timeout=5.0)
        release_runner.wait(timeout=5.0)
        return PNG_1X1

    pipe = MossImagePipeline(settings=settings, comfy_runner=blocking_runner)

    def _fake_verify():
        pipe._models_verified = True
        pipe._verified_weights = {"diffusion": "/models/moss.safetensors"}
        return pipe._verified_weights

    monkeypatch.setattr(pipe, "verify_models", _fake_verify)
    _fake_verify()
    pipe.runtime_identity = ident
    race_guard = LeaseFenceGuard(expected_consumer_kind="moss-image", identity=ident, state_path=race_state)
    monkeypatch.setattr(moss_main, "pipeline", pipe)
    monkeypatch.setattr(moss_main, "gpu_guard", race_guard)

    gen_results: list[int] = []
    with TestClient(moss_main.app) as client:
        def run_gen() -> None:
            r = client.post(
                "/v1/generate",
                json={"prompt": "courtyard", "width": 512, "height": 512, "steps": 4},
                headers=_grant_headers("lease-race", 50),
            )
            gen_results.append(r.status_code)

        t = threading.Thread(target=run_gen)
        t.start()
        entered.wait(timeout=5.0)
        reclaim_during_exec = client.post(
            "/internal/v1/gpu-guard/reclaim",
            json={"lease_id": "lease-race", "fencing_token": 50},
            headers={"Authorization": "Bearer runtime-secret"},
        )
        assert reclaim_during_exec.status_code == 409
        assert reclaim_during_exec.json()["reclaimed"] is False
        assert reclaim_during_exec.json()["token_sealed"] is True
        release_runner.set()
        t.join(timeout=5.0)
        assert gen_results == [200]

