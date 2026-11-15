import asyncio
from contextlib import asynccontextmanager
import json

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.gpu_guard import LeaseFenceError, LeaseFenceGuard, default_guard_state_path, is_cuda_mode
from src.schemas import FuseRequest, FuseResponse, DetectRequest, DetectResponse, HealthResponse
from src.pipeline import FaceFusionPipeline, FaceFusionPipelineError

pipeline = FaceFusionPipeline()
gpu_guard = LeaseFenceGuard(
    expected_consumer_kind="face-fusion",
    identity=pipeline.runtime_identity,
    state_path=default_guard_state_path("face-fusion"),
)


class RevokeLeaseRequest(BaseModel):
    lease_id: str
    fencing_token: int = 0


class ReclaimGuardRequest(BaseModel):
    lease_id: str = ""
    fencing_token: int = 0


@asynccontextmanager
async def lifespan(app):
    await asyncio.to_thread(pipeline.start)
    try:
        yield
    finally:
        await asyncio.to_thread(pipeline.close)


app = FastAPI(title="VerdantFlare Image Face Fusion", version="0.2.5",
              description="Internal service backed by official FaceFusion 3.9.0", lifespan=lifespan)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])


@app.exception_handler(FaceFusionPipelineError)
async def pipeline_exception_handler(request: Request, exc: FaceFusionPipelineError):
    return JSONResponse(status_code=exc.status_code,
                        content={"error_code": exc.error_code, "message": exc.message, "status": "error"})


@app.exception_handler(LeaseFenceError)
async def lease_fence_exception_handler(request: Request, exc: LeaseFenceError):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error_code": exc.error_code, "message": exc.message, "status": "error"},
    )


@app.get("/live")
def live():
    return {"status": "alive"}


@app.get("/health", response_model=HealthResponse)
def health_check():
    data = dict(pipeline.health())
    data.setdefault("active_tasks", pipeline.active_tasks)
    data.setdefault("residency_state", "S2" if (data.get("status") == "healthy" and is_cuda_mode(data)) else "S3")
    data.setdefault("runtime_identity", pipeline.runtime_identity.to_dict())
    return JSONResponse(status_code=200 if data["status"] == "healthy" else 503, content=data)


@app.post("/v1/detect", response_model=DetectResponse)
def detect_face(req: DetectRequest, request: Request):
    cuda_active = is_cuda_mode(pipeline.health())
    grant = gpu_guard.validate_request(request.headers, cuda_active=cuda_active)
    unheadered_exec_id = gpu_guard.begin_unheadered_execution() if (cuda_active and grant is None) else None
    try:
        result = pipeline.detect_face(req.image_path, req.target_face_index)
    except Exception:
        if unheadered_exec_id is not None:
            gpu_guard.end_unheadered_execution(unheadered_exec_id)
        if grant is not None:
            gpu_guard.finish_failed_request(grant)
        raise
    if unheadered_exec_id is not None:
        gpu_guard.end_unheadered_execution(unheadered_exec_id)
    headers = {}
    if grant is not None:
        proof = gpu_guard.seal_and_build_reclaim_proof(
            grant,
            active_tasks=pipeline.active_tasks,
            reclaimed=pipeline.active_tasks == 0,
        )
        headers["X-VF-Reclaim-Proof"] = json.dumps(proof)
    return JSONResponse(status_code=200, content=result, headers=headers)


@app.post("/v1/fuse", response_model=FuseResponse)
def fuse_face(req: FuseRequest, request: Request):
    cuda_active = is_cuda_mode(pipeline.health())
    grant = gpu_guard.validate_request(request.headers, cuda_active=cuda_active)
    unheadered_exec_id = gpu_guard.begin_unheadered_execution() if (cuda_active and grant is None) else None
    try:
        result = pipeline.fuse(**req.model_dump())
    except Exception:
        if unheadered_exec_id is not None:
            gpu_guard.end_unheadered_execution(unheadered_exec_id)
        if grant is not None:
            gpu_guard.finish_failed_request(grant)
        raise
    if unheadered_exec_id is not None:
        gpu_guard.end_unheadered_execution(unheadered_exec_id)
    headers = {}
    if grant is not None:
        proof = gpu_guard.seal_and_build_reclaim_proof(
            grant,
            active_tasks=pipeline.active_tasks,
            reclaimed=pipeline.active_tasks == 0,
        )
        headers["X-VF-Reclaim-Proof"] = json.dumps(proof)
    return JSONResponse(status_code=200, content=result, headers=headers)


@app.post("/internal/v1/gpu-guard/reclaim")
async def reclaim_guard(req: ReclaimGuardRequest, request: Request):
    gpu_guard.verify_runtime_token(request.headers.get("authorization"))
    proof = gpu_guard.reclaim_and_seal(
        lease_id=req.lease_id,
        fencing_token=req.fencing_token,
        active_tasks=pipeline.active_tasks,
    )
    return JSONResponse(status_code=200 if proof["reclaimed"] else 409, content=proof)


@app.post("/internal/v1/gpu-residency/revoke-lease")
async def revoke_lease(req: RevokeLeaseRequest, request: Request):
    gpu_guard.verify_runtime_token(request.headers.get("authorization"))
    return gpu_guard.revoke_lease(req.lease_id, req.fencing_token)


@app.post("/internal/v1/gpu-residency/evict-vram")
async def evict_vram(request: Request):
    gpu_guard.verify_runtime_token(request.headers.get("authorization"))
    await asyncio.to_thread(pipeline.evict_idle_vram)
    proof = gpu_guard.reclaim_and_seal(active_tasks=pipeline.active_tasks)
    proof["residency_state"] = pipeline.health().get("residency_state", "S3")
    return proof
