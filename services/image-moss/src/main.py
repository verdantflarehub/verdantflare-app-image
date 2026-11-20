"""FastAPI entrypoint for VerdantFlare image-moss."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import json

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.gpu_guard import LeaseFenceError, LeaseFenceGuard, default_guard_state_path
from src.pipeline import MossImagePipeline
from src.schemas import GenerateRequest, GenerateResponse, HealthResponse, MossApiError

pipeline = MossImagePipeline()
gpu_guard = LeaseFenceGuard(
    expected_consumer_kind="moss-image",
    identity=pipeline.runtime_identity,
    state_path=default_guard_state_path("moss-image"),
)


class RevokeLeaseRequest(BaseModel):
    lease_id: str
    fencing_token: int = 0


class ReclaimGuardRequest(BaseModel):
    lease_id: str = ""
    fencing_token: int = 0


@asynccontextmanager
async def lifespan(app: FastAPI):
    await asyncio.to_thread(pipeline.start)
    try:
        yield
    finally:
        await asyncio.to_thread(pipeline.close)


app = FastAPI(
    title="VerdantFlare Image MOSS",
    version="0.2.5",
    description="Internal microservice for MOSS-Image-SFT INT8-ConvRot inference backed by headless ComfyUI (pinned 19ae172, with MOSS-Image Edit support)",
    lifespan=lifespan,
)


@app.exception_handler(MossApiError)
async def moss_error_handler(request: Request, exc: MossApiError):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": "error",
            "error_code": exc.error_code,
            "message": exc.message,
        },
    )


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=400,
        content={
            "status": "error",
            "error_code": "INVALID_ARGUMENT",
            "message": str(exc),
        },
    )


@app.exception_handler(LeaseFenceError)
async def lease_fence_error_handler(request: Request, exc: LeaseFenceError):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": "error",
            "error_code": exc.error_code,
            "message": exc.message,
        },
    )


@app.get("/live")
def live():
    return {"status": "alive"}


@app.get("/health", response_model=HealthResponse)
def health_check():
    data = pipeline.health()
    return JSONResponse(
        status_code=200 if data["status"] == "healthy" else 503,
        content=data,
    )


@app.post("/v1/generate", response_model=GenerateResponse)
async def generate_image(req: GenerateRequest, request: Request):
    grant = gpu_guard.validate_request(request.headers)
    unheadered_exec_id = gpu_guard.begin_unheadered_execution() if grant is None else None
    try:
        result = await pipeline.generate(req)
    except Exception:
        if unheadered_exec_id is not None:
            gpu_guard.end_unheadered_execution(unheadered_exec_id)
        if grant is not None:
            gpu_guard.finish_failed_request(grant)
        raise
    if unheadered_exec_id is not None:
        gpu_guard.end_unheadered_execution(unheadered_exec_id)
    headers: dict[str, str] = {}
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
    proof["residency_state"] = pipeline.residency_state
    return proof
