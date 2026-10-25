"""FastAPI entrypoint for VerdantFlare image-moss-api."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from src.pipeline import MossImagePipeline
from src.schemas import GenerateRequest, GenerateResponse, HealthResponse, MossApiError

pipeline = MossImagePipeline()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await asyncio.to_thread(pipeline.start)
    try:
        yield
    finally:
        await asyncio.to_thread(pipeline.close)


app = FastAPI(
    title="VerdantFlare Image MOSS API",
    version="0.1.3",
    description="Internal microservice for MOSS-Image-SFT INT8-ConvRot inference backed by headless ComfyUI (pinned 067f853, with MOSS-Image Edit support)",
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
async def generate_image(req: GenerateRequest):
    return await pipeline.generate(req)
