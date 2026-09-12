import logging
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from src.schemas import (
    FuseRequest,
    FuseResponse,
    DetectRequest,
    DetectResponse,
    HealthResponse,
)
from src.pipeline import (
    FaceFusionPipeline,
    FaceFusionPipelineError,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("image-face-fusion-api")

app = FastAPI(
    title="VerdantFlare Image Face Fusion API",
    version="0.1.0",
    description="Internal microservice for high-fidelity face fusion, pose detection, and CodeFormer restoration.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = FaceFusionPipeline()


@app.exception_handler(FaceFusionPipelineError)
async def pipeline_exception_handler(request: Request, exc: FaceFusionPipelineError):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error_code": exc.error_code,
            "message": exc.message,
            "status": "error",
        },
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Kubernetes liveness and readiness probe endpoint."""
    return HealthResponse(
        status="healthy",
        cuda_available=pipeline.cuda_available,
        execution_provider=pipeline.execution_provider,
        models_loaded=pipeline.models_loaded,
    )


@app.post("/v1/detect", response_model=DetectResponse)
async def detect_face(req: DetectRequest):
    """Detect faces and verify head pose pitch/yaw/roll."""
    result = pipeline.detect_face(req.image_path)
    return DetectResponse(**result)


@app.post("/v1/fuse", response_model=FuseResponse)
async def fuse_face(req: FuseRequest):
    """Execute high-fidelity face fusion pipeline with zero-copy local paths."""
    result = pipeline.fuse(
        target_image_path=req.target_image_path,
        source_face_path=req.source_face_path,
        target_face_index=req.target_face_index,
        identity_strength=req.identity_strength,
        restore_face=req.restore_face,
        restoration_fidelity=req.restoration_fidelity,
        output_path=req.output_path,
    )
    return FuseResponse(**result)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=False)
