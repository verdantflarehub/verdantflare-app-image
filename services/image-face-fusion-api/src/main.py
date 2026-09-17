import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from src.schemas import FuseRequest, FuseResponse, DetectRequest, DetectResponse, HealthResponse
from src.pipeline import FaceFusionPipeline, FaceFusionPipelineError

pipeline = FaceFusionPipeline()


@asynccontextmanager
async def lifespan(app):
    await asyncio.to_thread(pipeline.start)
    try:
        yield
    finally:
        await asyncio.to_thread(pipeline.close)


app = FastAPI(title="VerdantFlare Image Face Fusion API", version="0.2.0",
              description="Internal API backed by official FaceFusion 3.9.0", lifespan=lifespan)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])


@app.exception_handler(FaceFusionPipelineError)
async def pipeline_exception_handler(request: Request, exc: FaceFusionPipelineError):
    return JSONResponse(status_code=exc.status_code,
                        content={"error_code": exc.error_code, "message": exc.message, "status": "error"})


@app.get("/live")
def live():
    return {"status": "alive"}


@app.get("/health", response_model=HealthResponse)
def health_check():
    data = pipeline.health()
    return JSONResponse(status_code=200 if data["status"] == "healthy" else 503, content=data)


@app.post("/v1/detect", response_model=DetectResponse)
def detect_face(req: DetectRequest):
    return pipeline.detect_face(req.image_path, req.target_face_index)


@app.post("/v1/fuse", response_model=FuseResponse)
def fuse_face(req: FuseRequest):
    return pipeline.fuse(**req.model_dump())
