from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class FuseRequest(BaseModel):
    """Request payload for high-fidelity face fusion."""
    target_image_path: str = Field(..., description="Absolute physical path of the canvas image to swap face on")
    source_face_path: str = Field(..., description="Absolute physical path of the baseline identity image")
    target_face_index: int = Field(default=0, ge=0, description="Target face index if multiple faces exist")
    identity_strength: float = Field(default=0.95, ge=0.5, le=1.0, description="ArcFace embedding injection weight")
    restore_face: bool = Field(default=True, description="Enable CodeFormer face super-resolution restoration")
    restoration_fidelity: float = Field(default=0.85, ge=0.0, le=1.0, description="Restoration fidelity weight")
    output_path: str = Field(..., description="Absolute physical path for the output masterpiece")


class FuseResponse(BaseModel):
    """Response payload for face fusion."""
    status: str = Field(default="success", description="Business status flag")
    output_path: str = Field(..., description="Absolute physical path of output image")
    detected_faces: int = Field(..., description="Number of detected faces in target image")
    arcface_similarity: float = Field(..., description="Cosine similarity score against source identity")
    inference_time_ms: int = Field(..., description="Total pipeline execution latency in milliseconds")
    pipeline: str = Field(default="retinaface+arcface512+inswapper128+codeformer", description="Executed pipeline signature")


class DetectRequest(BaseModel):
    """Request payload for face detection and pose verification."""
    image_path: str = Field(..., description="Absolute physical path of the image to detect")


class DetectResponse(BaseModel):
    """Response payload for face detection and pose analysis."""
    face_count: int = Field(..., description="Number of detected faces")
    pitch: float = Field(..., description="Head pitch angle in degrees (-35 to +35 safe)")
    yaw: float = Field(..., description="Head yaw angle in degrees (-45 to +45 safe)")
    roll: float = Field(..., description="Head roll angle in degrees (-30 to +30 safe)")
    confidence: float = Field(..., description="Detection confidence score (0.0 to 1.0)")
    bounding_box: List[int] = Field(..., description="[x1, y1, x2, y2] bounding box coordinates")
    pose_safe: bool = Field(..., description="Whether head pose is within non-distorted threshold")


class HealthResponse(BaseModel):
    """Response payload for health probe."""
    status: str = Field(default="healthy", description="Service health status")
    cuda_available: bool = Field(..., description="Whether CUDA execution provider is available")
    execution_provider: str = Field(..., description="Active ONNX Runtime execution provider")
    models_loaded: Dict[str, bool] = Field(..., description="Status of pipeline model weights")
