import os
import shutil
import tempfile
import unittest
from pathlib import Path
from PIL import Image
from starlette.testclient import TestClient

from src.main import app


class TestImageFaceFusionAPI(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.temp_dir = tempfile.mkdtemp(prefix="test_face_fusion_")
        self.temp_path = Path(self.temp_dir)

        # Create dummy target 4K canvas image
        self.target_img_path = self.temp_path / "canvas_4k.png"
        img_target = Image.new("RGB", (2160, 3840), color=(180, 150, 130))
        img_target.save(self.target_img_path, format="PNG")

        # Create dummy source face identity image
        self.source_img_path = self.temp_path / "xiaoyue_id.png"
        img_source = Image.new("RGB", (1024, 1536), color=(220, 200, 190))
        img_source.save(self.source_img_path, format="PNG")

        self.output_img_path = self.temp_path / "fused_output.png"

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_health_check(self):
        """Test GET /health probe returns valid status and provider."""
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "healthy")
        self.assertIn("cuda_available", data)
        self.assertIn("execution_provider", data)
        self.assertIn("models_loaded", data)

    def test_detect_face_success(self):
        """Test POST /v1/detect succeeds for valid image."""
        payload = {"image_path": str(self.target_img_path)}
        resp = self.client.post("/v1/detect", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertGreaterEqual(data["face_count"], 1)
        self.assertIn("pitch", data)
        self.assertIn("yaw", data)
        self.assertIn("roll", data)
        self.assertTrue(data["pose_safe"])
        self.assertEqual(len(data["bounding_box"]), 4)

    def test_detect_face_file_not_found(self):
        """Test POST /v1/detect returns 404 for nonexistent image."""
        payload = {"image_path": "/nonexistent/path/image.png"}
        resp = self.client.post("/v1/detect", json=payload)
        self.assertEqual(resp.status_code, 404)
        data = resp.json()
        self.assertEqual(data["error_code"], "FILE_PATH_UNREACHABLE")

    def test_fuse_face_success(self):
        """Test POST /v1/fuse executes end-to-end and outputs file with metrics."""
        payload = {
            "target_image_path": str(self.target_img_path),
            "source_face_path": str(self.source_img_path),
            "target_face_index": 0,
            "identity_strength": 0.95,
            "restore_face": True,
            "restoration_fidelity": 0.85,
            "output_path": str(self.output_img_path),
        }
        resp = self.client.post("/v1/fuse", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "success")
        self.assertTrue(self.output_img_path.is_file())
        self.assertGreaterEqual(data["arcface_similarity"], 0.85)
        self.assertGreaterEqual(data["inference_time_ms"], 0)
        self.assertEqual(data["pipeline"], "retinaface+arcface512+inswapper128+codeformer")

    def test_fuse_face_validation_error(self):
        """Test POST /v1/fuse rejects invalid identity_strength."""
        payload = {
            "target_image_path": str(self.target_img_path),
            "source_face_path": str(self.source_img_path),
            "identity_strength": 1.5,  # Out of range [0.5, 1.0]
            "output_path": str(self.output_img_path),
        }
        resp = self.client.post("/v1/fuse", json=payload)
        self.assertEqual(resp.status_code, 422)

    def test_fuse_face_missing_source(self):
        """Test POST /v1/fuse returns 404 when source image is missing."""
        payload = {
            "target_image_path": str(self.target_img_path),
            "source_face_path": "/missing/source_id.png",
            "output_path": str(self.output_img_path),
        }
        resp = self.client.post("/v1/fuse", json=payload)
        self.assertEqual(resp.status_code, 404)
        data = resp.json()
        self.assertEqual(data["error_code"], "FILE_PATH_UNREACHABLE")


if __name__ == "__main__":
    unittest.main()
