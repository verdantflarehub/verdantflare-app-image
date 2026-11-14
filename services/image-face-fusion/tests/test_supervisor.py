import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image
from src.pipeline import FaceFusionPipeline, FaceFusionPipelineError
from src.settings import Settings


def fixture_worker(connection, settings):
    connection.send(dict(status="healthy", cuda_available=False, execution_provider="CPUExecutionProvider",
                         models_loaded={"fixture": True}, backend="fixture", backend_version="test"))
    try:
        while True:
            operation, payload = connection.recv()
            if settings.swapper == "crash":
                os._exit(1)
            if settings.swapper == "slow":
                time.sleep(2)
            if operation == "fuse":
                Image.new("RGB", (16, 16)).save(payload["output_path"])
                connection.send({"result": dict(status="success", detected_faces=1, arcface_similarity=0.7,
                                                pipeline="fixture", execution={"weight": payload["identity_strength"]})})
            else:
                connection.send({"result": {"index": payload["target_face_index"]}})
    except EOFError:
        pass


class TestSupervisor(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.input = self.root / "input.png"
        self.output = self.root / "output.png"
        Image.new("RGB", (16, 16)).save(self.input)
        self.pipeline = FaceFusionPipeline(Settings(provider="cpu", startup_timeout=10, timeout=2), fixture_worker)

    def tearDown(self):
        self.pipeline.close()
        self.tmp.cleanup()

    def fuse(self):
        return self.pipeline.fuse(str(self.input), str(self.input), output_path=str(self.output))

    def test_reuses_process_and_publishes_png(self):
        self.pipeline.start()
        pid = self.pipeline._process.pid
        result = self.fuse()
        self.assertEqual(result["output_path"], str(self.output))
        self.assertEqual(self.pipeline.detect_face(str(self.input), 2), {"index": 2})
        self.assertEqual(self.pipeline._process.pid, pid)
        self.assertEqual(self.pipeline.health()["status"], "healthy")
        self.assertFalse(list(self.root.glob(".facefusion-*")))

    def test_existing_output_never_overwritten(self):
        self.output.write_bytes(b"original")
        with self.assertRaises(FaceFusionPipelineError) as ctx:
            self.fuse()
        self.assertEqual(ctx.exception.error_code, "OUTPUT_EXISTS")
        self.assertEqual(self.output.read_bytes(), b"original")

    def test_busy_is_bounded(self):
        with self.pipeline._lock:
            with self.assertRaises(FaceFusionPipelineError) as ctx:
                self.fuse()
        self.assertEqual(ctx.exception.error_code, "WORKER_BUSY")
        self.assertFalse(self.output.exists())
        self.assertFalse(list(self.root.glob(".facefusion-*")))

    def test_timeout_kills_worker_and_cleans_temp(self):
        self.pipeline.settings = Settings(provider="cpu", swapper="slow", timeout=0.1, startup_timeout=10)
        with self.assertRaises(FaceFusionPipelineError) as ctx:
            self.fuse()
        self.assertEqual(ctx.exception.status_code, 504)
        self.assertEqual(self.pipeline.health()["status"], "not_ready")
        self.assertFalse(self.output.exists())
        self.assertFalse(list(self.root.glob(".facefusion-*")))
        self.pipeline.settings = Settings(provider="cpu", startup_timeout=10)
        self.fuse()
        self.assertTrue(self.output.exists())

    def test_crash_never_publishes(self):
        self.pipeline.settings = Settings(provider="cpu", swapper="crash", startup_timeout=10)
        with self.assertRaises(FaceFusionPipelineError) as ctx:
            self.fuse()
        self.assertEqual(ctx.exception.error_code, "WORKER_EXITED")
        self.assertFalse(self.output.exists())

    def test_output_race_does_not_clobber(self):
        def fake_call(*args):
            self.output.write_bytes(b"other writer")
            Image.new("RGB", (8, 8)).save(args[1]["output_path"])
            return {}
        with patch.object(self.pipeline, "_call", side_effect=fake_call):
            with self.assertRaises(FaceFusionPipelineError):
                self.fuse()
        self.assertEqual(self.output.read_bytes(), b"other writer")

    def test_invalid_input_and_size_limit(self):
        self.pipeline.settings = Settings(max_pixels=1)
        with self.assertRaises(FaceFusionPipelineError) as ctx:
            self.fuse()
        self.assertEqual(ctx.exception.status_code, 413)
        self.input.write_bytes(b"not an image")
        with self.assertRaises(FaceFusionPipelineError) as ctx:
            self.fuse()
        self.assertEqual(ctx.exception.error_code, "INVALID_IMAGE")

    def test_no_startup_at_import(self):
        self.assertIsNone(self.pipeline._process)
        self.assertEqual(self.pipeline.health()["status"], "not_ready")
