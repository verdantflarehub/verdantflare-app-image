"""Use real pinned upstream modules. Synthetic ONNX graphs test the adapter only,
not face quality or the real model weights."""
import os
import tempfile
import unittest
import zlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.settings import Settings
from src.errors import FaceFusionPipelineError

SOURCE = os.getenv("FACEFUSION_SOURCE_ROOT", "")
if os.getenv("REQUIRE_FACEFUSION_TESTS") == "1" and not Path(SOURCE, "facefusion").is_dir():
    raise RuntimeError("Required official source checkout unavailable")


@unittest.skipUnless(Path(SOURCE, "facefusion").is_dir(), "Set FACEFUSION_SOURCE_ROOT to pinned checkout")
class TestOfficialBackend(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.settings = Settings(source_root=SOURCE, model_root=self.tmp.name, provider="cpu")
        from src.worker import OfficialBackend
        self.backend = OfficialBackend(self.settings, initialize=False)
        self.addCleanup(self.tmp.cleanup)

    def test_model_manifest_offline_and_no_source_modification(self):
        from facefusion import download
        with patch.object(download, "open_curl", side_effect=AssertionError("network access")):
            names = [Path(v["path"]).name for v in self.backend.manifest().values()]
            self.assertIn("hyperswap_1a_256.onnx", names)
            self.assertIn("codeformer.onnx", names)
            self.assertIn("arcface_w600k_r50.onnx", names)
            with self.assertRaisesRegex(ValueError, "Missing FaceFusion models"):
                self.backend.initialize()
        self.assertFalse(any(Path(self.tmp.name).iterdir()))

    def test_model_checksum_fails_before_sessions(self):
        first = next(iter(self.backend.manifest().values()))
        Path(first["path"]).write_bytes(b"bad weights")
        Path(first["hash_path"]).write_text("00000000")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            self.backend.initialize()
        self.assertEqual(Path(first["path"]).read_bytes(), b"bad weights")

    def test_all_sessions_really_load_on_cpu(self):
        import onnx
        import onnxruntime as ort
        from facefusion import inference_manager
        graph = onnx.helper.make_graph([onnx.helper.make_node("Identity", ["input"], ["output"])], "fixture",
            [onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1])],
            [onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1])])
        model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 13)], ir_version=8)
        blob = model.SerializeToString()
        for spec in self.backend.manifest().values():
            Path(spec["path"]).write_bytes(blob)
            Path(spec["hash_path"]).write_text(format(zlib.crc32(blob), "08x"))
        inference_manager.INFERENCE_POOL_SET = {"cli": {}, "ui": {}}
        def session(path, providers):
            options = ort.SessionOptions()
            options.intra_op_num_threads = options.inter_op_num_threads = 1
            return ort.InferenceSession(path, providers=providers, sess_options=options)
        with patch.object(inference_manager, "InferenceSession", side_effect=session):
            self.backend.initialize()
        health = self.backend.health()
        self.assertEqual(health["status"], "healthy")
        self.assertEqual(set(health["models_loaded"]), set(self.backend.manifest()))
        self.assertTrue(all(health["models_loaded"].values()))
        self.assertTrue(health["model_hashes"])
        inference_manager.INFERENCE_POOL_SET = {"cli": {}, "ui": {}}

    def test_non_integral_pixel_boost_rejected(self):
        from src.worker import OfficialBackend
        with self.assertRaisesRegex(ValueError, "integer multiple"):
            OfficialBackend(Settings(source_root=SOURCE, model_root=self.tmp.name, provider="cpu",
                                     swapper="blendswap_256", pixel_boost="384x384"), initialize=False)

    def test_pose_projection_and_selected_index(self):
        np, cv2 = self.backend.np, self.backend.cv2
        model = np.array([[0,0,0],[0,330,-65],[-225,-170,-135],[225,-170,-135],
                          [-150,150,-125],[150,150,-125]], dtype=np.float64)
        camera = np.array([[640,0,320],[0,640,240],[0,0,1]], dtype=np.float64)
        projected, _ = cv2.projectPoints(model, np.zeros((3,1)), np.array([[0.],[0.],[2000.]]), camera, np.zeros((4,1)))
        landmarks = np.zeros((68,2))
        landmarks[[30,8,36,45,48,54]] = projected.reshape(-1,2)
        face = SimpleNamespace(landmark_set={"68": landmarks}, score_set={"landmarker":0.9, "detector":0.99},
                               bounding_box=np.array([1,2,30,40]))
        angles = self.backend.pose(face, (480,640,3))
        self.assertTrue(all(abs(x)<0.1 for x in angles), angles)
        other = SimpleNamespace(landmark_set={}, score_set={"landmarker":0})
        result = self.backend.describe([other, face], 1, np.zeros((480,640,3)))
        self.assertTrue(result["pose_safe"])
        with self.assertRaises(FaceFusionPipelineError) as ctx:
            self.backend.describe([other, face], 0, np.zeros((480,640,3)))
        self.assertEqual(ctx.exception.error_code, "POSE_UNAVAILABLE")

    def test_official_swapper_and_enhancer_receive_controls(self):
        np = self.backend.np
        frame = np.zeros((32,32,3), dtype=np.uint8)
        face = SimpleNamespace(bounding_box=np.array([0,0,32,32]), embedding_norm=np.array([1.,0.]))
        self.backend.execution_provider = "CPUExecutionProvider"
        with patch.object(self.backend, "read", return_value=frame), \
             patch.object(self.backend.analyser, "analyse_frame", return_value=False), \
             patch.object(self.backend, "faces", return_value=[face]), \
             patch.object(self.backend, "describe", return_value={"pose_safe":True}), \
             patch.object(self.backend.swapper, "swap_face", return_value=frame) as swap, \
             patch.object(self.backend.enhancer, "enhance_face", return_value=frame) as enhance:
            result = self.backend.fuse("target", "source", 0, 0.95, False, 0.85, str(Path(self.tmp.name)/"out.png"))
            swap.assert_called_once()
            enhance.assert_not_called()
            self.assertAlmostEqual(self.backend.state.get_item("face_swapper_weight"), 0.9)
            self.assertNotIn("codeformer", result["pipeline"])
            self.assertEqual(result["execution"]["processors"], ["face_swapper"])
            result = self.backend.fuse("target", "source", 0, 0.5, True, 0.2, str(Path(self.tmp.name)/"out2.png"))
            enhance.assert_called_once()
            self.assertEqual(self.backend.state.get_item("face_swapper_weight"), 0)
            self.assertEqual(self.backend.state.get_item("face_enhancer_weight"), 0.2)

    def test_content_rejection_never_swaps(self):
        with patch.object(self.backend, "read", return_value=None), \
             patch.object(self.backend.analyser, "analyse_frame", return_value=True), \
             patch.object(self.backend.swapper, "swap_face") as swap:
            with self.assertRaises(FaceFusionPipelineError) as ctx:
                self.backend.fuse("t", "s", 0, 0.95, True, 0.85, "out")
            self.assertEqual(ctx.exception.error_code, "CONTENT_REJECTED")
            swap.assert_not_called()

    def test_actual_upstream_crop_pixel_boost_and_enhancer_tensor_contract(self):
        np = self.backend.np
        state = self.backend.state
        state.set_item("face_mask_types", ["box"])
        state.set_item("face_swapper_pixel_boost", "512x512")
        state.set_item("face_swapper_weight", 0.5)
        landmark = np.array([[180,180],[320,180],[250,250],[200,320],[300,320]], dtype=np.float32)
        face = SimpleNamespace(landmark_set={"5/68":landmark},
                               embedding=np.ones(512, dtype=np.float32),
                               embedding_norm=np.ones(512, dtype=np.float32)/np.sqrt(512))
        frame = np.full((512,512,3), 100, dtype=np.uint8)
        calls = []
        class SwapperSession:
            def get_inputs(inner):
                return [SimpleNamespace(name="source"), SimpleNamespace(name="target")]
            def run(inner, _, inputs):
                calls.append(inputs)
                return [inputs["target"]]
        with patch.object(self.backend.swapper, "get_inference_pool", return_value={"face_swapper":SwapperSession()}):
            swapped = self.backend.swapper.swap_face(face, face, frame, frame.copy())
        self.assertEqual(len(calls), 4)
        self.assertEqual(calls[0]["target"].shape, (1,3,256,256))
        self.assertEqual(calls[0]["source"].shape, (1,512))
        self.assertEqual(swapped.shape, frame.shape)
        self.assertTrue(np.isfinite(swapped).all())
        class EnhancerSession:
            def get_inputs(inner):
                return [SimpleNamespace(name="input"), SimpleNamespace(name="weight")]
            def run(inner, _, inputs):
                self.assertEqual(inputs["input"].shape, (1,3,512,512))
                self.assertEqual(inputs["weight"].dtype, np.float64)
                return [inputs["input"]]
        with patch.object(self.backend.enhancer, "get_inference_pool", return_value={"face_enhancer":EnhancerSession()}):
            enhanced = self.backend.enhancer.enhance_face(face, swapped)
        self.assertEqual(enhanced.shape, frame.shape)
        self.assertTrue(np.isfinite(enhanced).all())

    def test_real_worker_reports_missing_models_without_download(self):
        from PIL import Image
        from src.pipeline import FaceFusionPipeline
        image = Path(self.tmp.name) / "input.png"
        Image.new("RGB", (16,16)).save(image)
        pipeline = FaceFusionPipeline(self.settings)
        try:
            pipeline.start()
            self.assertEqual(pipeline.health()["status"], "not_ready")
            self.assertIn("Missing FaceFusion models", pipeline.health()["reason"])
            with self.assertRaises(FaceFusionPipelineError) as ctx:
                pipeline.detect_face(str(image))
            self.assertEqual(ctx.exception.error_code, "MODEL_NOT_READY")
            self.assertEqual(list(Path(self.tmp.name).iterdir()), [image])
        finally:
            pipeline.close()
