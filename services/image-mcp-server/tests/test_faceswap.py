import asyncio
import io
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from PIL import Image

from src.artifacts import ArtifactStore
from src.faceswap import handle_faceswap, resolve_source_face_path, resolve_target_image_path
from src.server import image_faceswap


class TestFaceSwap(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.store = ArtifactStore(root_dir=Path(self.tmp_dir))
        self.project_id = "test-project"

        # 创建一个测试用的虚拟 PNG
        img = Image.new("RGB", (100, 100), color=(255, 0, 0))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        self.png_data = buf.getvalue()

        # 在 ArtifactStore 中注册目标大片
        self.target_artifact = self.store.create_from_bytes(
            project_id=self.project_id,
            filename="target_model.png",
            data=self.png_data,
        )

        # 在 ArtifactStore 中注册小月基准角色卡
        self.source_artifact = self.store.create_from_bytes(
            project_id=self.project_id,
            filename="xiaoyue_id.png",
            data=self.png_data,
        )

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_resolve_paths_success(self):
        rec, path = resolve_target_image_path(self.store, self.target_artifact.artifact_id, self.project_id)
        self.assertEqual(rec.artifact_id, self.target_artifact.artifact_id)
        self.assertTrue(path.is_file())

        source_path = resolve_source_face_path(self.store, self.source_artifact.artifact_id, self.project_id)
        self.assertTrue(source_path.is_file())

    def test_handle_faceswap_target_not_found(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            res = loop.run_until_complete(
                handle_faceswap(
                    artifacts=self.store,
                    target_artifact_id="art-nonexistent",
                    source_identity_artifact_id=self.source_artifact.artifact_id,
                    project_id=self.project_id,
                )
            )
            self.assertEqual(res["status"], "failed")
            self.assertIn("目标大片解析失败", res["error"])
        finally:
            loop.close()

    def test_handle_faceswap_source_not_found(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            res = loop.run_until_complete(
                handle_faceswap(
                    artifacts=self.store,
                    target_artifact_id=self.target_artifact.artifact_id,
                    source_identity_artifact_id="nonexistent-face-id-xyz",
                    project_id=self.project_id,
                )
            )
            self.assertEqual(res["status"], "failed")
            self.assertIn("基准角色卡解析失败", res["error"])
        finally:
            loop.close()

    @patch("httpx.AsyncClient.post")
    def test_handle_faceswap_success(self, mock_post):
        # 模拟后端微服务执行生成输出文件
        async def fake_post(url, json=None, **kwargs):
            out_path = Path(json["output_path"])
            out_path.parent.mkdir(parents=True, exist_ok=True)
            # 写入一个融合后的测试图片
            out_img = Image.new("RGB", (2160, 3840), color=(0, 255, 0))
            out_img.save(out_path, format="PNG")

            from unittest.mock import MagicMock
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {
                "status": "success",
                "output_path": str(out_path),
                "detected_faces": 1,
                "arcface_similarity": 0.925,
                "inference_time_ms": 480,
                "pipeline": "retinaface+arcface512+inswapper128+codeformer",
            }
            return mock_resp

        mock_post.side_effect = fake_post

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            res = loop.run_until_complete(
                handle_faceswap(
                    artifacts=self.store,
                    target_artifact_id=self.target_artifact.artifact_id,
                    source_identity_artifact_id=self.source_artifact.artifact_id,
                    identity_strength=0.95,
                    restore_face=True,
                    restoration_fidelity=0.85,
                    project_id=self.project_id,
                )
            )
            self.assertEqual(res["status"], "completed")
            self.assertEqual(res["project_id"], self.project_id)
            self.assertIn("artifact", res)
            self.assertIn("download_path", res)
            self.assertEqual(res["metrics"]["arcface_similarity"], 0.925)
            self.assertEqual(res["metrics"]["width"], 2160)
            self.assertEqual(res["metrics"]["height"], 3840)

            # 验证 ArtifactStore 中能够检索到新记录
            fused_art_id = res["artifact"]["artifact_id"]
            retrieved = self.store.get(fused_art_id, self.project_id)
            self.assertIsNotNone(retrieved)
            self.assertEqual(retrieved.metadata["type"], "face_fusion")
            self.assertEqual(retrieved.metadata["arcface_similarity"], 0.925)
        finally:
            loop.close()

    @patch("httpx.AsyncClient.post")
    def test_handle_faceswap_timeout(self, mock_post):
        import httpx

        mock_post.side_effect = httpx.TimeoutException("Connection timed out")

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            res = loop.run_until_complete(
                handle_faceswap(
                    artifacts=self.store,
                    target_artifact_id=self.target_artifact.artifact_id,
                    source_identity_artifact_id=self.source_artifact.artifact_id,
                    project_id=self.project_id,
                )
            )
            self.assertEqual(res["status"], "failed")
            self.assertIn("超时", res["error"])
        finally:
            loop.close()

    @patch("src.server.handle_faceswap")
    def test_mcp_image_faceswap_tool(self, mock_handle):
        mock_handle.return_value = {
            "status": "completed",
            "project_id": "test-proj",
            "artifact": {"artifact_id": "art-fused-999"},
        }

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            tool_res = loop.run_until_complete(
                image_faceswap(
                    target_artifact_id="art-target-1",
                    source_identity_artifact_id="art-source-1",
                )
            )
            self.assertIsNotNone(tool_res)
            self.assertEqual(len(tool_res.content), 1)
            parsed = json.loads(tool_res.content[0].text)
            self.assertEqual(parsed["status"], "completed")
            self.assertEqual(parsed["artifact"]["artifact_id"], "art-fused-999")
        finally:
            loop.close()


if __name__ == "__main__":
    unittest.main()
