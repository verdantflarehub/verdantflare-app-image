import hashlib
import tempfile
import unittest
from pathlib import Path
from src.artifacts import ArtifactStore, ArtifactRecord, ArtifactNotFound, ArtifactError


class TestArtifacts(unittest.TestCase):
    def test_artifact_create_and_get(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            store = ArtifactStore(root_dir=tmp_path)
            store.ensure_ready()

            data = b"fake-image-content-12345"
            record = store.create_from_bytes(
                project_id="test-proj-01",
                filename="test.png",
                data=data,
                media_type="image/png",
                metadata={"engine": "codex"},
            )

            self.assertTrue(record.artifact_id.startswith("art-"))
            self.assertEqual(record.size_bytes, len(data))
            self.assertEqual(record.sha256, hashlib.sha256(data).hexdigest())

            # 读取验证
            fetched = store.get(record.artifact_id, project_id="test-proj-01")
            self.assertEqual(fetched.artifact_id, record.artifact_id)
            self.assertEqual(fetched.size_bytes, len(data))

            content_path = store.content_path(fetched)
            self.assertTrue(content_path.is_file())
            self.assertEqual(content_path.read_bytes(), data)

            # 404 测试
            with self.assertRaises(ArtifactNotFound):
                store.get("art-non-existent", project_id="test-proj-01")


if __name__ == "__main__":
    unittest.main()

