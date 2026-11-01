import base64
import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from src.artifacts import ArtifactError, ArtifactNotFound, ArtifactStore
from src.imports import ImportStore


PROJECT = "00000000-0000-7000-8000-000000000001"
OTHER = "00000000-0000-7000-8000-000000000002"


class ImportDeleteTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.artifacts = ArtifactStore(Path(self.tmp.name))
        self.store = ImportStore(self.artifacts)
        stream = io.BytesIO()
        Image.new("RGB", (80, 60), "blue").save(stream, format="PNG")
        self.data = stream.getvalue()
        self.digest = hashlib.sha256(self.data).hexdigest()
        self.args = dict(project_id=PROJECT, idempotency_key="photo", filename="photo.png",
                         size=len(self.data), sha256=self.digest, purpose="tryon-reference")
        self.receipt = self.store.prepare(**self.args)
        self.import_id = self.receipt["import_id"]

    def tearDown(self):
        self.artifacts._thumb_executor.shutdown(wait=True)
        self.tmp.cleanup()

    def chunk(self):
        return self.store.chunk(project_id=PROJECT, import_id=self.import_id, offset=0,
                                content_base64=base64.b64encode(self.data).decode(), sha256=self.digest)

    def commit(self):
        self.chunk()
        return self.store.commit(project_id=PROJECT, import_id=self.import_id)

    def test_committed_import_deletes_bytes_metadata_previews_and_caches(self):
        result = self.commit()
        native = result["artifact"]["artifact_id"]
        record = self.artifacts.get(native, PROJECT)
        original = self.artifacts.content_path(record)
        previews = [self.artifacts.get_or_create_thumbnail(native, edge, PROJECT)[0] for edge in (320, 640)]
        deleted = self.store.delete(project_id=PROJECT, import_id=self.import_id)
        self.assertEqual(deleted, {"project_id": PROJECT, "import_id": self.import_id, "status": "deleted"})
        self.assertFalse(original.exists())
        self.assertTrue(all(not p.exists() for p in previews))
        with self.assertRaises(ArtifactNotFound):
            self.artifacts.get(native, PROJECT)
        self.assertNotIn((PROJECT, native), self.artifacts._record_cache)
        self.assertEqual(self.store.delete(project_id=PROJECT, import_id=self.import_id), deleted)

    def test_uncommitted_import_cleans_parts_and_cannot_be_revived(self):
        self.chunk()
        self.assertTrue(self.store._part(self.import_id).exists())
        self.store.delete(project_id=PROJECT, import_id=self.import_id)
        self.assertFalse(self.store._part(self.import_id).exists())
        self.assertEqual(self.store.prepare(**self.args)["status"], "deleted")
        with self.assertRaisesRegex(ValueError, "import_not_active"):
            self.chunk()
        with self.assertRaisesRegex(ValueError, "import_not_active"):
            self.store.commit(project_id=PROJECT, import_id=self.import_id)

    def test_cleanup_failure_persists_tombstone_and_retry_finishes(self):
        result = self.commit()
        native = result["artifact"]["artifact_id"]
        content = self.artifacts.content_path(self.artifacts.get(native, PROJECT))
        original_unlink = Path.unlink
        def fail_content(path, *args, **kwargs):
            if path == content:
                raise OSError("simulated disk failure")
            return original_unlink(path, *args, **kwargs)
        with patch.object(Path, "unlink", fail_content):
            with self.assertRaisesRegex(ValueError, "import_cleanup_unavailable"):
                self.store.delete(project_id=PROJECT, import_id=self.import_id)
        self.assertEqual(self.store.status(project_id=PROJECT, import_id=self.import_id)["status"], "deleting")
        self.assertEqual(self.store.prepare(**self.args)["status"], "deleting")
        with self.assertRaises(ArtifactNotFound):
            self.artifacts.get(native, PROJECT)
        self.assertTrue(content.exists())
        recovered = ImportStore(ArtifactStore(Path(self.tmp.name)))
        try:
            self.assertEqual(recovered.delete(project_id=PROJECT, import_id=self.import_id)["status"], "deleted")
        finally:
            recovered.artifacts._thumb_executor.shutdown(wait=True)
        self.assertFalse(content.exists())

    def test_cross_project_delete_does_not_touch_photo(self):
        result = self.commit()
        with self.assertRaisesRegex(ValueError, "import_not_found"):
            self.store.delete(project_id=OTHER, import_id=self.import_id)
        self.assertEqual(self.store.status(project_id=PROJECT, import_id=self.import_id)["status"], "committed")
        self.assertEqual(self.artifacts.content_path(self.artifacts.get(result["artifact"]["artifact_id"], PROJECT)).read_bytes(), self.data)

    def test_queued_preview_cannot_recreate_after_partial_delete(self):
        result = self.commit()
        native = result["artifact"]["artifact_id"]
        self.artifacts._thumb_executor.shutdown(wait=True)
        self.artifacts.thumbnail_cache_path(self.digest, 320).unlink(missing_ok=True)
        original_get_lock = self.artifacts._get_thumb_lock
        original_unlink = Path.unlink
        content = self.artifacts.content_path(self.artifacts.get(native, PROJECT))
        def fail_content(path, *args, **kwargs):
            if path == content:
                raise OSError("simulated disk failure")
            return original_unlink(path, *args, **kwargs)
        def delete_before_preview_lock(key):
            with patch.object(self.artifacts, "_get_thumb_lock", original_get_lock):
                with patch.object(Path, "unlink", fail_content):
                    with self.assertRaisesRegex(ValueError, "import_cleanup_unavailable"):
                        self.store.delete(project_id=PROJECT, import_id=self.import_id)
            return original_get_lock(key)
        with patch.object(self.artifacts, "_get_thumb_lock", delete_before_preview_lock):
            with self.assertRaises(ArtifactNotFound):
                self.artifacts.get_or_create_thumbnail(native, 320, PROJECT)
        self.assertFalse(self.artifacts.thumbnail_cache_path(self.digest, 320).exists())

    def test_same_hash_other_artifact_original_survives(self):
        self.commit()
        other = self.artifacts.create_from_bytes(project_id=OTHER, filename="independent.png", data=self.data,
                    media_type="image/png", metadata={"origin": "generation"})
        self.store.delete(project_id=PROJECT, import_id=self.import_id)
        self.assertEqual(self.artifacts.content_path(self.artifacts.get(other.artifact_id, OTHER)).read_bytes(), self.data)
        self.assertTrue(self.artifacts.get_or_create_thumbnail(other.artifact_id, 320, OTHER)[0].exists())

    def test_direct_helper_refuses_nonimport_output(self):
        other = self.artifacts.create_from_bytes(project_id=PROJECT, filename="output.png", data=self.data,
                    media_type="image/png", metadata={"origin": "generation"})
        with self.assertRaisesRegex(ArtifactError, "not_the_requested_import"):
            self.artifacts.remove_imported(project_id=PROJECT, artifact_id=other.artifact_id,
                    import_id=self.import_id, filename=other.filename, sha256=self.digest)
        self.assertTrue(self.artifacts.content_path(other).exists())


if __name__ == "__main__":
    unittest.main()
