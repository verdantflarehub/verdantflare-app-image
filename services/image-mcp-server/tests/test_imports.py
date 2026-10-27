import base64
import hashlib
import io
import tempfile
import time
import unittest
from pathlib import Path

from PIL import Image

from src import imports as imports_module
from src.artifacts import ArtifactStore
from src.imports import MAX_CHUNK_BYTES, ImportStore

PROJECT = "01a16250-d1aa-7667-add1-588f58042bbc"


def png(width=64, height=32, noise=False) -> bytes:
    image = Image.effect_noise((width, height), 64).convert("RGB") if noise else Image.new("RGB", (width, height), (10, 20, 30))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class ImportTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.artifacts = ArtifactStore(Path(self.tmp.name))
        self.store = ImportStore(self.artifacts)

    def tearDown(self):
        self.tmp.cleanup()

    def upload(self, data: bytes, key="k1", filename="main.png", chunk=MAX_CHUNK_BYTES):
        prepared = self.store.prepare(project_id=PROJECT, idempotency_key=key, filename=filename,
                                      size=len(data), sha256=sha(data), purpose="T5.7 base image")
        for offset in range(0, len(data), chunk):
            part = data[offset:offset + chunk]
            self.store.chunk(project_id=PROJECT, import_id=prepared["import_id"], offset=offset,
                             content_base64=b64(part), sha256=sha(part))
        return prepared["import_id"]


class TestSingleImport(ImportTestCase):
    def test_imports_png_and_retry_is_idempotent(self):
        data = png()
        first = self.store.import_single(project_id=PROJECT, filename="crop.png",
                                         expected_sha256=sha(data), content_base64=b64(data))
        again = self.store.import_single(project_id=PROJECT, filename="crop.png",
                                         expected_sha256=sha(data), content_base64=b64(data))
        artifact = first["artifact"]
        self.assertEqual(artifact["artifact_id"], again["artifact"]["artifact_id"])
        self.assertRegex(artifact["artifact_id"], r"^art-[0-9a-f]{16}$")
        self.assertEqual((artifact["sha256"], artifact["size_bytes"]), (sha(data), len(data)))
        self.assertEqual(first["media"], {"media_type": "image/png", "width": 64, "height": 32})
        self.assertEqual(artifact["metadata"]["origin"], "user_import")
        stored = self.artifacts.get(artifact["artifact_id"], PROJECT)
        self.assertEqual(self.artifacts.content_path(stored).read_bytes(), data)

    def test_rejects_hash_mismatch_format_mismatch_and_corrupt_bytes(self):
        data = png()
        cases = [
            (dict(filename="a.png", expected_sha256="0" * 64, content_base64=b64(data)), "import_sha256_mismatch"),
            (dict(filename="a.jpg", expected_sha256=sha(data), content_base64=b64(data)), "import_format_mismatch"),
            (dict(filename="a.gif", expected_sha256=sha(data), content_base64=b64(data)), "unsupported_import_format"),
            (dict(filename="../a.png", expected_sha256=sha(data), content_base64=b64(data)), "invalid_import_filename"),
            (dict(filename="a.png", expected_sha256=sha(data[:40]), content_base64=b64(data[:40])), "import_decode_failed"),
            (dict(filename="a.png", expected_sha256=sha(data), content_base64="not base64!"), "invalid_import_base64"),
        ]
        for kwargs, code in cases:
            with self.subTest(code=code):
                with self.assertRaisesRegex(ValueError, f"^{code}$"):
                    self.store.import_single(project_id=PROJECT, **kwargs)

    def test_single_call_limit(self):
        original = imports_module.MAX_SINGLE_BYTES
        imports_module.MAX_SINGLE_BYTES = 100
        try:
            data = png()
            with self.assertRaisesRegex(ValueError, "^import_content_too_large_or_empty$"):
                self.store.import_single(project_id=PROJECT, filename="a.png",
                                         expected_sha256=sha(data), content_base64=b64(data))
        finally:
            imports_module.MAX_SINGLE_BYTES = original


class TestChunkedImport(ImportTestCase):
    def test_chunked_import_resume_and_commit(self):
        data = png(600, 400, noise=True)
        self.assertGreater(len(data), 3000)
        import_id = self.upload(data, chunk=1000)
        status = self.store.status(project_id=PROJECT, import_id=import_id)
        self.assertEqual((status["status"], status["offset"]), ("uploading", len(data)))
        committed = self.store.commit(project_id=PROJECT, import_id=import_id)
        self.assertEqual(committed["status"], "committed")
        self.assertEqual(committed["artifact"]["sha256"], sha(data))
        self.assertEqual(committed["media"]["width"], 600)
        self.assertEqual(self.store.commit(project_id=PROJECT, import_id=import_id)["artifact"],
                         committed["artifact"])
        record = self.artifacts.get(committed["artifact"]["artifact_id"], PROJECT)
        self.assertEqual(self.artifacts.content_path(record).read_bytes(), data)

    def test_repeated_chunk_is_safe_and_conflicts_are_rejected(self):
        data = png(300, 200, noise=True)
        prepared = self.store.prepare(project_id=PROJECT, idempotency_key="k", filename="m.png",
                                      size=len(data), sha256=sha(data), purpose="p")
        import_id, first = prepared["import_id"], data[:1000]
        self.store.chunk(project_id=PROJECT, import_id=import_id, offset=0, content_base64=b64(first), sha256=sha(first))
        repeated = self.store.chunk(project_id=PROJECT, import_id=import_id, offset=0,
                                    content_base64=b64(first), sha256=sha(first))
        self.assertEqual(repeated["offset"], 1000)
        other = bytes(1000)
        with self.assertRaisesRegex(ValueError, "^import_repeated_chunk_mismatch$"):
            self.store.chunk(project_id=PROJECT, import_id=import_id, offset=0, content_base64=b64(other), sha256=sha(other))
        gap = data[2000:2100]
        with self.assertRaisesRegex(ValueError, "^import_offset_or_status_conflict$"):
            self.store.chunk(project_id=PROJECT, import_id=import_id, offset=2000, content_base64=b64(gap), sha256=sha(gap))
        with self.assertRaisesRegex(ValueError, "^import_incomplete$"):
            self.store.commit(project_id=PROJECT, import_id=import_id)

    def test_prepare_idempotency_and_limits(self):
        data = png()
        first = self.store.prepare(project_id=PROJECT, idempotency_key="same", filename="m.png",
                                   size=len(data), sha256=sha(data), purpose="p")
        again = self.store.prepare(project_id=PROJECT, idempotency_key="same", filename="m.png",
                                   size=len(data), sha256=sha(data), purpose="p")
        self.assertEqual(first["import_id"], again["import_id"])
        with self.assertRaisesRegex(ValueError, "^import_idempotency_conflict$"):
            self.store.prepare(project_id=PROJECT, idempotency_key="same", filename="m.png",
                               size=len(data) + 1, sha256=sha(data), purpose="p")
        with self.assertRaisesRegex(ValueError, "^import_size_must_be_1_byte_to_20_MiB$"):
            self.store.prepare(project_id=PROJECT, idempotency_key="big", filename="m.png",
                               size=20 * 1024 * 1024 + 1, sha256=sha(data), purpose="p")
        self.assertIsNotNone(self.store.prepare(project_id=PROJECT, idempotency_key="max", filename="m.png",
                                                size=20 * 1024 * 1024, sha256=sha(data), purpose="p"))
        with self.assertRaisesRegex(ValueError, "^fixed_content_ref_required$"):
            self.store.prepare(project_id=PROJECT, idempotency_key="ref", filename="m.png", size=len(data),
                               sha256=sha(data), purpose="p", source_content_ref={"artifact_id": "x"})

    def test_imports_are_project_scoped(self):
        import_id = self.upload(png())
        with self.assertRaisesRegex(ValueError, "^import_not_found$"):
            self.store.status(project_id="another-project", import_id=import_id)

    def test_pixel_budget(self):
        original = imports_module.MAX_PIXELS
        imports_module.MAX_PIXELS = 1000
        try:
            data = png(64, 32)
            import_id = self.upload(data)
            with self.assertRaisesRegex(ValueError, "^import_pixel_budget_exceeded$"):
                self.store.commit(project_id=PROJECT, import_id=import_id)
        finally:
            imports_module.MAX_PIXELS = original

    def test_stale_uploads_expire(self):
        import_id = self.upload(png()[:100], key="old")
        original = imports_module.STALE_SECONDS
        imports_module.STALE_SECONDS = -1
        try:
            self.store.prepare(project_id=PROJECT, idempotency_key="new", filename="n.png",
                               size=10, sha256="a" * 64, purpose="p")
        finally:
            imports_module.STALE_SECONDS = original
        with self.assertRaisesRegex(ValueError, "^import_not_found$"):
            self.store.status(project_id=PROJECT, import_id=import_id)


if __name__ == "__main__":
    unittest.main()
