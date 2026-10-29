import hashlib
import tempfile
import unittest
from pathlib import Path
from src.artifacts import ArtifactStore, ArtifactRecord, ArtifactNotFound, ArtifactError


class TestArtifacts(unittest.TestCase):
    def test_nested_project_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = ArtifactStore(root_dir=Path(tmp))
            record = store.create_from_bytes(project_id="creator/xiaoyue", filename="test.png", data=b"image", media_type="image/png")
            self.assertEqual(store.get(record.artifact_id).project_id, "creator/xiaoyue")
            self.assertEqual(store.content_path(record).read_bytes(), b"image")

    def test_unsafe_project_paths(self):
        from src.project_paths import validate_project_id
        for value in ("", "/absolute", "../escape", "a/../b", "a//b", "a/", "a/./b", "a\\b"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_project_id(value)
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as outside:
            (Path(tmp) / "escape").symlink_to(outside, target_is_directory=True)
            with self.assertRaises(ArtifactError):
                ArtifactStore(root_dir=Path(tmp))._project_dir("escape/project")

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

    def test_artifact_index_and_lazy_spec_backfill(self):
        import io
        import json
        from PIL import Image

        buf = io.BytesIO()
        Image.new("RGB", (640, 360), color=(20, 40, 60)).save(buf, format="PNG")
        png_bytes = buf.getvalue()

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            store = ArtifactStore(root_dir=root)
            rec = store.create_from_bytes(
                project_id="nested/proj-a",
                filename="sample.png",
                data=png_bytes,
                media_type="image/png",
            )
            self.assertEqual(rec.metadata.get("width"), 640)
            self.assertEqual(rec.metadata.get("height"), 360)
            self.assertEqual(rec.metadata.get("format"), "PNG")
            # Newly created artifact is indexed in memory -> 0 rglob scans when fetched without project_id
            self.assertEqual(store.rglob_count, 0)
            fetched = store.get(rec.artifact_id)
            self.assertEqual(store.rglob_count, 0)
            self.assertEqual(fetched.project_id, "nested/proj-a")

            # Simulate cold start + legacy artifact missing width/height/format in JSON
            meta_path = store._meta_path("nested/proj-a", rec.artifact_id)
            legacy_dict = rec.to_dict()
            legacy_dict["metadata"] = {"engine": "gemini"}
            meta_path.write_text(json.dumps(legacy_dict), encoding="utf-8")
            original_content_bytes = store.content_path(rec).read_bytes()

            cold_store = ArtifactStore(root_dir=root)
            backfilled = cold_store.get(rec.artifact_id)
            self.assertEqual(cold_store.rglob_count, 1)
            self.assertEqual(backfilled.metadata.get("width"), 640)
            self.assertEqual(backfilled.metadata.get("height"), 360)
            self.assertEqual(backfilled.metadata.get("format"), "PNG")
            self.assertEqual(backfilled.metadata.get("engine"), "gemini")
            # Subsequent lookups hit the index without additional rglob
            cold_store.get(rec.artifact_id)
            self.assertEqual(cold_store.rglob_count, 1)
            # Immutable content binary is untouched
            self.assertEqual(store.content_path(rec).read_bytes(), original_content_bytes)
            persisted = json.loads(meta_path.read_text(encoding="utf-8"))
            self.assertEqual(persisted["metadata"]["width"], 640)
            self.assertEqual(persisted["metadata"]["height"], 360)
            self.assertEqual(persisted["metadata"]["format"], "PNG")

    def test_thumbnail_generation_cache_and_no_upscale(self):
        import concurrent.futures
        import io
        from PIL import Image

        # 1. Large 2048x1024 PNG -> 320x160 and 640x320 WebP thumbnails
        large_buf = io.BytesIO()
        Image.new("RGB", (2048, 1024), color=(120, 60, 180)).save(large_buf, format="PNG")
        large_bytes = large_buf.getvalue()

        # 2. Small 200x100 PNG -> stays 200x100 (no upscale)
        small_buf = io.BytesIO()
        Image.new("RGB", (200, 100), color=(40, 180, 90)).save(small_buf, format="PNG")
        small_bytes = small_buf.getvalue()

        with tempfile.TemporaryDirectory() as tmp_dir:
            store = ArtifactStore(root_dir=Path(tmp_dir))

            rec_large = store.create_from_bytes(
                project_id="thumb-proj",
                filename="large.png",
                data=large_bytes,
                media_type="image/png",
            )
            self.assertEqual(store.thumbnail_generate_count, 0)

            # Concurrent requests for the same 320 thumbnail must only generate once (Singleflight)
            with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
                results = list(
                    pool.map(
                        lambda _: store.get_or_create_thumbnail(rec_large.artifact_id, edge=320),
                        range(8),
                    )
                )
            self.assertEqual(store.thumbnail_generate_count, 1)
            thumb_320_path, etag_320 = results[0]
            self.assertTrue(thumb_320_path.is_file())
            self.assertEqual(etag_320, f'"{rec_large.sha256}-320-v1"')
            with Image.open(thumb_320_path) as t_img:
                self.assertEqual(t_img.format, "WEBP")
                self.assertEqual(t_img.size, (320, 160))

            # 640 thumbnail
            thumb_640_path, etag_640 = store.get_or_create_thumbnail(rec_large.artifact_id, edge=640)
            self.assertEqual(store.thumbnail_generate_count, 2)
            self.assertEqual(etag_640, f'"{rec_large.sha256}-640-v1"')
            with Image.open(thumb_640_path) as t_img:
                self.assertEqual(t_img.format, "WEBP")
                self.assertEqual(t_img.size, (640, 320))

            # Small image -> prewarm_thumbnails_sync generates both 320 and 640 without upscaling
            rec_small = store.create_from_bytes(
                project_id="thumb-proj",
                filename="small.png",
                data=small_bytes,
                media_type="image/png",
            )
            store.prewarm_thumbnails_sync(rec_small)
            self.assertEqual(store.thumbnail_generate_count, 4)
            small_thumb_path, _ = store.get_or_create_thumbnail(rec_small.artifact_id, edge=320)
            self.assertEqual(store.thumbnail_generate_count, 4)
            with Image.open(small_thumb_path) as s_img:
                self.assertEqual(s_img.format, "WEBP")
                self.assertEqual(s_img.size, (200, 100))

            # Invalid edge rejected
            with self.assertRaises(ValueError):
                store.get_or_create_thumbnail(rec_large.artifact_id, edge=500)


if __name__ == "__main__":
    unittest.main()

