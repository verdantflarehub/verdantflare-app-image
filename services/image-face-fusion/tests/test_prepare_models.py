import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zlib

from src.prepare_models import prepare


class Response(io.BytesIO):
    def geturl(self):
        return 'https://release-assets.githubusercontent.com/model'


class ModelPreparationTest(unittest.TestCase):
    def test_download_verify_reuse_and_reject_corruption(self):
        blob = b'fixture model'
        crc = format(zlib.crc32(blob), '08x').encode()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = dict(path=str(root / 'model.onnx'), hash_path=str(root / 'model.hash'),
                        url='https://github.com/model', hash_url='https://github.com/hash')
            with patch('src.prepare_models.urlopen', side_effect=[Response(crc), Response(blob)]) as request:
                result = prepare(spec)
                self.assertEqual(result['bytes'], len(blob))
                self.assertEqual(request.call_count, 2)
            with patch('src.prepare_models.urlopen', side_effect=AssertionError('unexpected network')):
                self.assertEqual(prepare(spec), result)
                Path(spec['path']).write_bytes(b'old corrupt bytes')
                with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                    prepare(spec)
                self.assertEqual(Path(spec['path']).read_bytes(), b'old corrupt bytes')

    def test_bad_download_is_not_published(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = dict(path=str(root / 'model.onnx'), hash_path=str(root / 'model.hash'),
                        url='https://github.com/model', hash_url='https://github.com/hash')
            with patch('src.prepare_models.urlopen', side_effect=[Response(b'00000000'), Response(b'bad')]):
                with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                    prepare(spec)
            self.assertFalse(Path(spec['path']).exists())
            self.assertEqual(list(root.glob('.prepare-*')), [])
