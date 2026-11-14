"""Explicit installation preparation; the serving worker remains offline.

Use the pinned upstream's selected model URLs and CRC sidecars. Existing files
are verified, never overwritten; SHA-256 is reported for provenance.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from urllib.parse import urlparse
from urllib.request import urlopen
import zlib


def checksums(path):
    digest, crc = hashlib.sha256(), 0
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
            crc = zlib.crc32(block, crc)
    return digest.hexdigest(), format(crc, '08x')


def download(path, url, validate, limit):
    path = Path(path)
    if path.is_symlink():
        raise ValueError('Model path cannot be a symlink')
    if path.exists():
        validate(path)
        return
    if not url or urlparse(url).scheme != 'https':
        raise ValueError('Pinned upstream HTTPS model URL is required')
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.prepare-', dir=path.parent)
    try:
        size = 0
        with os.fdopen(fd, 'wb') as output, urlopen(url, timeout=120) as response:
            if urlparse(response.geturl()).scheme != 'https':
                raise ValueError('Insecure model redirect')
            while block := response.read(1024 * 1024):
                size += len(block)
                if size > limit:
                    raise ValueError('Model download exceeds size limit')
                output.write(block)
        validate(Path(temporary))
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.is_symlink():
                raise ValueError('Model path cannot be a symlink')
            validate(path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def read_hash(path):
    value = Path(path).read_text().strip()
    if not re.fullmatch('[0-9a-f]{8}', value):
        raise ValueError('Invalid upstream CRC sidecar')
    return value


def prepare(spec):
    sidecar = Path(spec['hash_path'])
    download(sidecar, spec['hash_url'], read_hash, 1024)
    expected = read_hash(sidecar)

    def verify(path):
        if checksums(path)[1] != expected:
            raise ValueError('Model checksum mismatch')

    path = Path(spec['path'])
    download(path, spec['url'], verify, 8 * 1024**3)
    digest, _ = checksums(path)
    return {'file': path.name, 'sha256': digest, 'bytes': path.stat().st_size, 'crc32': expected}


def main():
    from src.settings import Settings, FACEFUSION_COMMIT
    from src.worker import OfficialBackend
    backend = OfficialBackend(Settings.from_env(), initialize=False, download_providers=('github',))
    result = [prepare(spec) for spec in backend.manifest().values()]
    print(json.dumps({'event': 'models_prepared', 'upstream_commit': FACEFUSION_COMMIT, 'models': result}))


if __name__ == '__main__':
    main()
