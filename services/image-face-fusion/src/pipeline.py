"""HTTP-facing process supervisor. No ONNX sessions or upstream global state here."""
import multiprocessing
import os
import tempfile
import threading
import time
from pathlib import Path

from PIL import Image

from src.errors import FaceFusionPipelineError
from src.settings import Settings, FACEFUSION_VERSION


def worker_main(connection, settings):
    # Import here so the parent's HTTP process never imports upstream or initializes CUDA.
    from src.worker import serve
    serve(connection, settings)


class FaceFusionPipeline:
    def __init__(self, settings=None, worker_target=None):
        self.settings = settings or Settings.from_env()
        self._target = worker_target or worker_main
        self._lock = threading.Lock()
        self._process = None
        self._connection = None
        self._health = self._not_ready("Worker has not started")

    @staticmethod
    def _not_ready(reason):
        return dict(status="not_ready", cuda_available=False, execution_provider="unavailable",
                    models_loaded={}, backend="facefusion", backend_version=FACEFUSION_VERSION,
                    reason=reason)

    def health(self):
        try:
            alive = self._process is not None and self._process.is_alive()
        except (ValueError, AttributeError):
            alive = False
        if not alive:
            return self._not_ready(self._health.get("reason", "Worker exited"))
        return dict(self._health)

    def _stop(self):
        if self._process is not None:
            if self._process.is_alive():
                self._process.terminate()
            self._process.join(timeout=3)
            if self._process.is_alive():
                self._process.kill()
                self._process.join(timeout=3)
            self._process.close()
        if self._connection is not None:
            self._connection.close()
        self._process = self._connection = None

    def close(self):
        with self._lock:
            self._stop()
            self._health = self._not_ready("Worker stopped")

    def _receive(self, timeout):
        try:
            if not self._connection.poll(timeout):
                raise FaceFusionPipelineError("FaceFusion worker timed out", "WORKER_TIMEOUT", 504)
            return self._connection.recv()
        except (EOFError, BrokenPipeError, OSError) as exc:
            raise FaceFusionPipelineError("FaceFusion worker exited", "WORKER_EXITED", 503) from exc

    def _start(self, timeout=None):
        if self._process is not None and self._process.is_alive():
            return
        self._stop()
        self._health = self._not_ready("Worker is initializing")
        context = multiprocessing.get_context("spawn")
        self._connection, child = context.Pipe()
        self._process = context.Process(target=self._target, args=(child, self.settings), daemon=True)
        self._process.start()
        child.close()
        try:
            self._health = self._receive(timeout if timeout is not None else self.settings.startup_timeout)
        except FaceFusionPipelineError as exc:
            self._health = self._not_ready(exc.message)
            self._stop()
            raise

    def start(self):
        with self._lock:
            try:
                self._start()
            except FaceFusionPipelineError:
                pass  # HTTP stays live; readiness exposes the startup failure.

    def _call(self, operation, payload):
        if not self._lock.acquire(blocking=False):
            raise FaceFusionPipelineError("FaceFusion worker is busy", "WORKER_BUSY", 503)
        try:
            deadline = time.monotonic() + self.settings.timeout
            self._start(min(self.settings.timeout, self.settings.startup_timeout))
            if self._health["status"] != "healthy":
                raise FaceFusionPipelineError(self._health.get("reason", "Models not ready"), "MODEL_NOT_READY", 503)
            try:
                self._connection.send((operation, payload))
                reply = self._receive(max(0, deadline - time.monotonic()))
            except (FaceFusionPipelineError, BrokenPipeError, OSError) as exc:
                self._health = self._not_ready("Worker failed; next request will restart it")
                self._stop()
                if isinstance(exc, FaceFusionPipelineError):
                    raise
                raise FaceFusionPipelineError("FaceFusion worker exited", "WORKER_EXITED", 503) from exc
            if "error" in reply:
                raise FaceFusionPipelineError(**reply["error"])
            return reply["result"]
        finally:
            self._lock.release()

    def _image_path(self, value):
        path = Path(value).resolve()
        if not path.is_file():
            raise FaceFusionPipelineError("Image path is unreachable", "FILE_PATH_UNREACHABLE", 404)
        try:
            with Image.open(path) as image:
                if image.width * image.height > self.settings.max_pixels:
                    raise FaceFusionPipelineError("Image exceeds pixel limit", "IMAGE_TOO_LARGE", 413)
                image.verify()
        except FaceFusionPipelineError:
            raise
        except Exception as exc:
            raise FaceFusionPipelineError("Invalid input image", "INVALID_IMAGE", 400) from exc
        return str(path)

    def detect_face(self, image_path, target_face_index=0):
        return self._call("detect", dict(image_path=self._image_path(image_path), target_face_index=target_face_index))

    def fuse(self, target_image_path, source_face_path, target_face_index=0,
             identity_strength=0.95, restore_face=True, restoration_fidelity=0.85, output_path=""):
        started = time.perf_counter()
        target = self._image_path(target_image_path)
        source = self._image_path(source_face_path)
        if not 0.5 <= identity_strength <= 1 or not 0 <= restoration_fidelity <= 1 or target_face_index < 0:
            raise FaceFusionPipelineError("Invalid fusion parameters", "INVALID_PARAMETERS", 422)
        output = Path(output_path).absolute()
        if not output_path or output.suffix.lower() != ".png":
            raise FaceFusionPipelineError("Output must be a PNG path", "INVALID_OUTPUT_PATH", 422)
        if output.exists() or output.is_symlink():
            raise FaceFusionPipelineError("Output already exists", "OUTPUT_EXISTS", 409)
        output.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=".facefusion-", suffix=".png", dir=output.parent)
        os.close(fd)
        try:
            result = self._call("fuse", dict(target_image_path=target, source_face_path=source,
                target_face_index=target_face_index, identity_strength=identity_strength,
                restore_face=restore_face, restoration_fidelity=restoration_fidelity, output_path=temporary))
            with Image.open(temporary) as image:
                if image.format != "PNG":
                    raise FaceFusionPipelineError("Worker returned invalid output", "INVALID_OUTPUT", 500)
                image.verify()
            # Atomic no-clobber publication, after worker success; timeouts never publish an artifact.
            try:
                os.link(temporary, output)
            except FileExistsError as exc:
                raise FaceFusionPipelineError("Output already exists", "OUTPUT_EXISTS", 409) from exc
            result.update(output_path=str(output.resolve()), inference_time_ms=int((time.perf_counter()-started)*1000))
            return result
        finally:
            Path(temporary).unlink(missing_ok=True)
