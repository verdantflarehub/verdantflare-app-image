"""Adapter to unmodified FaceFusion 3.9.0 modules, owned by one serial process."""
import hashlib
import importlib
import subprocess
import sys
import zlib
from pathlib import Path

from src.errors import FaceFusionPipelineError
from src.settings import FACEFUSION_COMMIT, FACEFUSION_VERSION


def load_upstream(settings):
    root = Path(settings.source_root).resolve()
    if (root / ".git").exists():
        revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"], text=True)
        if dirty:
            raise ValueError("FaceFusion tracked source has local changes")
    else:
        revision = (root / "UPSTREAM_COMMIT").read_text().strip()
    if revision != FACEFUSION_COMMIT:
        raise ValueError("Unexpected FaceFusion source revision")
    sys.path.insert(0, str(root))
    import facefusion
    if Path(facefusion.__file__).resolve().parent != root / "facefusion":
        raise ValueError("FaceFusion import resolved outside configured source")


class OfficialBackend:
    def __init__(self, settings, initialize=True):
        load_upstream(settings)
        import numpy as np
        import cv2
        from facefusion import choices, state_manager, face_creator, content_analyser
        from facefusion.processors.modules.face_swapper import core as swapper
        from facefusion.processors.modules.face_enhancer import core as enhancer
        self.np, self.cv2, self.state = np, cv2, state_manager
        self.creator, self.analyser = face_creator, content_analyser
        self.swapper, self.enhancer, self.settings = swapper, enhancer, settings
        defaults = {
            "download_providers": [], "download_scope": "full", "log_level": "error",
            "execution_device_ids": [0], "execution_providers": [settings.provider],
            "execution_thread_count": 1, "video_memory_strategy": "tolerant",
            "face_detector_model": "yolo_face", "face_detector_size": "640x640",
            "face_detector_margin": [0, 0, 0, 0], "face_detector_angles": [0], "face_detector_score": 0.5,
            "face_landmarker_model": "2dfan4", "face_landmarker_score": 0.5,
            "face_occluder_model": "xseg_1", "face_parser_model": "bisenet_resnet_34",
            "face_mask_types": list(settings.mask_types), "face_mask_blur": 0.3,
            "face_mask_padding": [0, 0, 0, 0], "face_mask_areas": choices.face_mask_areas,
            "face_mask_regions": choices.face_mask_regions, "face_swapper_model": settings.swapper,
            "face_swapper_pixel_boost": settings.pixel_boost, "face_swapper_weight": 0.5,
            "face_enhancer_model": settings.enhancer, "face_enhancer_blend": settings.enhancer_blend,
            "face_enhancer_weight": 0.85,
        }
        for key, value in defaults.items():
            state_manager.init_item(key, value)
        self.modules = [importlib.import_module("facefusion."+name) for name in
                        ("content_analyser", "face_classifier", "face_detector", "face_landmarker",
                         "face_masker", "face_recognizer")]+[swapper, enhancer]
        # The cached options dictionaries are the supported model path seam. No code patches,
        # download hooks or alternate inference implementation are installed.
        for module in self.modules:
            for options in module.create_static_model_set("full").values():
                for kind in ("hashes", "sources"):
                    for spec in options.get(kind, {}).values():
                        spec["path"] = str(Path(settings.model_root) / Path(spec["path"]).name)
        options = swapper.get_model_options()
        if options is None or enhancer.get_model_options() is None:
            raise ValueError("Unknown FaceFusion model")
        from facefusion.processors.modules.face_swapper.choices import face_swapper_set
        if settings.pixel_boost not in face_swapper_set[settings.swapper]:
            raise ValueError("Unsupported Pixel Boost size")
        width, height = map(int, settings.pixel_boost.split("x"))
        model_width, model_height = options["size"]
        if width % model_width or height % model_height or width // model_width != height // model_height:
            raise ValueError("Pixel Boost must be an integer multiple of model size")
        self.models_loaded = {}
        self.model_hashes = {}
        if initialize:
            self.initialize()

    def manifest(self):
        result = {}
        for module in self.modules:
            if hasattr(module, "collect_model_downloads"):
                hashes, sources = module.collect_model_downloads()
            else:
                options = module.get_model_options()
                hashes, sources = options["hashes"], options["sources"]
            for key, spec in sources.items():
                result[module.__name__ + ":" + key] = dict(path=spec["path"], hash_path=hashes[key]["path"])
        return result

    def initialize(self):
        import onnxruntime as ort
        requested = "CUDAExecutionProvider" if self.settings.provider == "cuda" else "CPUExecutionProvider"
        if requested not in ort.get_available_providers():
            raise ValueError("Requested ONNX execution provider is unavailable: " + requested)
        missing = []
        for key, spec in self.manifest().items():
            path, hash_path = Path(spec["path"]), Path(spec["hash_path"])
            self.models_loaded[key] = False
            if not path.is_file() or not hash_path.is_file():
                missing.append(path.name)
                continue
            digest, checksum = hashlib.sha256(), 0
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
                    checksum = zlib.crc32(block, checksum)
            if format(checksum, "08x") != hash_path.read_text().strip():
                raise ValueError("Model checksum mismatch: " + path.name)
            self.model_hashes[path.name] = digest.hexdigest()
        if missing:
            raise ValueError("Missing FaceFusion models/hash files: " + ", ".join(missing))
        for module in self.modules:
            pool = module.get_inference_pool()
            if not pool:
                raise ValueError("Empty inference pool: " + module.__name__)
            for key, session in pool.items():
                if requested not in session.get_providers():
                    raise ValueError("Model fell back from requested provider: " + key)
                self.models_loaded[module.__name__+":"+key] = True
        self.execution_provider = requested

    def health(self):
        return dict(status="healthy", cuda_available=self.settings.provider == "cuda",
                    execution_provider=self.execution_provider, models_loaded=self.models_loaded,
                    backend="facefusion", backend_version=FACEFUSION_VERSION,
                    model_hashes=self.model_hashes)

    def read(self, path):
        frame = self.cv2.imread(path)
        if frame is None:
            raise FaceFusionPipelineError("Unable to decode image", "INVALID_IMAGE", 400)
        if frame.shape[0]*frame.shape[1] > self.settings.max_pixels:
            raise FaceFusionPipelineError("Image exceeds pixel limit", "IMAGE_TOO_LARGE", 413)
        return frame

    def faces(self, frame):
        # No path-keyed image cache; get_many_faces performs detection on current pixels.
        return sorted(self.creator.get_many_faces([frame]),
                      key=lambda f: float((f.bounding_box[2]-f.bounding_box[0])*(f.bounding_box[3]-f.bounding_box[1])),
                      reverse=True)

    def select(self, faces, index):
        if index < 0 or index >= len(faces):
            raise FaceFusionPipelineError("Requested face was not detected", "FACE_NOT_FOUND", 400)
        return faces[index]

    def pose(self, face, shape):
        np, cv2 = self.np, self.cv2
        if face.score_set.get("landmarker", 0) <= self.state.get_item("face_landmarker_score"):
            raise FaceFusionPipelineError("Reliable pose landmarks unavailable", "POSE_UNAVAILABLE", 422)
        points = np.asarray(face.landmark_set["68"], dtype=np.float64)[[30, 8, 36, 45, 48, 54]]
        model = np.array([[0, 0, 0], [0, 330, -65], [-225, -170, -135],
                          [225, -170, -135], [-150, 150, -125], [150, 150, -125]], dtype=np.float64)
        h, w = shape[:2]
        camera = np.array([[w, 0, w/2], [0, w, h/2], [0, 0, 1]], dtype=np.float64)
        try:
            ok, rotation, translation = cv2.solvePnP(model, points, camera, np.zeros((4, 1)), flags=cv2.SOLVEPNP_ITERATIVE)
            if not ok or translation[2, 0] <= 0:
                raise ValueError("Pose solver failed")
            angles = tuple(float(v) for v in cv2.RQDecomp3x3(cv2.Rodrigues(rotation)[0])[0])
            if not np.isfinite(angles).all():
                raise ValueError("Non-finite pose")
            return angles
        except Exception as exc:
            raise FaceFusionPipelineError("Pose could not be estimated", "POSE_UNAVAILABLE", 422) from exc

    def describe(self, faces, index, frame):
        face = self.select(faces, index)
        pitch, yaw, roll = self.pose(face, frame.shape)
        return dict(face_count=len(faces), pitch=round(pitch, 2), yaw=round(yaw, 2), roll=round(roll, 2),
                    confidence=round(float(face.score_set["detector"]), 4),
                    bounding_box=[int(v) for v in face.bounding_box],
                    pose_safe=abs(pitch)<=35 and abs(yaw)<=45 and abs(roll)<=30)

    def detect(self, image_path, target_face_index=0):
        frame = self.read(image_path)
        return self.describe(self.faces(frame), target_face_index, frame)

    def fuse(self, target_image_path, source_face_path, target_face_index,
             identity_strength, restore_face, restoration_fidelity, output_path):
        target, source = self.read(target_image_path), self.read(source_face_path)
        if self.analyser.analyse_frame(target) or self.analyser.analyse_frame(source):
            raise FaceFusionPipelineError("FaceFusion content check rejected the input", "CONTENT_REJECTED", 422)
        targets, sources = self.faces(target), self.faces(source)
        target_face, source_face = self.select(targets, target_face_index), self.select(sources, 0)
        description = self.describe(targets, target_face_index, target)
        if not description["pose_safe"]:
            raise FaceFusionPipelineError("Target face exceeds pose limits", "POSE_ANGLE_EXCEEDED", 422)
        weight = (identity_strength-0.5)*2
        self.state.set_item("face_swapper_weight", weight)
        self.state.set_item("face_enhancer_weight", restoration_fidelity)
        fused = self.swapper.swap_face(source_face, target_face, source, target.copy())
        processors = ["face_swapper"]
        if restore_face:
            fused = self.enhancer.enhance_face(target_face, fused)
            processors.append("face_enhancer")
        output_faces = self.faces(fused)
        if not output_faces:
            raise FaceFusionPipelineError("No face in the output", "OUTPUT_FACE_NOT_FOUND", 422)
        from facefusion.face_helper import calculate_bounding_box_overlap
        output_face = max(output_faces, key=lambda f: calculate_bounding_box_overlap(target_face.bounding_box, f.bounding_box))
        if calculate_bounding_box_overlap(target_face.bounding_box, output_face.bounding_box) <= 0:
            raise FaceFusionPipelineError("Target face missing in output", "OUTPUT_FACE_NOT_FOUND", 422)
        similarity = float(self.np.dot(output_face.embedding_norm, source_face.embedding_norm))
        if not self.np.isfinite(similarity):
            raise FaceFusionPipelineError("Invalid output identity score", "INVALID_OUTPUT", 500)
        if not self.cv2.imwrite(output_path, fused):
            raise FaceFusionPipelineError("Unable to write output", "OUTPUT_WRITE_FAILED", 500)
        signature = f"facefusion-{FACEFUSION_VERSION}+{self.settings.swapper}"
        if restore_face:
            signature += "+"+self.settings.enhancer
        return dict(status="success", detected_faces=len(targets), arcface_similarity=round(similarity, 4),
                    pipeline=signature, execution=dict(backend="facefusion", version=FACEFUSION_VERSION,
                    commit=FACEFUSION_COMMIT, swapper=self.settings.swapper,
                    pixel_boost=self.settings.pixel_boost, mask_types=list(self.settings.mask_types),
                    processors=processors, face_swapper_weight=weight,
                    enhancer=self.settings.enhancer if restore_face else None,
                    enhancer_weight=restoration_fidelity if restore_face else None,
                    enhancer_blend=self.settings.enhancer_blend if restore_face else None,
                    provider=self.execution_provider, model_hashes=self.model_hashes))


def serve(connection, settings):
    try:
        backend = OfficialBackend(settings)
    except Exception as exc:
        connection.send(dict(status="not_ready", cuda_available=False, execution_provider="unavailable",
                             models_loaded={}, backend="facefusion", backend_version=FACEFUSION_VERSION,
                             reason=str(exc)))
        # Remain alive to expose a stable not-ready state; no automatic download/retry loop.
        try:
            while True:
                connection.recv()
                connection.send({"error": FaceFusionPipelineError("Models not ready", "MODEL_NOT_READY", 503).as_dict()})
        except EOFError:
            return
    else:
        connection.send(backend.health())
        try:
            while True:
                operation, payload = connection.recv()
                try:
                    if operation not in ("detect", "fuse"):
                        raise FaceFusionPipelineError("Unknown worker operation", "INVALID_OPERATION", 400)
                    result = getattr(backend, operation)(**payload)
                    connection.send({"result": result})
                except FaceFusionPipelineError as exc:
                    connection.send({"error": exc.as_dict()})
                except Exception:
                    # An unexpected inference failure invalidates this process and its sessions.
                    raise
        except EOFError:
            return
    finally:
        connection.close()
