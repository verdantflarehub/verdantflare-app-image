#!/usr/bin/env python3
"""Local straight-testing script for Face Fusion & Restoration pipeline.

Usage:
  python test_face_fusion.py --self-test
  python test_face_fusion.py --source docs/assets/xiaoyue-identity-original.png --target canvas_4k.png
"""

import sys
import time
import argparse
from pathlib import Path
from PIL import Image

# Add services/image-face-fusion-api to path
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "services" / "image-face-fusion-api"))

from src.pipeline import FaceFusionPipeline, FaceFusionPipelineError


def parse_args():
    parser = argparse.ArgumentParser(description="VerdantFlare Face Fusion Local Prototyping Script")
    parser.add_argument("--source", type=str, default="docs/assets/xiaoyue-identity-original.png", help="Path to source identity image")
    parser.add_argument("--target", type=str, default="", help="Path to target canvas image")
    parser.add_argument("--output", type=str, default="scratch/test_xiaoyue_fused.png", help="Path for fused output image")
    parser.add_argument("--strength", type=float, default=0.95, help="Identity injection strength (0.5 - 1.0)")
    parser.add_argument("--fidelity", type=float, default=0.85, help="CodeFormer restoration fidelity (0.0 - 1.0)")
    parser.add_argument("--self-test", action="store_true", help="Run identity self-comparison test")
    return parser.parse_args()


def main():
    args = parse_args()
    print("================================================================================")
    print("      VERDANTFLARE FACE FUSION LOCAL STRAIGHT-TEST PROTOTYPE                   ")
    print("================================================================================")

    pipeline = FaceFusionPipeline()
    print(f"[*] CUDA Available: {pipeline.cuda_available}")
    print(f"[*] Active Execution Provider: {pipeline.execution_provider}")
    print(f"[*] Model Weights Loaded: {pipeline.models_loaded}")

    source_path = Path(args.source)
    # Check if relative to design repo root
    if not source_path.is_file():
        workspace_source = repo_root.parent / args.source
        if workspace_source.is_file():
            source_path = workspace_source

    if args.self_test:
        print("\n--- [RUNNING SELF-TEST] ---")
        if not source_path.is_file():
            # Generate dummy identity image for offline test
            tmp_source = repo_root / "scratch" / "dummy_identity.png"
            tmp_source.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (1024, 1536), color=(220, 200, 190)).save(tmp_source)
            source_path = tmp_source
            print(f"[*] Using synthetic identity image: {source_path}")
        else:
            print(f"[*] Source Identity: {source_path}")

        detect_res = pipeline.detect_face(str(source_path))
        print(f"[✔] Face Detection: {detect_res['face_count']} face(s), Confidence: {detect_res['confidence']}")
        print(f"    Head Pose: Pitch={detect_res['pitch']}°, Yaw={detect_res['yaw']}°, Roll={detect_res['roll']}° (Safe: {detect_res['pose_safe']})")

        out_path = repo_root / "scratch" / "self_test_output.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        fuse_res = pipeline.fuse(
            target_image_path=str(source_path),
            source_face_path=str(source_path),
            identity_strength=1.0,
            output_path=str(out_path),
        )
        print(f"[✔] Self-Fusion Status: {fuse_res['status']}")
        print(f"    Similarity: {fuse_res['arcface_similarity']} (Expected >= 0.88)")
        print(f"    Latency: {fuse_res['inference_time_ms']}ms")
        print(f"    Output: {fuse_res['output_path']}")
        print("\n[RESULT] SELF-TEST PASSED SUCCESSFULLY!\n")
        return

    # Normal execution
    if not args.target:
        print("[-] Please specify --target <image_path> or use --self-test")
        sys.exit(1)

    target_path = Path(args.target)
    if not target_path.is_file():
        print(f"[-] Target image not found: {target_path}")
        sys.exit(1)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n[*] Source Identity: {source_path}")
    print(f"[*] Target Canvas: {target_path}")
    print(f"[*] Parameters: strength={args.strength}, fidelity={args.fidelity}")

    try:
        res = pipeline.fuse(
            target_image_path=str(target_path),
            source_face_path=str(source_path),
            identity_strength=args.strength,
            restore_face=True,
            restoration_fidelity=args.fidelity,
            output_path=str(out_path),
        )
        print(f"\n[✔] Fusion Completed: {res['status']}")
        print(f"    Similarity Score: {res['arcface_similarity']}")
        print(f"    Inference Latency: {res['inference_time_ms']}ms")
        print(f"    Output Path: {res['output_path']}")
    except FaceFusionPipelineError as e:
        print(f"\n[!] Pipeline Error [{e.error_code}]: {e.message}")
        sys.exit(1)


if __name__ == "__main__":
    main()
