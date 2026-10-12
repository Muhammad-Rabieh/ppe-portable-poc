#!/usr/bin/env python3
"""
Benchmark: measure 1-camera and 2-camera inference FPS for a hardware profile.

This is your go/no-go test before deploying to a box.
If 2-cam per-camera FPS >= 5, the box will run.  If not, lower imgsz or set
infer_every > 1 in configs/hardware.yaml.

Usage
-----
  python scripts/bench.py --profile dev_pt
  python scripts/bench.py --profile n100_cpu
  python scripts/bench.py --profile n100_cpu --imgsz 320
  python scripts/bench.py --profile rx580_vulkan --frames 120
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
EXPORTS = ROOT / "exports"

from ppe.backend import resolve_backend
from ppe.detector import Detector

PASS_FPS = 5.0   # minimum acceptable per-camera FPS for 2-cam deployment


def load_yaml(p):
    with open(p) as f:
        return yaml.safe_load(f)


def bench_single(detector: Detector, frames: list, warmup: int = 10) -> tuple:
    """Return (mean_ms, std_ms, fps) for single-stream inference."""
    for _ in range(warmup):
        detector.infer(frames[0])

    latencies = []
    for f in frames:
        t0 = time.perf_counter()
        detector.infer(f)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    lat = np.array(latencies)
    return float(lat.mean()), float(lat.std()), 1000.0 / float(lat.mean())


def bench_dual(detector: Detector, frame, n_frames: int) -> float:
    """Return aggregate FPS when sequentially serving 2 streams."""
    # Warmup
    for _ in range(5):
        detector.infer(frame)
        detector.infer(frame)

    t0 = time.perf_counter()
    for _ in range(n_frames):
        detector.infer(frame)
        detector.infer(frame)
    elapsed = time.perf_counter() - t0
    return (2 * n_frames) / elapsed


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default="dev_pt",
                    help="key in configs/hardware.yaml")
    ap.add_argument("--imgsz", type=int, default=None,
                    help="override imgsz from profile (useful for quick sweeps)")
    ap.add_argument("--frames", type=int, default=60,
                    help="number of frames to time (default: 60)")
    ap.add_argument("--hardware", default=str(ROOT / "configs" / "hardware.yaml"),
                    help="path to hardware.yaml")
    args = ap.parse_args()

    hw_cfg = load_yaml(args.hardware)
    if args.profile not in hw_cfg["profiles"]:
        available = ", ".join(hw_cfg["profiles"].keys())
        ap.error(f"Unknown profile {args.profile!r}. Available: {available}")

    hw = hw_cfg["profiles"][args.profile]
    if args.imgsz:
        hw["imgsz"] = args.imgsz

    model_path, predict_kwargs = resolve_backend(hw, EXPORTS)
    imgsz = predict_kwargs["imgsz"]

    print("=" * 60)
    print(f"  Profile  : {args.profile}")
    print(f"  Model    : {model_path}")
    print(f"  imgsz    : {imgsz}    half={predict_kwargs['half']}")
    print(f"  device   : {predict_kwargs['device']}")
    print(f"  frames   : {args.frames}")
    print("=" * 60)

    # Synthetic frames at model input size (worst-case preprocessing cost)
    frame = (np.random.rand(imgsz, imgsz, 3) * 255).astype(np.uint8)
    frames = [frame] * args.frames

    detector = Detector(model_path, conf=0.4, **predict_kwargs)

    # --- 1-cam benchmark ---
    mean_ms, std_ms, fps1 = bench_single(detector, frames)
    print(f"\n[1 cam]  mean {mean_ms:6.1f} ms  std {std_ms:5.1f} ms  "
          f"→ {fps1:6.1f} FPS")

    # --- 2-cam benchmark (sequential, shared model) ---
    dual_fps = bench_dual(detector, frame, args.frames)
    per_cam_fps = dual_fps / 2
    print(f"[2 cams] aggregate {dual_fps:6.1f} FPS  "
          f"→ {per_cam_fps:6.1f} FPS per camera")

    # --- Verdict ---
    print()
    if per_cam_fps >= PASS_FPS:
        print(f"[PASS ✓]  {per_cam_fps:.1f} FPS/cam ≥ {PASS_FPS} — this box can handle 2 cameras.")
    else:
        print(f"[FAIL ✗]  {per_cam_fps:.1f} FPS/cam < {PASS_FPS} — too slow for 2 cameras.")
        print("          Options:")
        print(f"            • Lower imgsz (try --imgsz 320 or 256)")
        print(f"            • Set infer_every: 2 in configs/hardware.yaml")
        print(f"            • Use a smaller model (yolov8n)")
    print()


if __name__ == "__main__":
    main()
