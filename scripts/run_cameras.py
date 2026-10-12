#!/usr/bin/env python3
"""
Run N cameras live using a named hardware profile.

Usage examples
--------------
  # Dev (CPU, pt format — works in Codespaces):
  python scripts/run_cameras.py --profile dev_pt

  # Cheap N100 mini PC:
  python scripts/run_cameras.py --profile n100_cpu

  # RX 580 (ncnn + Vulkan):
  python scripts/run_cameras.py --profile rx580_vulkan

  # Headless server (no display — FPS printed to stdout):
  python scripts/run_cameras.py --profile n100_cpu --headless
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
EXPORTS = ROOT / "exports"

from ppe.backend import resolve_backend
from ppe.detector import Detector
from ppe.pipeline import CameraWorker


def load_yaml(p):
    with open(p) as f:
        return yaml.safe_load(f)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default="dev_pt",
                    help="key in configs/hardware.yaml  (default: dev_pt)")
    ap.add_argument("--cameras", default=str(ROOT / "configs" / "cameras.yaml"),
                    help="path to cameras.yaml")
    ap.add_argument("--hardware", default=str(ROOT / "configs" / "hardware.yaml"),
                    help="path to hardware.yaml")
    ap.add_argument("--headless", action="store_true",
                    help="skip display; print FPS to stdout (for servers)")
    args = ap.parse_args()

    hw_cfg = load_yaml(args.hardware)
    if args.profile not in hw_cfg["profiles"]:
        available = ", ".join(hw_cfg["profiles"].keys())
        ap.error(f"Unknown profile {args.profile!r}. Available: {available}")

    hw = hw_cfg["profiles"][args.profile]
    cams = load_yaml(args.cameras)["cameras"]

    model_path, predict_kwargs = resolve_backend(hw, EXPORTS)
    print(f"[INFO] profile       = {args.profile}")
    print(f"[INFO] model         = {model_path}")
    print(f"[INFO] predict_kwargs= {predict_kwargs}")
    print(f"[INFO] infer_every   = {hw.get('infer_every', 1)}")

    # One detector shared across all cameras (model loaded once)
    detector = Detector(model_path, conf=hw.get("conf", 0.4), **predict_kwargs)

    workers = []
    for cam in cams:
        w = CameraWorker(
            name=cam["name"],
            source=cam["source"],
            detector=detector,
            infer_every=hw.get("infer_every", 1),
            draw=not args.headless,
        ).start()
        workers.append(w)
        print(f"[INFO] started {cam['name']} <- {cam['source']}")

    print("[INFO] running — Ctrl-C to stop")
    try:
        while True:
            time.sleep(1.0)
            line = " | ".join(
                f"{w.name}: cap {w.fps_capture:5.1f} fps  infer {w.fps_infer:5.1f} fps"
                for w in workers
            )
            print(line)

            if not args.headless:
                for w in workers:
                    if w.latest is not None:
                        cv2.imshow(w.name, w.latest)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        print("[INFO] shutting down …")
        for w in workers:
            w.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
