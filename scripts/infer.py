#!/usr/bin/env python3
"""
Run inference on a single image using a named hardware profile.

Usage
-----
  python scripts/infer.py --source path/to/image.jpg --profile dev_pt
  python scripts/infer.py --source path/to/image.jpg --profile n100_cpu
"""
import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
EXPORTS = ROOT / "exports"


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default=str(ROOT / "validation" / "bus.jpg"),
                    help="image path")
    ap.add_argument("--profile", default="dev_pt",
                    help="key in configs/hardware.yaml")
    ap.add_argument("--hardware", default=str(ROOT / "configs" / "hardware.yaml"))
    ap.add_argument("--show", action="store_true", help="display result window")
    ap.add_argument("--save", default="", help="path to save annotated image")
    return ap.parse_args()


def load_yaml(p):
    import yaml
    with open(p) as f:
        return yaml.safe_load(f)


def main():
    args = parse_args()
    hw = load_yaml(args.hardware)["profiles"][args.profile]

    from ppe.backend import resolve_backend
    from ppe.detector import Detector

    model_path, predict_kwargs = resolve_backend(hw, EXPORTS)
    print(f"[INFO] profile={args.profile}  model={model_path}")

    detector = Detector(model_path, conf=hw.get("conf", 0.4), **predict_kwargs)

    frame = cv2.imread(args.source)
    if frame is None:
        raise FileNotFoundError(f"Image not found: {args.source}")

    xyxy, confs, classes = detector.infer(frame)
    print(f"[INFO] detections: {len(xyxy)}")
    for box, c, k in zip(xyxy, confs, classes):
        print(f"  cls={k:2d}  conf={c:.3f}  xyxy={list(map(int, box))}")
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"cls{k} {c:.2f}", (x1, max(20, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    if args.save:
        cv2.imwrite(args.save, frame)
        print(f"[INFO] saved → {args.save}")

    if args.show:
        cv2.imshow("infer", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
