#!/usr/bin/env python3
"""
Export ppe.pt to all portable formats.

Formats exported by default: ONNX, ncnn.
Optional extras (GPU required): engine (TensorRT), openvino.

Usage
-----
  python scripts/export.py                        # ONNX + ncnn
  python scripts/export.py --formats onnx         # ONNX only
  python scripts/export.py --formats engine ncnn  # TRT + ncnn
  python scripts/export.py --imgsz 320            # smaller size
"""
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPORTS = ROOT / "exports"


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=str(ROOT / "ppe.pt"),
                    help="source .pt file (default: ppe.pt)")
    ap.add_argument("--formats", nargs="+",
                    default=["onnx", "ncnn"],
                    help="export formats (default: onnx ncnn)")
    ap.add_argument("--imgsz", type=int, default=416,
                    help="input image size (default: 416)")
    ap.add_argument("--half", action="store_true",
                    help="FP16 export (requires CUDA for engine)")
    return ap.parse_args()


def main():
    args = parse_args()
    EXPORTS.mkdir(exist_ok=True)

    from ultralytics import YOLO
    model = YOLO(args.model)

    for fmt in args.formats:
        print(f"\n[EXPORT] format={fmt}  imgsz={args.imgsz}  half={args.half}")
        try:
            out = model.export(
                format=fmt,
                imgsz=args.imgsz,
                half=args.half,
            )
            out_path = Path(out)
            dest_path = EXPORTS / out_path.name
            if out_path.resolve() != dest_path.resolve():
                import shutil
                if out_path.is_dir():
                    if dest_path.exists():
                        shutil.rmtree(dest_path)
                    shutil.move(str(out_path), str(dest_path))
                else:
                    shutil.move(str(out_path), str(dest_path))
                out = dest_path
            print(f"[EXPORT] done → {out}")
        except Exception as e:
            print(f"[EXPORT] FAILED for {fmt}: {e}")


if __name__ == "__main__":
    main()
