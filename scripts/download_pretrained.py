#!/usr/bin/env python3
"""
Download (or create a placeholder) ppe.pt for local testing.

In a real workflow you drop your trained ppe.pt here directly.
This script exists so the Codespaces/CI loop works without a real model:
  - If a real COCO-trained YOLOv8n is reachable, it uses that as a stand-in.
  - Otherwise it creates a tiny random-weight model so export/bench don't crash.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "ppe.pt"


def main():
    if DEST.exists():
        print(f"[INFO] {DEST} already exists — skipping download.")
        return

    print("[INFO] Downloading YOLOv8n as placeholder ppe.pt …")
    try:
        from ultralytics import YOLO
        model = YOLO("yolov8n.pt")   # downloads ~6 MB from Ultralytics CDN
        import shutil
        shutil.copy(model.ckpt_path, DEST)
        print(f"[INFO] Saved to {DEST}")
    except Exception as e:
        print(f"[WARN] Could not download: {e}")
        print("[INFO] Creating minimal placeholder ppe.pt …")
        _create_placeholder()


def _create_placeholder():
    """Write a tiny YOLOv8n with random weights so scripts can import it."""
    from ultralytics import YOLO
    import tempfile, shutil, os
    with tempfile.TemporaryDirectory() as tmp:
        m = YOLO("yolov8n.yaml")   # no pretrained weights — random init
        tmp_pt = Path(tmp) / "rand.pt"
        m.save(str(tmp_pt))
        shutil.copy(tmp_pt, DEST)
    print(f"[INFO] Placeholder saved to {DEST} (random weights — for testing only)")


if __name__ == "__main__":
    main()
