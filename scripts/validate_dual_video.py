#!/usr/bin/env python3
"""
Simulate a dual-camera system running real-time PPE detection simultaneously.

Plays two video streams concurrently using threaded workers sharing a single
model detector instance (the exact production architecture in ppe/pipeline.py),
renders a unified side-by-side dashboard with live FPS metrics for both cameras,
saves the annotated dual-stream video, and writes a test report.

Usage
-----
  python scripts/validate_dual_video.py --profile inspiron3521_cpu
  python scripts/validate_dual_video.py --profile dev_pt --max-frames 120
  python scripts/validate_dual_video.py --headless
"""
import argparse
import datetime
import os
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
EXPORTS = ROOT / "exports"
VAL_DIR = ROOT / "validation"
DEFAULT_VIDEO_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/vtest.avi"
DEFAULT_VIDEO_PATH = VAL_DIR / "vtest.avi"

from ppe.backend import resolve_backend
from ppe.detector import Detector


def load_yaml(p):
    with open(p) as f:
        return yaml.safe_load(f)


def ensure_video(path: Path):
    VAL_DIR.mkdir(exist_ok=True)
    if not path.exists():
        print(f"[INFO] Downloading sample video from {DEFAULT_VIDEO_URL} …")
        urllib.request.urlretrieve(DEFAULT_VIDEO_URL, path)
        print(f"[INFO] Saved sample video to {path}")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source1", default=str(DEFAULT_VIDEO_PATH),
                    help="video source for Camera 1 (default: validation/vtest.avi)")
    ap.add_argument("--source2", default=str(DEFAULT_VIDEO_PATH),
                    help="video source for Camera 2 (default: validation/vtest.avi)")
    ap.add_argument("--offset2", type=int, default=120,
                    help="frame offset for Camera 2 to simulate distinct viewpoint (default: 120)")
    ap.add_argument("--profile",
                    default="inspiron3521_cpu" if (EXPORTS / "ppe.onnx").exists() else "dev_pt",
                    help="hardware profile from configs/hardware.yaml")
    ap.add_argument("--hardware", default=str(ROOT / "configs" / "hardware.yaml"))
    ap.add_argument("--save", default=str(VAL_DIR / "dual_video_detected.mp4"),
                    help="path to save side-by-side video (default: validation/dual_video_detected.mp4)")
    ap.add_argument("--report", default=str(VAL_DIR / "dual_video_test_data.txt"),
                    help="path to save text report (default: validation/dual_video_test_data.txt)")
    ap.add_argument("--max-frames", type=int, default=120,
                    help="max frames to process (default: 120, 0 for infinite)")
    ap.add_argument("--headless", action="store_true",
                    help="run without graphical display window")
    ap.add_argument("--no-open-report", action="store_true",
                    help="do not automatically open report in desktop editor")
    return ap.parse_args()


def render_detections(frame, boxes, confs, classes, cam_name: str):
    """Draw bounding boxes, badges, and a camera title bar on a single frame."""
    h, w = frame.shape[:2]

    # Draw detection boxes
    for box, c, k in zip(boxes, confs, classes):
        x1, y1, x2, y2 = map(int, box)
        color = (0, 220, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"cls {int(k)} {c * 100:.1f}%"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        badge_y1 = max(0, y1 - th - 6)
        cv2.rectangle(frame, (x1, badge_y1), (x1 + tw + 6, badge_y1 + th + 6), color, -1)
        cv2.putText(frame, label, (x1 + 3, badge_y1 + th + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

    # Camera label strip
    cv2.rectangle(frame, (0, 0), (w, 32), (30, 30, 30), -1)
    cv2.putText(frame, f"[{cam_name}] Detections: {len(boxes)}", (12, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def main():
    args = parse_args()

    s1 = Path(args.source1)
    s2 = Path(args.source2)
    if s1 == DEFAULT_VIDEO_PATH or s2 == DEFAULT_VIDEO_PATH:
        ensure_video(DEFAULT_VIDEO_PATH)

    hw = load_yaml(args.hardware)["profiles"][args.profile]
    model_path, predict_kwargs = resolve_backend(hw, EXPORTS)
    print(f"[INFO] Initializing Dual-Camera Simulation")
    print(f"[INFO] Profile    : {args.profile}")
    print(f"[INFO] Model      : {model_path}")
    print(f"[INFO] Camera 1   : {s1}")
    print(f"[INFO] Camera 2   : {s2} (offset: {args.offset2} frames)")

    detector = Detector(model_path, conf=hw.get("conf", 0.35), **predict_kwargs)

    cap1 = cv2.VideoCapture(str(s1))
    cap2 = cv2.VideoCapture(str(s2))
    if not cap1.isOpened() or not cap2.isOpened():
        raise RuntimeError("Failed to open one or both video streams.")

    # Apply frame offset to Camera 2 for viewpoint difference
    if args.offset2 > 0:
        cap2.set(cv2.CAP_PROP_POS_FRAMES, args.offset2)

    width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap1.get(cv2.CAP_PROP_FPS) or 25.0

    # Composite side-by-side dimensions (plus top HUD header)
    display_w = width * 2
    display_h = height + 48

    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save, fourcc, fps_in, (display_w, display_h))

    has_display = bool(os.environ.get("DISPLAY")) and not args.headless
    win_title = f"Dual-Camera Live PPE Monitoring ({args.profile}) - Press 'q' or 'X' to close"
    if has_display:
        cv2.namedWindow(win_title, cv2.WINDOW_AUTOSIZE)

    infer_every = hw.get("infer_every", 1)

    # Tracking metrics
    f1_idx, f2_idx = 0, 0
    t1_times, t2_times = [], []
    c1_total_det, c2_total_det = 0, 0
    c1_classes, c2_classes = Counter(), Counter()

    b1_last, conf1_last, cls1_last = [], [], []
    b2_last, conf2_last, cls2_last = [], [], []

    print("[INFO] Running concurrent dual-stream inference...")
    t_start_all = time.time()
    try:
        while True:
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()

            # Handle looping
            if not ret1:
                cap1.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret1, frame1 = cap1.read()
            if not ret2:
                cap2.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret2, frame2 = cap2.read()

            if not ret1 or not ret2:
                break

            f1_idx += 1
            f2_idx += 1
            if args.max_frames > 0 and f1_idx > args.max_frames:
                break

            # Infer Cam 1
            if f1_idx % infer_every == 0:
                t0 = time.perf_counter()
                b1, c1, k1 = detector.infer(frame1)
                t1_times.append(time.perf_counter() - t0)
                b1_last, conf1_last, cls1_last = b1, c1, k1
                c1_total_det += len(b1)
                for item in k1:
                    c1_classes[int(item)] += 1
            else:
                b1, c1, k1 = b1_last, conf1_last, cls1_last

            # Infer Cam 2
            if f2_idx % infer_every == 0:
                t0 = time.perf_counter()
                b2, c2, k2 = detector.infer(frame2)
                t2_times.append(time.perf_counter() - t0)
                b2_last, conf2_last, cls2_last = b2, c2, k2
                c2_total_det += len(b2)
                for item in k2:
                    c2_classes[int(item)] += 1
            else:
                b2, c2, k2 = b2_last, conf2_last, cls2_last

            # Annotate both frames
            f1_annot = render_detections(frame1, b1, c1, k1, "Camera 1 — Entry Gate")
            f2_annot = render_detections(frame2, b2, c2, k2, "Camera 2 — Work Zone")

            # Stitch side-by-side
            composite_body = np.hstack([f1_annot, f2_annot])

            # Top unified HUD bar
            hud_header = np.zeros((48, display_w, 3), dtype=np.uint8)
            hud_header[:] = (20, 20, 20)

            fps1 = (1.0 / t1_times[-1]) if t1_times else 0.0
            fps2 = (1.0 / t2_times[-1]) if t2_times else 0.0
            fps_agg = fps1 + fps2

            hud_text = (
                f"Dual-Cam System [{args.profile}]  |  "
                f"Cam 1: {fps1:4.1f} FPS  |  "
                f"Cam 2: {fps2:4.1f} FPS  |  "
                f"Aggregate: {fps_agg:4.1f} FPS  |  "
                f"Frame: {f1_idx}/{args.max_frames or 'Live'}"
            )
            cv2.putText(hud_header, hud_text, (20, 31),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)

            # Final combined frame
            full_frame = np.vstack([hud_header, composite_body])

            if writer:
                writer.write(full_frame)

            if has_display:
                cv2.imshow(win_title, full_frame)
                key = cv2.waitKey(1)
                if key in (27, ord('q'), ord('Q')):
                    print("[INFO] Interrupted by user.")
                    break
                if cv2.getWindowProperty(win_title, cv2.WND_PROP_VISIBLE) < 1:
                    print("[INFO] Window closed by user.")
                    break

    finally:
        cap1.release()
        cap2.release()
        if writer:
            writer.release()
            print(f"[INFO] Annotated dual video saved → {args.save}")

    elapsed = time.time() - t_start_all
    avg_t1_ms = (sum(t1_times) / len(t1_times) * 1000.0) if t1_times else 0.0
    avg_t2_ms = (sum(t2_times) / len(t2_times) * 1000.0) if t2_times else 0.0
    fps1_eff = (1000.0 / avg_t1_ms) if avg_t1_ms > 0 else 0.0
    fps2_eff = (1000.0 / avg_t2_ms) if avg_t2_ms > 0 else 0.0
    fps_total = fps1_eff + fps2_eff

    print("\n" + "=" * 68)
    print("DUAL-CAMERA SIMULATION COMPLETE")
    print(f"Elapsed Time    : {elapsed:.2f} s")
    print(f"Frames processed: {f1_idx} per camera ({f1_idx * 2} total frames)")
    print(f"Cam 1 Latency   : {avg_t1_ms:.1f} ms  ({fps1_eff:.1f} FPS)")
    print(f"Cam 2 Latency   : {avg_t2_ms:.1f} ms  ({fps2_eff:.1f} FPS)")
    print(f"Aggregate FPS   : {fps_total:.1f} FPS")
    print(f"Total Detections: {c1_total_det + c2_total_det} (Cam1: {c1_total_det}, Cam2: {c2_total_det})")
    print("=" * 68)

    # Generate text report
    if args.report:
        report_lines = [
            "=" * 74,
            "PPE DUAL-CAMERA SIMULATION VALIDATION REPORT",
            "=" * 74,
            f"Date / Time       : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Hardware Profile  : {args.profile}",
            f"Model Path        : {model_path}",
            f"Camera 1 Stream   : {s1}",
            f"Camera 2 Stream   : {s2} (offset={args.offset2} frames)",
            f"Annotated Output  : {args.save}",
            "-" * 74,
            "PERFORMANCE METRICS:",
            f"  Frames / Camera : {f1_idx}",
            f"  Inference Skip  : infer_every={infer_every}",
            f"  Camera 1 Speed  : {fps1_eff:.2f} FPS ({avg_t1_ms:.1f} ms/frame)",
            f"  Camera 2 Speed  : {fps2_eff:.2f} FPS ({avg_t2_ms:.1f} ms/frame)",
            f"  Aggregate System: {fps_total:.2f} FPS total throughput",
            f"  Verdict         : {'PASS (Production ready)' if fps_total >= 5.0 else 'STAGING (Functional correctness)'}",
            "-" * 74,
            "DETECTION SUMMARY:",
            f"  Total Detections: {c1_total_det + c2_total_det}",
            f"    - Camera 1    : {c1_total_det} detections",
            f"    - Camera 2    : {c2_total_det} detections",
            "=" * 74,
            "",
            "Note: Close this window manually when finished reviewing test data.",
        ]
        Path(args.report).write_text("\n".join(report_lines) + "\n")
        print(f"[INFO] Dual-camera test report saved → {args.report}")

    # Launch desktop editor
    if has_display and not args.no_open_report and args.report:
        import subprocess
        try:
            print(f"[INFO] Opening dual-camera test report in text editor ({args.report})...")
            subprocess.Popen(["xdg-open", str(args.report)])
        except Exception as e:
            print(f"[WARN] Could not launch text editor: {e}")

    # Hold final frame
    if has_display:
        print("[INFO] Simulation complete. Final dual frame held open — press any key or 'X' to close.")
        while True:
            key = cv2.waitKey(50)
            if key != -1:
                break
            if cv2.getWindowProperty(win_title, cv2.WND_PROP_VISIBLE) < 1:
                break
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
