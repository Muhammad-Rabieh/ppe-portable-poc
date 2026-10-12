#!/usr/bin/env python3
"""
Simulate or run dual live cameras with real-time PPE detection simultaneously.

Supports:
  1. Live RTSP cameras (from configs/cameras.yaml or --source1 / --source2)
  2. Local video files (side-by-side simulation)

Runs threaded capture workers (ppe.stream.CameraStream) with TCP transport and
latest-frame semantics, renders a unified side-by-side dashboard with live FPS
and bounding boxes, and runs until manually closed by the user.

Usage
-----
  # Run live RTSP cameras from configs/cameras.yaml:
  python scripts/validate_dual_video.py --live --profile inspiron3521_cpu

  # Pass custom RTSP URLs directly:
  python scripts/validate_dual_video.py --source1 "rtsp://..." --source2 "rtsp://..."

  # Offline video simulation:
  python scripts/validate_dual_video.py --profile inspiron3521_cpu --max-frames 120
"""
import argparse
import datetime
import os
import re
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
from ppe.stream import CameraStream


def load_yaml(p):
    with open(p) as f:
        return yaml.safe_load(f)


def ensure_video(path: Path):
    VAL_DIR.mkdir(exist_ok=True)
    if not path.exists():
        print(f"[INFO] Downloading sample video from {DEFAULT_VIDEO_URL} …")
        urllib.request.urlretrieve(DEFAULT_VIDEO_URL, path)
        print(f"[INFO] Saved sample video to {path}")


def resolve_source_string(src: str) -> str:
    """Expand environment variables like ${RTSP_PASSWORD} or ${VAR:-default}."""
    if not isinstance(src, str):
        return src
    # Expand ${VAR:-default} with env var if present, else fallback
    def _sub_default(match):
        var_name = match.group(1)
        fallback = match.group(2)
        return os.environ.get(var_name, fallback)

    src = re.sub(r"\$\{([A-Za-z0-9_]+):-([^}]+)\}", _sub_default, src)
    # Expand simple ${VAR}
    src = os.path.expandvars(src)
    return src


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--live", action="store_true",
                    help="use live camera sources from configs/cameras.yaml")
    ap.add_argument("--cameras", default=str(ROOT / "configs" / "cameras.yaml"),
                    help="path to cameras.yaml")
    ap.add_argument("--source1", default=None,
                    help="video source for Camera 1 (RTSP URL, file path, or USB index)")
    ap.add_argument("--source2", default=None,
                    help="video source for Camera 2 (RTSP URL, file path, or USB index)")
    ap.add_argument("--offset2", type=int, default=120,
                    help="frame offset for Camera 2 when using the same video file (default: 120)")
    ap.add_argument("--profile",
                    default="inspiron3521_cpu" if (EXPORTS / "ppe.onnx").exists() else "dev_pt",
                    help="hardware profile from configs/hardware.yaml")
    ap.add_argument("--hardware", default=str(ROOT / "configs" / "hardware.yaml"))
    ap.add_argument("--save", default=str(VAL_DIR / "dual_video_detected.mp4"),
                    help="path to save side-by-side video (default: validation/dual_video_detected.mp4)")
    ap.add_argument("--report", default=str(VAL_DIR / "dual_video_test_data.txt"),
                    help="path to save text report (default: validation/dual_video_test_data.txt)")
    ap.add_argument("--max-frames", type=int, default=None,
                    help="max frames (default: 0 / infinite for live RTSP, 120 for offline test)")
    ap.add_argument("--headless", action="store_true",
                    help="run without graphical display window")
    ap.add_argument("--no-open-report", action="store_true",
                    help="do not automatically open report in desktop editor")
    return ap.parse_args()


def render_detections(frame, boxes, confs, classes, cam_name: str, src_label: str):
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
    label_text = f"[{cam_name}] Detections: {len(boxes)} | {src_label}"
    cv2.putText(frame, label_text, (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def mask_url_password(url: str) -> str:
    """Mask credentials in URL for logs/display."""
    if not isinstance(url, str):
        return str(url)
    return re.sub(r"://([^:]+):([^@]+)@", r"://\1:******@", url)


def is_window_open(win_name: str) -> bool:
    """Check if window is open without crashing on Qt (-27 Null pointer)."""
    try:
        return cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) >= 1
    except Exception:
        return False


def main():
    args = parse_args()

    # Determine sources
    if args.live or (args.source1 is None and args.source2 is None):
        cam_cfg = load_yaml(args.cameras)["cameras"]
        s1 = resolve_source_string(cam_cfg[0]["source"])
        s2 = resolve_source_string(cam_cfg[1]["source"]) if len(cam_cfg) > 1 else s1
        name1 = cam_cfg[0].get("name", "Cam 1")
        name2 = cam_cfg[1].get("name", "Cam 2") if len(cam_cfg) > 1 else "Cam 2"
    else:
        s1 = resolve_source_string(args.source1 or str(DEFAULT_VIDEO_PATH))
        s2 = resolve_source_string(args.source2 or str(DEFAULT_VIDEO_PATH))
        name1 = "Camera 1"
        name2 = "Camera 2"

    is_live = any(str(s).startswith("rtsp://") or str(s).startswith("http://") for s in (s1, s2))

    # Download default video if offline test
    if str(s1) == str(DEFAULT_VIDEO_PATH) or str(s2) == str(DEFAULT_VIDEO_PATH):
        ensure_video(DEFAULT_VIDEO_PATH)

    max_frames = args.max_frames
    if max_frames is None:
        max_frames = 0 if is_live else 120

    hw = load_yaml(args.hardware)["profiles"][args.profile]
    model_path, predict_kwargs = resolve_backend(hw, EXPORTS)
    print(f"\n====================================================================")
    print(f"PPE DUAL-CAMERA MONITORING INITIALIZING")
    print(f"Profile  : {args.profile}")
    print(f"Model    : {model_path}")
    print(f"Camera 1 : {mask_url_password(s1)}")
    print(f"Camera 2 : {mask_url_password(s2)}")
    print(f"Mode     : {'LIVE CONTINUOUS (runs until closed)' if max_frames == 0 else f'TEST ({max_frames} frames)'}")
    print(f"====================================================================\n")

    detector = Detector(model_path, conf=hw.get("conf", 0.35), **predict_kwargs)

    # Start threaded readers with latest-frame-only semantics
    print(f"[INFO] Connecting background streams...")
    stream1 = CameraStream(s1, name=name1).start()
    stream2 = CameraStream(s2, name=name2).start()

    # Wait briefly for initial frames
    print("[INFO] Waiting for first video frames from both cameras...")
    t_wait_start = time.time()
    frame1, frame2 = None, None
    while time.time() - t_wait_start < 8.0:
        if frame1 is None:
            frame1 = stream1.read()
        if frame2 is None:
            frame2 = stream2.read()
        if frame1 is not None and frame2 is not None:
            break
        time.sleep(0.1)

    if frame1 is None and frame2 is None:
        stream1.stop()
        stream2.stop()
        raise RuntimeError("Failed to receive video frames from cameras within timeout. "
                           "Please verify network connection and credentials.")

    # In case one camera connects slightly earlier
    if frame1 is None:
        frame1 = np.zeros_like(frame2)
    if frame2 is None:
        frame2 = np.zeros_like(frame1)

    # Normalize dimensions for side-by-side display
    target_h = min(frame1.shape[0], frame2.shape[0], 480)
    w1_norm = int(frame1.shape[1] * target_h / frame1.shape[0])
    w2_norm = int(frame2.shape[1] * target_h / frame2.shape[0])
    display_w = w1_norm + w2_norm
    display_h = target_h + 48

    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save, fourcc, 15.0, (display_w, display_h))

    has_display = bool(os.environ.get("DISPLAY")) and not args.headless
    win_title = f"Live Dual-Camera PPE Monitoring [{args.profile}] - Press 'q' or 'X' to close"
    if has_display:
        cv2.namedWindow(win_title, cv2.WINDOW_AUTOSIZE)

    infer_every = hw.get("infer_every", 1)

    f1_idx, f2_idx = 0, 0
    t1_times, t2_times = [], []
    c1_total_det, c2_total_det = 0, 0
    c1_classes, c2_classes = Counter(), Counter()

    b1_last, conf1_last, cls1_last = [], [], []
    b2_last, conf2_last, cls2_last = [], [], []

    print("[INFO] Live dual monitoring active! Display window open.")
    t_start_all = time.time()
    try:
        while True:
            f1 = stream1.read()
            f2 = stream2.read()

            if f1 is None:
                f1 = frame1
            else:
                frame1 = f1
            if f2 is None:
                f2 = frame2
            else:
                frame2 = f2

            f1_idx += 1
            f2_idx += 1

            if max_frames > 0 and f1_idx > max_frames:
                break

            # Infer Cam 1
            if f1_idx % infer_every == 0:
                t0 = time.perf_counter()
                b1, c1, k1 = detector.infer(f1)
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
                b2, c2, k2 = detector.infer(f2)
                t2_times.append(time.perf_counter() - t0)
                b2_last, conf2_last, cls2_last = b2, c2, k2
                c2_total_det += len(b2)
                for item in k2:
                    c2_classes[int(item)] += 1
            else:
                b2, c2, k2 = b2_last, conf2_last, cls2_last

            # Annotate frames
            f1_annot = render_detections(f1.copy(), b1, c1, k1, name1, mask_url_password(s1).split("@")[-1])
            f2_annot = render_detections(f2.copy(), b2, c2, k2, name2, mask_url_password(s2).split("@")[-1])

            # Resize to match target height cleanly
            if f1_annot.shape[0] != target_h:
                f1_annot = cv2.resize(f1_annot, (w1_norm, target_h))
            if f2_annot.shape[0] != target_h:
                f2_annot = cv2.resize(f2_annot, (w2_norm, target_h))

            # Horizontal side-by-side stitch
            composite_body = np.hstack([f1_annot, f2_annot])

            # Top unified HUD bar
            hud_header = np.zeros((48, display_w, 3), dtype=np.uint8)
            hud_header[:] = (20, 20, 20)

            fps1 = (1.0 / t1_times[-1]) if t1_times else stream1.fps
            fps2 = (1.0 / t2_times[-1]) if t2_times else stream2.fps
            fps_agg = fps1 + fps2

            hud_text = (
                f"Dual-Cam Live [{args.profile}]  |  "
                f"{name1}: {fps1:4.1f} FPS  |  "
                f"{name2}: {fps2:4.1f} FPS  |  "
                f"Agg: {fps_agg:4.1f} FPS  |  "
                f"Frames: {f1_idx}"
            )
            cv2.putText(hud_header, hud_text, (16, 31),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 255), 2, cv2.LINE_AA)

            full_frame = np.vstack([hud_header, composite_body])

            if writer:
                writer.write(full_frame)

            if has_display:
                cv2.imshow(win_title, full_frame)
                key = cv2.waitKey(1)
                if key in (27, ord('q'), ord('Q')):
                    print("[INFO] Monitoring stopped by user ('q' / Esc).")
                    break
                if not is_window_open(win_title):
                    print("[INFO] Window closed by user ('X' button).")
                    break

            time.sleep(0.005)

    except KeyboardInterrupt:
        print("[INFO] Monitoring interrupted by user (Ctrl-C).")
    except Exception as e:
        print(f"[WARN] Stopped: {e}")
    finally:
        print("[INFO] Stopping camera streams...")
        stream1.stop()
        stream2.stop()
        if writer:
            writer.release()
            print(f"[INFO] Annotated dual video saved → {args.save}")

    elapsed = max(0.001, time.time() - t_start_all)
    avg_t1_ms = (sum(t1_times) / len(t1_times) * 1000.0) if t1_times else 0.0
    avg_t2_ms = (sum(t2_times) / len(t2_times) * 1000.0) if t2_times else 0.0
    fps1_eff = (1000.0 / avg_t1_ms) if avg_t1_ms > 0 else 0.0
    fps2_eff = (1000.0 / avg_t2_ms) if avg_t2_ms > 0 else 0.0
    fps_total = fps1_eff + fps2_eff

    print("\n" + "=" * 68)
    print("LIVE DUAL-CAMERA SESSION SUMMARY")
    print(f"Elapsed Time    : {elapsed:.1f} s")
    print(f"Frames Captured : {f1_idx} ({f1_idx / elapsed:.1f} capture FPS)")
    print(f"Cam 1 Latency   : {avg_t1_ms:.1f} ms  ({fps1_eff:.1f} FPS)")
    print(f"Cam 2 Latency   : {avg_t2_ms:.1f} ms  ({fps2_eff:.1f} FPS)")
    print(f"Aggregate FPS   : {fps_total:.1f} FPS")
    print(f"Total Detections: {c1_total_det + c2_total_det} ({name1}: {c1_total_det}, {name2}: {c2_total_det})")
    print("=" * 68)

    # Generate text report
    if args.report:
        report_lines = [
            "=" * 74,
            "PPE DUAL LIVE CAMERA MONITORING REPORT",
            "=" * 74,
            f"Date / Time       : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Hardware Profile  : {args.profile}",
            f"Model Path        : {model_path}",
            f"Camera 1 Stream   : {mask_url_password(s1)}",
            f"Camera 2 Stream   : {mask_url_password(s2)}",
            f"Annotated Output  : {args.save}",
            "-" * 74,
            "OPERATIONAL METRICS:",
            f"  Session Duration: {elapsed:.1f} seconds",
            f"  Total Cycles    : {f1_idx} frames per camera",
            f"  Inference Skip  : infer_every={infer_every}",
            f"  Camera 1 Speed  : {fps1_eff:.2f} FPS ({avg_t1_ms:.1f} ms/frame)",
            f"  Camera 2 Speed  : {fps2_eff:.2f} FPS ({avg_t2_ms:.1f} ms/frame)",
            f"  Aggregate System: {fps_total:.2f} FPS total throughput",
            "-" * 74,
            "DETECTION SUMMARY:",
            f"  Total Detections: {c1_total_det + c2_total_det}",
            f"    - {name1:<12s}: {c1_total_det} detections",
            f"    - {name2:<12s}: {c2_total_det} detections",
            "=" * 74,
            "",
            "Note: Close this window manually when finished reviewing test data.",
        ]
        Path(args.report).write_text("\n".join(report_lines) + "\n")
        print(f"[INFO] Live camera test report saved → {args.report}")

    # Launch desktop editor
    if has_display and not args.no_open_report and args.report:
        import subprocess
        try:
            print(f"[INFO] Opening dual live report in text editor ({args.report})...")
            subprocess.Popen(["xdg-open", str(args.report)])
        except Exception as e:
            print(f"[WARN] Could not launch text editor: {e}")

    if has_display:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
