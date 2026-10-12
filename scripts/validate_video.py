#!/usr/bin/env python3
"""
Validate an exported model against a sample video stream.

Reads a test video (downloads validation/vtest.avi if not present), runs
inference using the specified hardware profile, renders live bounding boxes
with a visual HUD, saves the annotated video, and generates a test report.

Usage
-----
  python scripts/validate_video.py --profile inspiron3521_cpu
  python scripts/validate_video.py --profile dev_pt --max-frames 100
  python scripts/validate_video.py --headless
"""
import argparse
import os
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path

import cv2
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


def is_window_open(win_name: str) -> bool:
    """Check if window is open without crashing on Qt (-27 Null pointer)."""
    try:
        return cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) >= 1
    except Exception:
        return False


def ensure_video(path: Path):
    VAL_DIR.mkdir(exist_ok=True)
    if not path.exists():
        print(f"[INFO] Downloading sample video from {DEFAULT_VIDEO_URL} …")
        urllib.request.urlretrieve(DEFAULT_VIDEO_URL, path)
        print(f"[INFO] Saved sample video to {path}")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default=str(DEFAULT_VIDEO_PATH),
                    help="path to input video (default: validation/vtest.avi)")
    ap.add_argument("--profile",
                    default="inspiron3521_cpu" if (EXPORTS / "ppe.onnx").exists() else "dev_pt",
                    help="hardware profile from configs/hardware.yaml")
    ap.add_argument("--hardware", default=str(ROOT / "configs" / "hardware.yaml"))
    ap.add_argument("--save", default=str(VAL_DIR / "vtest_detected.mp4"),
                    help="path to save annotated video output (default: validation/vtest_detected.mp4)")
    ap.add_argument("--report", default=str(VAL_DIR / "video_test_data.txt"),
                    help="path to save text report (default: validation/video_test_data.txt)")
    ap.add_argument("--max-frames", type=int, default=120,
                    help="maximum frames to process (default: 120, 0 for entire video)")
    ap.add_argument("--headless", action="store_true",
                    help="run without graphical display window")
    ap.add_argument("--no-open-report", action="store_true",
                    help="do not automatically open report in desktop editor")
    return ap.parse_args()


def main():
    args = parse_args()
    source_path = Path(args.source)
    if source_path == DEFAULT_VIDEO_PATH:
        ensure_video(source_path)

    if not source_path.exists():
        raise FileNotFoundError(f"Input video not found: {source_path}")

    hw = load_yaml(args.hardware)["profiles"][args.profile]
    model_path, predict_kwargs = resolve_backend(hw, EXPORTS)
    print(f"[INFO] profile={args.profile}  model={model_path}")

    conf_thresh = hw.get("conf", 0.35)
    detector = Detector(model_path, conf=conf_thresh, **predict_kwargs)

    cap = cv2.VideoCapture(str(source_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video stream: {source_path}")

    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Video loaded: {width}x{height} @ {video_fps:.1f} FPS, {total_video_frames} frames")

    # Video writer for annotated output
    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save, fourcc, video_fps, (width, height))

    has_display = bool(os.environ.get("DISPLAY")) and not args.headless
    win_title = f"PPE Video Validation ({args.profile}) - Press 'q' or 'X' to exit"
    if has_display:
        cv2.namedWindow(win_title, cv2.WINDOW_AUTOSIZE)

    infer_every = hw.get("infer_every", 1)
    frame_idx = 0
    total_detections = 0
    class_counts = Counter()
    inference_times = []

    last_boxes, last_confs, last_classes = [], [], []

    print("[INFO] Starting active video inference...")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            if args.max_frames > 0 and frame_idx > args.max_frames:
                break

            # Frame skipping logic according to profile infer_every
            run_inference = (frame_idx % infer_every == 0)
            t_start = time.perf_counter()
            if run_inference:
                boxes, confs, classes = detector.infer(frame)
                t_infer = time.perf_counter() - t_start
                inference_times.append(t_infer)
                last_boxes, last_confs, last_classes = boxes, confs, classes
                total_detections += len(boxes)
                for c_id in classes:
                    class_counts[int(c_id)] += 1
            else:
                boxes, confs, classes = last_boxes, last_confs, last_classes

            # Render annotations on frame
            for box, c, k in zip(boxes, confs, classes):
                x1, y1, x2, y2 = map(int, box)
                color = (0, 220, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"cls {int(k)} {c * 100:.1f}%"
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                badge_y1 = max(0, y1 - h - 6)
                cv2.rectangle(frame, (x1, badge_y1), (x1 + w + 6, badge_y1 + h + 6), color, -1)
                cv2.putText(frame, label, (x1 + 3, badge_y1 + h + 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

            # On-screen HUD banner
            current_fps = (1.0 / inference_times[-1]) if inference_times else 0.0
            hud_text = (f"Profile: {args.profile} | Frame: {frame_idx}/{args.max_frames or total_video_frames} "
                        f"| FPS: {current_fps:4.1f} | Detections: {len(boxes)}")
            cv2.rectangle(frame, (8, 8), (width - 8, 38), (0, 0, 0), -1)
            cv2.putText(frame, hud_text, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)

            if writer:
                writer.write(frame)

            if has_display:
                cv2.imshow(win_title, frame)
                key = cv2.waitKey(1)
                if key in (27, ord('q'), ord('Q')):
                    print("[INFO] Playback interrupted by user.")
                    break
                if not is_window_open(win_title):
                    print("[INFO] Window closed by user.")
                    break

    except KeyboardInterrupt:
        print("[INFO] Playback interrupted by user (Ctrl-C).")
    except Exception as e:
        print(f"[WARN] Playback stopped: {e}")
    finally:
        cap.release()
        if writer:
            writer.release()
            print(f"[INFO] Annotated video saved → {args.save}")

    # Compute statistics
    avg_infer_ms = (sum(inference_times) / len(inference_times) * 1000.0) if inference_times else 0.0
    avg_fps = (1000.0 / avg_infer_ms) if avg_infer_ms > 0 else 0.0

    print("\n" + "=" * 60)
    print("VIDEO INFERENCE COMPLETE")
    print(f"Frames processed : {frame_idx}")
    print(f"Inference runs   : {len(inference_times)}")
    print(f"Avg infer time   : {avg_infer_ms:.1f} ms ({avg_fps:.1f} FPS)")
    print(f"Total detections : {total_detections}")
    print("=" * 60)

    # Generate text report
    if args.report:
        import datetime
        report_lines = [
            "=" * 72,
            "PPE SYSTEM VIDEO VALIDATION REPORT",
            "=" * 72,
            f"Date / Time       : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Hardware Profile  : {args.profile}",
            f"Model Path        : {model_path}",
            f"Video Source      : {source_path}",
            f"Annotated Output  : {args.save}",
            "-" * 72,
            f"Frames Processed  : {frame_idx}",
            f"Inference Cycles  : {len(inference_times)} (infer_every={infer_every})",
            f"Average Latency   : {avg_infer_ms:.2f} ms per frame",
            f"Inference Speed   : {avg_fps:.2f} FPS",
            f"Total Detections  : {total_detections}",
            "-" * 72,
            "Detected Objects Summary:",
        ]
        if not class_counts:
            report_lines.append("  (No detections recorded)")
        else:
            for cls_id, count in sorted(class_counts.items()):
                report_lines.append(f"  Class {cls_id:2d} : {count:5d} occurrences")
        report_lines.extend([
            "=" * 72,
            "",
            "Note: Close this window manually when finished reviewing test data.",
        ])
        Path(args.report).write_text("\n".join(report_lines) + "\n")
        print(f"[INFO] Video test report saved → {args.report}")

    # Open text report in system editor (FeatherPad)
    if has_display and not args.no_open_report and args.report:
        import subprocess
        try:
            print(f"[INFO] Opening video test report in text editor ({args.report})...")
            subprocess.Popen(["xdg-open", str(args.report)])
        except Exception as e:
            print(f"[WARN] Could not launch text editor: {e}")

    # Hold final frame until closed manually
    if has_display:
        print("[INFO] Video playback complete. Final frame held open — press any key or click 'X' to close.")
        while True:
            key = cv2.waitKey(50)
            if key != -1:
                break
            if not is_window_open(win_title):
                break
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
