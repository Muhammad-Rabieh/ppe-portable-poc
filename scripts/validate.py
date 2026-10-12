#!/usr/bin/env python3
"""
Validate an exported model against a reference image.

Downloads validation/bus.jpg from Ultralytics if not present, runs inference
with the given profile, and prints detection results.  Useful for confirming
an export is working after moving to a new machine.

Usage
-----
  python scripts/validate.py --profile dev_pt
  python scripts/validate.py --profile n100_cpu
"""
import argparse
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
EXPORTS = ROOT / "exports"
VAL_DIR = ROOT / "validation"
IMG_URL = "https://ultralytics.com/images/bus.jpg"
IMG_PATH = VAL_DIR / "bus.jpg"


def load_yaml(p):
    import yaml
    with open(p) as f:
        return yaml.safe_load(f)


def is_window_open(win_name: str) -> bool:
    """Check if window is open without crashing on Qt (-27 Null pointer)."""
    try:
        return cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) >= 1
    except Exception:
        return False


def ensure_image():
    VAL_DIR.mkdir(exist_ok=True)
    if not IMG_PATH.exists():
        print(f"[INFO] Downloading {IMG_URL} …")
        urllib.request.urlretrieve(IMG_URL, IMG_PATH)
        print(f"[INFO] Saved to {IMG_PATH}")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default="inspiron3521_cpu" if (EXPORTS / "ppe.onnx").exists() else "dev_pt",
                    help="hardware profile from configs/hardware.yaml")
    ap.add_argument("--hardware", default=str(ROOT / "configs" / "hardware.yaml"))
    ap.add_argument("--save", default=str(VAL_DIR / "bus_detected.jpg"),
                    help="path to save annotated image (default: validation/bus_detected.jpg)")
    ap.add_argument("--report", default=str(VAL_DIR / "test_data.txt"),
                    help="path to save text report (default: validation/test_data.txt)")
    ap.add_argument("--no-open-report", action="store_true",
                    help="do not open text report in editor")
    ap.add_argument("--headless", action="store_true",
                    help="do not open graphical window (save image only)")
    ap.add_argument("--wait", type=int, default=0,
                    help="ms to auto-close window (default: 0 = stay open until closed manually)")
    return ap.parse_args()


def main():
    args = parse_args()
    ensure_image()

    import os
    import cv2
    from ppe.backend import resolve_backend
    from ppe.detector import Detector

    hw = load_yaml(args.hardware)["profiles"][args.profile]
    model_path, predict_kwargs = resolve_backend(hw, EXPORTS)
    print(f"[INFO] profile={args.profile}  model={model_path}")

    detector = Detector(model_path, conf=hw.get("conf", 0.4), **predict_kwargs)
    frame = cv2.imread(str(IMG_PATH))
    if frame is None:
        raise FileNotFoundError(f"Failed to read image at {IMG_PATH}")

    xyxy, confs, classes = detector.infer(frame)
    print(f"[INFO] detections on bus.jpg: {len(xyxy)}")

    # Visual rendering: Draw clean bounding boxes and badges
    for box, c, k in zip(xyxy, confs, classes):
        x1, y1, x2, y2 = map(int, box)
        label = f"cls {k} {c * 100:.1f}%"
        print(f"  {label:14s}  xyxy=[{x1}, {y1}, {x2}, {y2}]")

        # Bounding box
        color = (0, 220, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Label badge background
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        badge_y1 = max(0, y1 - h - 8)
        cv2.rectangle(frame, (x1, badge_y1), (x1 + w + 8, badge_y1 + h + 8), color, -1)
        cv2.putText(frame, label, (x1 + 4, badge_y1 + h + 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

    # Top summary banner
    banner = f"Profile: {args.profile} | Detections: {len(xyxy)}"
    cv2.putText(frame, banner, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv2.LINE_AA)

    # Always save the visual output
    if args.save:
        cv2.imwrite(args.save, frame)
        print(f"[INFO] Visual test saved → {args.save}")

    # Generate and save the test data text report
    if args.report:
        import datetime
        report_lines = [
            "=" * 72,
            "PPE SYSTEM VALIDATION TEST REPORT",
            "=" * 72,
            f"Date / Time       : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Hardware Profile  : {args.profile}",
            f"Model Path        : {model_path}",
            f"Input Image       : {IMG_PATH}",
            f"Annotated Image   : {args.save}",
            "-" * 72,
            f"Total Detections  : {len(xyxy)}",
            f"Status            : {'[OK] Model is producing detections.' if len(xyxy) > 0 else '[WARN] No detections.'}",
            "-" * 72,
            "Detection Breakdown:",
        ]
        if len(xyxy) == 0:
            report_lines.append("  (No detections recorded)")
        else:
            for idx, (box, c, k) in enumerate(zip(xyxy, confs, classes), 1):
                x1, y1, x2, y2 = map(int, box)
                report_lines.append(
                    f"  #{idx:<2d} | Class: {k:2d} | Confidence: {c * 100:5.1f}% | Box: [{x1}, {y1}, {x2}, {y2}]"
                )
        report_lines.extend([
            "=" * 72,
            "",
            "Note: Close this window manually when finished reviewing test data.",
        ])
        report_content = "\n".join(report_lines) + "\n"
        Path(args.report).write_text(report_content)
        print(f"[INFO] Test data report saved → {args.report}")

    if len(xyxy) == 0:
        print("[WARN] No detections — model may be uninitialised (placeholder .pt).")
    else:
        print("[OK] Model is producing detections.")

    # Always show window if graphical display is present and not headless
    has_display = bool(os.environ.get("DISPLAY"))

    # Open text report in system editor (e.g. FeatherPad) so user can view & close manually
    if not args.headless and has_display and not args.no_open_report and args.report:
        import subprocess
        try:
            print(f"[INFO] Opening test data text report in text editor ({args.report})...")
            subprocess.Popen(["xdg-open", str(args.report)])
        except Exception as e:
            print(f"[WARN] Could not launch text editor: {e}")
    if not args.headless and has_display:
        win_title = "PPE Visual Validation Test (press any key or 'X' to close)"
        cv2.namedWindow(win_title, cv2.WINDOW_AUTOSIZE)
        cv2.imshow(win_title, frame)
        if args.wait > 0:
            print(f"[INFO] Displaying visual test window (auto-closes in {args.wait / 1000.0:.1f}s or on keypress)...")
            cv2.waitKey(args.wait)
        else:
            print("[INFO] Displaying visual test window (stays open until closed manually or any key is pressed)...")
            while True:
                key = cv2.waitKey(50)
                if key != -1:  # user pressed any key
                    break
                # check if user closed the window with 'X' button
                if not is_window_open(win_title):
                    break
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
