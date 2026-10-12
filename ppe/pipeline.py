"""Per-camera worker: pull latest frame, run inference, optionally draw boxes.

One CameraWorker per camera.  Workers share a single Detector instance so the
model is loaded into memory only once regardless of camera count.
"""
import threading
import time
import cv2

from .stream import CameraStream
from .detector import Detector

# Colour palette — distinct per class index (cycles if more classes than colours)
_COLOURS = [
    (0, 255, 0),    # green
    (0, 128, 255),  # orange
    (255, 0, 128),  # pink
    (255, 255, 0),  # cyan
    (128, 0, 255),  # violet
]


class CameraWorker:
    """
    Ties together a CameraStream and a Detector in a background thread.

    Parameters
    ----------
    name         : human-readable camera label (used in display + logs)
    source       : passed straight to CameraStream (int, path, RTSP URL)
    detector     : shared Detector instance
    infer_every  : run inference on every Nth frame (1 = every frame)
    draw         : annotate frames with bounding boxes
    conf         : confidence threshold label (informational; Detector owns it)
    """

    def __init__(
        self,
        name: str,
        source,
        detector: Detector,
        infer_every: int = 1,
        draw: bool = True,
        conf: float = 0.4,
    ):
        self.name = name
        self.stream = CameraStream(source, name=name)
        self.detector = detector
        self.infer_every = max(1, int(infer_every))
        self.draw = draw
        self.conf = conf

        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

        # Outputs readable from outside the thread
        self.latest = None       # most-recent annotated frame (BGR ndarray)
        self.fps_capture: float = 0.0
        self.fps_infer: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> "CameraWorker":
        """Start stream + inference thread. Returns self (fluent)."""
        self.stream.start()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        """Stop inference thread and underlying stream."""
        self._stop.set()
        self.stream.stop()
        if self._thread:
            self._thread.join(timeout=2.0)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _loop(self):
        n = 0
        last = time.time()
        count = 0

        while not self._stop.is_set():
            frame = self.stream.read()
            if frame is None:
                time.sleep(0.005)
                continue

            n += 1
            if n % self.infer_every != 0:
                continue

            xyxy, confs, classes = self.detector.infer(frame)

            if self.draw:
                for box, c, k in zip(xyxy, confs, classes):
                    x1, y1, x2, y2 = map(int, box)
                    colour = _COLOURS[int(k) % len(_COLOURS)]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
                    label = f"cls{k} {c:.2f}"
                    cv2.putText(
                        frame, label,
                        (x1, max(20, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1,
                    )

            self.latest = frame

            count += 1
            now = time.time()
            if now - last >= 1.0:
                self.fps_infer = count / (now - last)
                self.fps_capture = self.stream.fps
                count = 0
                last = now
