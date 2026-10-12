"""Threaded camera reader with drop-old-frames semantics.

On cheap hardware, blocking reads accumulate a growing queue of stale frames.
This reader keeps only the *latest* frame in a lock-guarded slot and discards
everything else — the inference thread always sees the newest image.
"""
import os
import threading
import time
import cv2


class CameraStream:
    """
    Background thread that continuously reads frames from ``source``.

    Parameters
    ----------
    source          : int | str — USB index, file path, or RTSP URL
    name            : label used in logs
    reconnect_delay : seconds to wait before retrying a dead capture
    """

    def __init__(self, source, name: str = "cam", reconnect_delay: float = 2.0):
        self.source = source
        self.name = name
        self.reconnect_delay = reconnect_delay

        self._cap: cv2.VideoCapture | None = None
        self._frame = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.fps: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> "CameraStream":
        """Start the background reader thread and return self (fluent)."""
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def read(self):
        """Return a copy of the latest frame, or None if not yet available."""
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def stop(self):
        """Signal the reader thread to exit and release the capture device."""
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._cap:
            self._cap.release()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _open(self) -> cv2.VideoCapture:
        # Force TCP transport for RTSP to avoid UDP packet-loss drops
        if isinstance(self.source, str) and self.source.startswith("rtsp"):
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
        cap = cv2.VideoCapture(self.source)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # keep only the newest frame
        except Exception:
            pass
        return cap

    def _loop(self):
        last = time.time()
        count = 0
        while not self._stop.is_set():
            if self._cap is None or not self._cap.isOpened():
                self._cap = self._open()
                if not self._cap.isOpened():
                    print(f"[{self.name}] could not open {self.source!r}, "
                          f"retrying in {self.reconnect_delay}s …")
                    time.sleep(self.reconnect_delay)
                    continue

            ok, frame = self._cap.read()
            if not ok:
                print(f"[{self.name}] read failed — reconnecting …")
                self._cap.release()
                self._cap = None
                time.sleep(self.reconnect_delay)
                continue

            with self._lock:
                self._frame = frame

            count += 1
            now = time.time()
            if now - last >= 1.0:
                self.fps = count / (now - last)
                count = 0
                last = now
