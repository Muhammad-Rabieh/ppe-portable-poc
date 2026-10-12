"""Thin wrapper around Ultralytics — same API for every backend."""
import numpy as np
from pathlib import Path
from ultralytics import YOLO


class Detector:
    """
    Wraps a YOLO model loaded from any supported format (pt / onnx / ncnn /
    engine / openvino).  Callers only see ``infer(frame)``.
    """

    def __init__(self, model_path: str, conf: float = 0.4, **predict_kwargs):
        self.model = YOLO(str(model_path))
        self.conf = conf
        self.predict_kwargs = predict_kwargs

    def infer(self, frame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run inference on one BGR frame (numpy array).

        Returns
        -------
        boxes  : (N, 4) float32  — xyxy pixel coordinates
        confs  : (N,)   float32  — confidence scores
        classes: (N,)   int      — class indices
        """
        results = self.model.predict(frame, conf=self.conf, **self.predict_kwargs)
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return (
                np.empty((0, 4), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=int),
            )
        return (
            r.boxes.xyxy.cpu().numpy(),
            r.boxes.conf.cpu().numpy(),
            r.boxes.cls.cpu().numpy().astype(int),
        )
