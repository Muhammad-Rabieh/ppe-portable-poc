# Portability Strategy

## The mental model

```
ppe.pt               ← master model (always keep this)
    │
    ├──► exports/ppe.onnx              ← for CPU boxes (ONNX Runtime)
    ├──► exports/ppe_ncnn_model/       ← for ARM / Vulkan (ncnn)
    ├──► exports/ppe.engine            ← for NVIDIA (TensorRT, not portable)
    └──► exports/ppe_openvino_model/   ← for Intel iGPU (OpenVINO)
```

`ppe.pt` is the source of truth. Exported files are **build artefacts** — they can always be regenerated with `python scripts/export.py`.

---

## The abstraction

```
configs/hardware.yaml
        │
        ▼
  ppe/backend.py::resolve_backend()
        │
        ▼
  ppe/detector.py::Detector.infer()   ← identical API for every format
        │
        ▼
  ppe/pipeline.py::CameraWorker       ← per-camera, shared detector
```

**Only `backend.py` knows about formats.** Everything above it sees `(boxes, confs, classes)`.

---

## Adding a new runtime

1. Add an `elif fmt == "yourformat":` block in `ppe/backend.py` that sets `model_path`.
2. Add a profile in `configs/hardware.yaml` with `format: yourformat`.
3. That's it. No other files change.

---

## Why ONNX as the default CPU format?

- Runs on any OS without additional drivers.
- `onnxruntime` is a single pip install.
- Slightly faster than PyTorch `.pt` on CPU for inference-only use.
- Same model file works on Windows, Linux, macOS.

## Why ncnn for Vulkan / ARM?

- ncnn has native Vulkan compute — it's the only realistic path to GPU acceleration on AMD cards without CUDA.
- ncnn CPU backend is often faster than ONNX on ARM (better NEON utilisation).
- Ultralytics has first-class `format=ncnn` support.

## Why keep .pt around?

- Re-export any format at any time.
- Run in Codespaces / dev without any conversion step.
- `.pt` is the fallback when no other format is available.
