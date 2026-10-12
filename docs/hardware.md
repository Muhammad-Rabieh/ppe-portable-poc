# Cheap hardware for 2-camera PPE detection

**Target**: 2 streams, YOLOv8n/s at 320–416 px.
≥ 5 FPS per camera is usable · ≥ 10 FPS is comfortable · < 5 FPS → drop `imgsz` or set `infer_every`.

---

## Hardware comparison

| Box | Price (approx) | Format | Expected 2-cam FPS/cam | Notes |
|-----|----------------|--------|------------------------|-------|
| Intel N100 mini PC (8–16 GB) | $130–180 | ONNX CPU | 5–10 @ 320 px | Cheapest sane option. Fanless. |
| Intel N150 / N305 mini PC | $180–260 | ONNX CPU | 8–15 @ 320 px | N305 has more cores — big jump. |
| Old office i5/i7 (8th gen+) | $80–150 used | ONNX CPU | 8–20 @ 416 px | Often cheapest per FPS. |
| RX 580 8 GB (AMD) | — (existing) | ncnn + Vulkan | 20–40 @ 416 px | Your current path. |
| GTX 1650 / RTX 3050 | $100–200 used | ONNX CUDA / TensorRT | 40–80 @ 640 px | Best value if a PCIe slot is free. |
| Jetson Orin Nano 8 GB | $250–500 | TensorRT | 30–60 @ 640 px | Low power, embedded form factor. |
| Raspberry Pi 5 (8 GB) | $80 | ncnn CPU | 2–5 @ 320 px | Works but marginal for 2 cams. |
| Orange Pi 5 / RK3588 | $100–200 | ncnn / RKNN | 10–20 @ 320 px | NPU helps a lot with RKNN. |
| Dell Inspiron 3521 (i5-3317U, 16 GB) | — | ONNX CPU | 3.6 @ 256 px (measured) | **Staging only.** HD 7500M is TeraScale-2 → no Vulkan. Used for pipeline verification. |

> ⚠️ **TeraScale-1/2 GPUs** (Radeon HD 5000/6000/7000M series) do **not** support Vulkan.
> Only GCN (HD 7000 desktop, Rx 200+) and newer work with ncnn Vulkan.

---

## Bandwidth reality check

| Setup | Bandwidth |
|-------|-----------|
| Two 1080p RTSP streams | ≈ 4–8 Mbps — fine on gigabit LAN |
| Two USB 2.0 cameras at 1080p | Fight for bus bandwidth — use RTSP/IP cams, USB 3.0, or drop to 720p |
| Two 720p RTSP streams | ≈ 2–4 Mbps — comfortable on any modern router |

---

## Rule of thumb

Run `scripts/bench.py` on the box before deploying:

```bash
python scripts/bench.py --profile n100_cpu
```

- **PASS (≥ 5 FPS/cam)** → run `scripts/run_cameras.py --profile n100_cpu`
- **FAIL (< 5 FPS/cam)** → try in order:
  1. `--imgsz 320` (or lower)
  2. Set `infer_every: 2` in `configs/hardware.yaml`
  3. Switch to `yolov8n` instead of `yolov8s`
  4. Upgrade the box

---

## Profile quick-pick guide

```
Box type                       → profile
──────────────────────────────────────────────────────
Intel N100 mini PC             → n100_cpu
Any old CPU-only machine       → old_i5_cpu  or  n100_cpu
AMD GPU (RX 470/480/580/...)   → rx580_vulkan
NVIDIA GPU (CUDA)              → nvidia_onnx_cuda
NVIDIA GPU (TensorRT, faster)  → nvidia_tensorrt
Raspberry Pi 5                 → rpi5_ncnn
Dell Inspiron 3521 (staging)   → inspiron3521_cpu
Dev / Codespaces / no GPU      → dev_pt
```
