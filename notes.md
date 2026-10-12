# PPE Portability POC — notes

## Model

- Base: YOLOv8n (placeholder — replace with trained PPE model)
- Training: see `notebooks/train_ppe_colab.ipynb`

## Export log

| Date | Ultralytics version | Format | imgsz | Notes |
|------|---------------------|--------|-------|-------|
| 2026-10-12 | 8.4.175 | ONNX (opset 18, onnxslim) | 416 | CPU-only wheel, exports/ppe.onnx (12.2 MB) |

## Hardware tested

| Box | Profile | bench result | Date |
|-----|---------|--------------|------|
| Dell Inspiron 3521 (i5-3317U, 16 GB) | inspiron3521_cpu | 1-cam: 9.4 FPS (106 ms) / 2-cam: 3.6 FPS/cam | 2026-10-12 |

## Machine profiles tested

### Dell Inspiron 3521 (staging/dev) — *this machine*
- **OS**: MX-25.2 XFCE, Debian 13 trixie, kernel 6.12
- **CPU**: i5-3317U (2C/4T, 1.7 GHz, 17 W)
- **RAM**: 16 GB DDR3
- **GPU**: Intel HD 4000 (iGPU) + Radeon HD 7500M (TeraScale-2, **no Vulkan**)
- **Format**: ONNX CPU only. `ncnn` Vulkan unavailable on this GPU architecture.
- **Project path**: `~/PRoJeCTs/ppe-portable-poc`
- **Profile**: `inspiron3521_cpu` (`imgsz=256`, `infer_every=5`)
- **Benchmark measured**: 1-cam: 9.4 FPS (106.4 ms) | 2-cam unthrottled: 3.6 FPS/cam
- **Role**: **Staging** — proves pipeline and export correctness, not throughput
- **Date tested**: 2026-10-12

### What "success" looks like on this box
1. `python scripts/bench.py --profile inspiron3521_cpu --frames 30` prints any FPS number without crashing. (PASSED - 9.4 FPS 1-cam / 3.6 FPS/cam 2-cam)
2. `python scripts/run_cameras.py --profile inspiron3521_cpu --headless` runs 2 streams for 60 s, printing per-camera FPS every second, and stops cleanly with Ctrl-C.
3. This row and `docs/hardware.md` have been updated with the actual bench result.

## To-do / known issues

- [ ] Replace placeholder ppe.pt with trained model
- [x] Run `bench.py --profile inspiron3521_cpu` and record result above
- [ ] Confirm ncnn Vulkan works on RX 580 (not applicable on Inspiron)
- [ ] Test with real RTSP cameras (swap `cameras.yaml` back when ready)

---
_Keep this file updated. It's the changelog that bench.py results and exports reference._
