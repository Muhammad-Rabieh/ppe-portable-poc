# PPE Portability POC

> **Goal: run 2 cameras on cheap hardware, and move to a different box with a config change — not a code change.**

---

## How it works

```
ppe.pt  ──export──►  exports/ppe.onnx
                     exports/ppe_ncnn_model
                     exports/ppe.engine      (NVIDIA only)

configs/hardware.yaml ──► ppe/backend.py ──► ppe/detector.py
         │                                        │
configs/cameras.yaml ──► ppe/pipeline.py ◄────────┘
                              │
                    scripts/run_cameras.py
```

- **`ppe.pt`** is the master. Everything is regenerated from it.
- **`ppe/backend.py`** picks the format. Nothing else in the codebase knows or cares about the runtime.
- **`configs/hardware.yaml`** says which runtime and settings to use on each box.
- **`scripts/bench.py`** tells you whether a box can handle 2 cameras before you commit to it.
- **`scripts/run_cameras.py`** runs them.

---

## Quick Start — CPU Baseline / Staging

The repository automatically manages a CPU-only virtual environment (`.venv`) with zero CUDA dependencies:

```bash
# 1. Install dependencies (CPU PyTorch + OpenCV + ONNX Runtime)
make setup

# 2. Download placeholder YOLOv8n weights
make download

# 3. Export model to ONNX format
make export

# 4. Run interactive visual validation on sample image
make validate
# or explicitly for Dell Inspiron 3521 / older dual-core CPUs:
make validate-inspiron

# 5. Run interactive visual validation on sample video stream
make validate-video
# or on Dell Inspiron 3521:
make validate-video-inspiron

# 6. Run dual-camera simulation test (side-by-side dashboard)
make validate-dual-video
# or on Dell Inspiron 3521:
make validate-dual-video-inspiron

# 7. Run performance benchmark
make bench
# or benchmark Dell Inspiron 3521:
make bench-inspiron
```

> **Visual Validation**: Running `make validate`, `make validate-video`, or `make validate-dual-video` will automatically:
> 1. Draw color-coded bounding boxes and class labels directly onto the test image/stream with live FPS HUD.
> 2. Open an active **graphical preview window** (with side-by-side view for dual streams; stays open until you press any key or click `X`).
> 3. Save the annotated output (`validation/bus_detected.jpg`, `validation/vtest_detected.mp4`, or `validation/dual_video_detected.mp4`).
> 4. Generate and open a structured text report in your desktop editor (`validation/test_data.txt`, `validation/video_test_data.txt`, or `validation/dual_video_test_data.txt`).
> *(To run silently in headless CI or SSH sessions, pass `--headless`)*.

---

## On Real Hardware with 2 Cameras

```bash
# Edit configs/cameras.yaml to point at your streams first

# AMD RX 580 (ncnn + Vulkan):
python scripts/run_cameras.py --profile rx580_vulkan

# Cheap CPU box (Intel N100):
python scripts/run_cameras.py --profile n100_cpu

# Dell Inspiron 3521 / older CPU staging:
python scripts/run_cameras.py --profile inspiron3521_cpu

# Headless server (no monitor):
python scripts/run_cameras.py --profile n100_cpu --headless
```

---

## Adding new hardware

Only add a stanza to `configs/hardware.yaml` — no code changes:

```yaml
  my_new_box:
    format: onnx       # or ncnn / engine / openvino
    device: cpu        # or 0, vulkan
    imgsz: 320
    half: false
    infer_every: 2
    conf: 0.4
```

Then run `bench.py` with `--profile my_new_box` to check if it passes.

---

## Train on Colab

Open `notebooks/train_ppe_colab.ipynb`, train, and save `ppe.pt` to Google Drive.
Drop it into the repo root, then:

```bash
python scripts/export.py
python scripts/validate.py
python scripts/bench.py --profile <your_box>
```

---

## Documentation

| Doc | What it covers |
|-----|----------------|
| [docs/portability-strategy.md](docs/portability-strategy.md) | The `.pt` / `exports/` mental model |
| [docs/hardware.md](docs/hardware.md) | Cheap hardware table + expected FPS per box |
| [docs/migration.md](docs/migration.md) | Moving to a new box — checklist |

---

## Makefile Targets

```bash
make setup              # Install pure CPU dependencies into .venv
make download           # Download placeholder ppe.pt (YOLOv8n)
make export             # Export ppe.pt to ONNX in exports/
make validate           # Visual validation with interactive preview & report
make validate-inspiron  # Visual validation for Dell Inspiron 3521 profile
make validate-video     # Visual validation on sample video stream
make validate-video-inspiron # Video visual validation on Inspiron 3521
make validate-dual-video     # Dual-camera simulation visual test (side-by-side)
make validate-dual-video-inspiron # Dual-camera simulation on Inspiron 3521
make bench              # Benchmark dev_pt profile (1-cam & 2-cam FPS)
make bench-inspiron     # Benchmark Inspiron 3521 profile (30 frames)
make run                # Run live camera pipeline with display
make run-headless       # Run live camera pipeline without display
make pack               # Pack repository context with Repomix (XML)
make pack-md            # Pack repository context with Repomix (Markdown)
make clean              # Remove build artifacts & test outputs
```

---

## What changed vs. "export scripts only"

| Before | Now |
|--------|-----|
| Export scripts only | Runtime package (`ppe/`) with backend abstraction |
| Single-image inference | Dual-camera threaded pipeline |
| No perf measurement | `bench.py` with 1-cam and 2-cam FPS + pass/fail verdict |
| Hardcoded model path | `configs/hardware.yaml` named profiles |
| No migration story | `docs/migration.md` + profile-only changes |
| No cheap-hardware guidance | `docs/hardware.md` with expected FPS per box |
