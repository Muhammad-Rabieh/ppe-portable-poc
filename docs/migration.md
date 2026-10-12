# Migrating between hardware

The whole point of this repo: **moving boxes is a config change, not a code change.**

---

## What to copy to the new machine

| File / Directory | Why |
|------------------|-----|
| `ppe.pt` | The master model — always carry it |
| `exports/` | ONNX + ncnn builds (or regenerate on new box) |
| `configs/hardware.yaml` | Named profiles for all target hardware |
| `configs/cameras.yaml` | Camera sources (usually unchanged between boxes) |
| `notes.md` | Records the Ultralytics version used to create exports |

---

## Steps on the new box

```bash
git clone <your-repo-url>
cd ppe-portability-poc

# 1. Setup environment (creates .venv with CPU-only wheels, zero CUDA bloat)
make setup

# 2. Download weights (or drop your trained ppe.pt in root)
make download

# 3. Export to target format (ONNX on CPU boxes, ncnn on Vulkan, etc.)
make export

# 4. Run visual validation check (opens detection preview window & test report)
make validate

# 5. Run the benchmark for that box's profile:
python scripts/bench.py --profile n100_cpu          # cheap CPU box
python scripts/bench.py --profile inspiron3521_cpu  # staging / old CPU box
python scripts/bench.py --profile rx580_vulkan      # AMD GPU
python scripts/bench.py --profile nvidia_tensorrt   # NVIDIA GPU

# 6. If PASS, start the cameras:
python scripts/run_cameras.py --profile n100_cpu
```

---

## Adding a new hardware type

**Only touch `configs/hardware.yaml`.**  No code changes.

Example — adding an RTX 3060 box:

```yaml
  my_rtx3060:
    format: engine        # export once: yolo export model=ppe.pt format=engine
    device: 0
    imgsz: 640
    half: true
    infer_every: 1
    conf: 0.4
```

Then:

```bash
python scripts/bench.py --profile my_rtx3060
python scripts/run_cameras.py --profile my_rtx3060
```

---

## Export compatibility table

| Format | `yolo export` flag | Works on |
|--------|-------------------|----------|
| `pt` | (none — native) | Any machine with PyTorch + Ultralytics |
| `onnx` | `format=onnx` | CPU, CUDA (onnxruntime-gpu) |
| `ncnn` | `format=ncnn` | ARM, Vulkan, CPU |
| `engine` | `format=engine` | NVIDIA only (TensorRT version must match) |
| `openvino` | `format=openvino` | Intel CPU/iGPU |

> [!WARNING]
> TensorRT `.engine` files are **not portable** across GPU generations or TensorRT versions.
> Always re-export `engine` files on the target machine from `ppe.pt`.

---

## Rules that keep migration cheap

1. **Never edit `ppe/`** for hardware reasons — edit the profile instead.
2. **Never share `.engine` files** across machines — always re-export from `ppe.pt`.
3. **Re-run `bench.py`** on every new box before trusting it in production.
4. **Keep `notes.md` updated** with the Ultralytics version used for each export.
