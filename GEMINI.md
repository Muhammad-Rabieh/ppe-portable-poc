# GEMINI.md — Agent & Context Guide for PPE Portable POC

This document serves as the persistent system instructions and context guide for Google Gemini (Gemini Code Assist, Gemini CLI, and Google DeepMind AI agents). It details project architecture, context packing via Repomix, documentation locations, and development workflows.

---

## 1. Project Overview

**PPE Portable POC** is an edge-optimized computer vision system designed for real-time Personal Protective Equipment (PPE) compliance monitoring across budget, heterogeneous edge devices (e.g., Intel N100 mini PCs, Orange Pi 5, Raspberry Pi 5, AMD RX 580 via Vulkan, NVIDIA Jetson).

### Core Architectural Principles
- **Decoupled Architecture**: Separation between trained model weights (`.pt`), hardware-optimized formats (`exports/`), and runtime pipeline code (`ppe/`).
- **Backend Abstraction**: All inference loads route through `ppe/backend.py` via `resolve_backend(profile, exports_dir)`. Model formats are never hardcoded into runtime scripts.
- **Declarative Hardware Profiles**: Hardware targets are defined in `configs/hardware.yaml`, enabling device migration without altering Python code.
- **Threaded Multi-Camera Streaming**: Decoupled ingestion (`ppe/stream.py`) and inference queue orchestration (`ppe/pipeline.py`) with configurable frame skipping (`infer_every`).

---

## 2. Using Repomix for AI Context Packing

The repository is configured with **Repomix** (`repomix.config.json`) to allow Gemini and other LLMs to ingest the entire codebase within a single, token-efficient context packet.

### Packing Commands

```bash
# Generate standard AI-friendly XML context (repomix-output.xml)
make pack
# or: npx --yes repomix

# Generate human/AI-readable Markdown context (repomix-output.md)
make pack-md
# or: npx --yes repomix --style markdown -o repomix-output.md

# Inspect token distribution per file
npx --yes repomix --token-count-tree

# Tree-sitter compressed packing (extracts signatures/classes, strips bodies to save ~70% tokens)
npx --yes repomix --compress
```

### When Using Repomix MCP Server
If running in an MCP-connected environment:
- Use `pack_codebase` targeting `/home/muhammad/PRoJeCTs/ppe-portable-poc`.
- Use `read_repomix_output` and `grep_repomix_output` to search and retrieve packed symbols.

### Exclusion Rules
Repomix automatically excludes:
- Large model binary weights (`*.pt`, `*.onnx`, `*.engine`, `*.rknn`, `*.mnn`, `*.bin`).
- Test imagery and videos (`validation/**`, `*.jpg`, `*.png`, `*.mp4`).
- Temporary runs and caches (`runs/**`, `__pycache__/**`).
Packing the codebase yields ~17k tokens, making it lightweight for large context windows.

---

## 3. Documentation Map

Consult the documentation files in `docs/` for detailed architectural rationale and hardware guidance:

| Document | Purpose & Contents |
|----------|-------------------|
| [`docs/portability-strategy.md`](file:///home/muhammad/PRoJeCTs/ppe-portable-poc/docs/portability-strategy.md) | Architectural mental model: decoupling `.pt`, `exports/`, and pipeline runtime. |
| [`docs/hardware.md`](file:///home/muhammad/PRoJeCTs/ppe-portable-poc/docs/hardware.md) | Comparison of budget edge devices, price-to-performance, and target FPS per camera. |
| [`docs/migration.md`](file:///home/muhammad/PRoJeCTs/ppe-portable-poc/docs/migration.md) | Step-by-step checklist and troubleshooting guide when onboarding a new edge box. |
| [`notes.md`](file:///home/muhammad/PRoJeCTs/ppe-portable-poc/notes.md) | Export logs, benchmark results per board, and project checklist. |
| [`README.md`](file:///home/muhammad/PRoJeCTs/ppe-portable-poc/README.md) | Top-level project summary and quickstart guide. |

---

## 4. Directory & Module Structure

```
ppe-portable-poc/
├── ppe/                          # Runtime package
│   ├── backend.py                # resolve_backend(): profile -> (model_path, predict_kwargs)
│   ├── detector.py               # PPEDetector: thread-safe inference wrapper around YOLO
│   ├── pipeline.py               # DualCameraPipeline: multi-camera threading & frame dropping
│   └── stream.py                 # CameraStream: OpenCV capture thread with automatic reconnection
├── configs/
│   ├── hardware.yaml             # Named hardware profiles (n100_cpu, rx580_vulkan, dev_pt, etc.)
│   └── cameras.yaml              # Camera URLs (RTSP/webcam), resolutions, and FPS targets
├── scripts/
│   ├── download_pretrained.py    # Downloads baseline YOLOv8n placeholder
│   ├── export.py                 # Exports ppe.pt -> ONNX and ncnn
│   ├── infer.py                  # Single-image inference test
│   ├── validate.py               # Validation script against sample image
│   ├── bench.py                  # Benchmark script (1-cam & 2-cam FPS with pass/fail)
│   └── run_cameras.py            # Main runner for multi-camera live pipeline
├── exports/                      # Target directory for exported model files
├── notebooks/
│   └── train_ppe_colab.ipynb     # Google Colab notebook for training custom PPE weights
├── repomix.config.json           # Repomix context packer configuration
├── Makefile                      # Standard CLI entry points
└── requirements-cpu.txt          # Python dependencies (CPU baseline)
```

---

## 5. Operational Commands

```bash
make setup              # Install dependencies into .venv (CPU PyTorch + OpenCV + ONNX Runtime)
make download           # Download placeholder ppe.pt (YOLOv8n)
make export             # Export ppe.pt to exports/ (ONNX)
make validate           # Validate inference visually (interactive window + test report)
make validate-inspiron  # Validate inference using inspiron3521_cpu profile
make bench              # Run performance benchmark using dev_pt profile
make bench-inspiron     # Run performance benchmark using inspiron3521_cpu profile
make run                # Run multi-camera pipeline with UI display
make run-headless       # Run multi-camera pipeline without display
make pack               # Repomix: Pack full repo to repomix-output.xml
make pack-md            # Repomix: Pack full repo to repomix-output.md
make clean              # Remove runtime build artifacts & repomix outputs
```

---

## 6. Guidelines & Conventions for AI Agents

1. **Adhere to `ppe/backend.py` abstraction**: Do not instantiate raw Ultralytics or engine loaders directly in runtime code. Route through `resolve_backend(profile, exports_dir)`.
2. **Configuration-driven hardware scaling**: Adjust hardware settings (such as `infer_every`, `imgsz`, `half`) in `configs/hardware.yaml`.
3. **No binary model commits**: Model weights (`.pt`, `.onnx`, `.engine`, etc.) must not be checked into Git.
4. **Preserve docstrings & type hints**: Keep all existing documentation and comments intact.
5. **Use Repomix for broad queries**: When needing broad context or cross-module analysis, leverage `repomix-output.xml` or Repomix tools.
