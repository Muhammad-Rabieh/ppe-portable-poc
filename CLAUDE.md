# CLAUDE.md — Agent & Context Guide for PPE Portable POC

This document serves as the persistent briefing document for AI assistants (Claude Code, Anthropic agents, and LLMs). It outlines project architecture, documentation locations, context packing via Repomix, and development conventions.

---

## 1. Project Overview

**PPE Portable POC** is a lightweight, modular proof-of-concept for detecting Personal Protective Equipment (hard hats, safety vests, etc.) across heterogeneous, low-cost edge devices (e.g., Intel N100 mini PCs, Orange Pi 5, Raspberry Pi 5, AMD RX 580 via Vulkan, NVIDIA Jetson).

### Core Architectural Principles
- **Decoupled Architecture**: Model weights (`.pt`), hardware-optimized formats (`exports/`), and the multi-stream pipeline runtime (`ppe/`) are strictly isolated.
- **Backend Abstraction**: All inference calls route through `ppe/backend.py` (`resolve_backend`), never through hardcoded model formats.
- **Configuration-Driven**: Adding or adjusting target hardware is done declaratively in `configs/hardware.yaml` without changing application code.
- **Dual-Camera Real-Time Pipeline**: Threaded capture (`ppe/stream.py`) with frame-skipping inference load management (`ppe/pipeline.py`).

---

## 2. Using Repomix for AI Context Packing

This repository is configured with **Repomix** (`repomix.config.json`) to quickly consolidate codebase context into a clean, token-efficient payload for LLMs.

### Packing the Repository

```bash
# Generate standard AI-friendly XML context (repomix-output.xml)
make pack
# or: npx --yes repomix

# Generate human/AI-readable Markdown context (repomix-output.md)
make pack-md
# or: npx --yes repomix --style markdown -o repomix-output.md

# Inspect file-by-file token counts
npx --yes repomix --token-count-tree

# Tree-sitter compressed packing (extracts signatures/classes, strips bodies to save ~70% tokens)
npx --yes repomix --compress
```

### When Using Repomix MCP Server
If running within an MCP-enabled agent environment:
- Use `pack_codebase` to pack `/home/muhammad/PRoJeCTs/ppe-portable-poc`.
- Use `read_repomix_output` or `grep_repomix_output` to search and retrieve packed symbols efficiently.

### Repomix Filter Rules
Configured in `repomix.config.json`:
- **Automatically excluded**: Model binaries (`*.pt`, `*.onnx`, `*.engine`, `*.rknn`, `*.mnn`, `*.bin`), sample images/videos (`validation/**`, `*.jpg`, `*.png`, `*.mp4`), cache directories, and `runs/**`.
- **Result**: Packing produces ~17k tokens, making the full project readily fit within standard context windows.

---

## 3. Documentation Map

Before modifying or investigating architectural decisions, consult the dedicated documentation in `docs/`:

| Document | Purpose & Contents |
|----------|-------------------|
| [`docs/portability-strategy.md`](file:///home/muhammad/PRoJeCTs/ppe-portable-poc/docs/portability-strategy.md) | The `.pt` vs `exports/` mental model; why and how models are ported. |
| [`docs/hardware.md`](file:///home/muhammad/PRoJeCTs/ppe-portable-poc/docs/hardware.md) | Matrix of target edge devices, prices, formats (ncnn, ONNX, TRT), and target FPS per camera. |
| [`docs/migration.md`](file:///home/muhammad/PRoJeCTs/ppe-portable-poc/docs/migration.md) | Checklist and troubleshooting steps for deploying to a new edge box. |
| [`notes.md`](file:///home/muhammad/PRoJeCTs/ppe-portable-poc/notes.md) | Running log of model exports, hardware benchmarks, and known tasks. |
| [`README.md`](file:///home/muhammad/PRoJeCTs/ppe-portable-poc/README.md) | Main project overview, quickstart instructions, and target table. |

---

## 4. Directory & Module Structure

```
ppe-portable-poc/
├── ppe/                          # Core runtime package
│   ├── backend.py                # resolve_backend(): maps profile -> model path & prediction args
│   ├── detector.py               # PPEDetector: thread-safe inference wrapper around YOLO
│   ├── pipeline.py               # DualCameraPipeline: thread coordination & frame dropping
│   └── stream.py                 # CameraStream: OpenCV capture thread with reconnect handling
├── configs/
│   ├── hardware.yaml             # Named hardware profiles (n100_cpu, rx580_vulkan, dev_pt, etc.)
│   └── cameras.yaml              # Camera URLs (RTSP/webcam), resolutions, and FPS targets
├── scripts/
│   ├── download_pretrained.py    # Downloads placeholder YOLOv8n baseline
│   ├── export.py                 # Exports ppe.pt -> ONNX and ncnn
│   ├── infer.py                  # Single-image inference test
│   ├── validate.py               # Validation script on test image
│   ├── bench.py                  # Benchmarking script (1-cam & 2-cam FPS with pass/fail)
│   └── run_cameras.py            # Real-time multi-camera runner
├── exports/                      # Target destination for exported models (.gitkeep)
├── notebooks/
│   └── train_ppe_colab.ipynb     # Google Colab notebook for training custom PPE weights
├── repomix.config.json           # Repomix context packer configuration
├── Makefile                      # Standard CLI entry points
└── requirements-cpu.txt          # Python dependencies (CPU baseline)
```

---

## 5. Standard Operational Commands

```bash
make setup              # Install dependencies into .venv (CPU PyTorch + OpenCV + ONNX Runtime)
make download           # Download placeholder ppe.pt (YOLOv8n)
make export             # Export ppe.pt to exports/ (ONNX)
make validate           # Validate inference visually (interactive window + test report)
make validate-inspiron  # Validate inference using inspiron3521_cpu profile
make validate-video     # Validate inference on video stream with live HUD
make validate-video-inspiron # Validate video on Inspiron 3521 profile
make validate-dual-video     # Validate dual cameras concurrently (side-by-side)
make validate-dual-video-inspiron # Validate dual cameras on Inspiron 3521
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

1. **Never bypass `ppe/backend.py`**: Pipeline and script code must not load Ultralytics or ONNX models directly. Always use `resolve_backend(profile, exports_dir)`.
2. **Configuration over code**: When adding support for a new hardware board or tuning FPS parameters (`infer_every`, `imgsz`, `half`), create a profile entry in `configs/hardware.yaml`.
3. **Keep binary files out of Git**: Model binaries (`.pt`, `.onnx`, `.engine`, `.rknn`, `.mnn`) belong in `exports/` or repo root, which are gitignored. Never stage binary models.
4. **Preserve docstrings & comments**: Maintain documentation integrity across all edits.
5. **Token Awareness**: When exploring the project context, consult `repomix-output.xml` or use Repomix commands instead of reading every individual directory iteratively.
