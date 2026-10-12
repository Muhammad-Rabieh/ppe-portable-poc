"""Backend abstraction — swap runtime without touching pipeline code."""
from pathlib import Path


def resolve_backend(profile: dict, exports_dir: Path):
    """
    Given a hardware profile like:
        {"format": "onnx", "device": "cpu", "half": False}
    or  {"format": "ncnn", "device": "vulkan", "half": True}
    return (model_path, predict_kwargs).

    Everything in the repo calls this. Nothing calls Ultralytics directly.
    """
    fmt = profile.get("format", "onnx").lower()
    device = profile.get("device", "cpu")
    half = bool(profile.get("half", False))
    imgsz = int(profile.get("imgsz", 416))

    if fmt == "pt":
        model_path = profile.get("pt_path", "ppe.pt")
    elif fmt == "onnx":
        model_path = exports_dir / "ppe.onnx"
    elif fmt == "ncnn":
        model_path = exports_dir / "ppe_ncnn_model"
    elif fmt == "engine":
        model_path = exports_dir / "ppe.engine"
    elif fmt == "openvino":
        model_path = exports_dir / "ppe_openvino_model"
    else:
        raise ValueError(f"Unknown format: {fmt!r}. "
                         f"Valid: pt, onnx, ncnn, engine, openvino")

    predict_kwargs = {
        "imgsz": imgsz,
        "device": device,
        "verbose": False,
    }
    if half:
        predict_kwargs["half"] = True
    return str(model_path), predict_kwargs
