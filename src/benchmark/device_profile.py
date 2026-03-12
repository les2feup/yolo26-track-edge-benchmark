"""
Device profile loader for edge benchmarking.

Reads a YAML profile from edge/profiles/ and exposes a typed DeviceProfile
dataclass. The active profile is selected via the DEVICE_PROFILE environment
variable (path to YAML) or falls back to auto-detection based on the current
platform. Notebooks pass the profile path explicitly.
"""

from __future__ import annotations

import os
import platform
import socket
from dataclasses import dataclass, field
from pathlib import Path

# PyYAML is not in the venv's requirements; fall back to stdlib json loader
# for environments where pyyaml is unavailable (Jetson JetPack minimal images).
try:
    import yaml as _yaml
    def _load_yaml(path: Path) -> dict:
        with open(path) as f:
            return _yaml.safe_load(f)
except ImportError:
    import json as _json
    def _load_yaml(path: Path) -> dict:  # type: ignore[misc]
        raise RuntimeError(
            "PyYAML not installed. Install it with: pip install pyyaml\n"
            f"Or convert {path} to JSON and rename to .json."
        )

_PROFILES_DIR = Path(__file__).parents[2] / "edge" / "profiles"


@dataclass
class DeviceProfile:
    device_id: str
    device_label: str
    os: str
    torch_device: str
    backend: str              # cpu | cuda | tensorrt | tensorrt_hq | hailo | qnn | ncnn
    model_variants: list[str]
    model_format: str         # pt | engine | hef | onnx | tflite | ncnn_model
    resolutions: list[int]
    warmup_frames: int
    result_tag: str

    # Derived helpers (not in YAML)
    profile_path: Path = field(default=Path("."), repr=False)


def load_profile(path: str | Path | None = None) -> DeviceProfile:
    """
    Load a device profile YAML.

    Resolution order:
    1. Explicit path argument
    2. DEVICE_PROFILE environment variable
    3. Auto-detect from hostname/platform (best-effort, logs a warning)
    4. Desktop fallback (no profile file — returns a desktop config)
    """
    if path is None:
        path = os.environ.get("DEVICE_PROFILE")

    if path is not None:
        resolved = Path(path)
        if not resolved.is_absolute():
            resolved = _PROFILES_DIR / resolved
        profile = _from_yaml(resolved)
        apply_device_workarounds(profile)
        return profile

    # Auto-detect: try to match hostname to a known profile
    hostname = socket.gethostname().lower()
    candidates = {
        "rpi5-b-hailo": "rpi5_hailo.yaml",
        "rpi5-b-hailo": "rpi5_cpu.yaml",
        "rpi4":         "rpi4.yaml",
        "jetson-nano":  "jetson_nano.yaml",
        "les2-seed-studio-j10": "jetson_nano.yaml",
        "uno-q":        "arduino_uno_q.yaml",
        "arduinoq":     "arduino_uno_q.yaml",
    }
    for fragment, yaml_name in candidates.items():
        if fragment in hostname:
            print(f"[device_profile] auto-detected '{yaml_name}' from hostname '{hostname}'")
            profile = _from_yaml(_PROFILES_DIR / yaml_name)
            apply_device_workarounds(profile)
            return profile

    # Desktop fallback — not an edge profile, used for local development
    print("[device_profile] no profile detected — using desktop fallback")
    return _desktop_fallback()


def _from_yaml(path: Path) -> DeviceProfile:
    if not path.exists():
        raise FileNotFoundError(
            f"Device profile not found: {path}\n"
            f"Available profiles: {[p.name for p in _PROFILES_DIR.glob('*.yaml')]}"
        )
    data = _load_yaml(path)
    return DeviceProfile(
        device_id=data["device_id"],
        device_label=data["device_label"],
        os=data["os"],
        torch_device=data["torch_device"],
        backend=data["backend"],
        model_variants=data["model_variants"],
        model_format=data["model_format"],
        resolutions=data["resolutions"],
        warmup_frames=data["warmup_frames"],
        result_tag=data["result_tag"],
        profile_path=path,
    )


# Backends with fixed input shapes baked in at compile time.
# Model paths encode the resolution in the filename (no resolution = 640px default).
_BAKED_RESOLUTION_BACKENDS = {"hailo", "tensorrt", "tensorrt_hq", "qnn", "ncnn"}


def baked_imgsz(model_path: str) -> int:
    """Derive the compile-time resolution from a model filename or folder name.

    Works for any backend that bakes resolution into the path:
        yolo26n.engine           → 640   (no resolution = canonical 640px)
        yolo26n_576.hef          → 576
        yolo26s_512.engine       → 512
        yolo26n_576_hq.hef       → 576   (quality tag stripped)
        yolo26n_640_ncnn_model   → 640   (_ncnn_model folder suffix stripped)
    """
    # Strip file extension if present; for directories the path has no extension
    stem = model_path.rsplit(".", 1)[0] if "." in Path(model_path).name else model_path
    # Strip known trailing tags before resolution parsing
    for tag in ("_ncnn_model", "_hq", "_lq", "_qnn"):
        if stem.endswith(tag):
            stem = stem[: -len(tag)]
            break
    parts = stem.rsplit("_", 1)
    return int(parts[-1]) if len(parts) == 2 and parts[-1].isdigit() else 640


def iter_model_imgsz(profile: DeviceProfile):
    """Yield (model_path, imgsz) for every benchmark condition.

    For backends with baked-in resolutions (hailo, tensorrt), each model variant
    encodes exactly one resolution — iterate model_variants and derive imgsz
    from the filename.

    For dynamic backends (cpu, cuda with .pt files), iterate the full
    model_variants × resolutions grid.
    """
    if profile.backend in _BAKED_RESOLUTION_BACKENDS:
        for mp in profile.model_variants:
            yield mp, baked_imgsz(mp)
    else:
        for mp in profile.model_variants:
            for imgsz in profile.resolutions:
                yield mp, imgsz


def resolve_model_path(model_name: str) -> Path:
    """Resolve a bare model filename to an absolute path under models/."""
    _models_dir = Path(__file__).parents[2] / "models"
    p = Path(model_name)
    return p if p.is_absolute() else _models_dir / model_name


def try_load_model(model_name: str, device: str) -> tuple:
    """
    Attempt to load a PyTorch YOLO model, returning (model, error_str).

    For .hef models use run_sequence_hailo() from hailo_runner instead —
    HEF files are not loadable via ultralytics.

    Returns (model_instance, None) on success.
    Returns (None, error_message) on any failure so callers can log and
    continue without crashing the benchmark loop.
    """
    resolved = resolve_model_path(model_name)

    if resolved.suffix == ".hef":
        return None, (
            f"{model_name} is a HEF file — use run_sequence_hailo() "
            "from benchmark.hailo_runner instead of try_load_model()."
        )

    # TRT HQ engines (surgered, 6 raw Conv outputs) use a dedicated runner
    # with custom post-processing — not loadable via ultralytics.
    if resolved.suffix == ".engine" and "_hq" in resolved.stem:
        return None, (
            f"{model_name} is a TRT HQ engine — use run_sequence_trt() "
            "from benchmark.trt_runner instead of try_load_model()."
        )

    # QNN ONNX models (surgered, 6 raw Conv outputs) use ONNX Runtime with
    # QNNExecutionProvider — not loadable via ultralytics.
    if resolved.suffix == ".onnx" and "_qnn" in resolved.stem:
        return None, (
            f"{model_name} is a QNN ONNX model — use run_sequence_qnn() "
            "from benchmark.qnn_runner instead of try_load_model()."
        )

    try:
        from ultralytics import YOLO
        # NCNN models are directories ending in _ncnn_model. Ultralytics requires
        # task="detect" explicitly because metadata.yaml is stripped from the folder
        # to reduce transfer size, so autobackend cannot infer the task.
        # NCNN runs CPU-only — skip .to(device).
        if resolved.is_dir() and resolved.name.endswith("_ncnn_model"):
            model = YOLO(str(resolved), task="detect")
            return model, None
        model = YOLO(str(resolved))
        # TensorRT engines manage their own device — skip .to() for .engine files
        if resolved.suffix != ".engine":
            model = model.to(device)
        return model, None
    except FileNotFoundError:
        return None, f"model file not found: {model_name}"
    except MemoryError as exc:
        return None, f"OOM loading {model_name}: {exc}"
    except RuntimeError as exc:
        return None, f"RuntimeError loading {model_name}: {exc}"
    except Exception as exc:  # noqa: BLE001
        return None, f"unexpected error loading {model_name}: {type(exc).__name__}: {exc}"


def apply_device_workarounds(profile: DeviceProfile) -> None:
    """Apply device-specific environment workarounds before importing frameworks.

    Jetson Nano (JetPack 4.6.x): preload libgomp to prevent ultralytics segfault.
    The system OpenMP library must be loaded before NumPy/torch pull in their own
    copy.  Setting LD_PRELOAD via os.environ works because ultralytics triggers
    the libgomp dlopen() lazily — the dynamic linker honours the variable at
    dlopen time, not only at process start.

    Must be called before any ultralytics/torch import.  No-op for non-Jetson profiles.
    """
    if profile.device_id in ("jetson_nano", "jetson_nano_trt"):
        _libgomp = "/usr/lib/aarch64-linux-gnu/libgomp.so.1"
        existing = os.environ.get("LD_PRELOAD", "")
        if _libgomp not in existing:
            os.environ["LD_PRELOAD"] = (
                f"{existing}:{_libgomp}" if existing else _libgomp
            )


def apply_thread_pinning(profile: DeviceProfile) -> None:
    """Pin CPU/BLAS thread counts for deterministic power measurements.

    Only active for the RPi 4 profile (quad-core Cortex-A72, CPU-only).
    Must be called before any torch/cv2 workload begins; env vars are
    set before importing torch so the BLAS backends pick them up.
    No-op for all other profiles.
    """
    if profile.device_id not in ("rpi4", "arduino_uno_q", "arduino_uno_q_qnn"):
        return

    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    import torch
    import cv2

    torch.set_num_threads(4)
    # set_num_interop_threads(1) combined with set_num_threads(4) silently
    # corrupts inference on Qualcomm torch 2.0.0.post4 (QRB2210 / Cortex-A53).
    # Each call works individually but the combination produces zero detections.
    # Skip interop pinning for Arduino UNO Q; RPi4's standard torch is unaffected.
    if profile.device_id not in ("arduino_uno_q", "arduino_uno_q_qnn"):
        torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)


def _desktop_fallback() -> DeviceProfile:
    """Desktop profile used when no edge profile is active."""
    import torch
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    return DeviceProfile(
        device_id="desktop",
        device_label=f"Desktop ({platform.node()})",
        os=platform.system().lower(),
        torch_device=device,
        backend="cuda" if "cuda" in device else "cpu",
        model_variants=["yolo26n.pt", "yolo26s.pt", "yolo26m.pt", "yolo26l.pt", "yolo26x.pt"],
        model_format="pt",
        resolutions=[640, 576, 512, 448, 384, 320],
        warmup_frames=30,
        result_tag="desktop",
        profile_path=Path("."),
    )
