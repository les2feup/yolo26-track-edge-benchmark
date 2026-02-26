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
    backend: str              # cpu | cuda | tensorrt | hailo
    model_variants: list[str]
    model_format: str         # pt | engine | hef | tflite
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
        return _from_yaml(resolved)

    # Auto-detect: try to match hostname to a known profile
    hostname = socket.gethostname().lower()
    candidates = {
        "rpi5":         "rpi5_cpu.yaml",
        "raspberrypi5": "rpi5_cpu.yaml",
        "rpi4":         "rpi4.yaml",
        "raspberrypi":  "rpi4.yaml",
        "jetson":       "jetson_nano.yaml",
        "uno-q":        "arduino_uno_q.yaml",
        "arduinoq":     "arduino_uno_q.yaml",
    }
    for fragment, yaml_name in candidates.items():
        if fragment in hostname:
            print(f"[device_profile] auto-detected '{yaml_name}' from hostname '{hostname}'")
            return _from_yaml(_PROFILES_DIR / yaml_name)

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

    try:
        from ultralytics import YOLO
        model = YOLO(str(resolved))
        model.to(device)
        return model, None
    except FileNotFoundError:
        return None, f"model file not found: {model_name}"
    except MemoryError as exc:
        return None, f"OOM loading {model_name}: {exc}"
    except RuntimeError as exc:
        return None, f"RuntimeError loading {model_name}: {exc}"
    except Exception as exc:  # noqa: BLE001
        return None, f"unexpected error loading {model_name}: {type(exc).__name__}: {exc}"


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
