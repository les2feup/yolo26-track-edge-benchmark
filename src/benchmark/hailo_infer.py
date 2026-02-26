"""
Hailo-8L inference wrapper using the HailoRT 4.17+ async API.

Loads a HEF file, runs frame-by-frame inference in synchronous-equivalent
mode (run_async + job.wait), and returns raw float32 output tensors.
No NMS or decoding is applied here — see hailo_postprocess.py.

hailo_platform is installed system-wide (python3-hailort apt package) and
made available inside the venv via a .pth file pointing to dist-packages.
"""

from __future__ import annotations

import threading
from pathlib import Path

import cv2
import numpy as np
from hailo_platform import (
    HEF,
    FormatType,
    HailoSchedulingAlgorithm,
    VDevice,
)


class HailoInfer:
    """
    Synchronous-equivalent Hailo-8L inference for one HEF model.

    Opens a VDevice once, configures the inference pipeline, and holds the
    configured context open for the lifetime of the object so that repeated
    calls to infer() share the same compiled pipeline without re-initialisation.

    Input:  BGR uint8 frame (any size) — resized internally to model input shape.
    Output: dict mapping output tensor name → float32 numpy array (no batch dim).
    """

    def __init__(self, hef_path: str | Path):
        params = VDevice.create_params()
        # ROUND_ROBIN required — NONE causes configure() segfault (HailoRT bug).
        params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
        self._vdevice = VDevice(params)

        hef_path = str(hef_path)
        self._hef = HEF(hef_path)
        self._infer_model = self._vdevice.create_infer_model(hef_path)
        self._infer_model.set_batch_size(1)

        # Pass raw RGB uint8 pixels; hardware applies quantisation constants.
        self._infer_model.input().set_format_type(FormatType.UINT8)

        # Request float32 outputs; HailoRT dequantises automatically.
        for info in self._hef.get_output_vstream_infos():
            self._infer_model.output(info.name).set_format_type(FormatType.FLOAT32)

        # (H, W, C) — no batch dimension in the buffer shape.
        self.input_shape: tuple = tuple(
            self._hef.get_input_vstream_infos()[0].shape
        )
        self._output_infos = self._hef.get_output_vstream_infos()

        # Pre-allocated output buffers reused across frames to avoid GC pressure.
        self._output_buffers: dict[str, np.ndarray] = {
            info.name: np.empty(
                self._infer_model.output(info.name).shape, dtype=np.float32
            )
            for info in self._output_infos
        }

        # Enter configure context once; holds pipeline state across all infer() calls.
        self._ctx = self._infer_model.configure()
        self._configured = self._ctx.__enter__()

    # ------------------------------------------------------------------
    def infer(self, frame_bgr: np.ndarray) -> dict[str, np.ndarray]:
        """
        Run one BGR frame through the Hailo-8L.

        Blocks until inference completes (synchronous-equivalent via job.wait).
        Returns a copy of each output buffer so the caller owns stable arrays
        even if infer() is called again before the results are consumed.
        """
        H, W, _ = self.input_shape
        frame_rgb = cv2.cvtColor(
            cv2.resize(frame_bgr, (W, H)), cv2.COLOR_BGR2RGB
        )  # (H, W, 3) uint8 RGB

        bindings = self._configured.create_bindings(
            output_buffers=self._output_buffers
        )
        bindings.input().set_buffer(frame_rgb)

        # Back-pressure gate: waits until the async queue has a free slot.
        self._configured.wait_for_async_ready(timeout_ms=10_000)

        done = threading.Event()
        error: list[Exception | None] = [None]

        def _cb(completion_info):
            if completion_info.exception:
                error[0] = completion_info.exception
            done.set()

        job = self._configured.run_async([bindings], _cb)
        job.wait(timeout_ms=10_000)

        if error[0]:
            raise RuntimeError(f"Hailo inference failed: {error[0]}")

        return {name: buf.copy() for name, buf in self._output_buffers.items()}

    # ------------------------------------------------------------------
    def close(self) -> None:
        self._ctx.__exit__(None, None, None)
        self._vdevice.release()

    def __enter__(self) -> "HailoInfer":
        return self

    def __exit__(self, *args) -> None:
        self.close()
