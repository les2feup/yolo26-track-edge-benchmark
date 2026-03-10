"""
TensorRT HQ engine inference wrapper for Jetson Nano.

Loads a .engine file compiled from the surgered ONNX (6 raw Conv outputs)
and runs synchronous frame-by-frame inference via the TensorRT Python API
and PyCUDA. No ultralytics dependency — the engine is self-contained.

Requirements (on the Jetson Nano, JetPack 4.6.x):
    tensorrt    — cp38 wheel: tensorrt-8.2.0.6-cp38-none-linux_aarch64.whl
    pycuda      — pip install pycuda (builds against system CUDA 10.2)

The wrapper pre-allocates host and device buffers at init time and reuses
them across frames to minimise allocation overhead.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

try:
    import tensorrt as trt
except ImportError:
    raise ImportError(
        "tensorrt Python bindings not found. On Jetson Nano (JetPack 4.6.x):\n"
        "  pip install tensorrt-8.2.0.6-cp38-none-linux_aarch64.whl\n"
        "Wheel location: /usr/lib/python3.8/dist-packages/ or download from NVIDIA."
    )

try:
    import pycuda.driver as cuda
    import pycuda.autoinit  # noqa: F401 — initialises CUDA context on import
except ImportError:
    raise ImportError(
        "pycuda not found. Install with: pip install pycuda\n"
        "Requires system CUDA toolkit (nvcc) — ships with JetPack 4.6.x."
    )


_TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# TensorRT dtype → numpy dtype mapping
_TRT_TO_NP = {
    trt.DataType.FLOAT: np.float32,
    trt.DataType.HALF:  np.float16,
    trt.DataType.INT32: np.int32,
    trt.DataType.INT8:  np.int8,
}


class TrtInfer:
    """
    Synchronous TensorRT inference for one HQ engine (6 raw Conv outputs).

    Opens the engine once, allocates host/device buffers for the single input
    and 6 outputs, and holds the execution context for the object's lifetime.
    Repeated infer() calls reuse all allocations.

    Input:  BGR uint8 frame (any size) — resized and normalised internally.
    Output: dict mapping binding name → float32 numpy array in NCHW layout.
    """

    def __init__(self, engine_path: str | Path):
        engine_path = str(engine_path)

        # Deserialise the TRT engine from file
        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(_TRT_LOGGER)
            self._engine = runtime.deserialize_cuda_engine(f.read())

        if self._engine is None:
            raise RuntimeError(f"Failed to deserialise TRT engine: {engine_path}")

        self._context = self._engine.create_execution_context()

        # Catalogue bindings: separate input from outputs, pre-allocate buffers
        self._input_name: str = ""
        self._input_shape: tuple = ()
        self._input_dtype = np.float32

        self._output_names: list[str] = []
        self._host_buffers: dict[str, np.ndarray] = {}
        self._device_allocs: dict[str, cuda.DeviceAllocation] = {}
        self._bindings: list[int] = []

        for i in range(self._engine.num_bindings):
            name  = self._engine.get_binding_name(i)
            shape = tuple(self._engine.get_binding_shape(i))
            dtype = _TRT_TO_NP.get(self._engine.get_binding_dtype(i), np.float32)

            # Allocate host (pinned) and device memory
            size  = int(np.prod(shape)) * np.dtype(dtype).itemsize
            host  = cuda.pagelocked_empty(int(np.prod(shape)), dtype)
            dev   = cuda.mem_alloc(size)

            self._host_buffers[name] = host
            self._device_allocs[name] = dev
            self._bindings.append(int(dev))

            if self._engine.binding_is_input(i):
                self._input_name  = name
                self._input_shape = shape   # (1, 3, H, W)
                self._input_dtype = dtype
            else:
                self._output_names.append(name)

        # Persistent CUDA stream for async H2D / inference / D2H overlap
        self._stream = cuda.Stream()

        # Expose input resolution for callers
        # Input shape is (1, 3, H, W) — extract spatial dims
        self.input_h: int = self._input_shape[2]
        self.input_w: int = self._input_shape[3]

    # ------------------------------------------------------------------
    def infer(self, frame_bgr: np.ndarray) -> dict[str, np.ndarray]:
        """
        Run one BGR frame through the TRT engine.

        Preprocessing: resize → RGB → float32 [0,1] → NCHW → contiguous.
        Returns dict mapping output binding name → float32 ndarray (NCHW).
        """
        # Preprocess: resize, BGR→RGB, normalise, HWC→CHW, add batch dim
        resized = cv2.resize(frame_bgr, (self.input_w, self.input_h))
        rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        blob    = rgb.astype(np.float32) / 255.0
        blob    = blob.transpose(2, 0, 1)           # HWC → CHW
        blob    = np.expand_dims(blob, axis=0)       # (1, 3, H, W)
        blob    = np.ascontiguousarray(blob)

        # Copy input to host buffer, then H2D transfer
        np.copyto(self._host_buffers[self._input_name],
                  blob.ravel().astype(self._input_dtype))
        cuda.memcpy_htod_async(
            self._device_allocs[self._input_name],
            self._host_buffers[self._input_name],
            self._stream,
        )

        # Execute inference
        self._context.execute_async_v2(
            bindings=self._bindings,
            stream_handle=self._stream.handle,
        )

        # D2H transfer for all outputs
        for name in self._output_names:
            cuda.memcpy_dtoh_async(
                self._host_buffers[name],
                self._device_allocs[name],
                self._stream,
            )

        self._stream.synchronize()

        # Reshape flat buffers to NCHW and return as float32 copies
        results = {}
        for name in self._output_names:
            idx   = self._engine.get_binding_index(name)
            shape = tuple(self._engine.get_binding_shape(idx))
            arr   = self._host_buffers[name].reshape(shape).astype(np.float32)
            results[name] = arr.copy()

        return results

    # ------------------------------------------------------------------
    def close(self) -> None:
        """Release CUDA resources."""
        del self._context
        del self._engine

    def __enter__(self) -> "TrtInfer":
        return self

    def __exit__(self, *args) -> None:
        self.close()
