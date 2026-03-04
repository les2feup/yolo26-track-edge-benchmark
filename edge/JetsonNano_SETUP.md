# Jetson Nano (Seeed reComputer J10) — Benchmark Setup

Target: NVIDIA Jetson Nano 4GB (Seeed Studio reComputer J10), JetPack 4.6.x, Maxwell GPU (128 CUDA cores, `sm_53`).

This guide walks through the exact steps to set up the YOLO26 tracking benchmark on a Jetson Nano 4GB (Seeed reComputer J10) running JetPack 4.6.x. The Nano's GPU and CUDA 10.2 environment require a custom setup path, including manual installation of compatible PyTorch wheels.

**Inference backend:** TensorRT FP16 via `.engine` files compiled on-device. The system `tensorrt.so` (Python 3.6 only) is bypassed by installing the ultralytics-hosted `tensorrt-8.2.0.6` wheel for Python 3.8.

**Precision note:** Inference runs at FP16 precision via TensorRT on the Maxwell GPU. `.engine` files have a fixed input shape baked in at compile time — each model×resolution pair produces a separate engine. The export script (`edge/export_tensorrt.py`) handles the `.pt` → `.onnx` → `.engine` pipeline.

All commands below run **on the Nano via SSH** unless otherwise noted.

---

## Why the Jetson Nano requires its own setup path

JetPack 4.6.x ships a fixed software stack:

| Component | Version |
|---|---|
| L4T / Ubuntu | R32.7.6 / 18.04 (bionic) |
| CUDA | 10.2 |
| cuDNN | 8.0 |
| TensorRT | 8.2.1.8 |
| Python (system) | 3.6 |

Two constraints follow:

1. **The system `tensorrt.so` rejects Python 3.8.** JetPack 4.6.x ships TensorRT Python bindings compiled for Python 3.6 only (pybind11 version check). **Workaround:** install the ultralytics-hosted `tensorrt-8.2.0.6-cp38` wheel, which provides compatible bindings for Python 3.8 (see Step 8b).
2. **ultralytics requires Python >= 3.8.** The system Python 3.6 is too old. We install Python 3.8 via `apt` and use pre-built PyTorch wheels hosted by the ultralytics project (torch 1.11 + torchvision 0.12, compiled for cp38 + CUDA 10.2 + aarch64).

**Result:** Inference runs via TensorRT FP16 on `.engine` files compiled on-device. The export pipeline (`.pt` → `.onnx` → `.engine`) is handled by `edge/export_tensorrt.py`.

---

## Step 1 — Verify JetPack and CUDA

```bash
cat /etc/nv_tegra_release            # confirm L4T R32.x (JetPack 4.6.x)
/usr/local/cuda/bin/nvcc --version   # CUDA 10.2 expected
```

If `nvcc` is not on your PATH, add CUDA to your shell profile:
```bash
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

---

## Step 2 — Performance mode

Lock CPU and GPU clocks to maximum for consistent benchmark results:

```bash
sudo nvpmodel -m 0                   # MAXN power mode (10W, all 4 cores)
sudo jetson_clocks                   # max out CPU/GPU/EMC clocks
```

**These settings do not persist across reboots.** Re-run both commands after every power cycle. Verify with `jtop` (see Step 3) — the status line should show `Jetson Clocks: active`.

To make jetson_clocks run automatically at boot:
```bash
sudo systemctl enable jetson_clocks
```

---

## Step 3 — Install jtop (recommended)

`jtop` provides a real-time dashboard for GPU utilisation, clock frequencies, memory, and temperature. Essential for verifying performance mode and diagnosing OOM during benchmarks.

```bash
sudo pip3 install jetson-stats
sudo systemctl restart jtop.service
# Then run: jtop
```

Verify that jtop shows `Jetson Clocks: active` and `NV Power Mode: MAXN`.

---

## Step 4 — Expand swap (recommended)

The Nano shares 4 GB between CPU and GPU. Extra swap reduces OOM kills during model loading. The eMMC (16 GB) is typically too full for a swap file — use the SD card instead:

```bash
# Create a 4 GB swap file on the SD card (adjust mount point as needed)
SD_MOUNT=/media/les2/SD-128-J10
sudo fallocate -l 4G "$SD_MOUNT/swapfile"
sudo chmod 600 "$SD_MOUNT/swapfile"
sudo mkswap "$SD_MOUNT/swapfile"
sudo swapon "$SD_MOUNT/swapfile"
echo "$SD_MOUNT/swapfile swap swap defaults 0 0" | sudo tee -a /etc/fstab
```

Verify: `free -h` should show ~6 GB swap total (2 GB existing + 4 GB new).

---

## Step 5 — Install Python 3.8 and system dependencies

```bash
sudo apt update
sudo apt install -y python3.8 python3.8-venv python3.8-dev

# System libraries required by the ultralytics-hosted PyTorch wheel
sudo apt install -y libopenblas-dev libomp-dev libopenmpi-dev
```

Verify: `python3.8 --version` should print `Python 3.8.x`.

---

## Step 6 — Transfer the repository to the SD card

The eMMC has very limited free space. The repository, venv, models, and dataset all live on the SD card.

From the **development machine**:

```bash
NANO_IP=<NANO_IP>
SD=/media/les2/SD-128-J10

# Sync the repo (exclude heavy/host-only directories)
rsync -avz --exclude='.venv' --exclude='__pycache__' --exclude='.git' --exclude='data' --exclude='edge/hailo-wheels/*.whl' --exclude='results' --exclude='models' \
  ~/Developer/yolo26-track-edge-benchmark/ \
  les2@${NANO_IP}:${SD}/yolo26-track-edge-benchmark/

# Transfer the .pt model weights
rsync -avz models/*.pt les2@${NANO_IP}:${SD}/yolo26-track-edge-benchmark/models/

# Transfer MOT17 dataset (~5 GB for three sequences)
rsync -avz --progress \
  data/MOT17/train/ \
  les2@${NANO_IP}:${SD}/yolo26-track-edge-benchmark/data/MOT17/train/
```

_Instead, you can download the dataset following the instruction in [README.md](../data/README.md)._

---

## Step 7 — Create virtual environment (on SD card)

```bash
SD=/media/les2/SD-128-J10
cd $SD/yolo26-track-edge-benchmark

python3.8 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
```

Add the `libgomp` workaround to the venv activation script so it applies automatically:
```bash
echo 'export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1' >> .venv/bin/activate
source .venv/bin/activate
```

---

## Step 8 — Install PyTorch and torchvision

These wheels are hosted by the ultralytics project, built for Python 3.8 + CUDA 10.2 + aarch64 (the exact combination that JetPack 4.6.x needs):

```bash
pip install \
  "https://github.com/ultralytics/assets/releases/download/v0.0.0/torch-1.11.0a0+gitbc2c6ed-cp38-cp38-linux_aarch64.whl" \
  "https://github.com/ultralytics/assets/releases/download/v0.0.0/torchvision-0.12.0a0+9b5a3fe-cp38-cp38-linux_aarch64.whl"
```

Verify:
```bash
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
# Expected: PyTorch 1.11.0a0+..., CUDA: True
```

If `torch.cuda.is_available()` returns `False`, something is wrong — do NOT proceed.

---

## Step 8b — Install TensorRT Python bindings

The system `tensorrt.so` only works with Python 3.6. Install the ultralytics-hosted wheel that provides Python 3.8-compatible bindings for the same TensorRT 8.2 version shipped with JetPack 4.6.x.

**Important:** If a previous setup attempt symlinked the system tensorrt into the venv (like the OpenCV symlink in Step 9), remove it first — the system bindings are Python 3.6 only:

```bash
# Remove stale symlink to system Python 3.6 tensorrt (if present)
SITE=$(python -c "import site; print(site.getsitepackages()[0])")
rm -f "$SITE/tensorrt"   # symlink, not a directory

pip install "https://github.com/ultralytics/assets/releases/download/v0.0.0/tensorrt-8.2.0.6-cp38-none-linux_aarch64.whl"
```

Verify:
```bash
python -c "import tensorrt; print(f'TensorRT {tensorrt.__version__}')"
# Expected: TensorRT 8.2.0.6
```

This unblocks `import tensorrt` so ultralytics can load `.engine` files for inference.

---

## Step 9 — Symlink system OpenCV

JetPack's system OpenCV is built with CUDA support. The pip `opencv-python` package is CPU-only and will conflict. Symlink the system build into the venv instead:

```bash
CV2_SO=$(find /usr -name "cv2*.so" 2>/dev/null | head -1)
VENV_SITE=$(python -c "import site; print(site.getsitepackages()[0])")
ln -sf "$CV2_SO" "$VENV_SITE/cv2.so"

python -c "import cv2; print(f'OpenCV {cv2.__version__}')"
```

---

## Step 10 — Install project dependencies

```bash
pip install onnx onnxslim
pip install -r edge/requirements-jetson.txt
pip install -e .
```

The `requirements-jetson.txt` pins all packages to versions compatible with Python 3.8, torch 1.11, and NumPy 1.x.

---

## Step 11 — Install ONNX runtime (aarch64 wheel)

The ONNX runtime GPU wheel must be installed **after** all other dependencies. ultralytics auto-installs a CPU-only `onnxruntime` (1.19.x) from PyPI which overwrites the GPU version. PyPI has no `onnxruntime-gpu` wheel for aarch64/cp38, so we install the ultralytics-hosted version manually as the final step.

```bash
pip uninstall -y onnxruntime onnxruntime-gpu 2>/dev/null
pip install "https://github.com/ultralytics/assets/releases/download/v0.0.0/onnxruntime_gpu-1.8.0-cp38-cp38-linux_aarch64.whl"
```

Verify:
```bash
python -c "import onnxruntime; print(f'onnxruntime {onnxruntime.__version__}'); print(onnxruntime.get_available_providers())"
# Expected: onnxruntime 1.8.0, ['CUDAExecutionProvider', 'CPUExecutionProvider']
```

If you see version 1.19.x or only `CPUExecutionProvider`, repeat the uninstall + install above.

---

## Step 12 — Export TensorRT engines

Compile `.engine` files on-device. Each model×resolution pair produces a separate engine with a fixed input shape baked in at compile time. The export pipeline is `.pt` → `.onnx` → `.engine` (FP16):

```bash
# Export all variants (yolo26n through yolo26x) at 640px and 576px
python edge/export_tensorrt.py

# Or export a single variant for a quick test
python edge/export_tensorrt.py --model yolo26n
```

Expected output in `models/`:
```
yolo26n.engine        yolo26n_576.engine
yolo26s.engine        yolo26s_576.engine
...
```

Larger models (yolo26m and above) may OOM during export — the script logs failures and continues. A summary is written to `edge/export_results_jetson.csv`.

---

## Step 13 — Quick inference test

Verify that `.engine` models load and run on the Maxwell GPU:

```bash
python -c "
from ultralytics import YOLO
model = YOLO('models/yolo26n.engine')
r = model.predict('data/MOT17/train/MOT17-09-SDP/img1/000001.jpg', device=0, verbose=False)
print(f'TensorRT engine: OK — {len(r[0].boxes)} detections')
"
```

If `import tensorrt` fails, revisit Step 8b. If CUDA is not available, revisit Step 8.

---

## Step 14 — Run the benchmark

Verify performance mode is active before starting:
```bash
sudo jetson_clocks --show   # all clocks should be at maximum
```

### Option A: Script-based (recommended for SSH)

```bash
DEVICE_PROFILE=jetson_nano.yaml python -c "
from benchmark.device_profile import load_profile
p = load_profile()
print(f'Device: {p.device_label}')
print(f'Backend: {p.backend}, torch_device: {p.torch_device}')
print(f'Models: {p.model_variants}')
print(f'Resolutions: {p.resolutions}')
"
```

Then open notebook 01 or run the benchmark loop programmatically.

### Option B: Jupyter Lab

```bash
DEVICE_PROFILE=$(pwd)/edge/profiles/jetson_nano.yaml jupyter lab \
  --ip=0.0.0.0 --no-browser --port=8888
```

Connect from the host browser using the URL printed in the terminal.

---

## Step 15 — Retrieve results

From the **development machine**:

```bash
NANO_IP=<NANO_IP>
SD=/media/les2/SD-128-J10
rsync -avz les2@${NANO_IP}:${SD}/yolo26-track-edge-benchmark/results/raw/ results/raw/
```

Results are tagged with `_jetson_nano` suffix and can be processed by notebooks 02–03 on the desktop.

---

## Known issues

**System `tensorrt.so` vs pip wheel**: JetPack 4.6.x ships `tensorrt.so` compiled for Python 3.6 only. Do NOT try to `import tensorrt` without first installing the ultralytics-hosted `tensorrt-8.2.0.6-cp38` wheel (Step 8b). The PyPI `tensorrt` package is x86_64/CUDA 13 only — do NOT install it either.

**FP16 vs FP32 precision**: The Jetson Nano runs TensorRT FP16 inference, while the RPi 4 and RPi 5 CPU backends run FP32. Desktop CUDA also runs FP32 (via `.pt` weights). FP16 reduces latency and memory but introduces small numerical differences in near-threshold detections compared to FP32 backends.

**`onnxruntime` version conflict**: ultralytics auto-installs `onnxruntime` 1.19.x (CPU-only) from PyPI, overwriting the GPU-enabled 1.8.0 wheel. Always install onnxruntime-gpu 1.8.0 *after* ultralytics and its transitive dependencies (Step 11). Verify with `python -c "import onnxruntime; print(onnxruntime.__version__, onnxruntime.get_available_providers())"`.

**`libgomp` conflicts**: ultralytics or ONNX operations can crash with a `libgomp` symbol error. Fix:
```bash
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1
```
This is added to the venv activation script in Step 7. If you created the venv before this step existed, add it manually to `.venv/bin/activate`.

**`jetson_clocks` resets on reboot**: Run `sudo nvpmodel -m 0 && sudo jetson_clocks` after each power cycle, or enable the systemd service (`sudo systemctl enable jetson_clocks`). Verify with `jtop` — status should show `Jetson Clocks: active`.

**ultralytics version pinning**: All devices must use the same ultralytics minor version (8.3.x) for comparable detection counts. NMS and post-processing logic changed between 8.3.x and 8.4.x, producing measurably different detection rates on identical inputs and weights. The Jetson is limited to `<8.4.0` (Python 3.8 compatibility), so all devices pin to `ultralytics>=8.3.0,<8.4.0`. If a device was set up with 8.4.x, downgrade with `pip install 'ultralytics>=8.3.0,<8.4.0'` and re-run.

**PyTorch version differences across devices**: The Jetson Nano runs torch 1.11 (the only available cp38/CUDA 10.2/aarch64 wheel), while the RPi 4 uses torch 2.0 (Miniforge, ARMv8.0-A compatible) and the RPi 5 / desktop may use torch 2.x+. Different torch versions can produce small numerical differences in floating-point operations, affecting detection counts at the margins (near-threshold boxes may be kept or dropped differently). This is an inherent limitation of cross-device benchmarking — each platform's hardware constraints dictate the available PyTorch wheel. The ultralytics pin controls the higher-level post-processing path; the torch-level numerical variation is documented but cannot be eliminated.

**`motmetrics` + NumPy 2.x**: Not an issue — NumPy is pinned to 1.x in `requirements-jetson.txt`.

**JupyterLab slow to start**: First cold start takes 60–90s on the Nano. Subsequent starts are faster.

**Shared memory budget**: The 4 GB is shared between CPU and GPU. Larger models (yolo26m and above) may fail to load at runtime. The benchmark runner logs these failures and skips them automatically.

**SD card I/O**: Dataset reads from the SD card are slow. If inference latency appears unreasonably high on the first sequence, it may be I/O-bound. The timing in `runner.py` measures only `model.track()`, not frame reads, so this should not affect results.

