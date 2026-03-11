# Jetson Nano (Seeed reComputer J10) — Benchmark Setup

Target: NVIDIA Jetson Nano 4GB (Seeed Studio reComputer J10), JetPack 4.6.x, Maxwell GPU (128 CUDA cores, `sm_53`).

This guide walks through the exact steps to set up the YOLO26 tracking benchmark on a Jetson Nano 4GB (Seeed reComputer J10) running JetPack 4.6.x. The Nano's GPU and CUDA 10.2 environment require a custom setup path, including manual installation of compatible PyTorch wheels.

**Inference backend:** PyTorch CUDA with `.pt` weights (FP32 precision, dynamic input shapes). TensorRT engines produce incorrect detection counts on TensorRT 8.2 / Maxwell — see [Known issues](#known-issues).

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

1. **ultralytics requires Python >= 3.8.** The system Python 3.6 is too old. We install Python 3.8 via `apt` and use pre-built PyTorch wheels hosted by the ultralytics project (torch 1.11 + torchvision 0.12, compiled for cp38 + CUDA 10.2 + aarch64).
2. **TensorRT engines produce incorrect results.** Both FP16 and FP32 engines compiled via `trtexec` on TensorRT 8.2 / Maxwell (sm_53) produce ~1/3 of the expected detections compared to the identical `.pt` model. This was verified experimentally: `.pt` FP32 gives 6 detections (matching the desktop baseline) while `.engine` FP32 gives only 2. The root cause is in the ONNX→TensorRT graph compilation, not precision.

**Result:** Inference runs via PyTorch CUDA on `.pt` weights at FP32 precision. No model compilation step — the same `.pt` file runs at any resolution.

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
pip install -r edge/requirements-jetson.txt
pip install -e .
```

The `requirements-jetson.txt` pins all packages to versions compatible with Python 3.8, torch 1.11, and NumPy 1.x.

---

## Step 11 — Quick inference test

Verify that `.pt` models run on the Maxwell GPU at multiple resolutions:

```bash
python -c "
from ultralytics import YOLO
model = YOLO('models/yolo26n.pt')
r = model.predict('data/MOT17/train/MOT17-09-SDP/img1/000001.jpg', imgsz=640, device=0, verbose=False)
print(f'640px: {len(r[0].boxes)} det, confs: {[round(c,3) for c in r[0].boxes.conf.tolist()]}')
r = model.predict('data/MOT17/train/MOT17-09-SDP/img1/000001.jpg', imgsz=576, device=0, verbose=False)
print(f'576px: {len(r[0].boxes)} det, confs: {[round(c,3) for c in r[0].boxes.conf.tolist()]}')
"
```

Both resolutions should produce detections. Expected: ~6 detections at 640px matching the desktop baseline. If CUDA is not available, revisit Step 8.

---

## Step 12 — Run the benchmark

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

DEVICE_PROFILE=$(pwd)/edge/profiles/jetson_nano_trt.yaml jupyter lab \
  --ip=0.0.0.0 --no-browser --port=8888
```

Connect from the host browser using the URL printed in the terminal.

---

## Step 13 — Retrieve results

From the **development machine**:

```bash
NANO_IP=<NANO_IP>
SD=/media/les2/SD-128-J10
rsync -avz les2@${NANO_IP}:${SD}/yolo26-track-edge-benchmark/results/raw/ results/raw/
```

Results are tagged with `_jetson_nano` suffix and can be processed by notebooks 02–03 on the desktop.

---

## Known issues

**TensorRT engines produce incorrect detection counts**: Both FP16 and FP32 `.engine` files compiled via `trtexec` on TensorRT 8.2 / Maxwell (sm_53) produce ~1/3 of the expected detections. Verified experimentally: `.pt` FP32 → 6 detections (matching desktop), `.engine` FP32 → 2 detections on the same frame. The root cause is in the ONNX→TensorRT graph compilation on TensorRT 8.2, not precision. Additionally, ultralytics 8.4.19 exports ONNX graphs with a `Mod` op that TensorRT 8.2 cannot parse. The ultralytics-hosted `tensorrt-8.2.0.6-cp38` wheel is available for `import tensorrt` but is not used in this benchmark due to the detection loss.

**`onnxruntime` version conflict**: ultralytics auto-installs `onnxruntime` 1.19.x (CPU-only) from PyPI, overwriting the GPU-enabled 1.8.0 wheel. Always install onnxruntime-gpu 1.8.0 *after* ultralytics and its transitive dependencies (Step 11). Verify with `python -c "import onnxruntime; print(onnxruntime.__version__, onnxruntime.get_available_providers())"`.

**`libgomp` conflicts**: ultralytics or ONNX operations can crash with a `libgomp` symbol error. Fix:
```bash
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1
```
This is added to the venv activation script in Step 7. If you created the venv before this step existed, add it manually to `.venv/bin/activate`.

**`jetson_clocks` resets on reboot**: Run `sudo nvpmodel -m 0 && sudo jetson_clocks` after each power cycle, or enable the systemd service (`sudo systemctl enable jetson_clocks`). Verify with `jtop` — status should show `Jetson Clocks: active`.

**ultralytics version pinning**: All devices pin `ultralytics==8.4.19` for cross-device reproducibility. NMS and post-processing logic differs between versions. ultralytics 8.4.19 is compatible with Python 3.8.

**PyTorch version differences across devices**: The Jetson Nano runs torch 1.11 (the only available cp38/CUDA 10.2/aarch64 wheel), while the RPi 4 uses torch 2.0 (Miniforge, ARMv8.0-A compatible) and the RPi 5 / desktop may use torch 2.x+. Verified experimentally: `.pt` FP32 produces identical detections (6 det, matching confidences to 3 decimal places) across torch 1.11 and torch 2.x, so this is not a practical concern for this model.

**`motmetrics` + NumPy 2.x**: Not an issue — NumPy is pinned to 1.x in `requirements-jetson.txt`.

**JupyterLab slow to start**: First cold start takes 60–90s on the Nano. Subsequent starts are faster.

**Shared memory budget**: The 4 GB is shared between CPU and GPU. Larger models (yolo26m and above) may fail to load at runtime. The benchmark runner logs these failures and skips them automatically.

**SD card I/O**: Dataset reads from the SD card are slow. If inference latency appears unreasonably high on the first sequence, it may be I/O-bound. The timing in `runner.py` measures only `model.track()`, not frame reads, so this should not affect results.

