# YOLO26 Track Edge Benchmark

Benchmarks YOLO26 + ByteTrack tracking accuracy and throughput across model
variants (n/s/m/l/x) and input resolutions on edge hardware. Experiments run
from a single set of Jupyter notebooks, with per-device behaviour controlled
by YAML profiles in `edge/profiles/`.

## Repository layout

```
data/           MOT17 sequences (not tracked — download separately)
edge/
  profiles/     YAML device profiles
  export_hailo.py       ONNX → HEF compiler (Docker, offline)
  export_tensorrt.py    .pt → .onnx → .engine (on-device, reference only)
  collect_calib.py      Calibration image sampler
  JetsonNano_SETUP.md   Jetson Nano 14-step setup guide
  RPi4_IMPLEMENTATION.md  RPi 4 Miniforge setup guide
models/         Model weights (.pt / .hef / .engine — not tracked)
notebooks/      00 setup · 01 profiling · 02 resolution · 03 figures · 04 power
results/        CSV benchmark outputs (tracked) · figures (not tracked)
src/benchmark/  Python package (editable install)
```

---

## Prerequisites (all devices)

### 1 — Dataset

Download MOT17 and place the sequences under `data/`:

```
data/MOT17/train/MOT17-02/
data/MOT17/train/MOT17-04/
data/MOT17/train/MOT17-09/
```

### 2 — Models

Download the YOLO26 weights and place them in `models/`:

```
models/yolo26n.pt   models/yolo26s.pt   models/yolo26m.pt
models/yolo26l.pt   models/yolo26x.pt
```

### 3 — Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

> Requires Python ≥ 3.12 on the development machine. Edge devices use
> device-specific environments described below.

---

## Running the notebooks

All experiment notebooks read a single environment variable to select the
active device profile:

```bash
DEVICE_PROFILE=edge/profiles/<profile>.yaml jupyter lab
```

If `DEVICE_PROFILE` is unset, the notebooks fall back to a desktop profile
(full resolution set, CUDA if available, `.pt` weights).

Notebook order:

| Notebook | Purpose |
|---|---|
| `00_setup_verify.ipynb` | Verify environment, model load, and data paths |
| `01_experiment1_profiling.ipynb` | Latency and memory across model variants |
| `02_experiment2_resolution.ipynb` | Tracking metrics vs. input resolution |
| `03_results_figures.ipynb` | Figures and tables (desktop / analysis machine) |

Run `00` first on every new device to confirm the environment before
starting the timed experiments.

---

## Accessing JupyterLab on a remote device

JupyterLab runs on the device but is accessed from your development machine
via **SSH port forwarding**. This is the most reliable approach — no firewall
changes, no IP configuration.

**Step 1 — On the device**, start JupyterLab from the project root:

```bash
cd ~/yolo26-track-edge-benchmark
DEVICE_PROFILE=$(pwd)/edge/profiles/<profile>.yaml \
  jupyter lab --no-browser --port=8888
```

Using `$(pwd)/...` produces an absolute path, which avoids resolution
errors when JupyterLab changes the working directory internally.

Copy the token from the output (looks like `?token=abc123...`).

**Step 2 — On your development machine**, open an SSH tunnel in a new terminal:

```bash
ssh -L 8888:localhost:8888 pi@<device-ip>
```

**Step 3** — Open `http://localhost:8888` in your browser and paste the token.

The tunnel stays open as long as the SSH session is active. To keep the
notebook running after you close your laptop, start JupyterLab inside `tmux`
or `screen` on the device before opening the tunnel.

```bash
# On the device — start a persistent session
tmux new -s bench
DEVICE_PROFILE=$(pwd)/edge/profiles/<profile>.yaml jupyter lab --no-browser --port=8888
# Detach with Ctrl-B D; reattach later with: tmux attach -t bench
```

---

## Cross-device reproducibility

All devices pin `ultralytics>=8.3.0,<8.4.0` — NMS/post-processing changed in
8.4.x, producing different detection counts on identical inputs.

PyTorch versions and precision vary across devices: the Jetson Nano runs
TensorRT FP16 (torch 1.11, cp38/CUDA 10.2), CPU-only devices (RPi 4/5) run
FP32 (torch 2.0+), and the desktop runs FP32 via CUDA or CPU. Small numerical
differences in near-threshold detections are expected and documented as a
known limitation.

---

## Device profiles

### Desktop / development machine

No profile needed — the notebooks auto-detect CUDA and use `.pt` weights.

```bash
jupyter lab
```

---

### Raspberry Pi 5 + Hailo-8L M.2 Hat

**Profile:** `edge/profiles/rpi5_hailo.yaml`
**Backend:** Hailo-8L (13 TOPS) via HailoRT
**Models:** `.hef` (compiled offline, see [HEF export](#hef-export-for-hailo-8l))

#### Setup on the Pi

```bash
# Install system dependencies
sudo apt update
sudo apt install -y python3-venv git

# Clone repo and create environment
git clone <repo-url> && cd yolo26-track-edge-benchmark
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

# Install HailoRT Python bindings (requires the .deb from Hailo Developer Zone)
# https://developer.hailo.ai → Software Downloads → HailoRT → Raspberry Pi
sudo dpkg -i hailort_<version>_arm64.deb
pip install hailort-<version>-cp311-cp311-linux_aarch64.whl

# Copy HEF models from development machine
rsync -avP user@devmachine:path/to/models/*.hef models/
```

#### Run

```bash
source .venv/bin/activate
DEVICE_PROFILE=$(pwd)/edge/profiles/rpi5_hailo.yaml jupyter lab --no-browser --port=8888
```

---

### Raspberry Pi 5 — CPU only

**Profile:** `edge/profiles/rpi5_cpu.yaml`
**Backend:** PyTorch CPU (Cortex-A76, 8 GB LPDDR4X)
**Models:** `.pt`

#### Setup on the Pi

```bash
git clone <repo-url> && cd yolo26-track-edge-benchmark
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

# Copy model weights
rsync -avP user@devmachine:path/to/models/*.pt models/
```

#### Run

```bash
source .venv/bin/activate
DEVICE_PROFILE=$(pwd)/edge/profiles/rpi5_cpu.yaml jupyter lab --no-browser --port=8888
```

> **Note:** yolo26m and above will be slow (~0.5–2 FPS). The notebooks skip
> variants that exceed available memory and log the failure.

---

### Raspberry Pi 4

**Profile:** `edge/profiles/rpi4.yaml`
**Backend:** PyTorch CPU (Cortex-A72, 4 GB LPDDR4)
**Models:** `.pt`
**Setup guide:** [`edge/RPi4_IMPLEMENTATION.md`](edge/RPi4_IMPLEMENTATION.md)

The RPi 4 uses Cortex-A72 (ARMv8.0-A), which lacks the ARMv8.2-A instructions
in modern PyPI wheels. The setup guide uses Miniforge (Conda) with pinned
PyTorch and NumPy versions compiled for broader ARM compatibility.

yolo26m and above are expected to fail on the 4 GB RAM ceiling — the runner
reports the error and continues.

```bash
DEVICE_PROFILE=$(pwd)/edge/profiles/rpi4.yaml jupyter lab --no-browser --port=8888
```

---

### Jetson Nano (JetPack 4.x)

**Profile:** `edge/profiles/jetson_nano.yaml`
**Backend:** TensorRT FP16 (Maxwell GPU, 4 GB shared)
**Models:** `.engine` (compiled on-device via `edge/export_tensorrt.py`)
**Setup guide:** [`edge/JetsonNano_SETUP.md`](edge/JetsonNano_SETUP.md)

JetPack 4.6.x ships Python 3.6 and TensorRT 8.2. The system `tensorrt.so` only
works with Python 3.6, but the ultralytics-hosted `tensorrt-8.2.0.6` wheel
provides Python 3.8-compatible bindings. `.engine` files are compiled on-device
(`edge/export_tensorrt.py`) with FP16 precision and fixed input shapes.
See the setup guide for the full 15-step installation process.

#### Run

```bash
source .venv/bin/activate
DEVICE_PROFILE=$(pwd)/edge/profiles/jetson_nano.yaml jupyter lab \
  --ip=0.0.0.0 --no-browser --port=8888
```

> **Note:** yolo26m and above may exceed the 4 GB shared memory budget
> during export or inference. The runner and export script log OOM failures
> and continue with the remaining variants.

---

### Arduino Uno Q (QRB2210)

**Profile:** `edge/profiles/arduino_uno_q.yaml`
**Backend:** PyTorch CPU (Cortex-A53, 2–4 GB LPDDR4)
**Models:** `.pt`

#### Setup on the board

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-dev git

git clone <repo-url> && cd yolo26-track-edge-benchmark
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

# Copy model weights from development machine
rsync -avP user@devmachine:path/to/models/*.pt models/
```

#### Run

```bash
source .venv/bin/activate
DEVICE_PROFILE=$(pwd)/edge/profiles/arduino_uno_q.yaml jupyter lab --no-browser --port=8888
```

> **Note:** yolo26n and yolo26s are expected to load; yolo26m and above will
> likely OOM on the 2 GB SKU. The runner skips gracefully and logs results for
> the variants that do load.

---

## HEF export for Hailo-8L

HEF compilation requires Ubuntu 22.04, Python 3.11, and the Hailo DFC 3.33
wheel. It runs in Docker — no native install needed.

### 1 — Place Hailo wheels

Download from [Hailo Developer Zone](https://developer.hailo.ai) and place in
`edge/hailo-wheels/`:

```
edge/hailo-wheels/hailo_dataflow_compiler-3.33.0-py3-none-linux_x86_64.whl
edge/hailo-wheels/hailort-4.23.0-cp311-cp311-linux_x86_64.whl
```

### 2 — Collect calibration images

```bash
source .venv/bin/activate
python edge/collect_calib.py --n 64
```

### 3 — Build the Docker image and run the export

```bash
docker build -f edge/Dockerfile.hailo-export -t hailo-export:3.33 .
docker compose -f edge/docker-compose.yml run --rm hailo-export
```

Output HEF files appear in `models/`. A provenance CSV is written to
`edge/export_results.csv`.

---

## Retrieving results

After running on a device, copy the result CSVs back to the development
machine:

```bash
rsync -avP pi@raspberrypi.local:~/yolo26-track-edge-benchmark/results/raw/ results/raw/
```

Result filenames include the `result_tag` from the device profile
(e.g. `MOT17-04_yolo26n.hef_640_rpi5_hailo.csv`), so outputs from
different devices can coexist in the same `results/raw/` directory without
collision.

Run `03_results_figures.ipynb` on the development machine to generate
combined figures across all devices.
