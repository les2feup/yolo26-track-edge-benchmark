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
  export_hailo.py   ONNX → HEF compiler (Docker, offline)
  collect_calib.py  Calibration image sampler
models/         Model weights (.pt / .hef / .engine — not tracked)
notebooks/      00 setup · 01 profiling · 02 resolution · 03 figures
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
DEVICE_PROFILE=edge/profiles/rpi5_hailo.yaml jupyter lab --no-browser --ip=0.0.0.0
```

Open the printed URL on your development machine.

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
DEVICE_PROFILE=edge/profiles/rpi5_cpu.yaml jupyter lab --no-browser --ip=0.0.0.0
```

> **Note:** yolo26m and above will be slow (~0.5–2 FPS). The notebooks skip
> variants that exceed available memory and log the failure.

---

### Raspberry Pi 4

**Profile:** `edge/profiles/rpi4.yaml`
**Backend:** PyTorch CPU (Cortex-A72, 4 GB LPDDR4)
**Models:** `.pt`

Same setup steps as RPi 5 CPU above. yolo26m and above are expected to fail
on the 4 GB RAM ceiling — the runner reports the error and continues.

```bash
DEVICE_PROFILE=edge/profiles/rpi4.yaml jupyter lab --no-browser --ip=0.0.0.0
```

---

### Jetson Nano (JetPack 4.x)

**Profile:** `edge/profiles/jetson_nano.yaml`
**Backend:** TensorRT (Maxwell GPU, 4 GB shared)
**Models:** `.engine` — **must be compiled on the Nano itself**

#### Setup on the Nano

```bash
# Enable max performance
sudo /usr/bin/jetson_clocks
sudo nvpmodel -m 0

# Python 3.8 venv (JetPack 4.x ships Python 3.6 as system default)
sudo apt-get install -y python3.8 python3.8-venv python3.8-dev
python3.8 -m venv ~/bench-venv && source ~/bench-venv/bin/activate
pip install --upgrade pip setuptools wheel

# PyTorch for JetPack 4.x (NVIDIA wheel — not on PyPI)
# Download from https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048
pip install torch-<version>-cp38-cp38-linux_aarch64.whl

# torchvision must be built from source to match the JetPack CUDA version
sudo apt-get install -y libjpeg-dev zlib1g-dev
pip install Cython
git clone --branch v0.15.2 https://github.com/pytorch/vision torchvision
cd torchvision && python setup.py install && cd ..

# Project dependencies (cv2 via system symlink on JetPack)
pip install ultralytics lap motmetrics pandas matplotlib psutil jupyterlab
ln -s /usr/lib/python3/dist-packages/cv2.cpython-*.so \
      ~/bench-venv/lib/python3.8/site-packages/cv2.so

git clone <repo-url> && cd yolo26-track-edge-benchmark
pip install -e .
```

#### Export TensorRT engines on the Nano

```python
# Run once on the Nano before the benchmark
from ultralytics import YOLO
for variant in ["yolo26n", "yolo26s", "yolo26m", "yolo26l", "yolo26x"]:
    YOLO(f"models/{variant}.pt").export(
        format="engine", imgsz=640, device=0, half=True
    )
```

#### Run

```bash
source ~/bench-venv/bin/activate
cd yolo26-track-edge-benchmark
DEVICE_PROFILE=edge/profiles/jetson_nano.yaml jupyter lab --no-browser --ip=0.0.0.0
```

> **Note:** yolo26m and above may exceed the 4 GB shared memory budget.
> The runner logs OOM failures and continues with the remaining variants.

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
DEVICE_PROFILE=edge/profiles/arduino_uno_q.yaml jupyter lab --no-browser --ip=0.0.0.0
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
