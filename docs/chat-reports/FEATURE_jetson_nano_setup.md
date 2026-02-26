# Jetson Nano — Experiment Setup

Target: NVIDIA Jetson Nano (JetPack 4.x), Maxwell GPU, 4 GB LPDDR4 shared memory.

---

## Why the Jetson Nano requires its own setup path

JetPack 4.x ships a fixed software stack that cannot be updated independently:

| Component | JetPack 4.6.x |
|---|---|
| CUDA | 10.2 |
| cuDNN | 8.0 |
| TensorRT | 7.x (4.6.1) or 8.x (4.6.2+) |
| Python (system) | 3.6 |
| L4T base | Ubuntu 18.04 |

Two constraints follow from this:

1. **TensorRT engines are not portable.** They are compiled for a specific CUDA compute capability and TRT version. The `.engine` files must be built on the Nano itself — not on a desktop and copied over.
2. **Standard pip packages may be incompatible.** PyTorch for JetPack 4.x is distributed by NVIDIA, not PyPI. Installing `pip install torch` will silently pull a CPU-only ARM wheel.

---

## Step 1 — Flash and first boot

Use NVIDIA SDK Manager on a host Ubuntu 20.04/22.04 machine to flash JetPack 4.6.1 (recommended — ships TRT 7.1 and is the most stable 4.x release for the nano).

After boot, expand the root filesystem if using an SD card:
```bash
sudo /usr/bin/jetson_clocks          # max out CPU/GPU clocks
sudo nvpmodel -m 0                   # MAXN power mode (10W)
```

Verify the CUDA stack is working:
```bash
nvcc --version                       # should print CUDA 10.2
python3 -c "import tensorrt; print(tensorrt.__version__)"
```

---

## Step 2 — Python 3.8 virtual environment

JetPack 4.x ships Python 3.6 as system Python. PyTorch wheels for JetPack 4.x target Python 3.8.

```bash
sudo apt-get update
sudo apt-get install -y python3.8 python3.8-venv python3.8-dev python3-pip

python3.8 -m venv ~/bench-venv
source ~/bench-venv/bin/activate
pip install --upgrade pip setuptools wheel
```

---

## Step 3 — PyTorch for JetPack 4.x (NVIDIA wheel)

The PyTorch wheel for JetPack 4.x must come from NVIDIA's index, not PyPI.

```bash
# torch 1.11.0 — tested against JetPack 4.6.x / CUDA 10.2 / Python 3.8
pip install \
  "https://developer.download.nvidia.com/compute/redist/jp/v461/pytorch/torch-1.11.0a0+17540c5-cp38-cp38-linux_aarch64.whl"
```

If that URL is no longer reachable, find the current wheel at:
`https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048`

Verify:
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# expected: 1.11.0a0+...  True
```

---

## Step 4 — torchvision (matching version)

```bash
sudo apt-get install -y libjpeg-dev zlib1g-dev libpython3-dev libopenblas-dev libavcodec-dev libavformat-dev libswscale-dev

git clone --branch v0.12.0 https://github.com/pytorch/vision torchvision
cd torchvision
export BUILD_VERSION=0.12.0
python setup.py install
cd .. && rm -rf torchvision
```

---

## Step 5 — Project dependencies

```bash
source ~/bench-venv/bin/activate
cd ~
git clone <your-repo-url> yolo26-track-edge-benchmark
cd yolo26-track-edge-benchmark

# OpenCV: use the system-built version with CUDA support from JetPack.
# The pip opencv-python wheel is CPU-only and will conflict.
# Link the system cv2 into the venv instead.
CV2_SO=$(find /usr -name "cv2*.so" 2>/dev/null | head -1)
VENV_SITE=$(python -c "import site; print(site.getsitepackages()[0])")
ln -sf "$CV2_SO" "$VENV_SITE/cv2.so"

# Install everything except opencv (already linked above)
pip install \
  "ultralytics>=8.3.0" \
  "lap>=0.5.12" \
  "motmetrics>=1.4.0" \
  "pandas>=2.0" \
  "matplotlib>=3.5" \
  "psutil>=6.0" \
  "pyyaml" \
  jupyterlab \
  ipykernel

# Editable install of the benchmark package
pip install -e .
```

---

## Step 6 — Export TensorRT engines on the Nano

Engines must be built on the device that will run them. For each model variant:

```bash
source ~/bench-venv/bin/activate
cd ~/yolo26-track-edge-benchmark

python - <<'EOF'
from ultralytics import YOLO
import os

variants = ["yolo26n.pt", "yolo26s.pt", "yolo26m.pt", "yolo26l.pt", "yolo26x.pt"]

for v in variants:
    print(f"\n── Exporting {v} ──")
    try:
        YOLO(v).export(
            format="engine",
            imgsz=640,
            device=0,
            half=True,       # FP16 — required on Maxwell (no INT8 calibration needed)
            simplify=True,
        )
        print(f"OK: {v.replace('.pt', '.engine')}")
    except Exception as exc:
        print(f"FAILED {v}: {exc}")
EOF
```

Expected results on the Nano (4 GB shared RAM):

| Model | Engine size | Outcome |
|---|---|---|
| yolo26n | ~8 MB | OK |
| yolo26s | ~22 MB | OK |
| yolo26m | ~52 MB | likely OK |
| yolo26l | ~90 MB | may OOM at runtime |
| yolo26x | ~155 MB | expected OOM |

The benchmark runner will skip any variant that fails to load — no manual intervention needed.

---

## Step 7 — Transfer MOT17 data

The Nano has a slow SD card I/O. Use `rsync` from the development machine:

```bash
# From dev machine — adjust IP/hostname as needed
rsync -avz --progress \
  data/MOT17/train/ \
  nano@jetson.local:~/yolo26-track-edge-benchmark/data/MOT17/train/
```

Transfer size: ~5 GB for the three sequences (MOT17-09, MOT17-02, MOT17-04).

---

## Step 8 — Run the notebook

```bash
source ~/bench-venv/bin/activate
cd ~/yolo26-track-edge-benchmark

DEVICE_PROFILE=jetson_nano.yaml jupyter lab \
  --ip=0.0.0.0 --no-browser --port=8888
```

On the dev machine, open the URL printed in the terminal (includes the token). Open [notebooks/01_experiment1_profiling.ipynb](../../notebooks/01_experiment1_profiling.ipynb).

Cell 1 will print:
```
Device  : Jetson Nano (JetPack 4.x)
Backend : tensorrt  |  torch device: cuda:0
Models  : ['yolo26n.engine', 'yolo26s.engine', ...]
Tag     : jetson_nano
```

Run all cells. Results land in `results/raw/` with filenames suffixed `_jetson_nano.csv`.

---

## Step 9 — Retrieve results

```bash
# From dev machine
rsync -avz nano@jetson.local:~/yolo26-track-edge-benchmark/results/raw/ results/raw/
```

Then run `notebooks/03_results_figures.ipynb` on the desktop to include the Jetson results in the cross-device comparison figures.

---

## Known issues

**`libgomp` conflicts on export**: Ultralytics TensorRT export on JetPack 4.x can crash with a `libgomp` symbol error. Fix:
```bash
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1
```
Add this to your `.bashrc` or prefix the export command with it.

**`motmetrics` + NumPy 2.x**: Not an issue on the Nano — NumPy from the NVIDIA PyTorch wheel is 1.x. The manual `_iou_distance()` workaround in `metrics.py` remains in place regardless.

**JupyterLab slow to start**: First cold start on the Nano takes 60–90 seconds. Subsequent starts are faster. Be patient.

**swap**: The Nano shares 4 GB between CPU and GPU. Creating a swap file reduces OOM kills during large model loads:
```bash
sudo fallocate -l 4G /var/swapfile
sudo chmod 600 /var/swapfile
sudo mkswap /var/swapfile
sudo swapon /var/swapfile
echo '/var/swapfile swap swap defaults 0 0' | sudo tee -a /etc/fstab
```
