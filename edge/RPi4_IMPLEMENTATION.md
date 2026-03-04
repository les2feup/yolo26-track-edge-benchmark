# Raspberry Pi 4 Setup: YOLO Edge Tracking Benchmark

This guide provides the exact steps to configure a Python environment for YOLO inference and MOT tracking on a Raspberry Pi 4 (or similar older ARM64 devices).

## ⚠️ The "Illegal Instruction" Problem
If you attempt to install modern machine learning libraries (like PyTorch > 2.1 or NumPy 2.x) using standard `pip` wheels on Python 3.13, you will likely encounter an `Illegal instruction` error, causing the Python process (and your Jupyter kernel) to crash instantly.



**Why this happens:** Modern `aarch64` Python wheels are increasingly compiled for the **ARMv8.2-A** instruction set (found in the Raspberry Pi 5 / Cortex-A76), which includes advanced NEON dot-product instructions. The Raspberry Pi 4 uses the older **Cortex-A72 (ARMv8.0-A)**, which lacks hardware support for these instructions. 

**The Solution:** We use **Miniforge (Conda)** to create a Python 3.11 environment and explicitly pin PyTorch and NumPy to older versions compiled for broader ARM compatibility.

---

## Installation Steps

### 1. Install Miniforge (ARM64)
Miniforge provides access to `conda-forge`, which hosts highly compatible binaries for ARM architecture.

```bash
# Download the Miniforge installer for ARM64 (aarch64)
wget [https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh](https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh)

# Run the installer silently
bash Miniforge3-Linux-aarch64.sh -b -p $HOME/miniforge3

# Initialize Conda for your shell (you may need to restart your terminal after this)
source $HOME/miniforge3/bin/activate
conda init

```

### 2. Create the Environment & Install Core ML Packages

We use `mamba` (included with Miniforge) to create the environment and install the heavy C++/math libraries.

_Crucially, we pin PyTorch < 2.1.0 and NumPy < 2.0.0 to avoid the architecture mismatch._

```bash
# Create the environment with Python 3.11
mamba create -n yolo-edge python=3.11 -y

# Activate the environment
conda activate yolo-edge

# Install the hardware-compatible core stack via conda-forge
mamba install "pytorch<2.1.0" "torchvision<0.16.0" "numpy<2.0.0" jupyterlab ipykernel opencv scipy pandas matplotlib psutil -c conda-forge -y
```

### 3. Verify the Core Installation

Before proceeding, verify that the core libraries load without crashing the CPU:

```bash
python -c "import numpy; print(f'NumPy OK: {numpy.__version__}')"
python -c "import torch; print(f'Torch OK: {torch.__version__}')"
```

_(If both print successfully without an `Illegal instruction` error, you are safe to proceed)._

### 4. Install Remaining Requirements

Now, use `pip` to install the pure-Python packages and specific tools from `requirements.txt`. Because the heavy ML packages are already installed via Conda, `pip` will safely skip them.

```bash
# Ensure you are in the root directory of the benchmark repository
pip install -r requirements.txt
pip install -e .  # Install the benchmark package itself
```

### 5. Register the Jupyter Kernel

Link this Conda environment to Jupyter so that notebooks can utilize it.

```bash
python -m ipykernel install --user --name yolo-edge --display-name "Python (YOLO Edge Pi4)"
```

## Running the Benchmark

Enable the 4 CPU cores for maximum performance:

```bash
import os
import torch
import cv2

# 1. Force OpenMP and OpenBLAS to use 4 cores (RPi 4 has 4 cores)
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

# 2. Tell PyTorch explicitly to parallelize intra-op and inter-op tasks
torch.set_num_threads(4)
torch.set_num_interop_threads(4)

# 3. Allow OpenCV to use multiple threads for image processing
cv2.setNumThreads(4)

print(f"PyTorch threads set to: {torch.get_num_threads()}")
```

To start the Jupyter Lab server with the correct hardware profile loaded:

```bash
# Ensure the environment is active
conda activate yolo-edge

# Launch Jupyter Lab with the device profile variable
DEVICE_PROFILE=$(pwd)/edge/profiles/rpi4.yaml jupyter lab --no-browser --port=8889
```

Then, open the provided notebooks in the `edge/notebooks/` directory to run the YOLO Edge Tracking Benchmark on your Raspberry Pi 4!