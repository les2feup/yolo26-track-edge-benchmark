# Miniforge Setup: ARMv8.0-A Devices

## Architecture Compatibility

ARMv8.0-A cores (Cortex-A72, Cortex-A53) lack the NEON dot-product instructions required by modern `aarch64` wheels (compiled for ARMv8.2-A+). Running standard PyPI wheels for PyTorch > 2.1 or NumPy 2.x causes an immediate `Illegal instruction` crash.

**Fix:** Use Miniforge (conda-forge) with PyTorch < 2.1 and NumPy < 2.0, which ship broader ARM-compatible binaries.

---

## Installation

### 1. Install Miniforge (ARM64)

```bash
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh
bash Miniforge3-Linux-aarch64.sh -b -p $HOME/miniforge3
source $HOME/miniforge3/bin/activate
conda init
```

### 2. Create the Environment

```bash
mamba create -n yolo-edge python=3.11 -y
conda activate yolo-edge
mamba install "pytorch<2.1.0" "torchvision<0.16.0" "numpy<2.0.0" jupyterlab ipykernel opencv scipy pandas matplotlib psutil -c conda-forge -y
```

### 3. Install Remaining Requirements

```bash
pip install -r requirements.txt
pip install -e .
```

### 4. Register the Jupyter Kernel

```bash
python -m ipykernel install --user --name yolo-edge --display-name "Python (YOLO Edge)"
```

---

## Running

```bash
conda activate yolo-edge
DEVICE_PROFILE=$(pwd)/edge/profiles/<device>.yaml jupyter lab --no-browser --ip=0.0.0.0 --port=8888
```
