# NCNN Model Export

NCNN models are exported **on the x86 desktop** (PNNX has no ARM binary) and transferred to each ARM device.

## Export

```bash
source .venv/bin/activate
python edge/export/export_ncnn.py
```

Exports `n/s/m` variants at 640 and 576 px by default. Each output is a folder:

```
models/yolo26n_640_ncnn_model/
    model.ncnn.param   # graph (resolution baked in)
    model.ncnn.bin     # weights
    metadata.yaml      # stride, task, names — required by ultralytics autobackend
```

**The resolution is baked into `model.ncnn.param` at export time.** Passing a different `imgsz` at inference is a silent correctness error.

To export a single model or override resolutions:

```bash
python edge/export/export_ncnn.py --model yolo26l.pt --resolutions 640
```

## Transfer to device

```bash
rsync -av models/yolo26n_640_ncnn_model models/yolo26s_640_ncnn_model ... user@host:/path/to/models/
```

Sync only `*_ncnn_model/` folders — the `.pt` files are not needed on the device.

## Device requirements

```bash
pip install ncnn
```

No other special setup. NCNN runs on CPU via ARM NEON on all supported targets (RPi 4, RPi 5, Arduino Uno Q).
