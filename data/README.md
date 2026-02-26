# Dataset Download Instructions

The only dataset required is MOT17 (sequences MOT17-02, MOT17-04, MOT17-09).

To download the dataset, please follow these steps:

```bash
# Download
curl -L https://motchallenge.net/data/MOT17.zip -o MOT17.zip

# Unzip
unzip MOT17.zip >> /dev/null
rm MOT17.zip

# Verify the three sequences you need are present
ls MOT17/train/ | grep -E "MOT17-02|MOT17-04|MOT17-09"
``` 

You should see six directories (one per public detector set — DPM, FRCNN, SDP — but the GT annotations are identical across them):

```text
MOT17-02-DPM  MOT17-02-FRCNN  MOT17-02-SDP
MOT17-04-DPM  MOT17-04-FRCNN  MOT17-04-SDP
MOT17-09-DPM  MOT17-09-FRCNN  MOT17-09-SDP
```

To validate the GT format and write the py-motmetrics integration before downloading 5.5 GB:

```bash
curl -L https://motchallenge.net/data/MOT17Labels.zip -o MOT17Labels.zip
unzip MOT17Labels.zip -d MOT17Labels
```

The GT file for each sequence is at:
```
MOT17/train/MOT17-04-FRCNN/gt/gt.txt
```

---

### GT format reminder (from your paper's checklist, Step 4)

Each line in `gt.txt`:
```
<frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <class>, <visibility>
```

Filter for evaluation:

- `class == 1` (pedestrian only)
- `visibility >= 0.25`
- `conf == 1` (exclude ignore regions where conf == 0)

You only need one detector variant's folder for the GT (they share the same `gt/gt.txt`), so `MOT17-04-FRCNN` is conventional. The frames (`img1/`) are identical across DPM/FRCNN/SDP — no need to download all three variants since we are providing our own detections via YOLO26.