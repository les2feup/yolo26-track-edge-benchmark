# Paper 1 — IEEE Embedded Systems Letters (4 pages)

## Working Title

Characterizing YOLO26 Tracking-Mode Perception for Urban Crowd Monitoring on Edge Devices

---

## Scope and Contribution Statement

This letter characterizes the operating envelope of YOLO26 in integrated tracking mode for urban crowd monitoring applications on resource-constrained edge devices. The contribution is not a detection benchmark — YOLO26's detection accuracy on standard datasets is established by its authors [CIT] — but rather a systematic characterization of how perception quality relevant to downstream crowd analytics degrades across device classes and operating configurations. Where prior edge-deployment evaluations of YOLO variants report detection accuracy and throughput in isolation [CIT], this work evaluates perception through the lens of tracking stability: the signal property that determines whether downstream spatiotemporal analysis (density estimation, motion characterization) can operate reliably on the detector's output.

**What this paper does:**

1. Profiles YOLO26 tracking-mode inference across device classes and model variants, measuring both conventional metrics (throughput, latency) and tracking-specific metrics (identity switch rate, track fragmentation) that are absent from standard edge benchmarks.
2. Characterizes how these signals degrade under progressive resolution reduction, identifying which signal degrades first and whether the degradation boundary is device-dependent.
3. Reports per-device operating envelopes stated descriptively from the observed data, without imposing a priori performance targets.

**What this paper does not do:**

- Propose a new detection or tracking method.
- Define universal performance thresholds for crowd monitoring applications.
- Build the downstream analytics pipeline (deferred to a companion study [CIT-self]).

---

## Hypothesis

When YOLO26 operates in tracking mode under progressive resolution reduction, track continuity (measured by identity-switch rate and track fragmentation) degrades at a different rate than detection stability (measured by detection count consistency). Specifically, the hypothesis predicts that identity assignment breaks down at a higher resolution than detection itself, because small reductions in bounding-box precision — insufficient to lose detections entirely — are sufficient to disrupt the IoU-based association that the tracker depends on. If confirmed, this implies that the operational boundary for crowd monitoring applications on edge devices is determined by tracking degradation rather than detection degradation, and that standard detection-only benchmarks overestimate the viable resolution range for tracking-dependent pipelines.

---

## Method Section (as it would appear in the paper)

### II-A. Perception Pipeline

The perception layer employs YOLO26 [CIT] in tracking mode, which performs object detection and multi-object tracking through a single integrated inference call. YOLO26 eliminates post-processing non-maximum suppression through its native end-to-end architecture [CIT], producing final tracked detections without intermediate filtering steps. The integrated tracker maintains persistent identity assignment across consecutive frames using a configurable backend.

Three model variants are evaluated — YOLO26n (nano), YOLO26s (small), and YOLO26m (medium) — spanning the efficiency–accuracy tradeoff relevant to edge deployment. The tracker backend is fixed to ByteTrack [CIT] throughout, isolating the effect of model variant and resolution from tracker configuration differences. ByteTrack performs two-stage association using IoU distance without appearance feature extraction, adding minimal computational overhead to the inference call. Tracker parameters are held constant across all conditions at their Ultralytics defaults: track_buffer = 30 frames, match_thresh = 0.8, track_high_thresh = 0.25, track_low_thresh = 0.1. The target detection class is person (COCO index 0), reflecting the primary object category for pedestrian-level urban monitoring.

### II-B. Edge Device Testbed

The evaluation spans three device classes representing distinct deployment cost points:

| Device | Accelerator | Representative scenario |
|---|---|---|
| Raspberry Pi 5 | CPU only (Cortex-A76) | Ultra-low-cost, high-volume deployment |
| Jetson Nano (4 GB) | 128-core Maxwell GPU | Cost-effective GPU-accelerated inference |
| Jetson Orin Nano | 1024-core Ampere GPU | Performance-oriented edge inference |

Inference is performed using the Ultralytics framework with YOLO26's native runtime: CPU execution on the Raspberry Pi, TensorRT-exported models on the Jetson devices. Each device runs the full tracking-mode pipeline (detection + ByteTrack association) within the same software stack. Power consumption is measured at the board level using a USB power meter during sustained inference, reported as the mean over a 60-second measurement window after thermal stabilization.

### II-C. Test Data

The evaluation employs three static-camera training sequences from the MOTChallenge MOT17 benchmark [CIT:MOT16_benchmark], selected to span a range of pedestrian densities relevant to urban crowd monitoring. MOT17 provides per-frame bounding-box annotations with persistent track identities following a standardized annotation protocol, enabling direct computation of multi-object tracking metrics against ground truth. Sequences from the training partition are used because ground-truth annotations for the test partition are withheld by the benchmark organizers.

| Sequence | Resolution | FPS | Frames (Duration) | Tracks | Density (ped/frame) | Scene description |
|---|---|---|---|---|---|---|
| MOT17-09 | 1920×1080 | 30 | 525 (18 s) | 26 | 10.1 | Pedestrian street, daytime, low angle |
| MOT17-02 | 1920×1080 | 30 | 600 (20 s) | 62 | 31.0 | Open square, moderate elevation |
| MOT17-04 | 1920×1080 | 30 | 1050 (35 s) | 83 | 45.3 | Pedestrian street, nighttime, elevated viewpoint |

The three sequences were selected on the basis of three criteria. First, all employ a static camera, which matches the fixed CCTV assumption of the target deployment scenario and avoids confounding tracker evaluation with ego-motion compensation. Second, they span a density range from sparse (10.1 pedestrians per frame) through moderate (31.0) to dense (45.3), enabling characterization of how perception quality varies with scene complexity. Third, they are captured at 1920×1080 resolution at 30 fps, providing sufficient spatial detail for the resolution degradation experiment (Section II-D).

MOT17 provides three public detection sets per sequence (DPM, Faster-RCNN, SDP), generated by legacy detectors. These are not used as input — YOLO26 performs its own detection — but may serve as contextual baselines for detection count comparison if needed.

For each sequence, YOLO26 tracking-mode output is evaluated against the MOT17 ground truth using standard MOT metrics computed via py-motmetrics [CIT]: MOTA, IDF1, identity switch count (IDSW), and mostly tracked / mostly lost ratios. Only the pedestrian class is evaluated, consistent with the MOT17 evaluation protocol that excludes static persons and non-pedestrian objects from metric computation.

### II-D. Experimental Design

The evaluation is structured around two experiments that directly test the hypothesis. No pass/fail thresholds are imposed on any metric; results are reported descriptively.

**Experiment 1 — Device × Model Profiling.** Each combination of 3 devices × 3 model variants is profiled at native input resolution (640 × 640) on all three MOT17 sequences. For each of the 9 configurations, the following quantities are recorded:

- *Throughput*: sustained frames per second (FPS), measured after a 30-second warm-up, excluding I/O.
- *Latency*: per-frame inference time (ms), reported as median and 95th percentile over the test sequence.
- *Memory*: peak resident memory consumption (MB).
- *Power*: board-level power draw during sustained inference (W).
- *Detection count*: per-frame person detection count, reported as mean and standard deviation.
- *Tracking quality*: MOTA, IDF1, identity switch count (IDSW), and track fragmentation ratio (tracks shorter than 5 frames / total tracks initiated), computed against MOT17 ground-truth annotations using py-motmetrics [CIT].

Tracking quality metrics are reported per sequence to expose the density dependence: MOT17-09 (sparse), MOT17-02 (moderate), MOT17-04 (dense). The profiling matrix is reported as a single table. Its purpose is to establish the baseline perception quality at full resolution for each device–model pair, providing the reference against which Experiment 2 measures degradation.

**Experiment 2 — Resolution Degradation Curves.** For each device, the model variant selected from Experiment 1 — defined as the variant that achieves the highest throughput while producing non-trivial tracking output (confirmed track count > 0 for the majority of frames) — is subjected to progressive resolution reduction from 640 × 640 to 320 × 320 in 64-pixel steps (6 levels).

At each resolution, three degradation signals are measured relative to the full-resolution (640 × 640) output of the same device–model pair:

1. *Detection stability*: mean absolute deviation of per-frame detection count relative to the 640 × 640 baseline. This quantifies whether the detector continues to find approximately the same number of objects as resolution decreases.
2. *Track continuity*: identity switch rate (IDSW per minute of video) and track fragmentation ratio. These quantify whether the tracker maintains stable identities — the prerequisite for any downstream per-individual motion analysis.
3. *Spatial precision*: mean footpoint displacement (pixels, at the reduced resolution) between matched detections and their 640 × 640 counterparts. This quantifies whether detected positions remain spatially accurate for density estimation.

All three signals are computed on each MOT17 sequence independently and plotted as a function of resolution on a common normalized scale. The hypothesis is tested by comparing the resolution at which each signal departs from its full-resolution value: if track continuity degrades at a higher resolution than detection stability, the hypothesis is supported. Results are reported both aggregated across sequences and per-sequence to assess whether the degradation pattern depends on scene density.

### II-E. Reporting

Results are reported as:

1. A 3 × 3 profiling table (Experiment 1) containing all measured quantities per device–model combination, with per-sequence tracking metrics for the three MOT17 sequences.
2. A per-device degradation figure (Experiment 2) with resolution on the horizontal axis and the three normalized degradation signals as curves. The figure visually identifies, for each device, which signal degrades first and at what resolution.
3. A summary statement per device describing the observed operating envelope — the resolution range within which all three perception signals remain close to their full-resolution values — derived from the degradation curves rather than imposed a priori.

---

## Assumptions and Limitations

The evaluation operates under a fixed tracker configuration (ByteTrack, Ultralytics defaults). Alternative backends (BoT-SORT with appearance re-identification) may shift the degradation boundaries; their evaluation is deferred to future work. The test sequences are drawn from the MOT17 benchmark, which captures outdoor pedestrian scenes in European urban settings. Generalization to other camera geometries, crowd densities, and geographic contexts requires additional validation. The sequences are short (18–35 seconds each), limiting the assessment of long-term tracking stability; this constraint is inherent to the benchmark design and is shared across the MOT17 literature. Power measurements reflect board-level consumption during inference and include system overhead beyond the model itself. The evaluation uses YOLO26's native Ultralytics runtime; alternative inference engines (ONNX Runtime, OpenVINO) may yield different throughput characteristics on the same hardware. The model variant selection for Experiment 2 is device-specific and pragmatic rather than optimal in any formal sense. The MOT17 ground truth annotates pedestrians only; multi-class tracking quality for bicycle and car categories is not evaluated against ground truth in this study.

---

## Implementation Checklist

| Step | Description | Deliverable |
|---|---|---|
| 1 | Install Ultralytics + YOLO26 on all 3 devices; verify `model.track()` runs | Confirmed install log per device |
| 2 | Export TensorRT models for Jetson devices (n, s, m variants) | `.engine` files per variant per device |
| 3 | Download MOT17 training set (5.5 GB); extract frames for MOT17-02, MOT17-04, MOT17-09 | Video frames + GT annotation files in MOT format |
| 4 | Verify GT format compatibility: map MOT17 annotations to py-motmetrics input; confirm class filtering (class=1 pedestrian only, visibility ≥ 0.25) | Validation script confirming correct metric computation |
| 5 | Run Experiment 1: 9 configurations × 3 sequences | Raw CSV: per-frame detections, track IDs, timing |
| 6 | Compute Experiment 1 metrics from raw CSV | Profiling table (Table I in paper) |
| 7 | Select model variant per device for Experiment 2 | Selection log with justification |
| 8 | Run Experiment 2: 3 devices × 6 resolutions × 3 sequences | Raw CSV: per-frame detections, track IDs, timing |
| 9 | Compute degradation signals from Experiment 2 raw data | Degradation figure (Fig. 1 in paper) |
| 10 | Measure power consumption per operating point | Power table or annotation on Fig. 1 |
| 11 | Write operating envelope summary per device | Summary paragraph for Discussion section |

### Reference Code Skeleton

```python
from ultralytics import YOLO
import time

# --- Experiment 1: Profile one device x model combination ---
model = YOLO("yolo26n.pt")  # or yolo26s, yolo26m
# On Jetson: model = YOLO("yolo26n.engine")  # TensorRT export

results_log = []
for frame in video_reader:
    t0 = time.perf_counter()
    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=0.25,
        classes=[0],           # person class only
        imgsz=640,             # vary in Experiment 2
        verbose=False
    )
    t1 = time.perf_counter()

    boxes = results[0].boxes
    if boxes.id is not None:
        track_ids = boxes.id.int().cpu().tolist()
        xyxy = boxes.xyxy.cpu().tolist()
        classes = boxes.cls.int().cpu().tolist()
        confs = boxes.conf.cpu().tolist()

        # Footpoints: bottom-center of each bbox
        footpoints = [
            ((x1 + x2) / 2, y2) for x1, y1, x2, y2 in xyxy
        ]
    else:
        track_ids, xyxy, classes, confs, footpoints = [], [], [], [], []

    results_log.append({
        "frame_id": frame_id,
        "inference_ms": (t1 - t0) * 1000,
        "n_detections": len(track_ids),
        "track_ids": track_ids,
        "footpoints": footpoints,
        "classes": classes,
        "confs": confs,
    })

# Post-process: compute MOTA/IDF1 via py-motmetrics against MOT17 GT
# Post-process: compute IDSW rate, fragmentation ratio
# Post-process: aggregate timing stats (median, p95)
```

### MOT17 Ground Truth Format Reference

MOT17 annotations are provided as CSV files with columns:
`<frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <class>, <visibility>`

Where class=1 indicates pedestrian (the evaluation target). The `conf` field is 0 for ground truth and 1 for ignore regions. Detections with visibility < 0.25 are typically excluded from evaluation. The py-motmetrics library accepts this format directly via `motmetrics.io.loadtxt()`.

### Data Access

MOT17 is freely available under Creative Commons Attribution-NonCommercial-ShareAlike 3.0 License from https://motchallenge.net/data/MOT17/. The full dataset (5.5 GB) or annotation files only (9.7 MB) can be downloaded. The dataset uses the same frame sequences as MOT16, with improved annotations and three public detection sets per sequence.

---

## Page Budget Estimate (IEEE two-column, 10pt)

| Section | Columns |
|---|---|
| Abstract | 0.3 |
| I. Introduction (context, gap, hypothesis, contribution) | 1.2 |
| II. Method (II-A through II-E as above) | 2.5 |
| III. Results (Table I + Fig. 1 + summary) | 2.5 |
| IV. Discussion & Conclusion | 0.8 |
| References (~15 entries) | 0.7 |
| **Total** | **~8 columns = 4 pages** |

The method section above is approximately 900 words excluding tables, fitting within the 2.5-column estimate. The results section must carry one table (~0.5 col), one two-column figure (~1.0 col), and interpretive text (~1.0 col).

---

## Known Limitations and Confounds

### Camera Angle Sensitivity (MOT17 Dataset)

The three MOT17 sequences exhibit different camera angles, which significantly affect tracker performance independent of resolution:

- **Low-angle sequences (MOT17-02, MOT17-09):** Ground-level or low-shoulder camera viewpoints create horizontal occlusion and perspective ambiguity. Pedestrians at the image margins are foreshortened and easily confused with neighbours. MOT17-09 exhibits a pathologically high IDSW baseline (1.81 switches/GT-track at 640px), suggesting that ByteTrack's IoU-based association is strained by this geometry even at native resolution. This sequence is useful for understanding edge-case tracker misconfiguration but should not be treated as representative of typical deployment scenarios.

- **Elevated-angle sequences (MOT17-04, MOT17-13):** Bird's-eye or high-shoulder viewpoints provide clearer spatial separation. MOT17-04 is already loss-dominated at 640px (low IDSW 0.71, high fragmentation 0.35), indicating a different failure regime than MOT17-09.

**Implication for hypothesis:** The hypothesis "track continuity degrades before detection" is measured across these heterogeneous camera angles. MOT17-09's high IDSW baseline and MOT17-04's loss-dominance suggest camera angle should be controlled in future studies. Future work will incorporate MOT20 elevated-angle sequences to separate camera-angle effects from resolution effects.

### Short Sequence Duration

MOT17-09 spans only 18 seconds (525 frames at 30 FPS). A handful of absolute IDSW events (e.g., 2–3 switches) map to a high rate when normalized to switches per GT track, amplifying the appearance of tracker instability. Longer sequences (MOT17-02: 20s, MOT17-04: 35s, MOT20: often 60+ seconds) provide more stable rate estimates.

### Fixed ByteTrack Configuration

All experiments use ByteTrack with fixed parameters (`track_buffer=30, match_thresh=0.8`). These settings are not tuned per-sequence or per-camera-angle. MOT17-09's high baseline IDSW suggests that `match_thresh=0.8` may be overly strict for low-angle geometry. Future work will explore per-sequence tuning and report its effect on the operating envelope.

---

## Key References for Dataset

- Milan, A., Leal-Taixé, L., Reid, I., Roth, S. & Schindler, K. MOT16: A Benchmark for Multi-Object Tracking. arXiv:1603.00831, 2016.
- Leal-Taixé, L., Milan, A., Reid, I., Roth, S. & Schindler, K. MOTChallenge 2015: Towards a Benchmark for Multi-Target Tracking. arXiv:1504.01942, 2015.
- Dendorfer, P. et al. MOT20: A benchmark for multi object tracking in crowded scenes. arXiv:2003.09003, 2020.
