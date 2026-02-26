# Paper 1 — IEEE Embedded Systems Letters (4 pages)

**Version:** v3
**Date:** 2026-02-24
**Supersedes:** `methodology_v2.md`
**Changes from v2:** See changelog at the end of this document.

---

## Working Title

Characterizing YOLO26 Tracking-Mode Perception for Urban Crowd Monitoring on Edge Devices

---

## Scope and Contribution Statement

This letter characterizes the operating envelope of YOLO26 in integrated tracking mode for urban crowd monitoring applications on resource-constrained edge devices. The contribution is not a detection benchmark — YOLO26's detection accuracy on standard datasets is established by its authors [CIT] — but rather a systematic characterization of how perception quality relevant to downstream crowd analytics degrades across device classes, model variants, and operating configurations. Where prior edge-deployment evaluations of YOLO variants report detection accuracy and throughput in isolation [CIT], this work evaluates perception through the lens of tracking stability: the signal property that determines whether downstream spatiotemporal analysis (density estimation, motion characterization) can operate reliably on the detector's output.

**What this paper does:**

1. Profiles YOLO26 tracking-mode inference across four device classes and three model variants, measuring both conventional metrics (throughput, latency, power) and tracking-specific metrics (identity switch rate and Mostly Tracked ratio) that are absent from standard edge benchmarks.
2. Characterizes how perception quality degrades under progressive resolution reduction across all model variants, identifying a consistent operating-envelope boundary at 576 px and documenting the density- and geometry-dependent failure modes that explain why the boundary exists.
3. Reports per-device, per-variant operating envelopes derived descriptively from the observed data, without imposing a priori performance targets, as a reference for deployment configuration decisions.

**What this paper does not do:**

- Propose a new detection or tracking method.
- Define universal performance thresholds for crowd monitoring applications.
- Build the downstream analytics pipeline (deferred to a companion study [CIT-self]).

---

## Framing Note

Earlier formulations of this paper advanced a falsifiable hypothesis — that track continuity degrades at a higher resolution than detection stability — as the primary contribution. Empirical evaluation across three MOT17 sequences and multiple model variants revealed that the degradation ordering is density- and geometry-dependent, and that IDSW alone is an insufficient proxy for tracking failure due to the detection-suppression artefact in dense scenes: at low resolutions, detection recall collapse reduces the association pool, driving IDSW toward zero even as most of the track population is simultaneously lost. The contribution has accordingly been reframed as an operating envelope characterization anchored in two complementary signals — identity confusion (IDSW/GT-track) and end-to-end track continuity (MT ratio) — whose joint interpretation correctly characterizes both failure modes. This framing is both more accurate with respect to what the data supports and more directly useful to practitioners selecting device–model–resolution configurations for deployment.

---

## Method Section (as it would appear in the paper)

### II-A. Perception Pipeline

The perception layer employs YOLO26 [CIT] operating in integrated tracking mode, wherein object detection and multi-object identity assignment are executed through a single inference call. YOLO26 eliminates post-processing non-maximum suppression through its native end-to-end architecture [CIT], producing tracked detections without intermediate filtering stages. The integrated tracker maintains persistent identity assignment across consecutive frames through a configurable association backend.

Three model variants are evaluated — YOLO26n (nano), YOLO26s (small), and YOLO26m (medium) — spanning the efficiency–accuracy trade-off space relevant to heterogeneous edge deployment. All three variants are evaluated across all experimental conditions rather than selecting a single variant per device, as the characterization objective is to establish the operating envelope across the full model-size axis rather than to identify a deployment-optimal configuration. The tracker backend is fixed to ByteTrack [CIT] throughout all conditions, isolating the effects of model variant, input resolution, and device class from tracker configuration differences. ByteTrack performs two-stage association using IoU distance without appearance feature extraction, contributing minimal computational overhead beyond the detection inference. Tracker parameters are held constant at Ultralytics framework defaults across all conditions: `track_buffer` = 30 frames, `match_thresh` = 0.8, `track_high_thresh` = 0.25, `track_low_thresh` = 0.1. The target detection class is restricted to person (COCO index 0), consistent with the pedestrian-centric scope of urban crowd monitoring.

---

### II-B. Edge Device Testbed

The evaluation spans four device classes representing distinct deployment cost and computational capability points:

| Device | Accelerator | Representative deployment scenario |
|---|---|---|
| Raspberry Pi 4 (4 GB) | CPU only (Cortex-A72) | Legacy low-cost fixed installation |
| Raspberry Pi 5 (8 GB) | CPU only (Cortex-A76) | Current-generation ultra-low-cost deployment |
| Jetson Nano (4 GB) | 128-core Maxwell GPU | Cost-effective GPU-accelerated inference |
| Arduino Portenta H7 | Dual-core M7/M4 + Vision Shield | Microcontroller-class ultra-edge inference |

The four devices span three orders of computational magnitude, from the microcontroller-class Arduino Portenta H7 through the general-purpose ARM CPU devices (Raspberry Pi 4 and Pi 5) to the GPU-accelerated Jetson Nano. This range deliberately encompasses deployment tiers that differ not only in throughput but in software ecosystem, inference runtime availability, and power envelope.

Inference is executed using the Ultralytics framework on the Raspberry Pi and Jetson devices. The Arduino Portenta H7 operates under a constrained inference runtime (TensorFlow Lite Micro or equivalent); model compatibility and precision constraints for this device are reported as characterization findings rather than controlled variables. TensorRT-exported models are employed on the Jetson Nano. Each device executes the complete tracking-mode pipeline — encompassing detection and ByteTrack association — within an identical software stack wherever runtime compatibility permits; deviations are documented per device. Power consumption is measured at the board level using a USB power meter during sustained inference, reported as the mean over a 60-second measurement window following thermal stabilisation.

---

### II-C. Test Data

The evaluation employs three static-camera training sequences from the MOTChallenge MOT17 benchmark [CIT:MOT16_benchmark], selected to span a range of pedestrian densities and scene geometries representative of urban monitoring deployments. MOT17 provides per-frame bounding-box annotations with persistent track identities following a standardised annotation protocol, enabling direct computation of multi-object tracking metrics against ground truth. Training partition sequences are employed exclusively, as ground-truth annotations for the test partition are withheld by the benchmark organisers.

Ground-truth annotations are filtered following the standard MOTChallenge evaluation protocol: pedestrian-class instances (class_id = 1) with per-frame visibility ≥ 0.25 are retained; rows with conf = 0 mark ignore/distractor regions in the MOT17 format and are excluded. Valid annotated instances carry conf = 1. Reported per-sequence densities reflect this evaluated subset and consequently differ from the advertised benchmark figures, which enumerate all annotated instances prior to visibility filtering.

| Sequence | Resolution | FPS | Frames (Duration) | GT Tracks | Evaluated density | Scene description |
|---|---|---|---|---|---|---|
| MOT17-09 | 1920×1080 | 30 | 525 (18 s) | 26 | 10.1 ped/fr | Pedestrian street, daytime, low angle |
| MOT17-02 | 1920×1080 | 30 | 600 (20 s) | 53 | 31.0 ped/fr | Open square, moderate elevation |
| MOT17-04 | 1920×1080 | 30 | 1050 (35 s) | 79 | 45.3 ped/fr | Pedestrian street, nighttime, elevated viewpoint |

Sequence selection was governed by three criteria. First, all three sequences employ a static, fixed camera, consistent with the stationary CCTV infrastructure assumed by the target deployment scenario and precluding confounds arising from ego-motion compensation. Second, the sequences span a density range from sparse through moderate to dense, enabling characterisation of how perception quality varies with scene complexity across operationally relevant conditions. Third, all sequences are captured at 1920×1080 at 30 fps, providing sufficient spatial resolution for the progressive resolution reduction experiment described in Section II-D.

It should be noted that the three sequences differ simultaneously in pedestrian density and camera elevation angle. These two variables cannot be fully decoupled within the available MOT17 static-camera sequences; this constraint is reflected in the reporting, which avoids attributing degradation pattern differences to density alone.

---

### II-D. Experimental Design

The evaluation is structured around two complementary experiments. No pass/fail thresholds are imposed on any metric; all results are reported descriptively, and the characterisation of operating envelopes is derived from the observed data rather than from a priori performance targets.

**Experiment 1 — Device × Model Profiling.** Every combination of 4 devices × 3 model variants is profiled at native input resolution (640 × 640) across all three MOT17 sequences, yielding a 12-configuration profiling matrix. The following quantities are recorded for each configuration:

- *Throughput*: sustained frames per second (FPS), measured following a 30-second warm-up period, excluding I/O operations.
- *Latency*: per-frame inference time (ms), reported as median and 95th percentile over the complete test sequence.
- *Memory*: peak resident memory consumption (MB).
- *Power*: board-level power draw during sustained inference (W), measured as described in Section II-B.
- *Detection count*: per-frame person detection count, reported as mean and standard deviation across frames.
- *Tracking quality*: MOTA, IDF1, identity switch count (IDSW), and Mostly Tracked (MT) ratio, computed against MOT17 ground-truth annotations via py-motmetrics [CIT].

Tracking quality metrics are disaggregated by sequence to expose density and geometry dependence. The profiling matrix establishes full-resolution perception quality baselines for each device–model pair, against which Experiment 2 quantifies degradation. Model variants exhibiting near-zero track initiation across the majority of frames for a given sequence are flagged as operating below a functional tracking threshold; such configurations are retained in the profiling table with appropriate annotation rather than excluded, as the characterisation of model-size limitations at specific device–scene combinations constitutes a finding in itself.

**Experiment 2 — Resolution Degradation Characterisation.** All three model variants are subjected independently to progressive resolution reduction from 640 × 640 to 320 × 320 in 64-pixel steps, yielding six resolution levels per variant. This experiment is conducted on each device and evaluated against all three MOT17 sequences, with MOT17-04 serving as the primary analytical focus given its combination of elevated viewpoint, high crowd density, and functional baseline tracking quality across multiple model variants. Results for MOT17-02 and MOT17-09 are presented as supporting characterisation of density and geometry dependence.

At each resolution level, a fresh model instance is instantiated to ensure complete tracker state reinitialisation, preventing residual association state from prior resolution levels from confounding the measurement. Two primary degradation signals are measured relative to the full-resolution (640 × 640) output of the same device–model pair:

1. *Identity confusion*: IDSW normalised by GT track count (`idsw_per_gt_track`), quantifying the rate at which the tracker reassigns identity to previously established tracks. This signal captures the confusion failure mode, wherein degraded bounding-box localisation produces IoU ambiguity sufficient to disrupt correct association. An important interpretation caveat applies to dense sequences: at severely reduced resolutions, detection recall collapse suppresses the active track population, reducing the association pool and driving raw IDSW counts toward zero even as tracking quality degrades. This artefact is identified by examining the two signals jointly — `idsw_per_gt_track` falling while `mostly_tracked_ratio` is simultaneously degrading indicates detection-pool suppression, not genuine tracking improvement. A ±1-switch noise floor band (±(1/N_GT) / baseline_IDSW) is computed per sequence to distinguish meaningful IDSW changes from counting artefacts.

2. *End-to-end track continuity*: fraction of GT tracks for which the pipeline correctly covers ≥ 80% of annotated frames (`mostly_tracked_ratio`), providing a holistic track persistence measure that captures both confusion and loss failure modes. This signal degrades monotonically with resolution across all three sequences and all three model variants, making it the consistent operating-envelope signal.

Both signals are normalised to the GT track count, making them density-invariant and directly comparable across sequences of differing crowd density. They are plotted together as relative change from the 640 baseline on a common axis, with absolute values reported in accompanying tables.

The joint behaviour of these two signals across the resolution sweep characterises the failure mode structure: in sparse low-angle scenes (MOT17-09), identity confusion onset precedes substantial continuity loss, with IDSW rising sharply from 512 px while MT degrades gradually; in dense elevated scenes (MOT17-04), confusion onset at 576–512 px is followed by detection-collapse suppression below 448 px, during which IDSW recovers toward baseline while MT continues to degrade monotonically; in moderate-density scenes (MOT17-02), IDSW remains within the noise floor across all resolutions while MT degrades from 576 px, characterising a loss-only failure regime. The resolution at which MT first exceeds a 10% relative degradation from baseline is identified per model variant and sequence, enabling derivation of the operating envelope boundary as a function of both resolution and model size.

Track fragmentation statistics (`frag_ratio`, `short_tracks_abs`, `total_initiated`) are computed and logged at each resolution level as diagnostic data. Post-implementation analysis established that ByteTrack almost never re-initiates a new track ID for a lost real person — losses are absorbed into the tracker's mostly-lost state and captured by the MT signal rather than appearing as short-track initiations. With GT-matched short-track counts ranging from 0–14 events per condition (out of 26–79 GT tracks), fragmentation carries insufficient signal for a primary claim and is retained exclusively for denominator-collapse diagnosis.

---

### II-E. Reporting

Results are reported as:

1. A 4 × 3 profiling table (Experiment 1) enumerating all measured quantities per device–model combination, with per-sequence tracking metrics disaggregated across the three MOT17 sequences. Model variants operating below a functional tracking threshold are annotated accordingly.

2. Per-device, per-variant degradation figures (Experiment 2) with input resolution on the horizontal axis and the two primary normalised degradation signals as curves — identity confusion (solid, with ±1-switch noise floor shading) and MT continuity loss (dotted). Both signals are co-plotted in every panel; presenting only IDSW for dense sequences would misrepresent the failure mode by making detection-pool suppression appear as tracking improvement. The MOT17-04 figure constitutes the primary analytical exhibit; results for MOT17-02 and MOT17-09 are presented in supporting figures.

3. A descriptive operating envelope summary per device–model pair, expressed as the resolution at which MT ratio first degrades more than 10% from the full-resolution reference. Empirically, this threshold is 576 px across all three model variants for the densest sequence (MOT17-04) and is therefore reported as a model-agnostic boundary. For sparse and moderate scenes, larger model variants show deferred MT onset by up to two resolution steps (128 px), characterising backbone capacity as a second-order modulating factor.

---

### II-F. Assumptions and Limitations

The evaluation employs a fixed tracker configuration (ByteTrack, Ultralytics defaults) throughout. Alternative backends incorporating appearance-based re-identification, such as BoT-SORT [CIT], may alter degradation boundaries and are deferred to future work. The MOT17 benchmark sequences represent outdoor European urban pedestrian scenes; generalisation to other geographic contexts, camera geometries, and crowd behavioural patterns requires supplementary validation. Sequence durations are constrained by benchmark design to 18–35 seconds, limiting assessment of long-term tracking stability — a constraint shared uniformly across the MOT17 literature.

Power measurements reflect board-level consumption including system overhead beyond the inference process itself. The YOLO26 model variants are evaluated using COCO-pretrained weights without scene-specific fine-tuning; domain shift between the COCO training distribution and the elevated CCTV perspective of MOT17-04 constitutes a recognised limitation, manifesting as reduced detection recall for pedestrians at steep viewing angles or under heavy mutual occlusion. This recall floor is acknowledged as a scene-geometry constraint rather than a pipeline deficiency.

The three evaluation sequences differ simultaneously in pedestrian density and camera elevation angle; these variables cannot be fully decoupled within the available MOT17 static-camera sequences, and degradation pattern differences across sequences therefore cannot be attributed exclusively to density. Ground-truth annotations with visibility below 0.25 are excluded from evaluation per standard MOTChallenge protocol; reported densities and metric values reflect this evaluated subset. The Arduino Portenta H7 operates under runtime and precision constraints that may preclude execution of the full YOLO26m and YOLO26s variants; findings for this device are reported as characterisation of the microcontroller-class deployment boundary rather than as fully controlled comparisons.

---

## Implementation Checklist

| Step | Description | Deliverable |
|---|---|---|
| 1 | Install Ultralytics + YOLO26 on all 4 devices; verify `model.track()` runs; document runtime constraints for Arduino Portenta H7 | Confirmed install log per device |
| 2 | Export TensorRT models for Jetson Nano (n, s, m variants); export compatible runtime format for Portenta H7 | `.engine` files (Jetson); quantised model files (Portenta) |
| 3 | Download MOT17 training set (5.5 GB); extract frames for MOT17-02, MOT17-04, MOT17-09 | Video frames + GT annotation files in MOT format |
| 4 | Verify GT format compatibility: map MOT17 annotations to py-motmetrics input; confirm class filtering (class_id=1, conf=1 to retain valid annotations, visibility ≥ 0.25) and 1-indexed frame ID alignment | Validation script confirming correct metric computation |
| 5 | Run Experiment 1: 12 configurations × 3 sequences | Raw CSV: per-frame detections, track IDs, timing |
| 6 | Compute Experiment 1 metrics from raw CSV | Profiling table (Table I in paper) |
| 7 | Run Experiment 2: 4 devices × 3 variants × 6 resolutions × 3 sequences; re-instantiate model per resolution level | Raw CSV: per-frame detections, track IDs, timing |
| 8 | Compute two primary degradation signals (IDSW/GT-track, MT ratio); log fragmentation diagnostic columns per resolution | Degradation figures (Fig. 1 in paper) |
| 9 | Measure power consumption per operating point | Power annotations on profiling table |
| 10 | Write operating envelope summary per device–model pair | Summary paragraph for Discussion section |

---

### Reference Code Skeleton

```python
from ultralytics import YOLO
import time

# --- Experiment 2: Resolution sweep for one device x model combination ---
for resolution in [640, 576, 512, 448, 384, 320]:
    # Re-instantiate per resolution to ensure clean tracker state
    model = YOLO("yolo26n.pt")  # or yolo26s, yolo26m
    # On Jetson: model = YOLO("yolo26n.engine")

    video_reader.reset()  # reset to frame 1 of sequence
    results_log = []

    for frame_idx, frame in enumerate(video_reader):
        t0 = time.perf_counter()
        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=0.25,
            classes=[0],           # person class only
            imgsz=resolution,
            verbose=False
        )
        t1 = time.perf_counter()

        boxes = results[0].boxes
        if boxes.id is not None:
            track_ids = boxes.id.int().cpu().tolist()
            xyxy      = boxes.xyxy.cpu().tolist()
            confs     = boxes.conf.cpu().tolist()
            footpoints = [((x1+x2)/2, y2) for x1, y1, x2, y2 in xyxy]
        else:
            track_ids, xyxy, confs, footpoints = [], [], [], []

        results_log.append({
            "frame_id":      frame_idx + 1,   # 1-indexed to match MOT17 GT
            "inference_ms":  (t1 - t0) * 1000,
            "n_detections":  len(track_ids),
            "track_ids":     track_ids,
            "footpoints":    footpoints,
            "confs":         confs,
        })

    # Post-process: compute MOTA/IDF1/IDSW via py-motmetrics against MOT17 GT
    # Post-process: compute fragmentation diagnostic columns (retained, not plotted as primary)
    # Post-process: aggregate timing stats (median, p95)
```

---

### MOT17 Ground Truth Format Reference

MOT17 annotations are provided as CSV files with columns:
`<frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <class>, <visibility>`

Where `class=1` indicates pedestrian. The `conf` field uses **1 for valid annotated instances** and **0 for ignore/distractor regions**. The standard evaluation filter retains rows where `class=1`, `conf=1` (valid annotations only), and `visibility >= 0.25`; rows with `conf=0` are excluded as ignore regions. Frame IDs are 1-indexed throughout; hypothesis frame IDs must match this convention for correct accumulator alignment.

Note: py-motmetrics' `motmetrics.distances.iou_matrix` uses `np.asfarray`, which was removed in NumPy 2.0. The IoU distance matrix is computed manually in the reference implementation (`_iou_distance` in `src/benchmark/metrics.py`) to maintain NumPy 2.x compatibility.

---

### Data Access

MOT17 is freely available under Creative Commons Attribution-NonCommercial-ShareAlike 3.0 License from https://motchallenge.net/data/MOT17/. The full dataset (5.5 GB) or annotation files only (9.7 MB) can be downloaded. The dataset uses the same frame sequences as MOT16, with improved annotations and three public detection sets per sequence.

---

## Page Budget Estimate (IEEE two-column, 10pt)

| Section | Columns |
|---|---|
| Abstract | 0.3 |
| I. Introduction (context, gap, contribution) | 1.2 |
| II. Method (II-A through II-F as above) | 2.5 |
| III. Results (Table I + Fig. 1 + summary) | 2.5 |
| IV. Discussion & Conclusion | 0.8 |
| References (~15 entries) | 0.7 |
| **Total** | **~8 columns = 4 pages** |

The method section above is approximately 1,200 words excluding tables and code, fitting within the 2.5-column estimate. The results section must carry one table (~0.5 col), one primary two-column figure for MOT17-04 (~1.0 col), and interpretive text (~1.0 col). Supporting figures for MOT17-02 and MOT17-09 may be deferred to supplementary material if the page budget is tight. The model-comparison figure from notebook 06 (parallel coordinates or per-model degradation curves) may substitute for one supporting figure.

---

## Key References for Dataset

- Milan, A., Leal-Taixé, L., Reid, I., Roth, S. & Schindler, K. MOT16: A Benchmark for Multi-Object Tracking. arXiv:1603.00831, 2016.
- Leal-Taixé, L., Milan, A., Reid, I., Roth, S. & Schindler, K. MOTChallenge 2015: Towards a Benchmark for Multi-Target Tracking. arXiv:1504.01942, 2015.
- Dendorfer, P. et al. MOT20: A benchmark for multi object tracking in crowded scenes. arXiv:2003.09003, 2020.

---

## Changelog: v2 → v3

### Substantive corrections

| Location | v2 text | v3 correction | Reason |
|---|---|---|---|
| Scope §1 bullet | "track fragmentation" listed as a primary metric | Removed; two metrics only (IDSW, MT) | Fragmentation carries 0–14 events per condition — insufficient for a primary claim |
| Scope §2 bullet | "transition from identity-confusion to track-loss failure modes" implied fragmentation as the loss signal | Reframed: transition characterised via IDSW spike-then-recovery + MT co-plot | Frag counts too low; the transition is observable through IDSW recovery while MT stays degraded |
| Framing Note | No mention of detection-suppression artefact | Added explicit description of the artefact (IDSW falls in dense scenes as detection pool collapses) | Multi-model data confirmed the artefact across all three variants; must be in the paper framing |
| Exp 2 signal list | Signal 3 = "Track fragmentation" with L_min/total_initiated as primary | Demoted to diagnostic-only paragraph; fragmentation is no longer a numbered primary signal | Implementation fix (GT-match filter + GT-count denominator) showed counts too low to trend |
| Exp 2 signal list | Signal 4 = MT (4th of 4) | MT is now signal 2 of 2; elevated to equal standing with IDSW | MT is the consistent operating-envelope signal across all conditions |
| Exp 2 narrative paragraph | "joint behaviour of identity confusion and track fragmentation … characterises the transition" | Rewritten: transition characterised by IDSW spike-then-recovery in MOT17-04 while MT stays degraded | Fragmentation is not part of the transition evidence |
| Reporting §2 | "four normalised degradation signals" | "two primary normalised degradation signals" with explicit co-plot requirement | Four signals reduced to two primaries; det stability and frag retained diagnostically |
| Reporting §3 | "jointly assessed through identity confusion, fragmentation, and MT ratio" | "MT ratio first degrades more than 10% from the full-resolution reference" | Two-signal framework; 10% threshold from methodology v3 operating envelope definition |
| Checklist step 4 | `conf > 0` as the GT filter condition | `conf == 1` to retain valid annotations | **Bug:** `conf=0` marks ignore regions in MOT17; `conf=1` marks valid GT instances. The code (`mot_gt.py` line 29) correctly uses `conf == 1`; the v2 doc was wrong |
| Checklist step 7 | "total_initiated, short_tracks_abs" in Exp 2 deliverables as primary outputs | Removed from the deliverable description; still computed but framed as diagnostic logging | Consistent with fragmentation demotion |
| Checklist step 8 | "Compute degradation signals; log fragmentation denominator diagnostics" | "Compute two primary degradation signals; log fragmentation diagnostic columns" | Wording aligned with v3 two-signal framework |
| GT format reference | "`conf > 0` … distractor regions" | "`conf=1` for valid annotations, `conf=0` for ignore regions" | Corrected inversion; now matches code and MOT17 spec |
| Reference code skeleton | `_fragmentation_ratio` with `total_initiated` as denominator | Removed; fragmentation function is not part of the primary pipeline skeleton | The implemented version uses GT-count denominator + GT-match filter and is not primary |

### Additions

| Location | Addition | Reason |
|---|---|---|
| Framing Note | Detection-suppression artefact explanation | Multi-model notebook 06 confirmed this is the dominant dense-sequence behaviour |
| Exp 2 narrative | Sequence-level failure mode characterisation (confusion-dominant / loss-only / mixed) | Established by single-model (notebook 02) and confirmed by multi-model (notebook 06) |
| Exp 2 narrative | Noise floor definition: `±(1/N_GT) / baseline_IDSW` | Required to distinguish real IDSW changes from counting artefacts in short sequences |
| Reporting §3 | Model-agnostic 576px boundary statement with second-order model size effect | Multi-model notebook 06: all three models agree at 576px for MOT17-04 |
| Reporting §2 | Explicit co-plot requirement (both signals in every panel) | Nano curves in dense scenes show negative `idsw_delta_norm` at low res; without MT co-plot the panel is actively misleading |
| GT format reference | NumPy 2.x / motmetrics compatibility note | Known issue documented in MEMORY.md; relevant to any reader reproducing the evaluation |
| Page budget | Mention of model-comparison figure (notebook 06) | New figure asset available |
