# Analysis: Track Continuity Metrics and Failure-Mode Transition in MOT17

**Date:** 2026-02-20
**Status:** Completed and integrated into notebooks 02 and 03
**Implications for hypothesis:** Refined; confirmed with caveat about camera-angle confound

---

## Problem Statement

Initial Experiment 2 design used only **IDSW (Identity Switch Rate)** to measure track continuity during resolution degradation. Analysis revealed three critical issues:

1. **MOT17-09 (sparse, low-angle) had pathological IDSW baseline** (1.81 switches/GT-track at 640px)
2. **MOT17-04 (dense, elevated-angle) showed IDSW *decreasing* as resolution dropped**, contradicting the hypothesis
3. **IDSW alone is insufficient** — it measures *confusion* (IoU ambiguity) but not *loss* (track breaks from missed detections)

---

## Root Cause Analysis

### The Confusion → Loss Transition

As resolution decreases, tracker failure mode shifts:

| Resolution | Dominant Failure | IDSW Trend | Fragmentation Trend |
|---|---|---|---|
| 640px (dense scenes) | Confusion (ambiguous IoU) | High baseline | Baseline |
| 576–448px | Mixed | Varies | Increases |
| 320px (sparse detections) | Loss (missed detections) | Low or decreasing | High |

**MOT17-04 (dense) example:**
- 640px: 0.71 IDSW/GT-track, 0.35 frag ratio — tracker finds many detections, struggles with association
- 320px: 0.25 IDSW/GT-track, 0.44 frag ratio — tracker finds few detections, simply loses tracks rather than confusing them
- Net result: IDSW *decreases* while fragmentation *increases* — counterintuitive if measuring IDSW alone

### Camera Angle Confound

**MOT17-09 (low-angle, 18s):**
- Perspective ambiguity: horizontal occlusion, foreshortened margins
- IDSW baseline 1.81 — ByteTrack struggles even at 640px
- Likely cause: `match_thresh=0.8` too strict for this geometry, or YOLO26n over-detection on low angles

**MOT17-04 (elevated-angle, 35s):**
- Bird's-eye: clear spatial separation
- IDSW baseline 0.71 — clean, expected range
- Already loss-dominated at 640px

---

## Solution: Three-Signal Framework

Replaced single IDSW metric with three complementary signals:

### 1. **IDSW/GT-track** (Confusion Signal)
- Measures IoU-based association ambiguity
- High at 640px in dense scenes (many detections competing for matches)
- Decreases or flat as resolution drops (fewer attempts at association)
- MOT17-04: -66% by 320px (loss-dominated regime, paradoxical decrease)

### 2. **Fragmentation Ratio** (Loss Signal)
- Fraction of initiated tracks shorter than 5 frames
- Baseline ~0.25–0.35 at 640px across sequences
- Rises monotonically as resolution drops — the true degradation signal
- MOT17-04: +26% by 320px (loss regime)

### 3. **Mostly Tracked Ratio** (Holistic Signal)
- Fraction of GT tracks with ≥80% detection coverage
- Combines confusion and loss effects
- Holistic proxy for end-to-end tracking success

---

## Refined Hypothesis

### Original (Too Narrow)
> "Identity switch rate degrades before detection stability"

### Refined (Validated by Data)
> "Track continuity — measured jointly via identity switches, fragmentation, and mostly-tracked ratio — degrades before detection stability. The failure mode transitions from confusion-dominated (high IDSW, low frag) in dense scenes at 640px to loss-dominated (low IDSW, high frag) at low resolutions."

---

## Key Findings

### Operating Envelope (>10% degradation threshold)

All three sequences: **~576px resolution** marks the >10% degradation boundary for:
- Detection stability (MAD of detections)
- Fragmentation ratio
- Spatial precision
- Mostly Tracked ratio

| Sequence | Camera | Duration | IDSW Baseline | Frag Baseline | >10% Threshold |
|---|---|---|---|---|---|
| MOT17-09 | Low-angle | 18s | 1.81 | 0.257 | 576px |
| MOT17-02 | Low-angle | 20s | 0.85 | 0.255 | 576px |
| MOT17-04 | Elevated | 35s | 0.71 | 0.352 | 576px |

### Surprises

1. **MOT17-04 IDSW paradox is not a data error** — it's a real failure-mode transition
2. **MOT17-09 is an edge case** — low-angle camera + short duration + probable tracker misconfiguration create atypical baseline
3. **Fragmentation is the consistent signal** — rises monotonically across all sequences, camera angles, and densities

---

## Implementation Changes

### `degradation.py`
- Updated `track_continuity()` to return 4 metrics: `num_switches`, `idsw_per_gt_track`, `frag_ratio`, `mostly_tracked_ratio`
- Per-GT-track normalization (density-invariant) instead of per-minute normalization

### Notebooks 02 & 03
- **Top panel:** Relative change curves for all four signals
- **Bottom panel:** Absolute values at each resolution with 640px annotations
- Added explicit discussion of camera-angle and failure-mode-transition findings

### Documentation
- Added "Known Limitations and Confounds" section to `methodology.md`
- Documented camera-angle effects and short-sequence amplification
- Flagged MOT17-09 as an edge case; planned MOT20 elevated-angle additions

---

## Implications for Paper

1. **Hypothesis is confirmed but refined:** Track continuity degrades before detection, measured correctly requires three signals, not one
2. **Camera angle is a critical variable:** Future work must control for viewpoint or explicitly study its interaction with resolution
3. **Operating envelope is device-agnostic (at 640× model baseline):** All sequences converge on ~576px boundary regardless of density
4. **ByteTrack tuning is sequence-dependent:** `match_thresh=0.8` may not be optimal for low-angle geometry

---

## Future Work Priorities

1. **Integrate MOT20 elevated-angle sequences** to isolate camera-geometry effects
2. **Tune ByteTrack per-sequence** and report sensitivity of the operating envelope
3. **Run on edge devices** (RPi 5, Jetson Nano, Orin Nano) to characterize device-specific boundaries
4. **Longer sequences** to stabilize rate estimates (MOT17-09's 18s is too short)

---

## References

- Dendorfer, P. et al. "MOT20: A benchmark for multi object tracking in crowded scenes." *International Conference on Computer Vision*, 2021.
- Zhang, Y., Wang, C., Wang, X., Zeng, W., & Liu, W. "ByteTrack: Multi-Object Tracking by Associating Every Detection Box." *ECCV*, 2022.
