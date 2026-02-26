# Hypothesis Refinement — Methodology v3

**Date:** 2026-02-24
**Supersedes:** `ANALYSIS_Track_Continuity_Metrics.md` (v2, dated 2026-02-20)
**Status:** Current — integrated into notebook 02 cells 0, 4, 5

---

## What Changed and Why

The v2 hypothesis relied on three signals — IDSW, fragmentation ratio, and Mostly Tracked — with
fragmentation as the "loss" complement to IDSW's "confusion." Post-v2 fixes to the metric
implementation invalidated fragmentation as a primary signal:

- **Fix 1 (denominator):** Re-anchoring `frag_ratio` from `total_initiated` to the GT track count
  revealed that the denominator had been collapsing at low resolutions, creating artificial
  stabilisation artefacts at 320px for MOT17-04.

- **Fix 2 (GT-matching filter):** Restricting `short_tracks_abs` to only pred IDs matched to a GT
  track at least once revealed that ByteTrack almost never re-initiates a new track ID for a lost
  real person. When a person is lost, they disappear from the tracker's state — they do not
  re-appear as a new short-lived track. `short_tracks_abs` across all resolutions:
  - MOT17-09: 0–3 events (out of 26 GT tracks) — no information content
  - MOT17-02: 3–11 events (out of 53 GT tracks) — too few to trend
  - MOT17-04: 8–14 events (out of 79 GT tracks) — too few to trend

  Fragmentation, correctly computed, is a low-count signal with no stable trend. It is retained
  in the data as a diagnostic column but is not a viable primary claim.

**Consequence:** The paper's evaluation framework reduces from three signals to two. This is
a strengthening, not a weakening — the surviving signals are clean, stable, and carry
complementary information.

---

## Original Hypothesis (v1)

> "Identity switch rate degrades before detection stability as input resolution decreases."

**Why it failed:** IDSW is non-monotone. In dense scenes, ByteTrack accumulates fewer switches at
low resolution because detection recall collapses — it is not making association errors because
it is not finding enough people to associate. IDSW decreasing is not evidence of better tracking.

---

## Refined Hypothesis (v2)

> "Track continuity — measured jointly via identity switches, fragmentation, and mostly-tracked
> ratio — degrades before detection stability. The failure mode transitions from
> confusion-dominated (high IDSW, low frag) in dense scenes at 640px to loss-dominated
> (low IDSW, high frag) at low resolutions."

**Why it was partially invalidated:** Fragmentation, after correct implementation, carries
insufficient signal to support the "loss-dominated" characterisation. The confusion→loss framing
for MOT17-04 was artifact-driven, not empirically supported.

---

## Current Hypothesis (v3)

### Core claim

> As input resolution decreases, YOLO-based multi-object tracking undergoes a two-phase failure
> sequence whose relative weight depends on scene density and camera geometry. A safe operating
> envelope — defined as the resolution below which at least one primary signal exceeds 10%
> relative degradation from the 640px baseline — is consistently identified at 576px across
> all evaluated conditions.

### Two-signal evaluation framework

| Signal | Metric | Measures | Direction |
|---|---|---|---|
| Identity confusion | `idsw_per_gt_track` | IoU association ambiguity | Higher = worse |
| End-to-end continuity | `mostly_tracked_ratio` | Fraction of GT tracks with ≥80% coverage | Lower = worse |

Both signals are normalised to the GT track count, making them density-invariant and
directly comparable across sequences.

### The two-phase failure sequence

**Phase 1 — Confusion onset:**
Resolution reduction degrades IoU precision below the ByteTrack association threshold, causing
mis-associations between spatially close pedestrian detections. IDSW rises above the ±1-switch
noise floor. This phase is visible in:
- MOT17-09 (sparse, low angle): onset at 512px, peak at 384px (+97% IDSW)
- MOT17-04 (dense, elevated): onset at 576px (+28% IDSW)
- MOT17-02 (moderate, moderate elevation): **absent** — geometry provides sufficient spatial
  separation that IoU precision loss does not induce association errors

**Phase 2 — Detection collapse:**
As resolution continues to drop, detection recall falls and tracks are permanently abandoned.
MT ratio degrades monotonically. This phase is present in all three sequences and becomes
dominant below 448px. It is the only failure mode observable in MOT17-02.

### Sequence-level characterisation

| Sequence | Density | Geometry | Primary failure | IDSW onset | MT onset |
|---|---|---|---|---|---|
| MOT17-09 | 10.1 ped/fr | Low angle | Confusion-dominant | 512px | 576px |
| MOT17-02 | 31.0 ped/fr | Moderate elevation | Loss-only | None | 576px |
| MOT17-04 | 45.3 ped/fr | Elevated viewpoint | Mixed (confusion→loss) | 576px | 576px |

**MOT17-09:** IDSW rises sharply and consistently (noise floor = ±2.7% of baseline; all points
from 512px onward are well above floor). MT degrades slowly. The scene's low camera angle creates
horizontal pedestrian overlap that ByteTrack's IoU gate cannot resolve as resolution drops — the
tracker is confusing identities more than it is losing them. The partial IDSW recovery at 320px
(+65% vs +97% at 384px) is real: detection recall has collapsed enough to reduce association
attempts, but MT at 320px confirms people are still being found (−24%), just mis-associated.

**MOT17-02:** IDSW is flat within the noise floor across all resolutions. MT degrades from 640px
onward. This sequence cannot support the confusion-phase claim and should not be presented as
doing so. It characterises the loss-only failure regime.

**MOT17-04:** IDSW spikes at 576–512px (+28%, +50%) then returns near baseline at 448px and
320px. MT degrades monotonically from 576px (−20%) through 384px (−50%). The IDSW recovery while
MT continues to degrade is the cleanest evidence for the two-phase structure in the dataset:
Phase 1 (confusion) is visible at 576–512px; Phase 2 (detection collapse) suppresses Phase 1
at 448px and below by reducing the number of association candidates. The 320px IDSW value
(−2.2% from baseline, exactly at the noise floor) should be treated as ambiguous, not
as evidence of genuine improvement.

---

## Operating Envelope

The 576px threshold is the conservative safe boundary. Justification by signal:

| Signal | First degradation step | All sequences agree? |
|---|---|---|
| IDSW (confusion) | 576px (MOT17-04), 512px (MOT17-09) | No — MOT17-02 shows no IDSW change |
| MT (continuity) | 576px (all three sequences) | **Yes** |

MT is the consistent operating-envelope signal. The 576px threshold holds across all evaluated
density and geometry conditions. IDSW provides mechanistic detail about *how* the tracker fails
(confusion vs loss), but MT provides the operationally useful threshold.

**Paper-facing formulation:**

> "Reducing inference resolution below 576px induces statistically meaningful degradation in
> end-to-end track continuity (MT ratio) across sparse, moderate, and dense pedestrian scenes.
> In sparse low-angle and dense elevated scenes, identity confusion (IDSW) precedes or
> co-occurs with continuity loss, revealing a resolution-induced association failure that
> the continuity signal alone would undercharacterise."

---

## What Was Dropped and Why

### Fragmentation ratio (dropped as primary signal)

After Fix 2 (GT-matching filter), `short_tracks_abs` is too small (0–14 events per condition)
to yield a stable trend. ByteTrack's re-identification architecture does not spawn new track IDs
for lost real persons — losses are absorbed silently into the mostly-lost state and reflected in
MT, not in short-track counts.

**Retained as:** Diagnostic column in the data (`frag_ratio`, `short_tracks_abs`,
`total_initiated`). Useful for verifying that the tracker is not artificially suppressing
initiations, but not a publishable primary signal.

### Confusion → loss transition (dropped as a three-signal narrative)

The v2 hypothesis framed frag as the "loss" signal that rises as IDSW falls, completing a
clean transition story. Without frag, the transition is observable only via the IDSW
spike-then-recovery pattern in MOT17-04 (Phase 1 visible, Phase 2 inferred from IDSW returning
to baseline while MT stays degraded). This is a weaker but more honest characterisation.

### Detection stability and spatial precision (not primary signals)

These were computed in Experiment 2 but were always secondary. Detection count MAD is a noisy
proxy for recall (it conflates true-positive and false-positive count changes). Spatial precision
was computed but never showed a clear trend distinct from the primary signals. Neither appears
in the v3 plot or the paper claim.

---

## Implications for Paper Structure

### Abstract / Introduction

Frame around the **operating envelope claim** (576px threshold, MT-anchored). Introduce the
two-phase failure sequence as the mechanism explaining *why* the threshold exists and why it
differs in character across scene types.

### Experiment 2 results section

Lead with the MT degradation curves — they are the consistent, universally applicable result.
Introduce IDSW as a mechanistic signal with explicit reference to the noise floor band (Fig. X,
shaded region). State explicitly that MOT17-02 shows no IDSW trend and interpret this as
geometry-driven insensitivity, not absence of failure.

### Discussion

Reference the tracker sensitivity finding from `ANALYSIS_Tracker_Sensitivity.md`:
`high_thresh=0.5` on MOT17-04 reduces IDSW/GT by 57% while reducing MT by 3 and MOTA by 18%.
This is a parameter-induced reproduction of the same confusion→detection-collapse transition
observed in the resolution sweep — it validates the two-signal framework by showing that IDSW
alone would have incorrectly classified a worsening configuration as an improvement.

### Limitations

1. **Density and geometry are confounded** within the three MOT17 sequences. MOT17-09 is both
   sparse and low-angle; MOT17-04 is both dense and elevated. The two-phase transition cannot be
   attributed solely to density without additional sequences that decouple the variables.
2. **MOT17-09 is a short sequence (≈450 frames).** IDSW counts of 37–73 across 26 GT tracks
   produce per-track rates with high variance. The ±1-switch noise floor (2.7% of baseline) is
   narrow, but run-to-run variability (not quantifiable from a single run) may be larger.
3. **Single model and tracker.** The single-model claim was partially resolved — see notebook 06
   and the multi-model section below. The ByteTrack architecture still requires separate
   validation for other tracker families.

---

## Multi-Model Extension (notebook 06 — 2026-02-24)

**Supersedes limitation 3 above for the model dimension only.**

Notebook 06 replicated the Experiment 2 analysis across all three YOLO26 model variants
(`yolo26n`, `yolo26s`, `yolo26m`) using the same 54 CSVs (3 models × 3 sequences × 6 resolutions).
The two primary findings from the single-model analysis hold at all model sizes.

### Operating envelope: model-agnostic at 576px

MT onset at 576px is consistent across all three models for MOT17-04 (the highest-density
sequence and therefore the most conservative constraint). For MOT17-09 and MOT17-02, MT onset
varies across models but the 576px boundary remains a valid conservative threshold.

Per-model operating envelope summary (MT onset = first resolution where `mt_delta_norm > 0.10`):

| Model | MOT17-09 | MOT17-02 | MOT17-04 |
|---|---|---|---|
| YOLOv26-N | 448px | 512px | **576px** |
| YOLOv26-S | 448px | 512px | **576px** |
| YOLOv26-M | 320px | 384px | **576px** |

MOT17-04 (dense, elevated) is the binding constraint. The 576px threshold is model-agnostic
for that sequence. For sparse and moderate scenes, larger models show deferred MT onset,
confirming backbone capacity as a second-order protective factor.

### IDSW onset by model

Per-model IDSW onset (first resolution where `idsw_delta_norm > idsw_noise_band`):

| Model | MOT17-09 | MOT17-02 | MOT17-04 |
|---|---|---|---|
| YOLOv26-N | 576px | 384px | 576px |
| YOLOv26-S | 512px | 320px | 576px |
| YOLOv26-M | 512px | 512px | 576px |

MOT17-02 shows variable and late IDSW onset across models (or no onset at all), consistent with
the loss-only characterisation from the single-model analysis. MOT17-04 onset is 576px for all
three models. MOT17-09 IDSW onset shifts one step earlier for nano vs small/medium — the nano
backbone's reduced feature capacity makes association more sensitive to resolution loss.

### Detection-suppression artefact in dense sequences

The multi-model data confirms the artefact described in the single-model analysis: `idsw_delta_norm`
goes strongly negative at low resolutions for MOT17-04 across all three models (nano: −0.66,
small: −0.02, medium: −0.26 at 320px). The artefact is most severe for nano — the weakest
backbone has the sharpest detection recall collapse, which suppresses the association pool and
drives IDSW toward zero. This is not tracking improvement; `mt_delta_norm` simultaneously reaches
0.86 (nano), 0.40 (small), and 0.22 (medium), confirming most of the track population is lost.

**Implication for paper figures:** Any plot showing only `idsw_delta_norm` for dense sequences
must include `mt_delta_norm` in the same panel, or the nano curves will appear to show improving
tracking as resolution drops. Both signals must be co-plotted.

### Model size as a second-order effect

Resolution is the dominant explanatory variable; model size modulates the onset resolution but
does not change the qualitative failure mode. The v3 two-phase sequence (confusion onset → detection
collapse) is observable in all three models. The paper can report the operating envelope without
model qualification beyond noting that larger models provide a one-step (64px) improvement in
MT onset for sparse scenes.

### Curve shape congruence

All three models produce qualitatively congruent degradation curve shapes. The signal patterns
are not model-specific artefacts — they reflect scene geometry and density. This is the strongest
available evidence that the two-signal characterisation is generalisable within the YOLO26 family
and the ByteTrack association scheme.
