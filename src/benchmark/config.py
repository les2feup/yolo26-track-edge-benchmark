from pathlib import Path

# ── Repository root relative to this file (src/benchmark/config.py → root) ──
_ROOT = Path(__file__).parents[2]

# ── Dataset and results paths ─────────────────────────────────────────────────
DATA_ROOT   = _ROOT / "data" / "MOT17" / "train"
RESULTS_RAW = _ROOT / "results" / "raw"

# ── MOT17 sequences evaluated, ordered sparse → medium → dense ───────────────
# Three static-camera sequences spanning pedestrian density and camera geometry.
# MOT17-09: pedestrian street, daytime, low angle,       density ≈ 10.1 ped/frame (sparse)
# MOT17-02: open square, moderate elevation,             density ≈ 31.0 ped/frame (medium)
# MOT17-04: pedestrian street, nighttime, elevated view, density ≈ 45.3 ped/frame (dense)
# Note: sequences differ in both density and camera elevation; these cannot be
# fully decoupled within the available MOT17 static-camera sequences.
SEQUENCES = ["MOT17-09", "MOT17-02", "MOT17-04"]

# ── Suffix used when constructing sequence directory paths ────────────────────
# All suffix variants (DPM/FRCNN/SDP) share identical gt/gt.txt and img1/.
SEQ_SUFFIX = "SDP"

# ── Baseline resolution for degradation normalisation ────────────────────────
# Model variants and resolution sweep are device-specific — defined in each
# device profile (edge/profiles/*.yaml) and the desktop fallback in device_profile.py.
IMGSZ_BASE  = 640

# ── Tracker and inference configuration (fixed across all conditions) ─────────
TRACKER = "bytetrack.yaml"
CONF    = 0.25
CLASSES = [0]              # COCO person index

# ── MOT17 ground-truth filtering criteria ────────────────────────────────────
# class==1 selects pedestrians; visibility>=0.25 excludes occluded annotations;
# conf==0 selects valid annotations (conf==1 marks ignore regions).
PEDESTRIAN_CLASS_GT  = 1
MIN_VISIBILITY       = 0.25

# ── Track fragmentation threshold ────────────────────────────────────────────
# Tracks shorter than this are counted as fragmented (paper definition: <5 frames).
MIN_TRACK_LEN_FRAMES = 5

# ── Warm-up frames discarded before timing measurements begin ────────────────
WARMUP_FRAMES = 30
