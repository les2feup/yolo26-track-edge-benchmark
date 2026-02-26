"""
Collect calibration frames for Hailo DFC quantisation.

Samples N frames evenly from each MOT17 sequence and copies them to
edge/calib_imgs/. These are unlabelled JPEG images used only during
the Hailo optimize step — no annotation is required.

Usage:
    python edge/collect_calib.py            # defaults: 64 frames per sequence
    python edge/collect_calib.py --n 128    # 128 frames per sequence
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

_ROOT      = Path(__file__).parents[1]
_DATA_ROOT = _ROOT / "data" / "MOT17" / "train"
_SEQUENCES = ["MOT17-09", "MOT17-02", "MOT17-04"]
_SEQ_SUFFIX = "SDP"
_OUT_DIR   = _ROOT / "edge" / "calib_imgs"


def collect(n_per_seq: int) -> None:
    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    for seq in _SEQUENCES:
        img_dir = _DATA_ROOT / f"{seq}-{_SEQ_SUFFIX}" / "img1"
        frames  = sorted(img_dir.glob("*.jpg"))
        if not frames:
            print(f"[warn] no frames found in {img_dir}")
            continue

        # Even stride across the full sequence length
        step     = max(1, len(frames) // n_per_seq)
        selected = frames[::step][:n_per_seq]

        for src in selected:
            dst = _OUT_DIR / f"{seq}_{src.name}"
            shutil.copy(src, dst)

        print(f"[{seq}] copied {len(selected)} / {len(frames)} frames → {_OUT_DIR}")

    total = len(list(_OUT_DIR.glob("*.jpg")))
    print(f"\nTotal calibration images: {total} → {_OUT_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect Hailo calibration images from MOT17")
    parser.add_argument("--n", type=int, default=64,
                        help="Frames to sample per sequence (default: 64)")
    args = parser.parse_args()
    collect(args.n)


if __name__ == "__main__":
    main()
