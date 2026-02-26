import configparser
from pathlib import Path

import pandas as pd

from benchmark.config import MIN_VISIBILITY, PEDESTRIAN_CLASS_GT


# MOT17 gt.txt column layout (1-indexed in the spec, 0-indexed here)
_GT_COLS = ["frame_id", "track_id", "x", "y", "w", "h", "conf", "class_id", "visibility"]


def load_gt(seq_dir: Path) -> pd.DataFrame:
    """Pedestrian ground-truth annotations for one MOT17 sequence.

    Applies the MOTChallenge evaluation filter: pedestrian class only
    (class_id == 1), visibility above threshold, and valid annotations
    only (conf == 1; conf == 0 marks ignore/distractor regions in MOT17).

    Returns a DataFrame with columns: frame_id, track_id, x, y, w, h.
    Coordinates are in pixels, top-left origin, width/height dimensions.
    """
    gt_path = seq_dir / "gt" / "gt.txt"
    df = pd.read_csv(gt_path, header=None, names=_GT_COLS)

    mask = (
        (df["class_id"] == PEDESTRIAN_CLASS_GT)
        & (df["visibility"] >= MIN_VISIBILITY)
        & (df["conf"] == 1)
    )
    return df.loc[mask, ["frame_id", "track_id", "x", "y", "w", "h"]].reset_index(drop=True)


def load_seqinfo(seq_dir: Path) -> dict:
    """Sequence metadata from seqinfo.ini.

    Returns a flat dict with keys: name, frameRate, seqLength, imWidth, imHeight.
    Values are cast to int except name (str).
    """
    ini_path = seq_dir / "seqinfo.ini"
    parser = configparser.ConfigParser()
    parser.read(ini_path)
    sec = parser["Sequence"]
    return {
        "name":       sec["name"],
        "frameRate":  int(sec["frameRate"]),
        "seqLength":  int(sec["seqLength"]),
        "imWidth":    int(sec["imWidth"]),
        "imHeight":   int(sec["imHeight"]),
    }
