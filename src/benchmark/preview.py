from __future__ import annotations

import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


class LivePreview:
    """Real-time annotated frame display for development and demo use.

    Draws track IDs and confidence scores onto each frame using a
    deterministic per-track colour derived from the track ID hash.
    Optionally writes the annotated output to an mp4 file.

    Intended for the development machine only; edge devices run headless.
    Guard usage with `if ENABLE_PREVIEW` in notebooks.
    """

    def __init__(
        self,
        window_name: str = "preview",
        scale: float = 0.5,
        save_path: str | Path | None = None,
        fps: float = 30.0,
    ) -> None:
        self._name      = window_name
        self._scale     = scale
        self._save_path = Path(save_path) if save_path else None
        self._fps       = fps
        self._writer: cv2.VideoWriter | None = None

        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    def show(self, frame_bgr: np.ndarray, result) -> bool:
        """Overlay tracking output on frame and display it.

        Args:
            frame_bgr: BGR image array from cv2.imread or video capture.
            result:    Single Ultralytics Results object from model.track().

        Returns:
            False if the user pressed 'q' (caller should stop the loop), True otherwise.
        """
        annotated = self._annotate(frame_bgr, result)

        # Write full-resolution frame before scaling for display
        if self._save_path is not None:
            self._ensure_writer(annotated)
            self._writer.write(annotated)

        h, w = annotated.shape[:2]
        display = cv2.resize(annotated, (int(w * self._scale), int(h * self._scale)))
        cv2.imshow(self._name, display)

        return cv2.waitKey(1) != ord("q")

    def close(self) -> None:
        """Release resources and close the display window."""
        cv2.destroyWindow(self._name)
        if self._writer is not None:
            self._writer.release()
            self._writer = None

    # ── Private helpers ───────────────────────────────────────────────────────

    def _annotate(self, frame: np.ndarray, result) -> np.ndarray:
        """Draw bounding boxes and track labels onto a copy of frame."""
        out   = frame.copy()
        boxes = result.boxes

        if boxes is None or boxes.id is None:
            return out

        ids   = boxes.id.int().cpu().tolist()
        xyxys = boxes.xyxy.cpu().tolist()
        confs = boxes.conf.cpu().tolist()

        for tid, (x1, y1, x2, y2), conf in zip(ids, xyxys, confs):
            color = _id_color(tid)
            cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            label = f"id:{tid} {conf:.2f}"
            cv2.putText(
                out, label,
                (int(x1), max(int(y1) - 5, 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
            )

        return out

    def _ensure_writer(self, frame: np.ndarray) -> None:
        """Initialise the VideoWriter on the first frame (size unknown until then)."""
        if self._writer is not None:
            return
        self._save_path.parent.mkdir(parents=True, exist_ok=True)
        h, w = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(str(self._save_path), fourcc, self._fps, (w, h))


def render_tracking_video(
    csv_path:  Path,
    img_dir:   Path,
    out_path:  Path,
    fps:       float = 30.0,
) -> None:
    """Replay a raw inference CSV as an annotated mp4.

    Reads bounding boxes and track IDs stored in the CSV produced by
    run_sequence(), overlays them on the original sequence frames, and
    writes the result to an mp4 file.  No model re-inference is performed.

    Box colours are deterministic per track ID (same palette as LivePreview).
    The output video is full-resolution (matches the source images), making it
    suitable for side-by-side comparison of resolution conditions.

    Args:
        csv_path: Raw inference CSV from results/raw/.
        img_dir:  Sequence img1/ directory containing the source JPEG frames.
        out_path: Destination mp4 file path (parent directories created if needed).
        fps:      Output frame rate; should match the sequence frameRate from seqinfo.ini.
    """
    df = pd.read_csv(csv_path)
    frame_paths = sorted(img_dir.glob("*.jpg"))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer: cv2.VideoWriter | None = None

    for img_path in frame_paths:
        # frame_id in CSV is 1-indexed; filename is zero-padded integer
        frame_id  = int(img_path.stem)
        frame_bgr = cv2.imread(str(img_path))

        row = df[df["frame_id"] == frame_id]
        if not row.empty:
            track_ids = json.loads(row.iloc[0]["track_ids"])
            bboxes    = json.loads(row.iloc[0]["bboxes_xyxy"])
            confs     = json.loads(row.iloc[0]["confs"])
            for tid, (x1, y1, x2, y2), conf in zip(track_ids, bboxes, confs):
                color = _id_color(tid)
                cv2.rectangle(frame_bgr, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(
                    frame_bgr, f"id:{tid} {conf:.2f}",
                    (int(x1), max(int(y1) - 5, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
                )

        if writer is None:
            h, w   = frame_bgr.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

        writer.write(frame_bgr)

    if writer is not None:
        writer.release()


def transcode_h264(src: Path, dst: Path) -> None:
    """Re-encode an mp4v file to H.264 (QuickTime / browser compatible).

    OpenCV's VideoWriter only produces mp4v (MPEG-4 Part 2) on this platform,
    which QuickTime Player and most web browsers reject.  This function shells
    out to ffmpeg with libx264 to produce a standards-compliant H.264 baseline
    stream that plays everywhere.

    The source file is preserved; dst is overwritten if it already exists.

    Args:
        src: Input mp4 written by render_tracking_video (mp4v codec).
        dst: Output path for the H.264 re-encoded file.

    Raises:
        RuntimeError: If ffmpeg exits with a non-zero status.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-i", str(src),
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",           # visually lossless at reasonable file size
        "-pix_fmt", "yuv420p",  # QuickTime requires 4:2:0 chroma subsampling
        "-movflags", "+faststart",  # moov atom at front for progressive playback
        str(dst),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{result.stderr}")


def _id_color(track_id: int) -> tuple[int, int, int]:
    """Deterministic BGR colour derived from a Knuth multiplicative hash of track_id."""
    h = (track_id * 2_654_435_761) & 0xFF_FFFF
    return (h & 0xFF, (h >> 8) & 0xFF, (h >> 16) & 0xFF)
